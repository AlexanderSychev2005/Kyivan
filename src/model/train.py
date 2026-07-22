"""
Kyivan Model Training Pipeline.

This script orchestrates the end-to-end training process for the multi-task
Kyivan model. It handles data loading, custom multi-task loss computation,
evaluation on dynamically masked text (Test A) and historical lacunae (Test B),
and generates detailed CSV prediction reports for linguists.

Key components:
1. `KyivanTrainer`: A custom HuggingFace Trainer that overrides `compute_loss`
   to calculate a weighted sum of 4 distinct losses (Restoration, Unk, Date, Region).
2. `TestBEvalCallback`: Automatically evaluates the model on the historical Test B
   dataset during the standard evaluation loop.
3. `generate_predictions_report`: Creates a human-readable CSV showing the model's
   top-5 predictions, confidence scores, and surrounding context for every lacuna.
"""

import argparse
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from collator import KyivanPhysicalCollator
from collator_v2 import KyivanPhysicalCollatorV2
from config import KyivanConfig
from datasets import load_from_disk
from model import Kyivan
from vocab_categories import maskable_ids
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

ALLOWED_PRED_IDS: Optional[set] = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Loads a JSON file from disk.

    Args:
        path (Union[str, Path]): Path to the JSON file.

    Returns:
        Dict[str, Any]: The parsed JSON data as a dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Saves a dictionary to disk as a JSON file.

    Args:
        data (Dict[str, Any]): The data to save.
        path (Union[str, Path]): The destination file path.

    Returns:
        None
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log.info(f"Saved: {path}")


def preprocess_logits_for_metrics(
    logits: Union[torch.Tensor, Tuple[torch.Tensor, ...]], labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepares the logits for accuracy metric calculations by extracting
    restoration, gap-expansion (unk), date, and region predictions.

    Args:
        logits (Union[torch.Tensor, Tuple[torch.Tensor, ...]]): The raw model outputs.
        labels (torch.Tensor): The target labels.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - Top 5 predicted token indices for restoration.
            - Per-token predicted class (0=stop, 1=expand) for the `[#]` unk head.
            - Date distribution predictions (softmax probabilities).
            - Region predicted class index.
    """
    # Extract logits from the multi-task output tuple
    if isinstance(logits, tuple):
        logits_restore = logits[0]
        logits_unk = logits[1] if len(logits) > 1 else None
        logits_date = logits[2] if len(logits) > 2 else None
        logits_region = logits[3] if len(logits) > 3 else None
    else:
        logits_restore = logits
        logits_unk = None
        logits_date = None
        logits_region = None

    # Top-5 restore predictions with illegal token masking
    if ALLOWED_PRED_IDS is not None:
        vocab_size = logits_restore.size(-1)
        allowed_mask = torch.zeros(
            vocab_size, dtype=torch.bool, device=logits_restore.device
        )
        allowed_idxs = torch.tensor(
            list(ALLOWED_PRED_IDS), dtype=torch.long, device=logits_restore.device
        )
        allowed_mask[allowed_idxs] = True
        masked_logits = logits_restore.clone()
        masked_logits[..., ~allowed_mask] = -1e9
        top5_restore = torch.topk(masked_logits, k=5, dim=-1).indices
    else:
        top5_restore = torch.topk(logits_restore, k=5, dim=-1).indices

    # Unk head: per-token argmax over the 2 gap-expansion classes
    if logits_unk is not None:
        unk_preds = torch.argmax(logits_unk, dim=-1)
    else:
        unk_preds = torch.zeros(
            top5_restore.shape[:-1], dtype=torch.long, device=logits_restore.device
        )

    # Date: softmax probabilities [batch, num_date_bins]
    if logits_date is not None:
        date_probs = torch.softmax(logits_date, dim=-1)
    else:
        date_probs = torch.zeros(
            logits_restore.size(0), 1, device=logits_restore.device
        )

    # Region: argmax class [batch]
    if logits_region is not None:
        region_preds = torch.argmax(logits_region, dim=-1)
    else:
        region_preds = torch.zeros(
            logits_restore.size(0), dtype=torch.long, device=logits_restore.device
        )

    return top5_restore, unk_preds, date_probs, region_preds


def compute_metrics(eval_preds: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    Calculates Top-1/3/5 accuracy for restoration, gap-expansion (unk) head
    accuracy, date MAE, and region accuracy.

    Args:
        eval_preds (Tuple[np.ndarray, np.ndarray]): A tuple containing the model
                                                    predictions and the true labels.

    Returns:
        Dict[str, float]: A dictionary containing the calculated metrics.
    """
    preds, labels = eval_preds

    # preds is a tuple: (top5_restore, unk_preds, date_probs, region_preds)
    # labels is a tuple: (labels_restore, unk_labels, date_labels, region_labels, date_labels_mask)
    if isinstance(preds, tuple):
        restore_preds, unk_preds, date_pred_probs, region_preds = preds
    else:
        restore_preds = preds
        unk_preds = None
        date_pred_probs = None
        region_preds = None

    if isinstance(labels, tuple):
        labels_restore = labels[0]
        unk_labels = labels[1] if len(labels) > 1 else None
        date_labels = labels[2] if len(labels) > 2 else None
        region_labels = labels[3] if len(labels) > 3 else None
        date_labels_mask = labels[4] if len(labels) > 4 else None
    else:
        labels_restore = labels
        unk_labels = None
        date_labels = None
        region_labels = None
        date_labels_mask = None

    metrics = {}

    # --- 1. Restoration Top-K Accuracy ---
    mask = labels_restore != -100
    masked_labels = labels_restore[mask]
    masked_preds = restore_preds[mask]

    if masked_labels.size == 0:
        metrics["top1_accuracy"] = 0.0
        metrics["top3_accuracy"] = 0.0
        metrics["top5_accuracy"] = 0.0
    else:
        metrics["top1_accuracy"] = float(np.mean(masked_preds[:, 0] == masked_labels))
        metrics["top3_accuracy"] = float(
            np.mean(np.any(masked_preds[:, :3] == masked_labels[:, None], axis=1))
        )
        metrics["top5_accuracy"] = float(
            np.mean(np.any(masked_preds[:, :5] == masked_labels[:, None], axis=1))
        )

    # --- 1b. Unk/Gap-Expansion Head Accuracy ---
    # Scored only at the single `[#]` token per example (unk_labels is -100
    # everywhere else) -- kept as its own metric block, separate from
    # top1/3/5_accuracy above, since it's a different task (binary "does this
    # gap need to grow", not character identity) over a disjoint set of
    # positions from the `[-]` restoration targets.
    if unk_preds is not None and unk_labels is not None:
        try:
            unk_mask = unk_labels != -100
            masked_unk_labels = unk_labels[unk_mask]
            masked_unk_preds = unk_preds[unk_mask]
            metrics["unk_total"] = int(masked_unk_labels.size)
            if masked_unk_labels.size > 0:
                metrics["unk_accuracy"] = float(
                    np.mean(masked_unk_preds == masked_unk_labels)
                )
                # Macro F1 since "expand" (span > 1 char) vs "stop" (span == 1
                # char) is usually imbalanced -- accuracy alone can look fine
                # while the head just always predicts the majority class.
                metrics["unk_macro_f1"] = float(
                    f1_score(
                        masked_unk_labels,
                        masked_unk_preds,
                        average="macro",
                        zero_division=0,
                    )
                )
        except Exception:
            pass

    # --- 2. Date MAE (Mean Absolute Error on predicted vs true date distribution peak) ---
    if date_pred_probs is not None and date_labels is not None:
        try:
            # date_pred_probs: [batch, num_bins], date_labels: [batch, num_bins]
            # Compare the peak bin (argmax) of predicted vs true distribution
            pred_peak_bin = np.argmax(date_pred_probs, axis=-1)  # [batch]
            true_peak_bin = np.argmax(date_labels, axis=-1)  # [batch]

            if date_labels_mask is not None:
                date_valid = np.asarray(date_labels_mask).astype(bool)
                pred_peak_bin = pred_peak_bin[date_valid]
                true_peak_bin = true_peak_bin[date_valid]

            if pred_peak_bin.size > 0:
                # MAE in bins (each bin = 50 years)
                metrics["date_bin_mae"] = float(
                    np.mean(np.abs(pred_peak_bin - true_peak_bin))
                )
                # MAE in years (multiply bin distance by 50)
                metrics["date_years_mae"] = float(
                    np.mean(np.abs(pred_peak_bin - true_peak_bin)) * 50
                )
                # Exact bin match accuracy
                metrics["date_exact_accuracy"] = float(
                    np.mean(pred_peak_bin == true_peak_bin)
                )
                # Macro F1 over the 20 date bins as classes -- accuracy alone
                # hides whether the model just always predicts the most
                # common bin(s) rather than actually discriminating dates.
                metrics["date_macro_f1"] = float(
                    f1_score(
                        true_peak_bin, pred_peak_bin, average="macro", zero_division=0
                    )
                )
        except Exception:
            pass

    # --- 3. Region Classification Accuracy ---
    if region_preds is not None and region_labels is not None:
        try:
            # region_labels: [batch], region_preds: [batch]
            valid_mask = region_labels != -100
            if np.any(valid_mask):
                true_region = region_labels[valid_mask]
                pred_region = region_preds[valid_mask]
                metrics["region_accuracy"] = float(
                    np.mean(pred_region == true_region)
                )
                # Macro F1 across the 4 dialect classes -- the corpus is
                # heavily imbalanced (OES dominates by document count), so
                # accuracy alone can look fine while the model just always
                # guesses the majority class.
                metrics["region_macro_f1"] = float(
                    f1_score(
                        true_region, pred_region, average="macro", zero_division=0
                    )
                )
        except Exception:
            pass

    return metrics


class KyivanTrainer(Trainer):
    """
    Custom Trainer integrating the multi-task loss function from the Aeneas paper.
    It balances character restoration, gap extension prediction, dating, and dialect classification.
    """

    def __init__(
        self,
        *args,
        log_path: Optional[Path] = None,
        eval_data_collator: Optional[Any] = None,
        loss_weight_restore: float = 5.0,
        loss_weight_unk: float = 1.0,
        loss_weight_date: float = 0.5,
        loss_weight_region: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.log_path = log_path
        self.log_history_custom = []
        # Distinct collator for eval (e.g. KyivanPhysicalCollatorV2(mode="valid")):
        # HF Trainer otherwise reuses self.data_collator for both train and eval,
        # which would score eval metrics (and early stopping) under the same
        # noisy, up-to-75%-mask-rate train regime instead of a fixed difficulty.
        self.eval_data_collator = eval_data_collator

        self.loss_weight_restore = loss_weight_restore
        self.loss_weight_unk = loss_weight_unk
        self.loss_weight_date = loss_weight_date
        self.loss_weight_region = loss_weight_region
        # Raw (unweighted) per-component loss, summed since the last
        # logging_steps flush -- mirrors how Trainer itself accumulates the
        # combined "loss" between logs, so loss_res/loss_unk/loss_date/
        # loss_region show up at the same cadence instead of only the
        # single combined number, letting you see e.g. whether loss_date
        # (unbounded KL, unlike the bounded CE terms) dominates the total.
        self._component_loss_sums = {
            "loss_res": 0.0,
            "loss_unk": 0.0,
            "loss_date": 0.0,
            "loss_region": 0.0,
        }
        self._component_loss_count = 0

    def get_eval_dataloader(self, eval_dataset: Any = None) -> Any:
        if self.eval_data_collator is None:
            return super().get_eval_dataloader(eval_dataset)
        original_collator = self.data_collator
        self.data_collator = self.eval_data_collator
        try:
            return super().get_eval_dataloader(eval_dataset)
        finally:
            self.data_collator = original_collator

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        # Only the periodic training-progress log call carries a bare "loss"
        # key (eval logging uses "eval_loss", the end-of-training summary
        # uses "train_loss") -- inject the averaged component losses there,
        # at the same cadence as the combined loss, then reset for the next
        # window.
        if "loss" in logs and self._component_loss_count > 0:
            for name in self._component_loss_sums:
                logs[name] = round(
                    self._component_loss_sums[name] / self._component_loss_count, 6
                )
                self._component_loss_sums[name] = 0.0
            self._component_loss_count = 0

        super().log(logs, *args, **kwargs)
        self.log_history_custom.append(
            {"timestamp": datetime.now().isoformat(), **logs}
        )
        if self.log_path and len(self.log_history_custom) % 10 == 0:
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump(self.log_history_custom, f, ensure_ascii=False, indent=2)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Computes the weighted multi-task loss.

        Args:
            model (nn.Module): The Kyivan model instance.
            inputs (Dict[str, torch.Tensor]): The batch of inputs and targets.
            return_outputs (bool): Whether to return the model outputs alongside the loss.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Any]]: The combined loss, and optionally the model outputs.
        """
        labels_restore = inputs.pop("labels")
        labels_unk = inputs.pop("unk_labels")
        labels_date = inputs.pop("date_labels", None)
        labels_date_mask = inputs.pop("date_labels_mask", None)
        labels_region = inputs.pop("region_labels", None)

        outputs = model(**inputs)

        # 1. Restoration Loss
        loss_res_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.05)
        loss_res = loss_res_fct(
            outputs.logits_restore.view(-1, model.config.vocab_char_size),
            labels_restore.view(-1),
        )

        # 2. Unknown Lacuna Length Loss
        loss_unk_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss_unk = loss_unk_fct(outputs.logits_unk.view(-1, 2), labels_unk.view(-1))

        total_loss = (self.loss_weight_restore * loss_res) + (
            self.loss_weight_unk * loss_unk
        )

        # 3. Date Distribution Loss (KL-Divergence). Unlike the bounded CE
        # terms above, KLDivLoss is unbounded and can spike much larger
        # early in training -- kept as a smaller weight by default
        # (loss_weight_date) since it was suspected of dominating the
        # shared encoder's gradient at the old weight (1.25) while
        # restoration accuracy stayed flat despite region/date metrics
        # looking reasonable.
        #
        # KLDivLoss has no ignore_index, so examples with a withheld date
        # label (e.g. 90% of Ostrog Bible chunks) are excluded manually via
        # labels_date_mask: per-example KL summed over bins, then averaged
        # only over the valid (mask==1) examples. Reduces to the exact same
        # value as the original `reduction="batchmean"` when every example is
        # valid, since sum-over-classes-then-mean-over-batch is the same sum
        # either order.
        loss_date = None
        if labels_date is not None:
            log_probs_date = F.log_softmax(outputs.logits_date, dim=-1)
            per_example_kldiv = F.kl_div(
                log_probs_date, labels_date, reduction="none"
            ).sum(dim=-1)
            if labels_date_mask is not None:
                valid = labels_date_mask.bool()
                if valid.any():
                    loss_date = per_example_kldiv[valid].mean()
                    total_loss += self.loss_weight_date * loss_date
            else:
                loss_date = per_example_kldiv.mean()
                total_loss += self.loss_weight_date * loss_date

        # 4. Region/Dialect Classification Loss. Also given a smaller
        # default weight (0.5, down from 2.0) -- region_accuracy was
        # already ~0.96 at the old weight, so it likely didn't need this
        # much of the shared encoder's gradient budget.
        loss_region = None
        if labels_region is not None:
            loss_region_fct = nn.CrossEntropyLoss(
                ignore_index=-100, label_smoothing=0.1
            )
            loss_region = loss_region_fct(outputs.logits_region, labels_region)
            total_loss += self.loss_weight_region * loss_region

        # Raw (unweighted) component losses, accumulated only for actual
        # training steps -- compute_loss is also called during evaluation
        # (Trainer.prediction_step), which would otherwise pollute the
        # train-time averages logged in `log()` below.
        if model.training:
            self._component_loss_sums["loss_res"] += float(loss_res.detach())
            self._component_loss_sums["loss_unk"] += float(loss_unk.detach())
            if loss_date is not None:
                self._component_loss_sums["loss_date"] += float(loss_date.detach())
            if loss_region is not None:
                self._component_loss_sums["loss_region"] += float(
                    loss_region.detach()
                )
            self._component_loss_count += 1

        return (total_loss, outputs) if return_outputs else total_loss


class TestBEvalCallback(TrainerCallback):
    """
    A callback that automatically runs evaluation on the historical Test B dataset
    whenever the standard evaluation (Test A) is triggered.
    """

    def __init__(
        self,
        test_b_dataset: Any,
        output_dir: Path,
        char_vocab: Dict[str, int],
        max_samples: Optional[int] = None,
    ) -> None:
        self.test_b_dataset = test_b_dataset
        self.output_dir = Path(output_dir)
        self.char_vocab = char_vocab
        self.max_samples = max_samples
        self._in_eval = False

    def on_evaluate(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        trainer = kwargs.get("trainer")
        if trainer is None or self._in_eval or self.test_b_dataset is None:
            return
        try:
            self._in_eval = True
            ds = self.test_b_dataset

            step = getattr(state, "global_step", None) or "final"
            report_path = self.output_dir / f"pred_report_test_b_step{step}.csv"

            metrics = generate_predictions_report(
                model=trainer.model,
                char_vocab=self.char_vocab,
                dataset=ds,
                output_path=report_path,
                max_samples=self.max_samples if self.max_samples else 1000,
                collator=None,
                batch_size=args.per_device_eval_batch_size,
            )

            formatted_metrics = {f"eval_test_b_{k}": v for k, v in metrics.items()}
            if hasattr(trainer, "log"):
                trainer.log(formatted_metrics)

            fname = self.output_dir / f"eval_metrics_test_b_step{step}.json"
            save_json(metrics, fname)
            log.info(f"Saved Test B metrics: {fname}")
        finally:
            self._in_eval = False


def generate_predictions_report(
    model: nn.Module,
    char_vocab: Dict[str, int],
    dataset: Any,
    output_path: Path,
    max_samples: int = 100,
    k_values: Tuple[int, ...] = (1, 3, 5),
    device: Optional[torch.device] = None,
    collator: Optional[Any] = None,
    batch_size: int = 8,
    context_window: int = 20,
) -> Dict[str, float]:
    """
    Generates a detailed CSV report containing the model's restoration predictions.
    Works for both Test A (requires collator for dynamic masking) and Test B (pre-masked).

    Args:
        model (nn.Module): The trained Kyivan model.
        char_vocab (Dict[str, int]): The character vocabulary dictionary.
        dataset (Any): The HuggingFace dataset to evaluate.
        output_path (Path): Destination path for the CSV report.
        max_samples (int): Maximum number of sequences to process.
        k_values (Tuple[int, ...]): Tuple of K values for Top-K hit tracking.
        device (Optional[torch.device]): Target device for inference.
        collator (Optional[Any]): The data collator (required for Test A).
        batch_size (int): Inference batch size.
        context_window (int): Number of characters to display around the target in the report.

    Returns:
        Dict[str, float]: A dictionary summarizing the total predictions, accuracy, and
            Top-K hits for `[-]` restoration, plus `unk_total`/`unk_correct`/`unk_accuracy`/
            `unk_macro_f1` for the `[#]` unk (gap-expansion) head, scored and reported
            separately. Also includes `date_total`/`date_bin_mae`/`date_years_mae`/
            `date_exact_accuracy`/`date_macro_f1` and `region_total`/`region_accuracy`/
            `region_macro_f1` whenever the dataset carries valid (non-withheld)
            date/region labels -- both Test A (via the collator's date_labels_mask/
            region_labels) and Test B.
    """
    if device is None:
        device = next(model.parameters()).device

    id_to_char = {int(v): k for k, v in char_vocab.items()}

    def decode_one_token(token_id: int) -> str:
        return id_to_char.get(int(token_id), "[UNK]")

    def _mask_pred_logits_for_allowed(pred_logits: torch.Tensor) -> torch.Tensor:
        if ALLOWED_PRED_IDS is None:
            return pred_logits
        vocab_size = pred_logits.size(-1)
        allowed_mask = torch.zeros(
            vocab_size, dtype=torch.bool, device=pred_logits.device
        )
        allowed_idxs = torch.tensor(
            list(ALLOWED_PRED_IDS), dtype=torch.long, device=pred_logits.device
        )
        allowed_mask[allowed_idxs] = True
        masked = pred_logits.clone()
        masked[..., ~allowed_mask] = -1e9
        return masked

    def get_context(input_ids: list, pos: int, context_window: int = 20) -> str:
        start = max(0, pos - context_window)
        end = min(len(input_ids), pos + context_window + 1)
        before = "".join(decode_one_token(int(cid)) for cid in input_ids[start:pos])
        target = decode_one_token(int(input_ids[pos]))
        after = "".join(decode_one_token(int(cid)) for cid in input_ids[pos + 1 : end])
        return before + ">>>" + target + "<<<" + after

    def decode_action(action: int) -> str:
        return "expand" if action == 1 else "stop"

    model.eval()
    rows = []
    # `[#]` gap-expansion (unk head) predictions are tracked separately from
    # the `[-]` restoration rows above: it's a different task (binary
    # "does this lacuna need to grow" over the single `[#]` position per
    # example) with a disjoint set of scored positions, so lumping it into
    # the same accuracy/CSV would silently blend two different things.
    gap_rows = []
    gap_correct = gap_used = 0
    # For unk_macro_f1 -- accuracy alone can look fine while the head just
    # always predicts the majority class (most `[#]` gaps are single-char).
    gap_true_actions: list = []
    gap_pred_actions: list = []
    hit_accum = {f"hit@{k}": 0 for k in k_values}
    correct = used = 0
    top_k_max = max(k_values)
    total_samples = min(len(dataset), max_samples)

    # Date/region for both Test A and Test B -- Test A's numbers here are
    # somewhat redundant with test_a_eval_metrics (compute_metrics via the
    # standard eval loop), but keeping the report itself self-contained
    # (date/region live in the same place regardless of which dataset you're
    # looking at) is worth the near-zero extra cost: outputs.logits_date/
    # logits_region are already computed by the same forward pass used for
    # restoration/unk above, just not read until now.
    has_date_col = "date_labels" in dataset.column_names
    has_region_col = "region_labels" in dataset.column_names
    date_true_bins, date_pred_bins = [], []
    region_true, region_pred = [], []

    log.info(f"Generating predictions report for {total_samples} samples...")

    # TEST B LOGIC (Pre-masked historical lacunae with existing labels)
    if "labels" in dataset.column_names:
        for i, sample in enumerate(dataset.select(range(total_samples))):
            if i % 20 == 0:
                log.info(f"  Processing {i}/{total_samples}...")

            input_ids = torch.tensor(
                sample["input_ids"], dtype=torch.long, device=device
            )
            attention_mask = torch.tensor(
                sample["attention_mask"], dtype=torch.long, device=device
            )
            labels = sample.get("labels", None)
            unk_labels = sample.get("unk_labels", None)
            has_gap_labels = bool(unk_labels) and any(l != -100 for l in unk_labels)

            date_labels_sample = sample.get("date_labels") if has_date_col else None
            has_date_label = bool(date_labels_sample) and sum(date_labels_sample) > 0
            region_label_sample = (
                sample.get("region_labels") if has_region_col else None
            )
            has_region_label = (
                region_label_sample is not None and region_label_sample != -100
            )

            no_restore = labels is None or -100 not in labels
            if no_restore and not (has_gap_labels or has_date_label or has_region_label):
                continue

            mask_positions = (
                [j for j, l in enumerate(labels) if l != -100] if labels else []
            )
            if not mask_positions and not (
                has_gap_labels or has_date_label or has_region_label
            ):
                continue

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids.unsqueeze(0),
                    attention_mask=attention_mask.unsqueeze(0),
                )
                logits = outputs.logits_restore[0]
                logits_unk = outputs.logits_unk[0]

            input_ids_list = input_ids.cpu().numpy().tolist()

            if has_date_label:
                date_true_bins.append(int(np.argmax(date_labels_sample)))
                date_pred_bins.append(int(torch.argmax(outputs.logits_date[0]).item()))

            if has_region_label:
                region_true.append(int(region_label_sample))
                region_pred.append(int(torch.argmax(outputs.logits_region[0]).item()))

            if has_gap_labels:
                for pos in (j for j, l in enumerate(unk_labels) if l != -100):
                    true_action = int(unk_labels[pos])
                    pred_logits_u = logits_unk[pos]
                    pred_action = int(torch.argmax(pred_logits_u).item())
                    pred_prob = float(
                        torch.softmax(pred_logits_u, dim=-1)[pred_action].item()
                    )
                    gap_used += 1
                    is_correct_gap = pred_action == true_action
                    if is_correct_gap:
                        gap_correct += 1
                    gap_true_actions.append(true_action)
                    gap_pred_actions.append(pred_action)
                    gap_rows.append(
                        {
                            "sample_idx": i,
                            "position": pos,
                            "context": get_context(
                                input_ids_list, pos, context_window
                            ),
                            "true_action": decode_action(true_action),
                            "pred_action": decode_action(pred_action),
                            "is_correct": is_correct_gap,
                            "pred_prob": round(pred_prob, 4),
                        }
                    )

            for pos in mask_positions:
                true_id = int(labels[pos])
                if true_id < 0:
                    continue

                pred_logits = logits[pos]
                pred_logits_masked = _mask_pred_logits_for_allowed(pred_logits)
                top_ids = (
                    torch.topk(pred_logits_masked, k=min(top_k_max, len(char_vocab)))
                    .indices.cpu()
                    .numpy()
                )

                pred_id = int(top_ids[0])
                true_char = decode_one_token(true_id)
                pred_char = decode_one_token(pred_id)

                # Skip evaluation for special context tokens
                if pred_char.startswith("[") or true_char.startswith("["):
                    continue

                used += 1
                is_correct = pred_id == true_id
                if is_correct:
                    correct += 1

                probs = torch.softmax(pred_logits, dim=-1)
                top1_prob = float(probs[pred_id].item())
                true_rank = next(
                    (r + 1 for r, tid in enumerate(top_ids) if int(tid) == true_id),
                    None,
                )
                top_chars = [decode_one_token(int(tid)) for tid in top_ids[:5]]
                context = get_context(input_ids_list, pos, context_window)

                for k in k_values:
                    if any(
                        decode_one_token(int(tid)) == true_char for tid in top_ids[:k]
                    ):
                        hit_accum[f"hit@{k}"] += 1

                rows.append(
                    {
                        "sample_idx": i,
                        "position": pos,
                        "context": context,
                        "true_char": true_char,
                        "pred_char": pred_char,
                        "is_correct": is_correct,
                        "true_rank": true_rank,
                        "top1_prob": round(top1_prob, 4),
                        "top5_preds": "|".join(top_chars),
                    }
                )

    # TEST A LOGIC (Clean text requiring dynamic collator masking)
    else:
        if collator is None:
            raise ValueError(
                "For test_a (no pre-existing labels), a collator must be provided."
            )

        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            batch_raw = [dataset[j] for j in range(start, end)]
            if start % (batch_size * 10) == 0:
                log.info(f"  Processing {start}/{total_samples}...")

            batch = collator(batch_raw)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"]
            unk_labels_batch = batch.get("unk_labels")
            date_labels_batch = batch.get("date_labels")
            date_labels_mask_batch = batch.get("date_labels_mask")
            region_labels_batch = batch.get("region_labels")

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits_restore
                logits_unk = outputs.logits_unk

            for b, labels in enumerate(labels_batch):
                labels = labels.tolist() if isinstance(labels, torch.Tensor) else labels
                mask_positions = [j for j, l in enumerate(labels) if l != -100]

                # Same accumulators as the Test B branch above -- the
                # collator already computes date_labels_mask/region_labels
                # (-100 sentinel) per example, so validity here reuses that
                # directly instead of re-deriving it.
                if (
                    date_labels_batch is not None
                    and date_labels_mask_batch is not None
                    and float(date_labels_mask_batch[b]) > 0.5
                ):
                    date_true_bins.append(int(torch.argmax(date_labels_batch[b]).item()))
                    date_pred_bins.append(
                        int(torch.argmax(outputs.logits_date[b]).item())
                    )

                if region_labels_batch is not None:
                    true_region_b = int(region_labels_batch[b])
                    if true_region_b != -100:
                        region_true.append(true_region_b)
                        region_pred.append(
                            int(torch.argmax(outputs.logits_region[b]).item())
                        )

                unk_labels = (
                    unk_labels_batch[b].tolist()
                    if unk_labels_batch is not None
                    else None
                )
                if unk_labels:
                    sample_input_ids_for_gap = input_ids[b].cpu().numpy().tolist()
                    sample_logits_unk = logits_unk[b]
                    for pos in (j for j, l in enumerate(unk_labels) if l != -100):
                        true_action = int(unk_labels[pos])
                        pred_logits_u = sample_logits_unk[pos]
                        pred_action = int(torch.argmax(pred_logits_u).item())
                        pred_prob = float(
                            torch.softmax(pred_logits_u, dim=-1)[pred_action].item()
                        )
                        gap_used += 1
                        is_correct_gap = pred_action == true_action
                        if is_correct_gap:
                            gap_correct += 1
                        gap_true_actions.append(true_action)
                        gap_pred_actions.append(pred_action)
                        gap_rows.append(
                            {
                                "sample_idx": start + b,
                                "position": pos,
                                "context": get_context(
                                    sample_input_ids_for_gap, pos, context_window
                                ),
                                "true_action": decode_action(true_action),
                                "pred_action": decode_action(pred_action),
                                "is_correct": is_correct_gap,
                                "pred_prob": round(pred_prob, 4),
                            }
                        )

                if not mask_positions:
                    continue

                sample_input_ids = input_ids[b].cpu().numpy().tolist()
                sample_logits = logits[b]

                for pos in mask_positions:
                    true_id = int(labels[pos])
                    if true_id < 0:
                        continue

                    pred_logits = sample_logits[pos]
                    pred_logits_masked = _mask_pred_logits_for_allowed(pred_logits)
                    top_ids = (
                        torch.topk(
                            pred_logits_masked, k=min(top_k_max, len(char_vocab))
                        )
                        .indices.cpu()
                        .numpy()
                    )

                    pred_id = int(top_ids[0])
                    true_char = decode_one_token(true_id)
                    pred_char = decode_one_token(pred_id)

                    if pred_char.startswith("[") or true_char.startswith("["):
                        continue

                    used += 1
                    is_correct = pred_id == true_id
                    if is_correct:
                        correct += 1

                    probs = torch.softmax(pred_logits, dim=-1)
                    top1_prob = float(probs[pred_id].item())
                    true_rank = next(
                        (r + 1 for r, tid in enumerate(top_ids) if int(tid) == true_id),
                        None,
                    )
                    top_chars = [decode_one_token(int(tid)) for tid in top_ids[:5]]
                    context = get_context(sample_input_ids, pos, context_window)

                    for k in k_values:
                        if any(
                            decode_one_token(int(tid)) == true_char
                            for tid in top_ids[:k]
                        ):
                            hit_accum[f"hit@{k}"] += 1

                    rows.append(
                        {
                            "sample_idx": start + b,
                            "position": pos,
                            "context": context,
                            "true_char": true_char,
                            "pred_char": pred_char,
                            "is_correct": is_correct,
                            "true_rank": true_rank,
                            "top1_prob": round(top1_prob, 4),
                            "top5_preds": "|".join(top_chars),
                        }
                    )

    if rows:
        pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
        log.info(f"Saved predictions report: {output_path}")

    if gap_rows:
        gap_output_path = output_path.with_name(
            output_path.stem + "_gaps" + output_path.suffix
        )
        pd.DataFrame(gap_rows).to_csv(gap_output_path, index=False, encoding="utf-8-sig")
        log.info(f"Saved gap-expansion ([#] unk head) report: {gap_output_path}")

    metrics = {
        "total_predictions": used,
        "correct": correct,
        "accuracy": round(correct / used, 4) if used > 0 else 0.0,
        **{k: round(v / used, 4) if used > 0 else 0.0 for k, v in hit_accum.items()},
        # Separate from the restoration accuracy above: scored only at `[#]`
        # positions, over the binary "does this gap need to grow" decision.
        # Named unk_* to match compute_metrics's naming for the same head
        # (outputs.logits_unk / unk_labels), rather than the gap_* used
        # internally in this function.
        "unk_total": gap_used,
        "unk_correct": gap_correct,
        "unk_accuracy": round(gap_correct / gap_used, 4) if gap_used > 0 else 0.0,
        "unk_macro_f1": round(
            float(
                f1_score(
                    gap_true_actions,
                    gap_pred_actions,
                    average="macro",
                    zero_division=0,
                )
            ),
            4,
        )
        if gap_used > 0
        else 0.0,
    }

    # Date/region: populated from either branch above (Test A or Test B --
    # whichever has valid, non-withheld labels). Mirrors compute_metrics's
    # date/region block -- peak-bin MAE/exact-match/macro-F1 for date,
    # accuracy/macro-F1 for region -- kept as its own block since it's a
    # whole-document, [SOS]-pooled prediction, unrelated to the
    # per-character rows above.
    if date_true_bins:
        date_true_arr = np.array(date_true_bins)
        date_pred_arr = np.array(date_pred_bins)
        bin_mae = float(np.mean(np.abs(date_pred_arr - date_true_arr)))
        metrics["date_total"] = len(date_true_bins)
        metrics["date_bin_mae"] = round(bin_mae, 4)
        metrics["date_years_mae"] = round(bin_mae * 50, 4)
        metrics["date_exact_accuracy"] = round(
            float(np.mean(date_pred_arr == date_true_arr)), 4
        )
        metrics["date_macro_f1"] = round(
            float(
                f1_score(
                    date_true_arr, date_pred_arr, average="macro", zero_division=0
                )
            ),
            4,
        )

    if region_true:
        region_true_arr = np.array(region_true)
        region_pred_arr = np.array(region_pred)
        metrics["region_total"] = len(region_true)
        metrics["region_accuracy"] = round(
            float(np.mean(region_pred_arr == region_true_arr)), 4
        )
        metrics["region_macro_f1"] = round(
            float(
                f1_score(
                    region_true_arr,
                    region_pred_arr,
                    average="macro",
                    zero_division=0,
                )
            ),
            4,
        )

    return metrics


def main() -> None:
    """
    Parses arguments, initializes the model and trainer, and executes the training loop.
    """
    parser = argparse.ArgumentParser(description="Kyivan Multi-Task Trainer")

    parser.add_argument(
        "--dataset_dir",
        default="prepared_datasets/hf_dataset",
        help="Path to the arrow dataset",
    )
    parser.add_argument(
        "--char_vocab_path",
        default="prepared_datasets/tokenizer/char_vocab.json",
        help="Path to char vocab",
    )
    parser.add_argument(
        "--output_dir",
        default="training_output",
        help="Directory for checkpoints and logs",
    )
    parser.add_argument(
        "--collator_version",
        choices=["v1", "v2"],
        default="v2",
        help="v1 = collator.py (fixed mlm_prob per-char coin flip). v2 = "
        "collator_v2.py (DeepMind Aeneas-aligned: per-example mask-rate "
        "sampling). Whether punctuation is masked/scored is controlled "
        "separately by vocab_categories.MASK_PUNCTUATION, not by this "
        "flag. v1 is kept only for comparison.",
    )
    parser.add_argument(
        "--char_mask_rate_min",
        type=float,
        default=0.0,
        help="v2 collator: lower bound of the per-example uniform sampling "
        "range for the overall character-masking rate.",
    )
    parser.add_argument(
        "--char_mask_rate_max",
        type=float,
        default=0.75,
        help="v2 collator: upper bound of the per-example uniform sampling "
        "range for the overall character-masking rate.",
    )
    parser.add_argument(
        "--span_mask_ratio",
        type=float,
        default=0.15,
        help="v2 collator: fraction of the per-example mask budget spent on "
        "contiguous (non-compressing, still per-character `[-]`) spans "
        "rather than scattered lone characters.",
    )
    parser.add_argument(
        "--span_mask_geometric_p",
        type=float,
        default=0.1,
        help="v2 collator: geometric-distribution parameter for "
        "non-compressing span length (mean length = 1/p).",
    )
    parser.add_argument(
        "--unk_geometric_p",
        type=float,
        default=0.25,
        help="v2 collator: geometric-distribution parameter for the single "
        "compressed unknown-length `[#]` gap's length (sampled as "
        "geometric(p) - 1, so ~p of examples get no gap at all).",
    )
    parser.add_argument(
        "--span_mask_eval_len",
        type=int,
        default=10,
        help="v2 collator, valid/eval mode only: masks exactly one span of "
        "size uniform(1, this) instead of the train-mode rate budget.",
    )
    parser.add_argument(
        "--edge_prob",
        type=float,
        default=0.1,
        help="v2 collator: probability of a simulated physical edge tear "
        "(birch-bark-specific; shares the compressed-gap slot with the "
        "unk_geometric_p gap -- at most one per example).",
    )

    parser.add_argument(
        "--loss_weight_restore",
        type=float,
        default=5.0,
        help="Weight for the [-] restoration cross-entropy loss (was "
        "hardcoded 3.0).",
    )
    parser.add_argument(
        "--loss_weight_unk",
        type=float,
        default=1.0,
        help="Weight for the [#] gap-expansion (unk head) cross-entropy loss.",
    )
    parser.add_argument(
        "--loss_weight_date",
        type=float,
        default=0.5,
        help="Weight for the date-distribution KL-divergence loss (was "
        "hardcoded 1.25). KL is unbounded/spiky unlike the CE terms and was "
        "suspected of dominating the shared encoder's gradient.",
    )
    parser.add_argument(
        "--loss_weight_region",
        type=float,
        default=0.5,
        help="Weight for the region/dialect cross-entropy loss (was "
        "hardcoded 2.0). region_accuracy was already ~0.96 at the old "
        "weight, so it likely didn't need this much gradient share.",
    )

    parser.add_argument(
        "--max_len", type=int, default=1024, help="Maximum sequence length"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=512, help="Hidden dimension size"
    )
    parser.add_argument(
        "--num_layers", type=int, default=6, help="Number of encoder layers"
    )
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )

    parser.add_argument("--epochs", type=int, default=10, help="Total training epochs")
    parser.add_argument(
        "--train_bs", type=int, default=16, help="Training batch size per device"
    )
    parser.add_argument(
        "--eval_bs", type=int, default=16, help="Evaluation batch size per device"
    )
    parser.add_argument(
        "--grad_accum", type=int, default=4, help="Gradient accumulation steps"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate")
    parser.add_argument(
        "--warmup_steps", type=int, default=1000, help="Linear warmup steps"
    )
    parser.add_argument(
        "--eval_steps", type=int, default=400, help="Steps between evaluations"
    )
    parser.add_argument(
        "--early_stopping_patience_epochs",
        type=float,
        default=3,
        help="Stop training if eval top1_accuracy hasn't improved for this "
        "many epochs' worth of evaluations. Set to 0 to disable.",
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        action="store_true",
        default=True,
        help="Enable mixed precision training (default: enabled)",
    )
    parser.add_argument(
        "--no_fp16",
        dest="fp16",
        action="store_false",
        help="Disable mixed precision training",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # Speed and Memory Optimizations
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="Compile model via torch.compile (requires PyTorch 2.0+)",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save VRAM",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw_torch",
        help="Optimizer to use (e.g. adamw_bnb_8bit)",
    )

    parser.add_argument(
        "--report_test_a", action="store_true", help="Generate CSV report for Test A"
    )
    parser.add_argument(
        "--report_test_b", action="store_true", help="Generate CSV report for Test B"
    )
    parser.add_argument(
        "--report_test_b_every_eval",
        action="store_true",
        help="Also generate a full per-case Test B CSV report (up to "
        "--max_report_samples rows) on every eval step during training, not "
        "just the final run. Off by default -- this reruns inference over "
        "the whole Test B sample at every eval, which adds real wall-clock "
        "cost on top of the standard eval loop.",
    )
    parser.add_argument(
        "--max_report_samples",
        type=int,
        default=1000,
        help="Max samples for the CSV report",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint folder to resume training from, or 'True' to auto-resume from the latest in output_dir",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(output_dir / "training.log", mode="a", encoding="utf-8")
    fh.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    log.addHandler(fh)

    log.info("=" * 80)
    log.info("Starting Kyivan Training")
    log.info("=" * 80)

    dataset = load_from_disk(args.dataset_dir)
    char_vocab = load_json(args.char_vocab_path)

    global ALLOWED_PRED_IDS
    ALLOWED_PRED_IDS = maskable_ids(char_vocab)

    log.info("Creating model...")
    config = KyivanConfig(
        vocab_char_size=len(char_vocab),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.hidden_size * 4,
        max_position_embeddings=args.max_len,
        pad_token_id=char_vocab["[PAD]"],
    )
    # Configure the number of date bins (20) and dialect regions (4) based on the dataset taxonomy
    model = Kyivan(config, num_date_bins=20, num_regions=4)

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"  Total params: {n_params:,}")

    eval_collator = None
    if args.collator_version == "v2":
        v2_kwargs = dict(
            char_vocab=char_vocab,
            char_mask_rate_min=args.char_mask_rate_min,
            char_mask_rate_max=args.char_mask_rate_max,
            span_mask_ratio=args.span_mask_ratio,
            span_mask_geometric_p=args.span_mask_geometric_p,
            unk_geometric_p=args.unk_geometric_p,
            span_mask_eval_len=args.span_mask_eval_len,
            edge_prob=args.edge_prob,
        )
        collator = KyivanPhysicalCollatorV2(**v2_kwargs)
        # Fixed, comparable eval difficulty (one span of size 1..span_mask_eval_len)
        # instead of reusing the train collator's per-batch random mask rate --
        # otherwise eval_top1_accuracy (early stopping / best-checkpoint metric)
        # swings with whatever rate happened to be sampled for that batch.
        eval_collator = KyivanPhysicalCollatorV2(**v2_kwargs, mode="valid")
    else:
        collator = KyivanPhysicalCollator(
            char_vocab=char_vocab,
            mlm_prob=0.15,
            span_mask_ratio=0.2,
            span_geometric_p=0.2,
            edge_prob=0.1,
        )
    log.info(f"Using collator: {args.collator_version}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = output_dir / "checkpoints"
    log_path = output_dir / f"training_log_{timestamp}.json"

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        save_total_limit=3,
        logging_steps=100,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,
        weight_decay=0.01,
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        gradient_checkpointing=args.gradient_checkpointing,
        torch_compile=args.torch_compile,
        optim=args.optim,
        report_to=[],
        label_names=[
            "labels",
            "unk_labels",
            "date_labels",
            "region_labels",
            "date_labels_mask",
        ],
        load_best_model_at_end=True,
        metric_for_best_model="top1_accuracy",
        greater_is_better=True,
        remove_unused_columns=False,
        seed=args.seed,
    )

    log.info("Creating trainer...")
    callbacks = []
    if "test_b" in dataset and args.report_test_b_every_eval:
        callbacks.append(
            TestBEvalCallback(
                dataset["test_b"], output_dir, char_vocab, args.max_report_samples
            )
        )

    if args.early_stopping_patience_epochs > 0:
        # early_stopping_patience counts *evaluations*, not epochs, so convert
        # using the actual steps-per-epoch for this run (dataset size / batch
        # size / grad accumulation all affect how many evals happen per epoch).
        updates_per_epoch = math.ceil(
            math.ceil(len(dataset["train"]) / args.train_bs) / args.grad_accum
        )
        evals_per_epoch = max(1, round(updates_per_epoch / args.eval_steps))
        patience = max(1, round(evals_per_epoch * args.early_stopping_patience_epochs))
        log.info(
            f"Early stopping: patience={patience} evals "
            f"(~{args.early_stopping_patience_epochs} epochs at "
            f"{evals_per_epoch} evals/epoch)"
        )
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer = KyivanTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test_a"],
        data_collator=collator,
        eval_data_collator=eval_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        log_path=log_path,
        callbacks=callbacks,
        loss_weight_restore=args.loss_weight_restore,
        loss_weight_unk=args.loss_weight_unk,
        loss_weight_date=args.loss_weight_date,
        loss_weight_region=args.loss_weight_region,
    )

    log.info("Starting training...")

    resume_checkpoint = args.resume_from_checkpoint
    if resume_checkpoint is not None and resume_checkpoint.lower() == "true":
        resume_checkpoint = True

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(trainer.log_history_custom, f, ensure_ascii=False, indent=2)

    log.info("Computing final metrics on test_a...")
    eval_metrics = trainer.evaluate()
    save_json(eval_metrics, output_dir / f"eval_metrics_{timestamp}.json")

    report_specs = [
        (
            args.report_test_a,
            "test_a",
            dataset.get("test_a"),
            output_dir / f"pred_report_test_a_{timestamp}.csv",
            eval_collator if eval_collator is not None else collator,
        ),
        (
            args.report_test_b,
            "test_b",
            dataset.get("test_b"),
            output_dir / f"pred_report_test_b_{timestamp}.csv",
            None,
        ),
    ]

    report_metrics = {}
    for enabled, name, ds, report_path, report_collator in report_specs:
        if not enabled or ds is None:
            continue
        metrics = generate_predictions_report(
            model=model,
            char_vocab=char_vocab,
            dataset=ds,
            output_path=report_path,
            max_samples=args.max_report_samples,
            collator=report_collator,
            batch_size=8,
        )
        save_json(metrics, output_dir / f"pred_report_{name}_metrics_{timestamp}.json")
        report_metrics[name] = metrics

    model_save_path = output_dir / "final_model"
    model_save_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_save_path))
    save_json(vars(args), output_dir / "training_config.json")

    final_report = {
        "timestamp": timestamp,
        # train_result.training_loss is the final *loss* value, not a
        # duration -- train_result.metrics carries the actual wall-clock
        # runtime (train_runtime, seconds) alongside it.
        "training_loss": train_result.training_loss,
        "training_duration_seconds": train_result.metrics.get("train_runtime"),
        # This is trainer.evaluate() run on test_a (via compute_metrics),
        # not metrics on the train split -- named for what it actually is.
        "test_a_eval_metrics": eval_metrics,
        "args": vars(args),
    }
    if "test_a" in report_metrics:
        final_report["test_a_metrics"] = report_metrics["test_a"]
    if "test_b" in report_metrics:
        final_report["test_b_metrics"] = report_metrics["test_b"]

    save_json(final_report, output_dir / f"final_report_{timestamp}.json")
    log.info("Training completed successfully!")


if __name__ == "__main__":
    main()
