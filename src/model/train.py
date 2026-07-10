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
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collator import KyivanPhysicalCollator
from config import KyivanConfig
from datasets import load_from_disk
from model import Kyivan
from transformers import Trainer, TrainerCallback, TrainingArguments

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
) -> torch.Tensor:
    """
    Prepares the logits for accuracy metric calculations by extracting the
    restoration logits and masking out illegal tokens (e.g., special tags).

    Args:
        logits (Union[torch.Tensor, Tuple[torch.Tensor, ...]]): The raw model outputs.
        labels (torch.Tensor): The target labels.

    Returns:
        torch.Tensor: The top 5 predicted token indices.
    """
    # Extract only the restoration logits (index 0) from the multi-task output tuple
    if isinstance(logits, tuple):
        logits_restore = logits[0]
    else:
        logits_restore = logits

    if ALLOWED_PRED_IDS is None:
        return torch.topk(logits_restore, k=5, dim=-1).indices

    vocab_size = logits_restore.size(-1)
    allowed_mask = torch.zeros(
        vocab_size, dtype=torch.bool, device=logits_restore.device
    )
    allowed_idxs = torch.tensor(
        list(ALLOWED_PRED_IDS), dtype=torch.long, device=logits_restore.device
    )
    allowed_mask[allowed_idxs] = True

    # Clone and penalize illegal tokens with a highly negative value
    masked_logits = logits_restore.clone()
    masked_logits[..., ~allowed_mask] = -1e9

    return torch.topk(masked_logits, k=5, dim=-1).indices


def compute_metrics(eval_preds: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    Calculates Top-1, Top-3, and Top-5 accuracy metrics for character restoration.

    Args:
        eval_preds (Tuple[np.ndarray, np.ndarray]): A tuple containing the model
                                                    predictions and the true labels.

    Returns:
        Dict[str, float]: A dictionary containing the calculated accuracy metrics.
    """
    preds, labels = eval_preds

    if isinstance(labels, tuple):
        labels = labels[0]

    # Ignore padding and unmasked tokens
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]

    if labels.size == 0:
        return {"top1_accuracy": 0.0, "top3_accuracy": 0.0, "top5_accuracy": 0.0}

    return {
        "top1_accuracy": float(np.mean(preds[:, 0] == labels)),
        "top3_accuracy": float(
            np.mean(np.any(preds[:, :3] == labels[:, None], axis=1))
        ),
        "top5_accuracy": float(
            np.mean(np.any(preds[:, :5] == labels[:, None], axis=1))
        ),
    }


class KyivanTrainer(Trainer):
    """
    Custom Trainer integrating the multi-task loss function from the Aeneas paper.
    It balances character restoration, gap extension prediction, dating, and dialect classification.
    """

    def __init__(self, *args, log_path: Optional[Path] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.log_path = log_path
        self.log_history_custom = []

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
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
        labels_region = inputs.pop("region_labels", None)

        outputs = model(**inputs)

        # 1. Restoration Loss (Weight: 3.0)
        loss_res_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.05)
        loss_res = loss_res_fct(
            outputs.logits_restore.view(-1, model.config.vocab_char_size),
            labels_restore.view(-1),
        )

        # 2. Unknown Lacuna Length Loss (Weight: 1.0)
        loss_unk_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss_unk = loss_unk_fct(outputs.logits_unk.view(-1, 2), labels_unk.view(-1))

        total_loss = (3.0 * loss_res) + (1.0 * loss_unk)

        # 3. Date Distribution Loss (Weight: 1.25, KL-Divergence)
        if labels_date is not None:
            log_probs_date = F.log_softmax(outputs.logits_date, dim=-1)
            loss_date_fct = nn.KLDivLoss(reduction="batchmean")
            loss_date = loss_date_fct(log_probs_date, labels_date)
            total_loss += 1.25 * loss_date

        # 4. Region/Dialect Classification Loss (Weight: 2.0)
        if labels_region is not None:
            loss_region_fct = nn.CrossEntropyLoss(
                ignore_index=-100, label_smoothing=0.1
            )
            loss_region = loss_region_fct(outputs.logits_region, labels_region)
            total_loss += 2.0 * loss_region

        return (total_loss, outputs) if return_outputs else total_loss


class TestBEvalCallback(TrainerCallback):
    """
    A callback that automatically runs evaluation on the historical Test B dataset
    whenever the standard evaluation (Test A) is triggered.
    """

    def __init__(
        self, test_b_dataset: Any, output_dir: Path, char_vocab: Dict[str, int], max_samples: Optional[int] = None
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
            from utils.common import save_json
            save_json(metrics, fname)
            from utils.logger import log
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
        Dict[str, float]: A dictionary summarizing the total predictions, accuracy, and Top-K hits.
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

    model.eval()
    rows = []
    hit_accum = {f"hit@{k}": 0 for k in k_values}
    correct = used = 0
    top_k_max = max(k_values)
    total_samples = min(len(dataset), max_samples)

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

            if labels is None or -100 not in labels:
                continue

            mask_positions = [j for j, l in enumerate(labels) if l != -100]
            if not mask_positions:
                continue

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids.unsqueeze(0),
                    attention_mask=attention_mask.unsqueeze(0),
                )
                logits = outputs.logits_restore[0]

            input_ids_list = input_ids.cpu().numpy().tolist()

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

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits_restore

            for b, labels in enumerate(labels_batch):
                labels = labels.tolist() if isinstance(labels, torch.Tensor) else labels
                mask_positions = [j for j, l in enumerate(labels) if l != -100]
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

    metrics = {
        "total_predictions": used,
        "correct": correct,
        "accuracy": round(correct / used, 4) if used > 0 else 0.0,
        **{k: round(v / used, 4) if used > 0 else 0.0 for k, v in hit_accum.items()},
    }
    return metrics


def main() -> None:
    """
    Parses arguments, initializes the model and trainer, and executes the training loop.
    """
    parser = argparse.ArgumentParser(description="Kyivan Multi-Task Trainer")

    parser.add_argument(
        "--dataset_dir",
        default="novgorodets/artifacts/kyivan_dataset",
        help="Path to the arrow dataset",
    )
    parser.add_argument(
        "--char_vocab_path",
        default="novgorodets/artifacts/char_tokenizer/char_vocab.json",
        help="Path to char vocab",
    )
    parser.add_argument(
        "--output_dir",
        default="novgorodets/artifacts/training_output",
        help="Directory for checkpoints and logs",
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
        "--fp16", action="store_true", help="Enable mixed precision training"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--report_test_a", action="store_true", help="Generate CSV report for Test A"
    )
    parser.add_argument(
        "--report_test_b", action="store_true", help="Generate CSV report for Test B"
    )
    parser.add_argument(
        "--max_report_samples",
        type=int,
        default=1000,
        help="Max samples for the CSV report",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 80)
    log.info("Starting Kyivan Training")
    log.info("=" * 80)

    dataset = load_from_disk(args.dataset_dir)
    char_vocab = load_json(args.char_vocab_path)

    global ALLOWED_PRED_IDS
    ALLOWED_PRED_IDS = {
        int(v)
        for k, v in char_vocab.items()
        if len(k) == 1 and unicodedata.category(k) in ("Ll", "Lu", "Lo", "Zs")
    }

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

    collator = KyivanPhysicalCollator(
        char_vocab=char_vocab,
        mlm_prob=0.15,
        span_mask_ratio=0.2,
        span_geometric_p=0.2,
        edge_prob=0.1,
    )

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
        fp16=True,
        weight_decay=0.01,
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        report_to=[],
        label_names=["labels", "unk_labels", "date_labels", "region_labels"],
        load_best_model_at_end=True,
        metric_for_best_model="top1_accuracy",
        greater_is_better=True,
        remove_unused_columns=False,
        seed=args.seed,
    )

    log.info("Creating trainer...")
    callbacks = []
    if "test_b" in dataset:
        callbacks.append(
            TestBEvalCallback(dataset["test_b"], output_dir, char_vocab, args.max_report_samples)
        )

    trainer = KyivanTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test_a"],
        data_collator=collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        log_path=log_path,
        callbacks=callbacks,
    )

    log.info("Starting training...")
    train_result = trainer.train()

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
            collator,
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
        "training_duration": train_result.training_loss,
        "train_metrics": eval_metrics,
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
