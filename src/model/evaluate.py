"""
Kyivan Model Evaluation Script.

A standalone script for comprehensive evaluation of trained Kyivan checkpoints
on Test A (dynamically masked) and Test B (real historical lacunae) datasets.

Metrics computed:
  - Test A: Restoration Top-1/3/5 accuracy, Date MAE (bins & years),
            Date exact accuracy, Region accuracy. CSV prediction report.
  - Test B: Restoration Top-1/3/5 accuracy on real archaeological lacunae.
            CSV prediction report with context windows.

Usage:
    python evaluate.py \
        --checkpoint_dir checkpoints/checkpoints/checkpoint-2700 \
        --dataset_dir prepared_datasets/hf_dataset \
        --char_vocab_path prepared_datasets/tokenizer/char_vocab.json \
        --output_dir evaluation_results
"""

import argparse
import json
import logging
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from collator import KyivanPhysicalCollator
from config import KyivanConfig
from model import Kyivan

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

# Region label index -> human-readable name
REGION_NAMES = {0: "NW (North-Western/Novgorod)", 1: "SW (South-Western/Ruthenian)",
                2: "OES (Old East Slavic)", 3: "CS (Church Slavonic)"}

# Date bin index -> human-readable period (20 bins, 50 years each, 800-1800 AD)
def bin_to_period(bin_idx: int) -> str:
    start = 800 + bin_idx * 50
    return f"{start}-{start + 50} AD"


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log.info(f"Saved: {path}")


def load_model(checkpoint_dir: str, device: torch.device) -> Kyivan:
    """Load a Kyivan model from a training checkpoint."""
    log.info(f"Loading model from {checkpoint_dir}...")
    config = KyivanConfig.from_pretrained(checkpoint_dir)
    model = Kyivan(config, num_date_bins=20, num_regions=4)
    
    from safetensors.torch import load_file
    state_dict = load_file(Path(checkpoint_dir) / "model.safetensors")
    model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters())
    log.info(f"  Loaded model with {param_count:,} parameters")
    return model


def get_allowed_pred_ids(char_vocab: Dict[str, int]) -> set:
    """Get the set of allowed prediction IDs (real characters only, no special tokens)."""
    return {
        int(v)
        for k, v in char_vocab.items()
        if len(k) == 1 and unicodedata.category(k) in ("Ll", "Lu", "Lo", "Zs")
    }


def mask_logits(logits: torch.Tensor, allowed_ids: set) -> torch.Tensor:
    """Mask out illegal tokens (special tags) from prediction logits."""
    vocab_size = logits.size(-1)
    allowed_mask = torch.zeros(vocab_size, dtype=torch.bool, device=logits.device)
    allowed_idxs = torch.tensor(list(allowed_ids), dtype=torch.long, device=logits.device)
    allowed_mask[allowed_idxs] = True
    masked = logits.clone()
    masked[..., ~allowed_mask] = -1e9
    return masked


# ============================================================================
# TEST A EVALUATION: Dynamic masking + Restoration + Date + Region
# ============================================================================

def evaluate_test_a(
    model: Kyivan,
    dataset: Any,
    char_vocab: Dict[str, int],
    allowed_ids: set,
    output_dir: Path,
    collator: KyivanPhysicalCollator,
    batch_size: int = 16,
    max_samples: int = 2528,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """
    Evaluate on Test A using dynamic masking from the collator.
    
    Computes:
      - Restoration: Top-1/3/5 accuracy
      - Date: MAE in bins, MAE in years, exact bin accuracy
      - Region: classification accuracy
      - CSV report with per-character predictions
    """
    log.info("=" * 60)
    log.info("EVALUATING TEST A (Dynamic Masking)")
    log.info("=" * 60)

    id_to_char = {int(v): k for k, v in char_vocab.items()}
    total_samples = min(len(dataset), max_samples)

    # --- Restoration metrics ---
    all_restore_correct = 0
    all_restore_total = 0
    hit_at = {1: 0, 3: 0, 5: 0}
    rows = []

    # --- Date metrics ---
    date_pred_peaks = []
    date_true_peaks = []

    # --- Region metrics ---
    region_preds_all = []
    region_labels_all = []

    for start in tqdm(range(0, total_samples, batch_size), desc="Test A"):
        end = min(start + batch_size, total_samples)
        batch_raw = [dataset[j] for j in range(start, end)]
        batch = collator(batch_raw)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_restore = batch["labels"]  # [batch, seq_len]
        date_labels = batch.get("date_labels", None)   # [batch, 20]
        region_labels = batch.get("region_labels", None)  # [batch]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # --- Restoration ---
        logits_restore = outputs.logits_restore  # [batch, seq, vocab]
        for b in range(input_ids.size(0)):
            labels_b = labels_restore[b].tolist()
            mask_positions = [j for j, l in enumerate(labels_b) if l != -100]
            if not mask_positions:
                continue

            for pos in mask_positions:
                true_id = int(labels_b[pos])
                if true_id < 0:
                    continue

                pred_logits = mask_logits(logits_restore[b, pos], allowed_ids)
                top_ids = torch.topk(pred_logits, k=5).indices.cpu().numpy()
                pred_id = int(top_ids[0])

                true_char = id_to_char.get(true_id, "?")
                pred_char = id_to_char.get(pred_id, "?")
                if pred_char.startswith("[") or true_char.startswith("["):
                    continue

                all_restore_total += 1
                is_correct = pred_id == true_id
                if is_correct:
                    all_restore_correct += 1

                for k in [1, 3, 5]:
                    if true_id in top_ids[:k]:
                        hit_at[k] += 1

                # Context for CSV
                ids_list = input_ids[b].cpu().numpy().tolist()
                ctx_start = max(0, pos - 20)
                ctx_end = min(len(ids_list), pos + 21)
                before = "".join(id_to_char.get(int(c), "?") for c in ids_list[ctx_start:pos])
                after = "".join(id_to_char.get(int(c), "?") for c in ids_list[pos+1:ctx_end])
                context = before + ">>>" + id_to_char.get(int(ids_list[pos]), "?") + "<<<" + after

                probs = torch.softmax(logits_restore[b, pos], dim=-1)
                top_chars = [id_to_char.get(int(tid), "?") for tid in top_ids[:5]]

                rows.append({
                    "sample_idx": start + b, "position": pos, "context": context,
                    "true_char": true_char, "pred_char": pred_char,
                    "is_correct": is_correct, "top1_prob": round(float(probs[pred_id].item()), 4),
                    "top5_preds": "|".join(top_chars),
                })

        # --- Date ---
        if date_labels is not None:
            log_probs = F.softmax(outputs.logits_date, dim=-1).cpu().numpy()
            true_dates = date_labels.numpy()
            for b in range(log_probs.shape[0]):
                date_pred_peaks.append(np.argmax(log_probs[b]))
                date_true_peaks.append(np.argmax(true_dates[b]))

        # --- Region ---
        if region_labels is not None:
            r_preds = torch.argmax(outputs.logits_region, dim=-1).cpu().numpy()
            r_labels = region_labels.numpy()
            for b in range(r_preds.shape[0]):
                if int(r_labels[b]) != -100:
                    region_preds_all.append(int(r_preds[b]))
                    region_labels_all.append(int(r_labels[b]))

    # --- Compile Metrics ---
    metrics = {}

    # Restoration
    n = all_restore_total
    metrics["restoration"] = {
        "total_predictions": n,
        "top1_accuracy": round(hit_at[1] / n, 4) if n > 0 else 0.0,
        "top3_accuracy": round(hit_at[3] / n, 4) if n > 0 else 0.0,
        "top5_accuracy": round(hit_at[5] / n, 4) if n > 0 else 0.0,
    }

    # Date
    if date_pred_peaks:
        pred_bins = np.array(date_pred_peaks)
        true_bins = np.array(date_true_peaks)
        bin_mae = float(np.mean(np.abs(pred_bins - true_bins)))
        metrics["date"] = {
            "total_samples": len(pred_bins),
            "bin_mae": round(bin_mae, 4),
            "years_mae": round(bin_mae * 50, 1),
            "exact_bin_accuracy": round(float(np.mean(pred_bins == true_bins)), 4),
        }

    # Region
    if region_preds_all:
        r_preds_arr = np.array(region_preds_all)
        r_labels_arr = np.array(region_labels_all)
        metrics["region"] = {
            "total_samples": len(r_preds_arr),
            "accuracy": round(float(np.mean(r_preds_arr == r_labels_arr)), 4),
        }
        # Per-class accuracy
        per_class = {}
        for cls_idx, cls_name in REGION_NAMES.items():
            mask = r_labels_arr == cls_idx
            if mask.sum() > 0:
                per_class[cls_name] = {
                    "count": int(mask.sum()),
                    "accuracy": round(float(np.mean(r_preds_arr[mask] == cls_idx)), 4),
                }
        metrics["region"]["per_class"] = per_class

    # Save CSV report
    if rows:
        csv_path = output_dir / "test_a_predictions.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
        log.info(f"Saved Test A predictions CSV: {csv_path}")

    return metrics


# ============================================================================
# TEST B EVALUATION: Real historical lacunae
# ============================================================================

def evaluate_test_b(
    model: Kyivan,
    dataset: Any,
    char_vocab: Dict[str, int],
    allowed_ids: set,
    output_dir: Path,
    max_samples: int = 1188,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """
    Evaluate on Test B — real historical lacunae (pre-masked with known labels).
    
    Computes:
      - Restoration: Top-1/3/5 accuracy on genuine archaeological gaps
      - CSV report with detailed per-lacuna predictions and context
    """
    log.info("=" * 60)
    log.info("EVALUATING TEST B (Real Historical Lacunae)")
    log.info("=" * 60)

    id_to_char = {int(v): k for k, v in char_vocab.items()}
    total_samples = min(len(dataset), max_samples)

    correct = 0
    total = 0
    hit_at = {1: 0, 3: 0, 5: 0}
    rows = []

    # Date & Region for Test B
    date_pred_peaks = []
    date_true_peaks = []
    region_preds_all = []
    region_labels_all = []

    for i in tqdm(range(total_samples), desc="Test B"):
        sample = dataset[i]
        input_ids = torch.tensor(sample["input_ids"], dtype=torch.long, device=device)
        attention_mask = torch.tensor(sample["attention_mask"], dtype=torch.long, device=device)
        labels = sample.get("labels", None)

        if labels is None:
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

        ids_list = input_ids.cpu().numpy().tolist()

        for pos in mask_positions:
            true_id = int(labels[pos])
            if true_id < 0:
                continue

            pred_logits = mask_logits(logits[pos], allowed_ids)
            top_ids = torch.topk(pred_logits, k=5).indices.cpu().numpy()
            pred_id = int(top_ids[0])

            true_char = id_to_char.get(true_id, "?")
            pred_char = id_to_char.get(pred_id, "?")
            if pred_char.startswith("[") or true_char.startswith("["):
                continue

            total += 1
            is_correct = pred_id == true_id
            if is_correct:
                correct += 1

            for k in [1, 3, 5]:
                if true_id in top_ids[:k]:
                    hit_at[k] += 1

            # Context
            ctx_start = max(0, pos - 20)
            ctx_end = min(len(ids_list), pos + 21)
            before = "".join(id_to_char.get(int(c), "?") for c in ids_list[ctx_start:pos])
            after = "".join(id_to_char.get(int(c), "?") for c in ids_list[pos+1:ctx_end])
            context = before + ">>>" + id_to_char.get(int(ids_list[pos]), "?") + "<<<" + after

            probs = torch.softmax(logits[pos], dim=-1)
            top_chars = [id_to_char.get(int(tid), "?") for tid in top_ids[:5]]
            true_rank = next((r + 1 for r, tid in enumerate(top_ids) if int(tid) == true_id), None)

            rows.append({
                "sample_idx": i, "position": pos, "context": context,
                "true_char": true_char, "pred_char": pred_char,
                "is_correct": is_correct, "true_rank": true_rank,
                "top1_prob": round(float(probs[pred_id].item()), 4),
                "top5_preds": "|".join(top_chars),
            })

        # Date & Region for this sample
        if "date_labels" in sample and sample["date_labels"] is not None:
            date_probs = F.softmax(outputs.logits_date[0], dim=-1).cpu().numpy()
            true_date = np.array(sample["date_labels"])
            date_pred_peaks.append(np.argmax(date_probs))
            date_true_peaks.append(np.argmax(true_date))

        if "region_labels" in sample and sample["region_labels"] is not None:
            r_pred = torch.argmax(outputs.logits_region[0]).item()
            r_label = int(sample["region_labels"])
            if r_label != -100:
                region_preds_all.append(r_pred)
                region_labels_all.append(r_label)

    # --- Compile Metrics ---
    n = total
    metrics = {
        "restoration": {
            "total_predictions": n,
            "top1_accuracy": round(hit_at[1] / n, 4) if n > 0 else 0.0,
            "top3_accuracy": round(hit_at[3] / n, 4) if n > 0 else 0.0,
            "top5_accuracy": round(hit_at[5] / n, 4) if n > 0 else 0.0,
        }
    }

    # Date
    if date_pred_peaks:
        pred_bins = np.array(date_pred_peaks)
        true_bins = np.array(date_true_peaks)
        bin_mae = float(np.mean(np.abs(pred_bins - true_bins)))
        metrics["date"] = {
            "total_samples": len(pred_bins),
            "bin_mae": round(bin_mae, 4),
            "years_mae": round(bin_mae * 50, 1),
            "exact_bin_accuracy": round(float(np.mean(pred_bins == true_bins)), 4),
        }

    # Region
    if region_preds_all:
        r_preds_arr = np.array(region_preds_all)
        r_labels_arr = np.array(region_labels_all)
        metrics["region"] = {
            "total_samples": len(r_preds_arr),
            "accuracy": round(float(np.mean(r_preds_arr == r_labels_arr)), 4),
        }
        per_class = {}
        for cls_idx, cls_name in REGION_NAMES.items():
            mask = r_labels_arr == cls_idx
            if mask.sum() > 0:
                per_class[cls_name] = {
                    "count": int(mask.sum()),
                    "accuracy": round(float(np.mean(r_preds_arr[mask] == cls_idx)), 4),
                }
        metrics["region"]["per_class"] = per_class

    # Save CSV report
    if rows:
        csv_path = output_dir / "test_b_predictions.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
        log.info(f"Saved Test B predictions CSV: {csv_path}")

    return metrics


# ============================================================================
# MAIN
# ============================================================================

def print_metrics_summary(test_name: str, metrics: Dict[str, Any]) -> None:
    """Pretty-print metrics to the console."""
    print(f"\n{'=' * 60}")
    print(f"  {test_name} RESULTS")
    print(f"{'=' * 60}")

    if "restoration" in metrics:
        r = metrics["restoration"]
        print(f"\n  [Restoration]")
        print(f"    Total predictions:   {r['total_predictions']}")
        print(f"    Top-1 Accuracy:      {r['top1_accuracy']:.4f}  ({r['top1_accuracy']*100:.2f}%)")
        print(f"    Top-3 Accuracy:      {r['top3_accuracy']:.4f}  ({r['top3_accuracy']*100:.2f}%)")
        print(f"    Top-5 Accuracy:      {r['top5_accuracy']:.4f}  ({r['top5_accuracy']*100:.2f}%)")

    if "date" in metrics:
        d = metrics["date"]
        print(f"\n  [Date Prediction]")
        print(f"    Samples evaluated:   {d['total_samples']}")
        print(f"    Bin MAE:             {d['bin_mae']:.2f} bins")
        print(f"    Years MAE:           {d['years_mae']:.1f} years")
        print(f"    Exact bin accuracy:  {d['exact_bin_accuracy']:.4f}  ({d['exact_bin_accuracy']*100:.2f}%)")

    if "region" in metrics:
        rg = metrics["region"]
        print(f"\n  [Region/Dialect Classification]")
        print(f"    Samples evaluated:   {rg['total_samples']}")
        print(f"    Overall accuracy:    {rg['accuracy']:.4f}  ({rg['accuracy']*100:.2f}%)")
        if "per_class" in rg:
            for cls_name, cls_data in rg["per_class"].items():
                print(f"      {cls_name}: {cls_data['accuracy']:.4f} ({cls_data['accuracy']*100:.1f}%)  [n={cls_data['count']}]")

    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Kyivan Model Evaluation")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to the checkpoint directory")
    parser.add_argument("--dataset_dir", type=str, default="prepared_datasets/hf_dataset",
                        help="Path to the HF dataset on disk")
    parser.add_argument("--char_vocab_path", type=str,
                        default="prepared_datasets/tokenizer/char_vocab.json",
                        help="Path to character vocabulary JSON")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory for output metrics and CSV reports")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for Test A evaluation")
    parser.add_argument("--skip_test_a", action="store_true",
                        help="Skip Test A evaluation")
    parser.add_argument("--skip_test_b", action="store_true",
                        help="Skip Test B evaluation")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (default: auto-detect)")
    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Load resources
    char_vocab = load_json(args.char_vocab_path)
    allowed_ids = get_allowed_pred_ids(char_vocab)
    dataset = load_from_disk(args.dataset_dir)
    model = load_model(args.checkpoint_dir, device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_metrics = {"checkpoint": args.checkpoint_dir, "timestamp": timestamp}

    # --- Test A ---
    if not args.skip_test_a and "test_a" in dataset:
        collator = KyivanPhysicalCollator(
            char_vocab=char_vocab,
            mlm_prob=0.15,
            span_mask_ratio=0.2,
            span_geometric_p=0.2,
            edge_prob=0.1,
        )
        test_a_metrics = evaluate_test_a(
            model=model,
            dataset=dataset["test_a"],
            char_vocab=char_vocab,
            allowed_ids=allowed_ids,
            output_dir=output_dir,
            collator=collator,
            batch_size=args.batch_size,
            device=device,
        )
        all_metrics["test_a"] = test_a_metrics
        print_metrics_summary("TEST A (Dynamic Masking)", test_a_metrics)

    # --- Test B ---
    if not args.skip_test_b and "test_b" in dataset:
        test_b_metrics = evaluate_test_b(
            model=model,
            dataset=dataset["test_b"],
            char_vocab=char_vocab,
            allowed_ids=allowed_ids,
            output_dir=output_dir,
            device=device,
        )
        all_metrics["test_b"] = test_b_metrics
        print_metrics_summary("TEST B (Real Historical Lacunae)", test_b_metrics)

    # Save all metrics
    metrics_path = output_dir / f"eval_metrics_{timestamp}.json"
    save_json(all_metrics, metrics_path)
    log.info(f"All metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
