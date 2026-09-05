"""
Two follow-up evaluations requested for the paper, both reusing
evaluate_cer.py's machinery instead of duplicating it:

1. CER on Test A, under the same deterministic eval-mode masking already
   used for its other reported metrics (top1/3/5, region, date, unk) --
   KyivanPhysicalCollatorV2(mode="valid") seeds each document from its own
   tokens XOR --seed, so materializing it once here reproduces exactly the
   masking those other Test A numbers were already scored against.

2. Gap-expansion (unk) step accuracy/macro-F1 on Test B. Test B's real
   lacunae are always revealed-length by construction (process_test_b_line
   only ever emits individual [-] marks -- see prepare_splits.py), so unlike
   Test A they never contain a genuine [#] token for compute_metrics/
   generate_predictions_report to score the unk head against. Each real
   lacuna is instead recast as a single [#], labeled multi (1) vs single (0)
   by the exact same rule collator_v2.py itself uses when it builds a
   compressed gap (gap_is_multi = len(positions) > 1), and scored with one
   forward pass -- isolating one lacuna at a time with every other lacuna
   filled with its ground truth, identically to evaluate_cer.py's CER loop.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from datasets import load_from_disk

from src.model.collator_v2 import KyivanPhysicalCollatorV2
from src.model.evaluate_cer import build_clean_sequence, evaluate_cer, find_lacunae
from src.model.inference_beam import KyivanBeamRestorer
from src.model.vocab_categories import tokenize_text


def materialize_test_a(dataset_dir: str, vocab: Dict[str, int], seed: int, max_len: int,
                        crop_min_len: int, edge_prob: float, unk_geometric_p: float,
                        span_mask_eval_len: int) -> List[Dict]:
    ds = load_from_disk(dataset_dir)["test_a"]
    collator = KyivanPhysicalCollatorV2(
        char_vocab=vocab, crop_max_len=max_len, crop_min_len=crop_min_len,
        edge_prob=edge_prob, unk_geometric_p=unk_geometric_p,
        span_mask_eval_len=span_mask_eval_len, eval_seed=seed, mode="valid",
    )
    records = []
    for row in ds:
        ids = tokenize_text(row["text"], vocab)
        tokens, labels_res, _ = collator._process_one(ids)
        records.append({"input_ids": tokens, "labels": labels_res})
    return records


def evaluate_gap_expansion(restorer: KyivanBeamRestorer, dataset, max_samples=None) -> Dict:
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    class_stats = {0: {"tp": 0, "fp": 0, "fn": 0}, 1: {"tp": 0, "fp": 0, "fn": 0}}
    correct = total = 0

    for i in range(n):
        if i % 40 == 0:
            print(f"Processing {i}/{n}...")
        ex = dataset[i]
        clean = build_clean_sequence(ex["input_ids"], ex["labels"])

        for start, end in find_lacunae(ex["labels"]):
            true_label = 1 if (end - start) > 1 else 0
            query = clean[:start] + [restorer.unk_id] + clean[end:]
            input_ids = torch.tensor([query], dtype=torch.long, device=restorer.device)
            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                out = restorer.model(input_ids=input_ids, attention_mask=attention_mask)
            pred_label = int(out.logits_unk[0, start].argmax().item())

            total += 1
            correct += int(pred_label == true_label)
            for c in (0, 1):
                if true_label == c and pred_label == c:
                    class_stats[c]["tp"] += 1
                elif true_label == c and pred_label != c:
                    class_stats[c]["fn"] += 1
                elif true_label != c and pred_label == c:
                    class_stats[c]["fp"] += 1

    f1s = []
    for c in (0, 1):
        s = class_stats[c]
        prec = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) else 0.0
        rec = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)

    return {
        "n": total,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "macro_f1": round(sum(f1s) / len(f1s), 4),
        "class_stats": class_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test A CER / Test B gap-expansion follow-ups")
    parser.add_argument("--task", choices=["cer_test_a", "gap_expansion_test_b"], required=True)
    parser.add_argument("--model_dir", default="runs/kyivan_h224_mask018_300ep/final_model")
    parser.add_argument("--vocab", default="prepared_datasets/tokenizer/char_vocab.json")
    parser.add_argument("--dataset_dir", default="prepared_datasets/hf_dataset")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--crop_min_len", type=int, default=128)
    parser.add_argument("--edge_prob", type=float, default=0.1)
    parser.add_argument("--unk_geometric_p", type=float, default=0.25)
    parser.add_argument("--span_mask_eval_len", type=int, default=10)
    parser.add_argument("--beam_width", type=int, default=1)
    parser.add_argument("--compress_unk", action="store_true")
    parser.add_argument("--max_bucket_len", type=int, default=20)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    with open(args.vocab, encoding="utf-8") as f:
        vocab = json.load(f)

    print(f"Loading model from {args.model_dir}...")
    restorer = KyivanBeamRestorer(model_dir=args.model_dir, char_vocab_path=args.vocab, device=args.device)

    if args.task == "cer_test_a":
        print("Materializing Test A under deterministic eval-mode masking...")
        records = materialize_test_a(
            args.dataset_dir, vocab, args.seed, args.max_len, args.crop_min_len,
            args.edge_prob, args.unk_geometric_p, args.span_mask_eval_len,
        )
        results = evaluate_cer(
            restorer, records, max_samples=args.max_samples, beam_width=args.beam_width,
            compress_unk=args.compress_unk, max_bucket_len=args.max_bucket_len,
        )
        print(f"\n--- Test A CER (compress_unk={args.compress_unk}) ---")
        print(f"Lacunae scored: {results['n_lacunae']} (skipped: {results['n_skipped']})")
        print(f"Overall CER (micro-average): {results['overall_cer']}")
        for length, stats in results["per_length"].items():
            print(f"  {length:>3}: n={stats['n']:<5} CER={stats['cer']}")
    else:
        print("Loading Test B...")
        dataset = load_from_disk(args.dataset_dir)["test_b"]
        results = evaluate_gap_expansion(restorer, dataset, max_samples=args.max_samples)
        print("\n--- Test B gap-expansion (unk) accuracy ---")
        print(f"n={results['n']}  accuracy={results['accuracy']}  macro_f1={results['macro_f1']}")
        print(f"class_stats: {results['class_stats']}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved: {args.output_json}")


if __name__ == "__main__":
    main()
