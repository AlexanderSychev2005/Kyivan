"""
Kyivan Iterative-Decode Evaluator.

Scores a checkpoint the way it actually restores text in production: via
KyivanRestorer.iterative_decode's confidence-based greedy loop (fill the
single [-] the model is currently most confident about, refeed the whole
sequence, repeat) -- not the one-shot "predict every masked position
independently in a single forward pass" scoring that compute_metrics/
generate_predictions_report use during training. Kyivan is a bidirectional
encoder (attends both directions), not a decoder like Aeneas, so later
fills legitimately condition on the model's own earlier guesses; one-shot
scoring can't measure whether that actually helps.

Test B (real historical lacunae) is the primary target here: every real
gap is already expanded into individual [-] tokens by prepare_splits.py's
process_test_b_line (the true reconstructed length is known from the
editorial brackets themselves), so decoding never needs to grow the
sequence and every position stays aligned with `labels` throughout --
iterative_decode can be used unmodified with no position bookkeeping.

Test A (dynamic masking) can *also* draw a [#] unknown-length gap
(KyivanPhysicalCollatorV2._pick_compressed_gap runs in eval mode too),
which does grow the sequence as the model expands it -- those draws are
skipped here rather than tracked through the insertion, since a synthetic
gap has no real "correct expansion length" to score against anyway (the
collator's own unk_label is just one binary expand/stop bit, not a full
target string).

Slow by design: one forward pass per resolved [-] position, not one pass
per document. Use --max_samples to bound runtime.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from datasets import load_from_disk

from src.model.collator_v2 import KyivanPhysicalCollatorV2
from src.model.inference import KyivanRestorer


def evaluate_test_b(restorer: KyivanRestorer, dataset: Any, max_samples: int, max_steps: int) -> Dict[str, float]:
    correct = used = 0
    n = min(len(dataset), max_samples)
    for i in range(n):
        sample = dataset[i]
        tokens = sample["input_ids"]
        labels = sample["labels"]
        if restorer.mask_id not in tokens:
            continue

        final_tokens = restorer.iterative_decode(tokens, max_steps=max_steps)
        for pos, true_id in enumerate(labels):
            if true_id == -100:
                continue
            used += 1
            if final_tokens[pos] == true_id:
                correct += 1

        if (i + 1) % 20 == 0:
            print(f"  test_b {i + 1}/{n}...")

    return {
        "total": used,
        "correct": correct,
        "accuracy": round(correct / used, 4) if used else 0.0,
    }


def evaluate_test_a(
    restorer: KyivanRestorer,
    dataset: Any,
    char_vocab: Dict[str, int],
    max_samples: int,
    max_steps: int,
) -> Dict[str, float]:
    collator = KyivanPhysicalCollatorV2(char_vocab, mode="valid")
    correct = used = 0
    skipped_gap = 0
    n = min(len(dataset), max_samples)
    for i in range(n):
        batch = collator([dataset[i]])
        length = int(batch["attention_mask"][0].sum().item())
        tokens = batch["input_ids"][0][:length].tolist()
        labels = batch["labels"][0][:length].tolist()

        if restorer.unk_id in tokens:
            # Sequence would grow mid-decode -- see module docstring.
            skipped_gap += 1
            continue
        if restorer.mask_id not in tokens:
            continue

        final_tokens = restorer.iterative_decode(tokens, max_steps=max_steps)
        for pos, true_id in enumerate(labels):
            if true_id == -100:
                continue
            used += 1
            if final_tokens[pos] == true_id:
                correct += 1

        if (i + 1) % 20 == 0:
            print(f"  test_a {i + 1}/{n}...")

    return {
        "total": used,
        "correct": correct,
        "accuracy": round(correct / used, 4) if used else 0.0,
        "skipped_gap_draws": skipped_gap,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Kyivan Iterative-Decode Evaluator")
    parser.add_argument("--model_dir", required=True, help="Path to the trained model directory")
    parser.add_argument(
        "--vocab", default="prepared_datasets/tokenizer/char_vocab.json",
        help="Path to the character vocabulary JSON",
    )
    parser.add_argument("--dataset_dir", default="prepared_datasets/hf_dataset")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=100, help="Safety cap per document")
    parser.add_argument("--skip_test_a", action="store_true")
    parser.add_argument("--skip_test_b", action="store_true")
    args = parser.parse_args()

    restorer = KyivanRestorer(
        model_dir=args.model_dir, char_vocab_path=args.vocab, device=args.device
    )
    dataset = load_from_disk(args.dataset_dir)

    if not args.skip_test_a and "test_a" in dataset:
        with open(args.vocab, "r", encoding="utf-8") as f:
            char_vocab = json.load(f)
        metrics_a = evaluate_test_a(
            restorer, dataset["test_a"], char_vocab, args.max_samples, args.max_steps
        )
        print(f"\nTEST A (iterative decode): {metrics_a}")

    if not args.skip_test_b and "test_b" in dataset:
        metrics_b = evaluate_test_b(
            restorer, dataset["test_b"], args.max_samples, args.max_steps
        )
        print(f"\nTEST B (iterative decode): {metrics_b}")


if __name__ == "__main__":
    main()
