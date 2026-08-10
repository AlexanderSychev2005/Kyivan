"""
Kyivan CER (Character Error Rate) Evaluation.

Unlike generate_predictions_report (train.py) -- which scores independent
per-position top-1/3/5 accuracy from a single forward pass -- this measures
whole-lacuna reconstruction quality: for each contiguous masked run
("lacuna") in Test B, actually run the real decoder (KyivanBeamRestorer.
restore_text_beam) to produce a predicted STRING, then compare it to the
true string via Levenshtein edit distance. This is the standard metric for
comparing against other text-restoration work (Aeneas/Ithaca included),
and unlike per-position accuracy it naturally handles length mismatches --
relevant for the `[#]` (unknown-length) case, where the model might guess
a different length than the truth.

beam_width=1 (default) still routes through KyivanBeamRestorer.
restore_text_beam rather than the simpler KyivanRestorer.iterative_decode
that inference.py now exposes: the two aren't equivalent once a `[#]`
expands. iterative_decode hard-prioritizes any open `[#]` over every `[-]`
in the sequence, while restore_text_beam scores `[#]` and `[-]` branches in
the same pool and lets the highest-scoring one win -- same result while a
query has only one open marker type (true here, since each query isolates
a single lacuna), but not guaranteed to match once expansion creates both
kinds of open position at once. Using restore_text_beam uniformly for
beam_width=1 and >1 keeps CER numbers comparable across --beam_width runs.

Two modes, both using the same decoder:
  - Default (revealed length): each lacuna is presented as `true_len`
    individual `[-]` marks -- the model knows exactly how many characters
    are missing, only their identity is unknown.
  - --compress_unk: each lacuna is collapsed into a single `[#]` instead --
    the model must also discover the length itself via the unk head,
    matching what a real, undocumented physical gap looks like.

Every OTHER lacuna in a sample is filled with its ground truth (not left
masked) while evaluating a given one, so one lacuna's reconstruction error
never cascades into another's -- CER is measured per lacuna in isolation,
not confounded by unresolved neighbors.

Results are bucketed by true lacuna length (1..--max_bucket_len, plus a
">max_bucket_len" catch-all) so you can see how CER degrades with gap size.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from datasets import load_from_disk

from src.model.inference_beam import KyivanBeamRestorer


def levenshtein(a: str, b: str) -> int:
    """Standard O(len(a)*len(b)) edit distance -- no new dependency for
    what's always a short string (lacunae top out around 20-25 chars)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def build_clean_sequence(input_ids: List[int], labels: List[int]) -> List[int]:
    """Every masked position filled with its true character -- the fully
    reconstructed original document, used as the "everything else is
    ground truth" context when isolating one lacuna at a time."""
    return [
        labels[i] if labels[i] != -100 else input_ids[i] for i in range(len(input_ids))
    ]


def find_lacunae(labels: List[int]) -> List[Tuple[int, int]]:
    """Contiguous runs of masked positions -> [(start, end), ...] (end exclusive)."""
    runs = []
    start = None
    for i, label in enumerate(labels):
        if label != -100:
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(labels)))
    return runs


def evaluate_cer(
    restorer: KyivanBeamRestorer,
    dataset,
    max_samples: Optional[int],
    beam_width: int,
    compress_unk: bool,
    max_bucket_len: int,
) -> Dict:
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    buckets: Dict[int, Dict[str, int]] = {}
    total_edit = total_len = n_lacunae = n_skipped = 0

    for i in range(n):
        if i % 20 == 0:
            print(f"Processing {i}/{n}...")
        ex = dataset[i]
        input_ids = ex["input_ids"]
        labels = ex["labels"]
        clean = build_clean_sequence(input_ids, labels)

        for start, end in find_lacunae(labels):
            true_len = end - start
            true_str = restorer.decode(labels[start:end])
            if len(true_str) != true_len:
                # A true label decoded to "" (e.g. [UNK] -- an original
                # character outside the vocab) -- can't score CER against
                # an undefined reference, skip this lacuna.
                n_skipped += 1
                continue

            # Built and sliced entirely in token space -- no decode->text->
            # _tokenize round trip, which isn't safe here: re-running
            # normalize_historical_text on already-clean context text isn't
            # guaranteed to be a no-op (it can still match patterns like
            # _HYPHEN_GAP_RE/_ELLIPSIS_RE in genuine historical text and
            # alter it), which would silently corrupt the "everything else
            # is ground truth" context this evaluation depends on.
            before_tokens = clean[:start]
            after_tokens = clean[end:]
            marker_tokens = (
                [restorer.unk_id] if compress_unk else [restorer.mask_id] * true_len
            )
            query_tokens = before_tokens + marker_tokens + after_tokens

            try:
                results = restorer.restore_text_beam(
                    tokens=query_tokens,
                    beam_width=beam_width,
                    top_n=1,
                    unk_restoration_max_len=max(true_len * 3, 20),
                )
            except ValueError:
                n_skipped += 1
                continue
            if not results:
                n_skipped += 1
                continue

            final_tokens = results[0].history[-1]
            end_cut = (
                len(final_tokens) - len(after_tokens)
                if after_tokens
                else len(final_tokens)
            )
            pred_str = restorer.decode(final_tokens[len(before_tokens) : end_cut])

            edit = levenshtein(pred_str, true_str)
            bucket = min(true_len, max_bucket_len + 1)
            b = buckets.setdefault(bucket, {"edit": 0, "len": 0, "n": 0})
            b["edit"] += edit
            b["len"] += true_len
            b["n"] += 1
            total_edit += edit
            total_len += true_len
            n_lacunae += 1

    per_bucket = {}
    for length, b in sorted(buckets.items()):
        key = f">{max_bucket_len}" if length == max_bucket_len + 1 else str(length)
        per_bucket[key] = {
            "n": b["n"],
            "cer": round(b["edit"] / b["len"], 4) if b["len"] > 0 else 0.0,
        }

    return {
        "n_lacunae": n_lacunae,
        "n_skipped": n_skipped,
        "overall_cer": round(total_edit / total_len, 4) if total_len > 0 else 0.0,
        "per_length": per_bucket,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Kyivan CER Evaluation (Test B)")
    parser.add_argument("--model_dir", default="outputs/kyivan_run5/final_model")
    parser.add_argument("--vocab", default="prepared_datasets/tokenizer/char_vocab.json")
    parser.add_argument("--dataset_dir", default="prepared_datasets/hf_dataset")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--beam_width",
        type=int,
        default=1,
        help="1 = greedy decoding (default), >1 = real beam search",
    )
    parser.add_argument(
        "--compress_unk",
        action="store_true",
        help="Collapse each lacuna into a single [#] (unknown length, model "
        "must discover it) instead of revealing the true length as "
        "individual [-] marks (default).",
    )
    parser.add_argument(
        "--max_bucket_len",
        type=int,
        default=20,
        help="Report CER per true-length bucket 1..this, plus a '>this' catch-all",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap on Test B documents processed (default: all)",
    )
    parser.add_argument("--output_json", default=None)

    args = parser.parse_args()

    print(f"Loading model from {args.model_dir}...")
    restorer = KyivanBeamRestorer(
        model_dir=args.model_dir, char_vocab_path=args.vocab, device=args.device
    )

    print(f"Loading Test B from {args.dataset_dir}...")
    dataset = load_from_disk(args.dataset_dir)["test_b"]

    results = evaluate_cer(
        restorer,
        dataset,
        max_samples=args.max_samples,
        beam_width=args.beam_width,
        compress_unk=args.compress_unk,
        max_bucket_len=args.max_bucket_len,
    )

    print(f"\n--- CER RESULTS (beam_width={args.beam_width}, "
          f"compress_unk={args.compress_unk}) ---")
    print(f"Lacunae scored: {results['n_lacunae']} (skipped: {results['n_skipped']})")
    print(f"Overall CER (micro-average): {results['overall_cer']}")
    print("\nBy true length:")
    for length, stats in results["per_length"].items():
        print(f"  {length:>3}: n={stats['n']:<5} CER={stats['cer']}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved: {args.output_json}")


if __name__ == "__main__":
    main()
