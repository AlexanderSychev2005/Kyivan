"""
Character-Level Tokenizer Builder.

This script builds a character-level vocabulary required for the non-autoregressive
Kyivan model. Unlike standard subword tokenizers (like WordPiece or BPE),
this tokenizer operates strictly on individual characters. This approach is highly
efficient for handling heavily inflected ancient Slavic languages and physical text
degradations (e.g., missing fragments on birch bark).

Key Kyivan-specific special tokens included:
- `[SOS]`: Start of Sequence. Used globally by the model to predict dialect and date.
- `[-]`: Single character mask (replaces the traditional `[MASK]`).
- `[#]`: Lacuna token representing an unknown number of missing characters.
"""

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterator, List

# Define the exhaustive list of special tokens to be added to the vocabulary.
# These tokens will not be split into characters during tokenization.
SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[SOS]",
    "[-]",
    "[#]",
    "[GAP]",
]

# Regular expression used to isolate the bracket-wrapped special tokens so
# they remain intact (never split into individual characters) and are
# excluded from the character-frequency count below -- they're added to the
# vocab separately, as whole tokens, via SPECIAL_TOKENS.
#
# Deliberately does NOT list punctuation here (this used to also match
# `+:·`, which meant they got skipped by the `SPECIAL_RE.fullmatch(part):
# continue` check in collect_chars below exactly like a real special token,
# silently dropping real corpus characters -- e.g. `·`, the actual word
# divider in these texts, occurring ~240k times -- out of the vocab
# entirely, regardless of min_freq). Whether a character makes it into the
# vocab should depend only on how often it actually occurs in the corpus;
# whether it's then eligible for masking/prediction is a separate, later
# decision governed by vocab_categories.MASKABLE_CATEGORIES.
SPECIAL_RE = re.compile(r"(\[GAP\]|\[SOS\]|\[-\]|\[#\]|\[PAD\]|\[UNK\])")

# Cased-letter code-point ranges excluded from the vocabulary entirely (not
# just from masking/prediction). Old East Slavic manuscripts are the
# restoration target; Latin and Greek letters in this corpus are ~99.98%
# single-source noise -- OCR misreads and citations/bibliography in the
# scholarly source editions, not intentional text in the restored language
# (measured: 70 Latin + 25 Greek vocab entries cover 3,993 of 15.6M corpus
# characters, and 24 of those 95 entries occur exactly once). Excluding them
# here means they become [UNK] wherever they occur, in input context as well
# as in prediction targets -- stronger than just gating them out of
# MASKABLE_CATEGORIES, which would still leave them cluttering the embedding
# table and the restore head's output classes.
_EXCLUDED_SCRIPT_RANGES = (
    (0x0041, 0x024F),  # Latin: Basic Latin, Latin-1 Supplement, Extended-A/B
    (0x0370, 0x03FF),  # Greek and Coptic
    (0x1F00, 0x1FFF),  # Greek Extended
)


def _is_excluded_script(ch: str) -> bool:
    if unicodedata.category(ch) not in ("Ll", "Lu"):
        return False
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _EXCLUDED_SCRIPT_RANGES)


def iter_lines(path: Path) -> Iterator[str]:
    """
    Reads a file line by line and yields textual content.
    Gracefully handles both plain text and JSONL formats.

    Args:
        path (Path): The path to the input corpus file.

    Yields:
        str: A single text line, or the 'text' field content if reading a JSONL file.
    """
    is_jsonl = path.suffix == ".jsonl"

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if is_jsonl:
                try:
                    data = json.loads(line)
                    # Extract only the target text to avoid adding JSON syntax to the vocab
                    yield data.get("text", "")
                except json.JSONDecodeError:
                    continue
            else:
                yield line


def collect_chars(path: Path, min_freq: int) -> List[str]:
    """
    Extracts and counts individual characters from the corpus, ignoring predefined
    special tokens, case-folded to lowercase, with Latin/Greek letters dropped.

    Args:
        path (Path): The path to the training corpus.
        min_freq (int): The minimum occurrence frequency required for a character to be included.

    Returns:
        List[str]: A sorted list of unique characters that meet the frequency threshold.
    """
    counter = Counter()

    for line in iter_lines(path):
        # Split the sequence while keeping special tokens separated
        parts = SPECIAL_RE.split(line)

        for part in parts:
            if not part:
                continue

            # Skip counting if the segment exactly matches a special token
            if SPECIAL_RE.fullmatch(part):
                continue

            # Count each raw character in the remaining standard text.
            # Case-fold to lowercase -- these manuscripts had no case
            # distinction of their own; upper/lower is a modern
            # transcription artifact, so an uppercase occurrence should
            # count toward the same vocab entry as its lowercase form
            # rather than splitting the restore head's target classes (and
            # the model's training signal) in two for no linguistic reason.
            # Latin/Greek letters are dropped outright, not merely folded.
            for ch in part:
                ch = ch.lower()
                if _is_excluded_script(ch):
                    continue
                counter[ch] += 1

    # Filter characters by the provided frequency threshold
    chars = [ch for ch, c in counter.items() if c >= min_freq]
    chars.sort()

    return chars


def main() -> None:
    """
    Parses CLI arguments, extracts the character vocabulary from the corpus,
    and saves the resulting vocabulary and configuration JSON files to disk.
    """
    parser = argparse.ArgumentParser(
        description="Build Character Vocabulary for Kyivan"
    )
    parser.add_argument(
        "--train_path", default="splits/train.jsonl", help="Path to the training data"
    )
    parser.add_argument(
        "--out_dir", default="artifacts/char_tokenizer", help="Output directory"
    )
    parser.add_argument(
        "--min_freq", type=int, default=1, help="Minimum character frequency"
    )
    args = parser.parse_args()

    train_path = Path(args.train_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Collect valid characters from the corpus
    chars = collect_chars(train_path, args.min_freq)

    # 2. Merge special tokens and collected characters while maintaining uniqueness
    vocab_tokens = []
    seen = set()

    for tok in SPECIAL_TOKENS + chars:
        if tok not in seen:
            seen.add(tok)
            vocab_tokens.append(tok)

    # 3. Create the final vocabulary mapping (token -> integer ID)
    vocab = {tok: i for i, tok in enumerate(vocab_tokens)}

    # 4. Define the tokenizer configuration map for HuggingFace compatibility
    cfg = {
        "special_tokens": SPECIAL_TOKENS,
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
        "sos_token": "[SOS]",
        "mask_token": "[-]",
        "unk_gap_token": "[#]",
    }

    # 5. Export artifacts to disk
    (out_dir / "char_vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "tokenizer_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Saved char vocab: {out_dir / 'char_vocab.json'}")
    print(f"Vocab size: {len(vocab)}")


if __name__ == "__main__":
    main()
