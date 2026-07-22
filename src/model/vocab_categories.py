"""
Shared character-category policy for the Kyivan vocabulary.

Single source of truth for which characters are eligible to be masked as
restoration targets and which the restore head is allowed to predict.
Before this module existed, the same `("Ll", "Lu", "Lo", "Zs")` tuple was
duplicated independently in four places -- collator_v2.py (what gets
masked), train.py (what the restore head is allowed to predict/score),
inference.py (what restoration is forbidden from generating), and
prepare_splits.py (what Test B's real editorial brackets count as
maskable) -- that could silently drift out of sync. Changing
MASKABLE_CATEGORIES here now updates all four at once.
"""

import unicodedata
from typing import Dict, Set

# Letters + plain space are always maskable.
_LETTER_CATEGORIES = ("Ll", "Lu", "Lo", "Zs")

# Single on/off switch for punctuation: flip this and MASKABLE_CATEGORIES
# below (and therefore collator_v2/train.py/inference.py/prepare_splits.py)
# all follow, with no other file to edit.
MASK_PUNCTUATION = True

# Po covers this corpus's real word-dividers (`·` middle dot, `:` colon);
# Sm covers `+` (the cross at the start of many letters). Chosen narrowly to
# match what's actually meaningful punctuation here, not the full Unicode
# punctuation surface -- digits, brackets, hyphen, etc. stay excluded even
# when MASK_PUNCTUATION is on.
_PUNCTUATION_CATEGORIES = ("Po", "Sm")

MASKABLE_CATEGORIES = _LETTER_CATEGORIES + (
    _PUNCTUATION_CATEGORIES if MASK_PUNCTUATION else ()
)


def is_maskable_char(ch: str) -> bool:
    """True for a single character whose Unicode category is maskable."""
    return len(ch) == 1 and unicodedata.category(ch) in MASKABLE_CATEGORIES


def maskable_ids(char_vocab: Dict[str, int]) -> Set[int]:
    """Vocab ids of single characters whose Unicode category is maskable."""
    return {
        int(v) for k, v in char_vocab.items() if is_maskable_char(k)
    }
