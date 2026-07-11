import re
import unicodedata

_LAT_TO_CYR = {
    "A": "А",
    "a": "а",
    "B": "В",
    "b": "в",
    "E": "Е",
    "e": "е",
    "K": "К",
    "k": "к",
    "M": "М",
    "m": "м",
    "H": "Н",
    "n": "н",
    "O": "О",
    "o": "о",
    "P": "Р",
    "p": "р",
    "C": "С",
    "c": "с",
    "T": "Т",
    "t": "т",
    "y": "у",
    "x": "х",
    "X": "Х",
    "i": "і",
    "I": "І",
}

# Canonicalize rare chars
_RARE_CHAR_MAP = {
    "†": "+",
    "×": "+",
    "⁘": ":",
    "⁙": ":",
    "⁞": ":",
    "¦": ":",
    "∙": "·",
    "*": "·",
    ".": "·",
    "\uf13f": "·",
    "҇": "҃",
    "\uf222": "҃",
    "\uf23a": "҃",
    "\uf2b4": "҃",
    "\uf2b5": "҃",
    "\uf4a5": "҃",
    "\uf074": "ꙅ",
    "\uf130": "ꙩ",
    "\uf48e": "ꙩ",
    "\uf147": "ѡ",
    "\uf14e": "ѿ",
    "\uf42e": "ѿ",
    "\uf467": "ѯ",
    "\uf47e": "ꙋ",
    "\uf480": "ꙋ",
}

_DELETE_CHARS = {
    "⃝",
    "⟦",
    "⟧",
    "/",
    "\\",
    "|",
    "?",
    "!",
    '"',
    ";",
    ",",
    "̇",
    "̈",
    "̴",
    "͘",
    "\u200e",  # LRM
    "\uf080",
    "\uf245",
    "\uf265",
    "\uf27a",
    "\uf2db",
    "\uf4a4",
}
_DELETE_RE = re.compile("[" + re.escape("".join(_DELETE_CHARS)) + "]")

# legacy-GAP relics
_LEGACY_GAP_RE = re.compile(r"___G[АA][РP]___")


def normalize_historical_text(line: str) -> str:
    if not line:
        return ""

    text = unicodedata.normalize("NFC", line)
    text = text.replace("\ufeff", "").replace("\u200b", "")

    # Residual PUA noise
    text = re.sub(r"[\ue000-\uf8ff]", "", text)

    # Modern quotes
    text = re.sub(r'["\'«»„“”]', "", text)

    # Remove standard diacritics (accents) but keep titlo (outside this range)
    text = re.sub(r"[\u0300-\u036f]", "", text)

    # 1) Normalize rare paleographic symbols
    for src, dst in _RARE_CHAR_MAP.items():
        text = text.replace(src, dst)

    # 2) Legacy GAP
    text = _LEGACY_GAP_RE.sub("[GAP]", text)

    # 3) Latin -> Cyrillic typos
    for lat, cyr in _LAT_TO_CYR.items():
        text = text.replace(lat, cyr)

    # 4) Delete garbage chars
    text = _DELETE_RE.sub(" ", text)

    # 5) Remove all punctuation except: letters, digits, _, spaces, brackets, +, :, ·, titlo (҃)
    text = re.sub(r"[^\w\s:\[\]\(\)·+҃\-]", " ", text)  # Also keeping hyphens

    # 6) Compress spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text
