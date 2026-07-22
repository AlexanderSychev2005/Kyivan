import re
import unicodedata

KEPT_TITLO = "҃"  # COMBINING CYRILLIC TITLO -- the one combining mark we keep

# Latin letters that are visual homoglyphs of Cyrillic ones -- OCR/copy-paste
# noise, not intentional Latin text. Only ever applied to an ISOLATED Latin
# letter (a run of exactly 1) between/around Cyrillic text; a run of 2+ Latin
# letters in a row is left untouched, since that's real Latin/foreign text.
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
_LATIN_RUN_RE = re.compile(r"[A-Za-z]+")


def _fix_isolated_latin_homoglyphs(text: str) -> str:
    def repl(m):
        run = m.group(0)
        return _LAT_TO_CYR.get(run, run) if len(run) == 1 else run

    return _LATIN_RUN_RE.sub(repl, text)


# Single-codepoint diacritic letters with no place in Old Slavic text -- OCR
# noise, not intentional. ӑ/ӓ/ӣ/ӥ/ӧ/ӱ are Cyrillic vowels used only for
# non-Slavic (Turkic/Uralic) languages; ı/İ are Latin dotless/dotted i, used
# in some paleographic fonts for the dotless form of Cyrillic і; ς is Greek
# final sigma, seen only word-finally in Cyrillic-spelled Greek loanwords
# (e.g. "агіоς"). NOT included: ѐ/ѝ/ѷ, which are genuine OCS/South Slavic
# stress/orthographic marks still in active use in this corpus.
_STRAY_DIACRITIC_MAP = {
    "ӑ": "а",
    "Ӑ": "А",
    "ӓ": "а",
    "Ӓ": "А",
    "ӣ": "и",
    "Ӣ": "И",
    "ӥ": "и",
    "Ӥ": "И",
    "ӧ": "о",
    "Ӧ": "О",
    "ӱ": "у",
    "Ӱ": "У",
    "ı": "і",
    "İ": "І",
    "ς": "с",
}
_STRAY_DIACRITIC_RE = re.compile("[" + "".join(_STRAY_DIACRITIC_MAP) + "]")

# Canonicalize rare paleographic/punctuation symbols
_RARE_CHAR_MAP = {
    "†": "+",
    "×": "+",
    "ꙇ": "і",  # CYRILLIC LETTER IOTA -- positional allograph of і, folded for spelling normalization
    "Ꙇ": "І",
    # LATIN LETTER VOICED LARYNGEAL SPIRANT -- homoglyph mis-encoding of
    # Cyrillic dzelo/zelo (ѕ), seen used as the Cyrillic numeral for 6 in
    # dates (e.g. "лѣто ᴤ҃ и҃ ф҃ п҃ в҃" alongside other letter-numerals).
    "ᴤ": "ѕ",
    "⁘": ":",
    "⁙": ":",
    "⁞": ":",
    "⁖": ":",
    "¦": ":",
    "∙": "·",
    "*": "·",
    ".": "·",
    "": "·",
    "—": "-",
    "–": "-",
    "҇": "҃",
    "": "҃",
    "": "҃",
    "": "҃",
    "": "҃",
    "": "҃",
    "": "ꙅ",
    "": "ꙩ",
    "": "ꙩ",
    "": "ѡ",
    "": "ѿ",
    "": "ѿ",
    "": "ѯ",
    "": "ꙋ",
    "": "ꙋ",
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
    "",
    "",
    "",
    "",
    "",
    "",
}
_DELETE_RE = re.compile("[" + re.escape("".join(_DELETE_CHARS)) + "]")

_QUOTE_RE = re.compile(r'["\'«»„“”‘’]')

# OCR noise inserted mid-word (e.g. "ѻ³ч҃е" for "ѻч҃е", "ѿдалечеº" for
# "ѿдалече") -- stripped with no trace, unlike _DELETE_CHARS below which
# replaces with a space (correct for stray punctuation at word boundaries,
# wrong here since it would split one word into two fake ones).
_STRAY_MARK_RE = re.compile("[³º]")

_STRAY_BRACKETS_RE = re.compile(r"[\[\]\(\)<>]")

# legacy-GAP relics
_LEGACY_GAP_RE = re.compile(r"___G[АA][РP]___")
_GAP_RE = re.compile(r"\[GAP\]")
_ELLIPSIS_RE = re.compile(r"\.\.\.+|…")
# NKRYA's raw source marks an illegible/truncated word fragment with a run of
# underscores (e.g. "[UNK] _гина"); the character has no other use anywhere
# in the corpus, so any run is a gap marker.
_UNDERSCORE_RE = re.compile(r"_+")
# A single hyphen is legitimate punctuation across the corpus (ordinal
# numbers like "177-го", dialogue dashes, date ranges, ledger notation) and
# is left alone. A run of 2+ -- contiguous ("--------") or whitespace-
# separated ("- - - -"), as several datasets use to count missing letters --
# is a gap marker, same convention as birchbark's "·" runs.
_HYPHEN_GAP_RE = re.compile(r"-(?:\s*-)+")
_UNK_TOKEN = "[UNK]"
_UNK_PLACEHOLDER = (
    ""  # private-use codepoint, protects [UNK] from the bracket-structure pass below
)

_EMPTY_BRACKET_RE = re.compile(r"\(\)|\[\]")
_BRACKET_PAIRS = {")": "(", "]": "["}
_BRACKET_OPENERS = set(_BRACKET_PAIRS.values())
_BRACKET_CLOSERS = {v: k for k, v in _BRACKET_PAIRS.items()}
_DOUBLE_PLACEHOLDER_RE = re.compile(_UNK_PLACEHOLDER + r"\s*" + _UNK_PLACEHOLDER)


def _split_gaps_out_of_brackets(text: str) -> str:
    """A gap must never be nested inside () or [] -- "(бо[UNK]рис)" reads as
    "the whole span бо?рис was reconstructed", which is false; it must be
    "(бо)[UNK](рис)": two separate reconstructed spans around a bare gap.
    Close whichever brackets are open before the gap and reopen the same
    ones after (assumes text is already well-formed, i.e. already through
    the unmatched-bracket pass above)."""
    out = []
    open_stack = []
    for ch in text:
        if ch == _UNK_PLACEHOLDER and open_stack:
            out.extend(_BRACKET_CLOSERS[b] for b in reversed(open_stack))
            out.append(ch)
            out.extend(open_stack)
        else:
            out.append(ch)
            if ch in _BRACKET_OPENERS:
                open_stack.append(ch)
            elif ch in _BRACKET_PAIRS and open_stack:
                open_stack.pop()
    return "".join(out)


def _fix_bracket_structure(text: str) -> str:
    """Diplomatic transcriptions (keep_brackets=True) occasionally carry two
    source-data defects: an empty pair "()"/"[]" (illegible span the editor
    couldn't even guess at -- semantically a gap) and a stray unmatched
    bracket (a transcription typo). Fold the former into a gap and drop the
    latter; well-formed (possibly nested) brackets are left untouched except
    to pull a nested gap out to bracket-pair level (see
    _split_gaps_out_of_brackets)."""
    text = text.replace(_UNK_TOKEN, _UNK_PLACEHOLDER)
    text = _EMPTY_BRACKET_RE.sub(_UNK_PLACEHOLDER, text)

    result = list(text)
    stack = []
    for i, ch in enumerate(text):
        if ch in _BRACKET_OPENERS:
            stack.append(i)
        elif ch in _BRACKET_PAIRS:
            if stack and text[stack[-1]] == _BRACKET_PAIRS[ch]:
                stack.pop()
            else:
                result[i] = ""
    for i in stack:
        result[i] = ""
    text = "".join(result)

    text = _split_gaps_out_of_brackets(text)
    # The split above can leave an empty pair right against the gap (e.g. a
    # gap at the very edge of a span, or two gaps that were already adjacent
    # getting a "()" reopened/closed between them) or two gaps back to back;
    # clean both up, repeating since collapsing one can expose another
    # (e.g. three gaps in a row only fully collapses over multiple passes).
    prev = None
    while prev != text:
        prev = text
        text = _EMPTY_BRACKET_RE.sub(_UNK_PLACEHOLDER, text)
        text = _DOUBLE_PLACEHOLDER_RE.sub(_UNK_PLACEHOLDER, text)

    return text.replace(_UNK_PLACEHOLDER, _UNK_TOKEN)


def _normalize_segment(text: str, keep_brackets: bool) -> str:
    """Char-level canonicalization of a stretch of text that is guaranteed
    to contain no [UNK] marker (see normalize_historical_text)."""
    text = _QUOTE_RE.sub("", text)
    text = _STRAY_MARK_RE.sub("", text)
    text = _fix_isolated_latin_homoglyphs(text)
    text = _STRAY_DIACRITIC_RE.sub(lambda m: _STRAY_DIACRITIC_MAP[m.group(0)], text)

    # Drop every combining mark except titlo: accents, breathing marks
    # (dasia/psili pneumata), payerok, vzmet, bridge, etc. are font/OCR-level
    # artifacts, not part of the base letter.
    text = "".join(
        c for c in text if not (unicodedata.category(c) == "Mn" and c != KEPT_TITLO)
    )

    for src, dst in _RARE_CHAR_MAP.items():
        text = text.replace(src, dst)

    # ьі digraph -> ы (same letter, two encodings)
    text = text.replace("ьі", "ы").replace("ЬІ", "Ы")

    text = _DELETE_RE.sub(" ", text)

    if keep_brackets:
        text = re.sub(r"[^\w\s:\[\]\(\)·+҃\-]", " ", text)
    else:
        text = _STRAY_BRACKETS_RE.sub("", text)
        text = re.sub(r"[^\w\s:·+҃\-]", " ", text)

    return text


def normalize_historical_text(line: str, keep_brackets: bool = False) -> str:
    """Canonicalize a piece of historical Slavic text so the same letter
    looks the same regardless of source, and apply the corpus-wide
    gap/bracket policy.

    Literal "[GAP]" and any run of "..."/"…" become "[UNK]". Stray (), [],
    <> are stripped everywhere EXCEPT diplomatic transcriptions (birchbark,
    epigraphy) where keep_brackets=True preserves them -- there they mark
    genuine reconstructed text, not editorial noise.
    """
    if not line:
        return ""

    text = unicodedata.normalize("NFC", line)
    text = text.replace("﻿", "").replace("​", "")

    # Residual PUA noise (font-specific artifacts already decoded upstream)
    text = re.sub(r"[-]", "", text)

    # Canonical gap markers -- resolved before any char-level substitution
    # can corrupt them (e.g. Latin K/A/P inside "GAP"/"UNK" -> Cyrillic).
    text = _LEGACY_GAP_RE.sub("[GAP]", text)
    text = _GAP_RE.sub(_UNK_TOKEN, text)
    text = _ELLIPSIS_RE.sub(_UNK_TOKEN, text)
    text = _UNDERSCORE_RE.sub(_UNK_TOKEN, text)
    text = _HYPHEN_GAP_RE.sub(_UNK_TOKEN, text)

    segments = [
        _normalize_segment(seg, keep_brackets) for seg in text.split(_UNK_TOKEN)
    ]
    text = _UNK_TOKEN.join(segments)

    if keep_brackets:
        text = _fix_bracket_structure(text)

    text = re.sub(r"\s+", " ", text).strip()
    return text
