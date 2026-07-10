# -*- coding: utf-8 -*-
"""
normalize_histdict.py
Normalization pipeline for texts crawled from https://histdict.uni-sofia.bg
(Church Slavonic / Old East Slavic), used by the DIACU-derived corpus.

Pipeline (order matters):
  0. drop_editorial_tail : cut trailing editors' commentary blocks (modern
                        Bulgarian notes / critical apparatus / footnote lists)
                        that follow the main text after a blank line.
                        NB: a blank line alone is NOT a reliable boundary
                        (several docs contain blank lines mid-text), so each
                        blank-line-separated segment is classified by content.
  0b. remove_editorial_marks : delete inline editorial marks -- (!), (sic),
                        (sic!), /!/, /?/, '*', verse/line references
                        (19 ред), (ред 19), (ст. 1836).
  1. clean_lines      : remove foliation/page markers (67v, -67v-, //67v),
                        strip editorial verse-number prefixes (8., 44.2.),
                        rejoin hyphenated line-breaks (incl. a page number
                        wedged between hyphen and word), reattach drop caps
                        (an initial letter typeset on its own line) to the
                        following line WITHOUT a space.
  2. decode_chars     : histdict PUA font -> real Unicode (letters, one titlo);
                        inline linearization of superscript letters (vynosnye).
  3. strip_diacritics : remove all combining marks EXCEPT titlo (U+0483);
                        remove spacing jer-markers (payerok, vertical tilde);
                        remove Arabic digits (copy artifacts).
  4. unify            : diaeresis-vowel junk -> base vowel, Latin i-diaeresis
                        -> Cyrillic yi, dot-separators -> one middle dot,
                        underscore/vertical-bar/nbsp -> space; NFC; squeeze.

PRESERVED: the OCS / Old-Russian alphabet; Latin/Greek apparatus & parallels
(manuscript sigla, sic, Greek source words) which are genuine editorial content.
"""
import re
import json
import unicodedata

KEPT_TITLO = "\u0483"   # COMBINING CYRILLIC TITLO -- the single titlo we keep

# --- 1. PUA font decoding table (histdict). Full detail: pua_decoding.csv ---
PUA_MAP = {
    '\ue205': 'и',
    '\ue20d': 'ч',
    '\ue033': '',
    '\ue201': 'ѥ',
    '\ue010': '҃',
    '\ue014': '҃',
    '\ue012': '҃',
    '\ue018': '',
    '\ue031': '',
    '\ue016': '',
    '\ue215': 'ѹ',
    '\ue342': '·',
    '\ue204': 'ѥ',
    '\ue017': '',
    '\ue027': '',
    '\ue343': '·',
    '\ue02e': '',
    '\ue21d': 'с',
    '\ue223': 'ꙑ',
    '\ue032': '',
    '\ue344': '·',
    '\ue100': '',
    '\ue225': '·',
    '\ue20c': 'ч',
    '\ue221': 'ф',
    '\ue340': '·',
    '\ue213': 'ꙗ',
    '\ue105': 'о',
    '\ue21b': 'о',
    '\ue347': '',
    '\ue028': '',
    '\ue209': 'ꙑ',
    '\ue02c': '',
    '\ue21c': 'с',
    '\ue34a': '',
    '\ue219': '',
    '\ue349': '·',
    '\ue019': '',
    '\ue211': 'ѧ',
    '\ue348': '·',
    '\ue216': 'ѹ',
    '\ue02f': '',
    '\ue227': '·',
    '\ue02d': '',
    '\ue30d': '·',
    '\ue114': 'з',
    '\ue218': 'ѫ',
    '\ue345': '',
    '\ue346': '',
    '\ue21f': 'ѥ',
    '\ue106': '҃',
    '\ue309': '',
    '\ue002': '҃',
    '\ue20b': 'о',
    '\ue104': 'ъ',
    '\ue101': 'и',
    '\ue011': '҃',
    '\ue208': 'ы',
    '\ue28a': '',
    '\uf025': '',
    '\ue203': '҃',
    '\ue10e': '',
    '\ue212': '·',
    '\uf021': '',
    '\ue102': '',
    '\uf076': '·',
    '\uf080': '·'
}

# --- superscript combining Cyrillic letters (vynosnye bukvy) -> base letter ---
SUP_MAP = {
    'ⷭ': 'с',
    'ⷦ': 'к',
    'ⷪ': 'о',
    'ⷶ': 'а',
    'ⷢ': 'г',
    'ⷱ': 'ч',
    'ⷣ': 'д',
    'ⷧ': 'л',
    'ⷲ': 'ш',
    'ⷨ': 'м',
    'ⷮ': 'т',
    'ⷬ': 'р',
    'ꙸ': 'ъ',
    'ⷠ': 'б',
    'ꙷ': 'ѹ',
    'ⷡ': 'в',
    'ⷩ': 'н',
    'ⷽ': 'ѧ',
    'ⷤ': 'ж',
    'ⷳ': 'щ',
    'ⷴ': 'ѳ',
    'ⷯ': 'х',
    'ⷾ': 'ѫ',
    'ⷫ': 'п',
    'ⷷ': 'е',
    'ⷥ': 'з',
    'ⷰ': 'ц',
    'ꙴ': 'є',
    'ⷺ': 'ѣ',
    'ꙵ': 'и',
    'ⷹ': 'ѹ',
    'ⷼ': 'ꙗ',
    'ⷻ': 'ю',
    'ꙹ': 'ы',
    'ꙺ': 'ь',
    'ꙶ': 'ї',
    'ꙻ': 'ѡ',
    'ⷵ': 'ст',
}

# --- 2. combined char-translate table (PUA + superscripts) ---
_translate = {}
for _ch, _t in PUA_MAP.items():
    _translate[_ch] = _t
for _ch, _b in SUP_MAP.items():
    _translate[_ch] = _b
_TRANS = {ord(k): (v if v != "" else None) for k, v in _translate.items()}

# --- 3. diacritics / spacing marks / digits ---
_SPACING_MARKS = {"\ua67f", "\u2e2f"}   # payerok, vertical tilde (omitted-jer)
_DIGIT_RE = re.compile(r"\d+")

# spacing modifier letters / stray marks to delete outright (erik, palatal apostrophe,
# soft hyphen inside words, stray modifier o)
_DELETE_CHARS = {"\u02b9", "\u02bc", "\u00ad", "\u1d55"}

# combining overline abbreviation/numeral marks that carry the SAME function as
# U+0483 titlo (different font encodings of the same phenomenon) -> fold to titlo
_TITLO_FOLD = {"\u0360", "\u035e", "\u0304", "\ua66f", "\u1dcd"}

# --- 4. equivalence / whitespace unification ---
_MIDDOT = "\u00b7"
_EQUIV = {
    "\u04f9": "\u044b", "\u04e5": "\u0438", "\u04f3": "\u0443", "\u04f1": "\u0443",
    "\u04d3": "\u0430", "\u04e7": "\u043e",
    "\u00ef": "\u0457",
    # stray Glagolitic letters used as Cyrillic substitutes
    "\u2c54": "\u0454", "\u2c34": "\u0434",
    "\u2022": _MIDDOT, "\u2e31": _MIDDOT, "\u2219": _MIDDOT, "\uf13f": _MIDDOT,
    "\u00a0": " ", "\u2008": " ", "\u2007": " ", "\u202f": " ",
    "\u2009": " ", "\t": " ", "_": " ",
}
_EQUIV_TRANS = {ord(k): v for k, v in _EQUIV.items()}

_HYPHENS = "\u002d\u2010\u2011\u2013\u2014"
_MARKER_LINE = re.compile(
    r"^(?:"
    r"-?\d+[abcdrv]?-?"
    r"|-?\d+[\u0430\u0431\u0432\u0433\u0434]?-?"
    r"|/{1,2}\d*[abcdrv\u0430\u0431]?/{0,2}"     # //67v, /68r/, //68r, bare //
    r"|[f\u043bF\u041b]\.\s*\d+[rv\u0430\u0431]?"
    r"|\d+"
    r")$"
)
# verse/chapter refs at line start: "8.", "8.11.", "44.2." (also glued to text)
_LEADING_LINENUM = re.compile(r"^(?:\d+\.)+\s*")
_END_HYPHEN = re.compile(r"[" + _HYPHENS + r"]\s*\d*\s*$")
# _LEAD_DIGITS = re.compile(r"^\s*\d+[ab]?\s*")
_DIGITS = re.compile(r"\b\d+[ab]?\b")
_SEPARATOR = re.compile(r"※{2,}")
_MULTISPACE = re.compile(r"[ ]{2,}")

# --- 0. trailing editorial commentary ------------------------------------
# Modern-Bulgarian editor-note markers (checked on the first lines of each
# blank-line-separated segment; the main text is medieval and never uses
# these modern words/spellings).
_BG_COMMENT_HINTS = (
    "думата", "буквата", "буквите", "текстът", "текста е", "лигатур",
    "препис", "червенослов", "реставрац", "грешка", "издание", "изданието",
    "нормализаци", "задраскан", "изтрит", "възстанов", "не се чете",
    "ркп.", "доб. по", "заглавието", "похвалн", "поучението", "празникът",
    "паметта", "службата", "наставленията", "приписка", "липсва",
    "е написан", "надписан", "пренесена", "огледално", "празен",
    "в полето", "горното поле", "бележка",
    "ff. ", "f. r", "f. v", "sqq.", "lege ", "scil. ", "sic pro",
    "corr. ", "add. ", "om. ", "falso rep.", "recte ", "cf. ",
    "infra in marg.", "signatura folii", "apud ", "pro ", "item in",
    "verisimiliter", "originaliter", "Vind", "Cod", "Ss. patrum"
)
# critical-apparatus line: "бы́въшхъ Z : би́вшихь K, M, S" (Latin sigla + colon)
_APPARATUS_LINE = re.compile(r"[A-Z][a-z]?\s*:|:\s*[^:]*\b[A-Z][a-z]?\b")
# footnote-marker line: "[64])"
_FOOTNOTE_LINE = re.compile(r"^\[\d+\]\)?$")
# bare biblical reference line: "Иоан. 8. 12.", "Дан 3:54"
_BIBLE_REF_LINE = re.compile(r"^[\u0400-\u04ff]{2,6}\.?\s?\d+\s?[.:]\s?\d+\.?$")

# drop cap: a single letter typeset on its own line, belonging to the first
# word of the next line.  Uppercase => always a drop cap in this corpus
# (incl. І, Х); lowercase => join only when the next line starts with a
# letter that cannot begin a word (jers/ery).  Ѿ/ѿ are excluded: they are
# ambiguous between the preposition "отъ" (keep space) and the prefix "от-"
# (no space) and would need a lexicon to resolve.
_DROPCAP_EXCLUDE = {"\u047e", "\u047f"}          # Ѿ ѿ
_NONINITIAL = "\u044a\u044c\u044b\ua651"          # ъ ь ы ꙑ


def _is_editorial_segment(seg):
    """Heuristic: does this blank-line-separated segment consist of editors'
    notes / critical apparatus / footnotes rather than the source text?"""
    lines = [ln.strip("\u00a0 \t\r") for ln in seg.split(chr(10))]
    lines = [ln for ln in lines if ln]
    if not lines:
        return True
    head = lines[:12]
    low = chr(10).join(head).casefold()
    if any(h in low for h in _BG_COMMENT_HINTS):
        return True
    hits = sum(
        1 for ln in head
        if _FOOTNOTE_LINE.match(ln)
        or _BIBLE_REF_LINE.match(ln)
        or (" : " in ln and _APPARATUS_LINE.search(ln))
    )
    return hits >= max(1, len(head) // 3)



def drop_editorial_tail(content):
    """Step 0: keep blank-line-separated segments up to the first one that
    classifies as editorial commentary; drop it and everything after."""
    parts = content.split(chr(10) * 2)
    kept = [parts[0]]
    for seg in parts[1:]:
        if _is_editorial_segment(seg):
            break
        kept.append(seg)
    return chr(10).join(kept)


# --- 0b. inline editorial marks & references ------------------------------
_EDITORIAL_MARKS = re.compile(
    r"\(\s*sic\s*!?\s*\)"                      # (sic), (sic!)
    r"|\(\s*!\s*\)"                            # (!)
    r"|/\s*[!?]\s*/"                           # /!/  /?/
    r"|\(\s*(?:\d+\s*\u0440\u0435\u0434|\u0440\u0435\u0434\s*\d+)\s*\)"  # (19 ред), (ред 19), (ред1 )
    r"|\(\s*\u0441\u0442\.\s*\d+\s*\)"         # (ст. 1836)
    r"|(?<!\S)/{1,2}\d+[abcdrv\u0430\u0431]?/{0,2}(?!\S)"  # inline //3б, /12/
    r"|\*"                                     # asterisks
    r"|\b\d+\.\s*\d+[ab–-]*[ab]?\b"  # 13. 15a, 13. 15a-b, 13. 15a–b
    r"|(?:Слово)\s*№\s*\d+\b"  # Слово № 1
    r"|\(\s*\d+[A-Za-z\u0400-\u04ff]?\s*/\s*\d+[A-Za-z\u0400-\u04ff]?\s*\)"
    r"|<\s*(?:\d+\.?\s*)*\d+[a-c–\-]*[a-c]?\s*>"  # <0. 1>, <1. 1–3>
    r"|<\s*>"
    r"|\\",
    flags=re.DOTALL
)

# inline modern-Bulgarian editorial remarks embedded in the source text
# (e.g. "(това заглавие на параграф стои на последния ред ...)",
#  "Тук се възстановява реда на обърканите текстове").
_INLINE_COMMENT_HINTS = (
    # Bulgarian editors' notes (modern vocabulary/spellings; the source text
    # -- incl. scribal colophons with "преписахъ", "преписа сѧ" -- never
    # uses these exact forms, so bare "препис(а)" must NOT be a hint here)
    "възстановява", "възстановен", "се представят", "се привежда",
    "преписвач", "преписи", "по преписа", "препис от", "др. препис",
    "осн.препис", "осн. препис", "основния", "съкратената ред",
    "разночетени", "изданието", "заглавие на", "заглавието", "няма заглавие",
    "това заглавие", "чудесата на св", "страници", "нечетим", "лигатура",
    "липсва", "липсват", "липсващите", "задраскан", "грешка",
    "това се обозначава", "тук се", "спазва се", "правопис",
    # Russian apparatus notes (doc_166--168 marginal descriptions)
    "далее в", "миниатюра", "киноварью", "полуустав", "в ркп.",
    "см. вар", "см. прим",
)
_PARENTHETICAL = re.compile(r"\(([^()]*)\)")
_BRACKETED = re.compile(r"\[([^\[\]]*)\]")


def _drop_inline_comments(text):
    """Remove parentheticals/bracketed insertions and whole lines that
    contain modern-Bulgarian (or Russian) editorial vocabulary; the medieval
    text itself never does.  Brackets WITHOUT such vocabulary are kept: they
    carry genuine reconstructed text."""
    def _cond(m):
        return "" if any(h in m.group(1).casefold()
                         for h in _INLINE_COMMENT_HINTS) else m.group(0)
    text = _PARENTHETICAL.sub(_cond, text)
    text = _BRACKETED.sub(_cond, text)
    kept = []
    for line in text.split(chr(10)):
        low = line.casefold()
        if any(h in low for h in _INLINE_COMMENT_HINTS):
            continue
        kept.append(line)
    return chr(10).join(kept)


def remove_editorial_marks(text):
    """Step 0b: delete inline editorial marks; must run BEFORE digit removal
    so that (19 ред)/(ст. 1) disappear entirely instead of leaving '( )'."""
    text = _drop_inline_comments(text)
    return _EDITORIAL_MARKS.sub("", text)


def _is_dropcap(s):
    """Single-letter line that is a drop cap (initial) of the next line."""
    if len(s) != 1 or not s.isalpha() or s in _DROPCAP_EXCLUDE:
        return False
    if not ("\u0400" <= s <= "\u04ff" or "\ua640" <= s <= "\ua69f"):
        return False
    # uppercase initials are always drop caps in this corpus (incl. І, Х);
    # lowercase ones are resolved later against the next line's first char
    return True


def clean_lines(content):
    """Step 1: strip foliation/lineation markup; rejoin hyphenated breaks;
    reattach drop caps to the following line without a space."""
    out = []
    pending_join = False
    dropcap = []
    for raw in content.split(chr(10)):
        s = raw.strip("\u00a0 \t\r")
        if not s:
            continue
        s2 = _LEADING_LINENUM.sub("", s, count=1)
        if s2 != s:
            s = s2.strip()
            if not s:
                continue
        if _MARKER_LINE.match(s):
            continue
        if _is_dropcap(s):
            dropcap.append(s)
            continue
        if dropcap:
            glued = ""
            for piece in dropcap:
                if piece.isupper() or (s and s[0] in _NONINITIAL):
                    glued += piece
                else:
                    out.append(piece)
            s = glued + s
            dropcap = []

        # for all lines
        s = _DIGITS.sub("", s)

        if pending_join:
            if out:
                out[-1] = out[-1] + s
            else:
                out.append(s)
            pending_join = False
        else:
            out.append(s)
        if _END_HYPHEN.search(out[-1]):
            out[-1] = _END_HYPHEN.sub("", out[-1])
            pending_join = True
    if dropcap:
        out.extend(dropcap)
    return " ".join(out)


def decode_chars(text):
    """Step 2: PUA font -> Unicode; linearize superscript letters."""
    return text.translate(_TRANS)


def strip_diacritics(text):
    """Step 3: fold titlo-variant overlines to U+0483; remove all other
    combining marks except titlo; spacing jer-marks; digits."""
    text = "".join(KEPT_TITLO if c in _TITLO_FOLD else c for c in text)
    text = "".join(
        c for c in text
        if not (unicodedata.category(c) == "Mn" and c != KEPT_TITLO)
        and c not in _SPACING_MARKS
        and c not in _DELETE_CHARS
    )
    text = _DIGIT_RE.sub("", text)
    return text


def unify(text):
    """Step 4: unify equivalents & whitespace; line-break bar; NFC; squeeze."""
    text = re.sub(r"(?<=\w)\|(?=\w)", "", text)
    text = text.replace("|", " ")
    text = text.translate(_EQUIV_TRANS)
    # ы
    text = re.sub("ьі", "ы", text)
    text = re.sub("ЬІ", "Ы", text)
    text = re.sub("\u0483{2,}", "\u0483", text)   # collapse redundant titlos
    text = unicodedata.normalize("NFC", text)
    text = _MULTISPACE.sub(" ", text).strip()
    return text


def normalize(content):
    """Full pipeline: raw histdict `content` -> normalized text."""
    text = drop_editorial_tail(content)
    text = remove_editorial_marks(text)
    text = clean_lines(text)

    text = re.sub(r"\n\s*\n+", "\n", text)

    text = decode_chars(text)
    text = strip_diacritics(text)
    text = unify(text)
    return text


def normalize_corpus(docs, content_key="content"):
    """Normalize a list of doc-dicts; returns a new list (content replaced)."""
    out = []
    for d in docs:
        nd = dict(d)
        nd[content_key] = normalize(d.get(content_key, "") or "")
        out.append(nd)
    return out


if __name__ == "__main__":
    import sys
    inp = sys.argv[1] if len(sys.argv) > 1 else "histdict_corpus.json"
    outp = sys.argv[2] if len(sys.argv) > 2 else "histdict_normalized.json"
    with open(inp, encoding="utf-8") as f:
        data = json.load(f)
    data = normalize_corpus(data)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print("wrote", outp, "(%d docs)" % len(data))