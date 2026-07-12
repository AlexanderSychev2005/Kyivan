import csv
import json
import re
import unicodedata

GAP = '[GAP]'

# Structural bookkeeping that leaks into the CSV 'text' column on
# multi-inscription rows -- "Текст 1:"/"text 2" labels (Cyrillic or Latin
# spelling), trailing "Доп. интерпретации" commentary, the "im."/"vac."
# editorial abbreviations, and positional labels ("Лицевая сторона:",
# "(левый столбец)"). None of this is inscription content.
_ADDL_INTERP_RE = re.compile(r'Доп\.\s*интерпретации.*$', re.S)
_TEXT_LABEL_RE = re.compile(r'(?i)(?:text|текст)\s*\d+\s*:?')
_IM_RE = re.compile(r'(?i)\bim\.\s*')
_VAC_RE = re.compile(r'(?i)\bvac\.\s*')
_POSITION_LABEL_RE = re.compile(
    r'(?i);?\(?(?:лицевая|оборотная)\s+сторона\)?\s*:?|'
    r';?\(?(?:левый|правый)\s+столбец\)?\s*:?'
)

# "⸗ =~ ̴" mark a word split across two inscribed lines (e.g. "Хр[GAP]⸗ стос"
# is one word, "Христос"); join with no space rather than a word boundary.
_LINE_BREAK_RE = re.compile(r'[⸗=~̴]\s*')

_DASH = '\\-‐‑–—−'  # -, ‐, ‑, –, —, −
# Bracket-wrapped illegible spans (dots, ellipsis, dash runs) collapse
# straight to a bare gap.
_WRAPPED_GAP_RE = re.compile(r'[\(\[][.…' + _DASH + r']+[\)\]]')
# Bare ellipsis / 2+ dot runs / 2+ dash runs -> gap
_BARE_GAP_RE = re.compile(r'…|\.{2,}|[' + _DASH + r']{2,}')
# Any remaining lone dash (one illegible letter) -> gap
_LONE_DASH_RE = re.compile(r'[' + _DASH + ']')
# Normalize any malformed bracket nesting directly around a bare GAP token
_STRAY_GAP_BRACKETS_RE = re.compile(r'\[*GAP\]*')

# () [] {} <> ⟦⟧ ⟨⟩ all mark restored/uncertain letters in this diplomatic
# convention: kept in 'original', unwrapped (content kept) in 'target'.
_BRACKET_CHARS_RE = re.compile(r'[\(\)\[\]\{\}<>⟦⟧⟨⟩]')


def clean_labels(raw_text):
    """Strip editorial bookkeeping that leaked into the 'text' column of
    multi-inscription rows -- it is not part of any inscription."""
    text = _ADDL_INTERP_RE.sub('', raw_text)
    text = _TEXT_LABEL_RE.sub('', text)
    text = _POSITION_LABEL_RE.sub('', text)
    text = _IM_RE.sub('', text)
    text = _VAC_RE.sub('', text)
    return re.sub(r'\s+', ' ', text).strip()


_SCRIPT_PREFIXES = ('CYRILLIC', 'GREEK', 'LATIN', 'GLAGOLITIC')


def is_mostly_cyrillic(text):
    """Whole inscriptions in Greek, Latin or Glagolitic graffiti do occur in
    the corpus but aren't Cyrillic Slavic text; drop them rather than let
    foreign letters pass through the (script-agnostic) normalizer untouched."""
    counts = {p: 0 for p in _SCRIPT_PREFIXES}
    for ch in text:
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, '')
        for prefix in _SCRIPT_PREFIXES:
            if name.startswith(prefix):
                counts[prefix] += 1
                break
    total = sum(counts.values())
    if total == 0:
        return True
    return counts['CYRILLIC'] >= total * 0.5


def collapse_duplicate_gaps(text):
    while GAP + GAP in text or GAP + ' ' + GAP in text:
        text = text.replace(GAP + GAP, GAP).replace(GAP + ' ' + GAP, GAP)
    return text


def build_original(raw_text):
    """Diplomatic reading: keeps () [] {} <> ⟦⟧ ⟨⟩ around restored/uncertain
    letters, collapses illegible spans (dash runs, ellipses) to a bare gap."""
    text = _LINE_BREAK_RE.sub('', raw_text)
    text = _WRAPPED_GAP_RE.sub(GAP, text)
    text = _BARE_GAP_RE.sub(GAP, text)
    text = _LONE_DASH_RE.sub(GAP, text)
    text = _STRAY_GAP_BRACKETS_RE.sub(GAP, text)
    text = re.sub(r'\s+', ' ', text).strip()
    return collapse_duplicate_gaps(text)


def build_target(original_text):
    """Reconstructed reading: same as original but with the editorial
    delimiters removed (their content is kept), gap markers left as-is."""
    parts = original_text.split(GAP)
    parts = [_BRACKET_CHARS_RE.sub('', p) for p in parts]
    text = GAP.join(parts)
    text = re.sub(r'\s+', ' ', text).strip()
    return collapse_duplicate_gaps(text)


def main():
    with open('data/epigraphica/epigraphica_full_data.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    dataset = []
    skipped_empty = 0
    skipped_non_cyrillic = 0

    for row in rows:
        raw_text = row['text'].strip()
        if not raw_text:
            skipped_empty += 1
            continue

        raw_text = clean_labels(raw_text)
        if not raw_text:
            skipped_empty += 1
            continue

        if not is_mostly_cyrillic(raw_text):
            skipped_non_cyrillic += 1
            continue

        row_id = row.get('﻿id', row.get('id'))
        year = row.get('date', '').strip() or 'unknown'

        original = build_original(raw_text)
        target = build_target(original)

        dataset.append({
            'doc_id': f'epigraphica_{row_id}',
            'original': original,
            'target': target,
            'dialect': 'epigraph',
            'year': year,
            'category': 'DAILY',
        })

    print(f'Total CSV rows: {len(rows)}')
    print(f'Skipped (empty text): {skipped_empty}')
    print(f'Skipped (non-Cyrillic): {skipped_non_cyrillic}')
    print(f'Written: {len(dataset)}')

    with open('data/epigraphica/epigraphica.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print('Saved to data/epigraphica/epigraphica.json')


if __name__ == '__main__':
    main()
