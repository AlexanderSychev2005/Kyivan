"""Convert bldr_all.jsonl (raw scrape output) into the {doc_id, text, year,
dialect, source} list-of-dict schema prepare_datasets.py expects as input
(same shape as e.g. data/sofia/sofia_cleaned.json).

Year: each BLDR volume covers a known, roughly chronological period. Most
of the table below was read directly off the "Т. N: <period>." citation
line each work page carries (now stripped from the scraped text itself as
noise -- see TAIL_MARKERS in scraper.py); the citation wasn't reachable for
every volume (some works always hit a "Примечания" section first), so
those four are interpolated from their confirmed neighbors (BLDR volumes
run in strict chronological order) and marked accordingly. Fed as a
"YYYY-YYYY" string, same convention as e.g. sofia_cleaned.json, so
prepare_datasets.py's existing parse_year regex handles it unchanged.
"""

import json
import re
from pathlib import Path

# (start_year, end_year, confirmed_from_citation)
VOLUME_PERIODS = {
    1: (1001, 1200, True),
    2: (1001, 1200, True),
    3: (1001, 1200, True),
    4: (1101, 1200, True),
    5: (1201, 1300, True),
    6: (1301, 1450, True),
    7: (1450, 1500, True),
    8: (1401, 1500, False),  # interpolated: between tom7 (2nd half XV) and tom9
    9: (1490, 1550, True),
    10: (1501, 1600, True),
    11: (1501, 1600, True),
    12: (1501, 1600, True),
    13: (1550, 1600, False),  # interpolated: between tom12 (XVI) and tom14
    14: (1580, 1620, True),
    15: (1601, 1700, True),
    16: (1601, 1700, False),  # interpolated: between tom15/tom18, both XVII
    17: (1601, 1700, False),  # interpolated: same
    18: (1601, 1700, True),
    19: (1701, 1800, True),
    20: (1701, 1900, True),
}

# Genre words in the (Russian) work title that signal Church Slavonic
# liturgical/hagiographic register rather than vernacular/chronicle OES --
# same genre-based convention prepare_datasets.py already uses to hardcode
# sofia/epigraphica/bible_ostrog to "CS" regardless of individual document.
_CS_GENRE_RE = re.compile(
    r"Житие|Поучение|Служба|Похвала|Моление|Устав|Слово о|Правда|"
    r"Послание|Чудеса|Хождение",
    re.IGNORECASE,
)


def guess_dialect(title: str) -> str:
    return "CS" if _CS_GENRE_RE.search(title) else "OES"


records = []
for line in open(Path(__file__).parent / "bldr_all.jsonl", encoding="utf-8"):
    d = json.loads(line)
    start, end, _confirmed = VOLUME_PERIODS[d["volume"]]
    records.append(
        {
            "doc_id": f"bldr_tom{d['volume']}_{d['work_num']}",
            "text": d["text"],
            "year": f"{start}-{end}",
            "dialect": guess_dialect(d["title"]),
            "source": "bldr_azbyka",
        }
    )

out_path = Path(__file__).parent / "bldr_azbyka_cleaned.json"
out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"{len(records)} records -> {out_path}")
