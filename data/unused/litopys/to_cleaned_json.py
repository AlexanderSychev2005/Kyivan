"""Convert litopys_all.jsonl (raw scrape output) into the {doc_id, text,
year, dialect, source} schema prepare_datasets.py expects. Year comes from
year_map.json (built by re-fetching each page and reading its own printed
"[YYY] - [YYY]" annalistic column-range header, e.g. "[907] - [944]" on
lavr02.htm) -- 70/75 pages have one; the other 5 (mostly pre-annalistic
front matter, e.g. lavr01's flood/tribes narrative) are left undated.
Dialect: uniformly OES, same as torot/UD-RNC -- these are secular
chronicles throughout, no genre split like BLDR's."""

import json
from pathlib import Path

year_map = json.load(open(Path(__file__).parent / "year_map.json", encoding="utf-8"))

records = []
for line in open(Path(__file__).parent / "litopys_all.jsonl", encoding="utf-8"):
    d = json.loads(line)
    year_range = year_map.get(d["page_id"])
    records.append(
        {
            "doc_id": f"litopys_{d['chronicle']}_{d['page_id']}",
            "text": d["text"],
            "year": f"{year_range[0]}-{year_range[1]}" if year_range else "",
            "dialect": "OES",
            "source": "litopys",
        }
    )

out_path = Path(__file__).parent / "litopys_cleaned.json"
out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"{len(records)} records -> {out_path}")
