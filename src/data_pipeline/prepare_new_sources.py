"""Runs the same normalize/prepare logic as prepare_datasets.py's generic
per-source loop, but only for the two newly-scraped sources (bldr_azbyka,
litopys) -- process_datasets() itself touches every source unconditionally
(including an unseeded random draw for bible_ostrog), so re-running it here
would needlessly reshuffle already-pushed sources this task has nothing to
do with."""

import json
import os

from prepare_datasets import get_date_target, get_macro_dialect, parse_year
from normalization import normalize_historical_text

NEW_SOURCES = [
    ("bldr_azbyka", "../../data/bldr_azbyka/bldr_azbyka_cleaned.json"),
    ("litopys", "../../data/litopys/litopys_cleaned.json"),
]


def main() -> None:
    for ds_name, json_path in NEW_SOURCES:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        out_path = f"../../prepared_datasets/{ds_name}_prepared.jsonl"
        with open(out_path, "w", encoding="utf-8") as f_out:
            for doc in data:
                text = normalize_historical_text(doc.get("text", ""), keep_brackets=False)

                raw_year = doc.get("year", "")
                source = doc.get("source", "")
                dialect = doc.get("dialect", "")
                doc_id = doc.get("doc_id", "")

                macro_dialect = get_macro_dialect(ds_name, dialect, source)
                interval = parse_year(raw_year)
                target = get_date_target(interval)

                new_doc = {
                    "doc_id": doc_id,
                    "text": text,
                    "macro_dialect": macro_dialect,
                    "date_interval": interval,
                    "date_target": target,
                    "date_number": raw_year
                    if interval and interval[0] == interval[1]
                    else None,
                    "category": doc.get("category", "unknown"),
                    "original_dialect": dialect,
                }
                f_out.write(json.dumps(new_doc, ensure_ascii=False) + "\n")

        print(f"Prepared {ds_name} -> {out_path} ({len(data)} docs)")


if __name__ == "__main__":
    main()
