"""Offline: build a small supplementary metadata file for the web demo's
"similar documents" cards -- source-specific fields that exist in a raw
source but never made it into prepare_datasets.py's flattened schema
(doc_id/text/macro_dialect/date_interval/category/...), because that schema
has to stay uniform across all 7 corpora. Same reasoning/shape as akkadian/
src/analysis/build_line_tables.py's doc_lines.json: a separate file keyed by
doc_id, loaded alongside doc_embeddings.npy/doc_meta.json, instead of adding
source-specific columns to the core pipeline.

Two raw sources, two schemas of extra fields (only non-empty fields are kept
per doc, so a source that doesn't have a given field just omits it rather
than writing nulls):

1. birchbark_classes.jsonl (the corpus prepare_datasets.py already reads)
   has 'region' (the find-site town, e.g. "Новгород"/"Псков"/"Тверь") and
   'genre' (a fine-grained genre string, e.g. "частное письмо", much richer
   than category_mapped's 5 buckets) -- but its own 'number' field isn't
   directly the doc's citation number for non-Novgorod cities (e.g.
   "Пск.\xa01", matching doc_id "birchbark_Пск. 1").

2. data/active/birchbark_gramoty_ru.csv (gramoty.ru's own per-letter export)
   carries the proper city-scoped citation ('title', e.g. "Грамота № Пск. 1"
   -- what birchbark_classes.jsonl's own number-based doc_id can't give us
   for non-Novgorod letters), a source URL, a short human 'content' summary,
   and (for some letters) a modern-Russian translation. Joined to
   birchbark_classes.jsonl by (city, trailing digits of 'number') since the
   two use different id conventions; ~68% of birchbark_classes.jsonl's 1259
   letters match (the rest are letters gramoty.ru's own site doesn't have a
   page for -- lost/merged/renumbered, same as a CDLI-bulk P-number with no
   matching ORACC page in build_line_tables.py).

   data/active/epigraphica/epigraphica_full_data.csv (epigraphica.ru's own
   export -- the file process_epigraphica.py already expects at this exact
   path, previously missing from disk) adds a source URL, a short 'content'
   summary, the physical carrier type/location, and a translation when
   epigraphica.ru's own editors provide one. Joined to
   epigraphica_prepared.jsonl 1:1 by id (doc_id is "epigraphica_{id}").

Run once (re-run only if the raw sources change):

    python -m src.model.build_doc_extra

Output: runs/kyivan_h224_mask018_300ep/embeddings/doc_extra.json,
{doc_id: {...}} -- app.py loads it alongside doc_embeddings.npy/doc_meta.json
and merges present fields into the "similar documents" response; a doc_id
with no entry here just shows what doc_meta.json already has.
"""
import csv
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
BIRCHBARK_CLASSES_PATH = BASE_DIR / "data" / "active" / "birchbark_classes.jsonl"
BIRCHBARK_GRAMOTY_CSV = BASE_DIR / "data" / "active" / "birchbark_gramoty_ru.csv"
EPIGRAPHICA_CSV = BASE_DIR / "data" / "active" / "epigraphica" / "epigraphica_full_data.csv"
OUT_PATH = BASE_DIR / "runs" / "kyivan_h224_mask018_300ep" / "embeddings" / "doc_extra.json"


def _clean(v):
    v = (v or "").strip()
    return v or None


def build_birchbark_extra() -> dict:
    gramoty_by_key = {}
    if BIRCHBARK_GRAMOTY_CSV.exists():
        with open(BIRCHBARK_GRAMOTY_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                gramoty_by_key[(row["city"], row["id"])] = row

    result = {}
    if not BIRCHBARK_CLASSES_PATH.exists():
        return result
    with open(BIRCHBARK_CLASSES_PATH, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            number = doc.get("number") or ""
            region = doc.get("region")
            doc_id = f"birchbark_{number}"

            entry = {}
            if region:
                entry["region_city"] = region
            genre = doc.get("genre")
            if genre and genre != "-":
                entry["genre"] = genre

            m = re.search(r"(\d+)$", number)
            gramoty_row = gramoty_by_key.get((region, m.group(1))) if m else None
            if gramoty_row:
                if _clean(gramoty_row.get("title")):
                    entry["title"] = gramoty_row["title"]
                if _clean(gramoty_row.get("url")):
                    entry["source_url"] = gramoty_row["url"]
                if _clean(gramoty_row.get("content")):
                    entry["summary"] = gramoty_row["content"]
                if _clean(gramoty_row.get("translation_ru")):
                    entry["translation_ru"] = gramoty_row["translation_ru"]

            if entry:
                result[doc_id] = entry
    return result


def build_epigraphica_extra() -> dict:
    result = {}
    if not EPIGRAPHICA_CSV.exists():
        return result
    with open(EPIGRAPHICA_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row_id = row.get("﻿id", row.get("id"))
            doc_id = f"epigraphica_{row_id}"
            entry = {}
            if _clean(row.get("url")):
                entry["source_url"] = row["url"]
            if _clean(row.get("content")):
                entry["summary"] = row["content"]
            if _clean(row.get("place")):
                entry["place"] = row["place"]
            if _clean(row.get("carrier_category")):
                entry["object_type"] = row["carrier_category"]
            if _clean(row.get("translation")):
                entry["translation_ru"] = row["translation"]
            if entry:
                result[doc_id] = entry
    return result


def main() -> None:
    result = build_birchbark_extra()
    result.update(build_epigraphica_extra())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    print(f"Saved {len(result)} entries to {OUT_PATH}")


if __name__ == "__main__":
    main()
