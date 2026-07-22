import json
import os
import glob
import re
from normalization import normalize_historical_text

BUCKETS_START = 800
BUCKET_SIZE = 50
NUM_BUCKETS = 20


def get_date_target(interval):
    # None (not an all-zero vector) for "no usable date" -- an all-zero
    # vector looks like a real, valid target to prepare_splits.py/the
    # collator (which only treats a literal None as "withhold this label"),
    # so returning [0.0]*NUM_BUCKETS here would train/score the date head
    # against a fake "definitely bin 0" answer for every undated document.
    if not interval:
        return None
    start, end = interval
    start = max(BUCKETS_START, start)
    end = min(BUCKETS_START + NUM_BUCKETS * BUCKET_SIZE - 1, end)

    if start > end:
        return None

    total_years = end - start + 1
    target = [0.0] * NUM_BUCKETS

    for y in range(start, end + 1):
        idx = (y - BUCKETS_START) // BUCKET_SIZE
        if 0 <= idx < NUM_BUCKETS:
            target[idx] += 1.0 / total_years

    return target


def parse_year(year_str):
    if not year_str or str(year_str).lower() == "unknown":
        return None
    year_str = str(year_str)

    # Find all 3 or 4 digit numbers in the string
    numbers = [int(n) for n in re.findall(r"\d{3,4}", year_str)]

    if len(numbers) >= 2:
        s, e = numbers[0], numbers[1]
        return [min(s, e), max(s, e)]
    elif len(numbers) == 1:
        return [numbers[0], numbers[0]]

    return None


def get_macro_dialect(dataset_name, dialect_str, file_source=""):
    dialect = str(dialect_str).lower()
    file_source = str(file_source).lower()

    if dataset_name == "sofia":
        return "CS"

    if dataset_name == "epigraphica":
        return "CS"
    if dataset_name == "bible_ostrog":
        return "CS"
    if dataset_name == "UD_Old_East_Slavic-Ruthenian":
        return "SW"
    if dataset_name == "birchbark":
        return "NW"
    if dataset_name == "UD_Old_East_Slavic-RNC":
        return "OES"

    if dataset_name == "NKRYA":
        if "pskov" in file_source or "novgorod" in file_source:
            return "NW"
        return "OES"

    if dataset_name in ["pushkin_texts", "torot"]:
        # Improved logic for descriptions
        is_oes_base = (
            "древнерусск" in dialect
            or "старорусск" in dialect
            or "московск" in dialect
            or "русском языке" in dialect
            or "приказн" in dialect
        )
        is_cs_base = (
            "церковнославянск" in dialect
            or "старославянск" in dialect
            or "среднеболгарск" in dialect
            or "южнославянск" in dialect
        )

        # Explicit starts
        if (
            dialect.startswith("церковнославянский")
            or dialect.startswith("старославянский")
            or "на старославянском" in dialect
            or "на церковнославянском" in dialect
        ):
            return "CS"

        if (
            "влияни" in dialect
            or "элемент" in dialect
            or "смешени" in dialect
            or "традици" in dialect
        ):
            if is_oes_base:
                return "OES"

        if is_oes_base:
            return "OES"
        if is_cs_base:
            return "CS"
        return "OES"

    return "Unknown"


_BB_DOT_RUN_RE = re.compile(r"·(?:\s*·)+")  # 2+ consecutive · = a run of lost letters
_BB_HYPHEN_RE = re.compile(r"-+")  # birchbark also marks a lost letter with '-'


def birchbark_gap_fix(text):
    """birchbark's own diplomatic convention marks each illegible letter with
    a raised dot or hyphen. A single '·' between legible words is ordinary
    punctuation (by far the common case -- 2676 of ~2700 occurrences) and is
    left alone; a run of 2+ is a deliberate letter count and becomes one gap,
    same as an already-explicit [GAP]. Every '-' is a lost-letter marker."""
    text = _BB_DOT_RUN_RE.sub("[GAP]", text)
    text = _BB_HYPHEN_RE.sub("[GAP]", text)
    while "[GAP][GAP]" in text or "[GAP] [GAP]" in text:
        text = text.replace("[GAP][GAP]", "[GAP]").replace("[GAP] [GAP]", "[GAP]")
    return text


def visible_length(text):
    """Length of text with whitespace stripped, counting each [UNK] gap
    as a single character rather than the 5 chars of its literal spelling."""
    return len(re.sub(r"\[UNK\]", "#", re.sub(r"\s+", "", text)))


global_rnc_ngrams = set()


def get_ngrams(text, n=5):
    words = re.sub(r"[\W_]+", " ", text.lower()).split()
    return set([" ".join(words[i : i + n]) for i in range(max(1, len(words) - n + 1))])


def process_datasets():
    datasets = [
        ("UD_Old_East_Slavic-RNC", "data/UD_Old_East_Slavic-RNC/rnc_cleaned.json"),
        ("NKRYA", "data/NKRYA/nkrya_scraped_cleaned.json"),
        ("epigraphica", "data/epigraphica/epigraphica.json"),
        (
            "UD_Old_East_Slavic-Ruthenian",
            "data/UD_Old_East_Slavic-Ruthenian/ruthenian_cleaned.json",
        ),
        ("pushkin_texts", "data/pushkin_texts/pushkin_texts.json"),
        ("torot", "data/TOROT/torot.json"),
        ("sofia", "data/sofia/sofia_cleaned.json"),
        ("bible_ostrog", "data/bible_ostrog/bible_ostrog.json"),
        # No date/dialect metadata -- get_macro_dialect falls back to
        # "Unknown" and get_date_target(None) returns None for docs with no
        # "dialect"/"year" keys, both correctly excluded from the
        # date/region losses downstream rather than mislabeled.
        ("byliny", "data/byliny/byliny.json"),
    ]

    stats = {"CS": 0, "OES": 0, "NW": 0, "SW": 0, "Unknown": 0}

    for ds_name, json_path in datasets:
        if not os.path.exists(json_path):
            continue

        out_path = f"prepared_datasets/{ds_name}_prepared.jsonl"
        if "nkrya_scraped_cleaned" in out_path:
            out_path = out_path.replace(
                "nkrya_scraped_cleaned_prepared", "nkrya_prepared"
            )

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        with open(out_path, "w", encoding="utf-8") as f_out:
            for doc in data:
                # epigraphica has no bare 'text' field -- it ships 'original'
                # (diplomatic, with () [] restorations) and 'target' (the
                # reconstructed reading); 'text' downstream is the latter.
                text = doc.get("text", "") or doc.get("target", "")
                # Apply advanced paleographic normalization; epigraphy keeps
                # its () [] (genuine reconstructed text), everything else
                # has stray brackets stripped.
                text = normalize_historical_text(
                    text, keep_brackets=(ds_name == "epigraphica")
                )
                # Extract n-grams if this is RNC
                if "rnc_cleaned" in json_path:
                    global_rnc_ngrams.update(get_ngrams(text))

                # Deduplicate NKRYA against RNC
                if "NKRYA" in json_path:
                    n_ngrams = get_ngrams(text)
                    # If NKRYA text shares at least 3 5-grams with RNC, we consider it a duplicate and drop it
                    if len(n_ngrams.intersection(global_rnc_ngrams)) >= 3:
                        continue

                # Drop inscriptions/fragments too short to carry any signal
                # ([UNK] gaps count as 1 char, not their 5-char spelling).
                if (
                    ds_name in ("epigraphica", "bible_ostrog")
                    and visible_length(text) < 5
                ):
                    continue

                raw_year = doc.get("year", "")
                source = doc.get("source", "")
                dialect = doc.get("dialect", "")
                doc_id = doc.get("doc_id", "")

                macro_dialect = get_macro_dialect(ds_name, dialect, source)
                interval = parse_year(raw_year)
                target = get_date_target(interval)

                # Prevent massive CS/1581 imbalance by clearing labels on 90% of Bible
                import random

                if ds_name == "bible_ostrog" and random.random() > 0.1:
                    macro_dialect = "Unknown"
                    target = [0.0] * 20
                    interval = None

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
                if "original" in doc:
                    new_doc["original"] = normalize_historical_text(
                        doc["original"], keep_brackets=True
                    )
                if "target" in doc:
                    new_doc["target"] = normalize_historical_text(
                        doc["target"], keep_brackets=True
                    )

                f_out.write(json.dumps(new_doc, ensure_ascii=False) + "\n")
                stats[macro_dialect] += 1

        print(f"Prepared {ds_name} -> {out_path}")

    # Process birchbark
    birch_path = "data/birchbark_classes.jsonl"
    if os.path.exists(birch_path):
        out_path = "prepared_datasets/birchbark_classes_prepared.jsonl"
        with (
            open(birch_path, "r", encoding="utf-8") as f,
            open(out_path, "w", encoding="utf-8") as f_out,
        ):
            for line in f:
                if not line.strip():
                    continue
                doc = json.loads(line)

                macro_dialect = "NW"
                interval = doc.get("date_interval")
                target = get_date_target(interval)

                doc["macro_dialect"] = macro_dialect
                doc["date_target"] = target
                if "text" not in doc and "target" in doc:
                    doc["text"] = doc["target"]

                # Normalize + clean up newlines; birchbark keeps its () []
                # (genuine reconstructed text).
                for key in ["text", "target", "original", "masked"]:
                    if key in doc and isinstance(doc[key], str):
                        doc[key] = normalize_historical_text(
                            birchbark_gap_fix(doc[key]), keep_brackets=True
                        )

                if visible_length(doc.get("text", "")) < 5:
                    continue

                f_out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                stats[macro_dialect] += 1
        print(f"Prepared birchbark -> {out_path}")

        # Same rows, flattened to the epigraphica_prepared.jsonl field layout
        # (doc_id/text/macro_dialect/date_interval/date_target/date_number/
        # category/original_dialect/original/target) instead of birchbark's
        # own class-specific fields (masked/number/date/region/genre).
        flat_out_path = "prepared_datasets/birchbark_prepared.jsonl"
        with (
            open(birch_path, "r", encoding="utf-8") as f,
            open(flat_out_path, "w", encoding="utf-8") as f_out,
        ):
            for line in f:
                if not line.strip():
                    continue
                doc = json.loads(line)

                interval = doc.get("date_interval")
                date_target = get_date_target(interval)
                raw_date = doc.get("date", "")
                original = normalize_historical_text(
                    birchbark_gap_fix(doc.get("original", "")), keep_brackets=True
                )
                target = normalize_historical_text(
                    birchbark_gap_fix(doc.get("target", doc.get("original", ""))),
                    keep_brackets=True,
                )

                if visible_length(target) < 5:
                    continue

                new_doc = {
                    "doc_id": f"birchbark_{doc.get('number')}",
                    "text": target,
                    "macro_dialect": "NW",
                    "date_interval": interval,
                    "date_target": date_target,
                    "date_number": raw_date
                    if interval and interval[0] == interval[1]
                    else None,
                    "category": doc.get(
                        "category_mapped", doc.get("category", "unknown")
                    ),
                    "original_dialect": "birchbark",
                    "original": original,
                    "target": target,
                }
                f_out.write(json.dumps(new_doc, ensure_ascii=False) + "\n")
        print(f"Prepared birchbark (flat) -> {flat_out_path}")

    # Process epigraphica Test B
    epi_brackets_path = "data/epigraphica/epigraphica_final_cleaned_with_brackets.txt"
    if os.path.exists(epi_brackets_path):
        out_path = "prepared_datasets/epigraphica_classes_prepared.jsonl"
        with (
            open(epi_brackets_path, "r", encoding="utf-8") as f,
            open(out_path, "w", encoding="utf-8") as f_out,
        ):
            for line in f:
                if not line.strip():
                    continue
                original = normalize_historical_text(line.strip(), keep_brackets=True)
                if visible_length(original) < 5:
                    continue
                doc = {
                    "original": original,
                    "macro_dialect": "CS",
                    "date_target": [0.0] * 20,
                }
                f_out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                stats["CS"] += 1
        print(f"Prepared epigraphica brackets -> {out_path}")

    print("\n--- MACRO DIALECT STATS ---")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    process_datasets()
