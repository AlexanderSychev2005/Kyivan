import json
import os
import re

from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


def load_data(path):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


DIALECT_MAP = {"OES": 0, "CS": 1, "NW": 2, "SW": 3}
ROUND_PAT = re.compile(r"\(([^)]+)\)")
SQUARE_PAT = re.compile(r"\[(?!(?:GAP|MASK|PAD|UNK|CLS|SEP|SOS|#|-)\]|CTX_)([^\]]+)\]")


def chunk_text(text, max_len=512, stride=256):
    chunks = []
    text_len = len(text)
    if text_len <= max_len:
        return [text]
    for i in range(0, text_len, stride):
        chunk = text[i : i + max_len]
        chunks.append(chunk)
        if i + max_len >= text_len:
            break
    return chunks


def tokenize_text(text, vocab):
    tokens = []
    i = 0
    text_len = len(text)
    unk_id = vocab.get("[UNK]", 1)

    while i < text_len:
        if text[i] == "[":
            end = text.find("]", i)
            if end != -1:
                special = text[i : end + 1]
                if special in vocab:
                    tokens.append(vocab[special])
                    i = end + 1
                    continue
        char = text[i]
        if char in vocab:
            tokens.append(vocab[char])
        else:
            tokens.append(unk_id)
        i += 1
    return tokens


def process_test_b_line(line, vocab):
    line = line.strip()
    if not line:
        return None

    has_round = bool(ROUND_PAT.search(line))
    has_square = bool(SQUARE_PAT.search(line))
    if not has_round and not has_square:
        return None

    target_str = ROUND_PAT.sub(r"\1", line)
    target_str = SQUARE_PAT.sub(r"\1", target_str)

    def replace_func(m):
        inner = m.group(1)
        return "[-]" * len(inner)

    masked_str = ROUND_PAT.sub(replace_func, line)
    masked_str = SQUARE_PAT.sub(replace_func, masked_str)

    target_ids = tokenize_text(target_str, vocab)
    masked_ids = tokenize_text(masked_str, vocab)

    if len(target_ids) != len(masked_ids):
        return None

    mask_id = vocab.get("[-]")
    labels = []
    valid_mask = False
    for i in range(len(masked_ids)):
        if masked_ids[i] == mask_id:
            labels.append(target_ids[i])
            valid_mask = True
        else:
            labels.append(-100)

    if not valid_mask:
        return None

    # The model's date/region heads always pool from position 0, and the
    # collator always prepends [SOS] for train/eval/test_a -- test_b's
    # pre-built input_ids must carry the same convention, or the model sees
    # an out-of-distribution layout (wrong pooling token, shifted RoPE
    # positions) whenever test_b is evaluated.
    sos_id = vocab["[SOS]"]
    masked_ids = [sos_id] + masked_ids
    labels = [-100] + labels

    return masked_ids, labels, target_str, masked_str


def _clean_text_b_target(text):
    text = re.sub(r"[\n\r]+", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def collect_test_b(path, vocab, region_label, test_b_texts, test_b_records):
    """Every doc whose 'original' has a real () or [] must be excluded from
    the general train/eval/test_a pool, whether or not we can also build a
    usable masked test_b record from it (process_test_b_line's regex-based
    masking can fail on edge cases like nested brackets). So the exclusion
    text always comes from the pipeline's own canonical 'text'/'target'
    field -- never from process_test_b_line's independently re-derived
    target string, which can diverge from it (different whitespace
    collapsing) and silently break the exact-match leak check."""
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            original = doc.get("original")
            if not original:
                continue
            if not (ROUND_PAT.search(original) or SQUARE_PAT.search(original)):
                continue

            canonical = _clean_text_b_target(doc.get("text", doc.get("target", "")))
            if canonical:
                test_b_texts.add(canonical)

            res = process_test_b_line(original, vocab)
            if not res:
                continue
            m_ids, labels, tgt_str, masked_str = res
            test_b_records.append({
                "input_ids": m_ids,
                "attention_mask": [1] * len(m_ids),
                "labels": labels,
                "date_labels": doc.get("date_target", [0.0] * 20),
                "region_labels": region_label,
                "original_text": tgt_str,
                "text_with_missing": masked_str,
                "metadata": json.dumps(doc, ensure_ascii=False)
            })


def main():
    with open("prepared_datasets/tokenizer/char_vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)

    print("Extracting Test B from birchbarks and epigraphica...")
    test_b_records = []
    test_b_texts = set()

    collect_test_b(
        "prepared_datasets/epigraphica_prepared.jsonl", vocab, DIALECT_MAP["CS"],
        test_b_texts, test_b_records,
    )
    collect_test_b(
        "prepared_datasets/birchbark_prepared.jsonl", vocab, DIALECT_MAP["NW"],
        test_b_texts, test_b_records,
    )

    print(f"Extracted {len(test_b_records)} segments for Test B.")

    print("Loading final_dataset.jsonl...")
    docs = list(load_data('prepared_datasets/final_dataset.jsonl'))

    records = []
    overlap_count = 0
    for doc in docs:
        text = doc["text"]
        # Prevent data leakage: skip text if it's already used in Test B
        if text.strip() in test_b_texts:
            overlap_count += 1
            continue

        date_target = doc.get("date_target", [0.0] * 20)
        dialect_str = doc.get("macro_dialect", "OES")
        dialect_id = DIALECT_MAP.get(dialect_str, 0)

        # Create metadata without the massive full text to avoid MemoryError when duplicating across chunks
        meta_doc = {k: v for k, v in doc.items() if k != "text"}
        meta_json = json.dumps(meta_doc, ensure_ascii=False)

        chunks = chunk_text(text, max_len=1024, stride=512)
        for c in chunks:
            input_ids = tokenize_text(c, vocab)
            if len(input_ids) > 0:
                records.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": [1]
                        * len(input_ids),  # Needed for HF, overridden by collator
                        "date_labels": date_target,
                        "region_labels": dialect_id,
                        "original_text": c,
                        "metadata": meta_json
                    }
                )

    print(f"Excluded {overlap_count} documents due to Test B overlap.")
    print(f"Total chunks created for Train/Test A: {len(records)}")

    labels_strat = [r["region_labels"] for r in records]

    # 90% train / 5% eval / 5% test_a, stratified by macro_dialect throughout
    # (two-stage: hold out 10% first, then split that in half).
    train_records, holdout_records, _, holdout_labels = train_test_split(
        records, labels_strat, test_size=0.1, random_state=42, stratify=labels_strat
    )
    eval_records, test_a_records = train_test_split(
        holdout_records, test_size=0.5, random_state=42, stratify=holdout_labels
    )

    print(f"Train chunks: {len(train_records)}")
    print(f"Eval chunks: {len(eval_records)}")
    print(f"Test A chunks: {len(test_a_records)}")

    # ---------------------------------------------------------
    # EXPORT METADATA JSON FOR TEST SPLITS
    # ---------------------------------------------------------
    os.makedirs("test_eval_datasets", exist_ok=True)
    print("Exporting rich JSON files for test splits to test_eval_datasets/...")

    def clean_meta(meta):
        meta.pop("date_target", None)
        interval = meta.pop("date_interval", None)
        if interval:
            meta["date"] = f"{interval[0]} - {interval[1]}"
        elif meta.get("date_number"):
            meta["date"] = str(meta["date_number"])
        return meta

    def export_plain(records, out_name):
        export = [
            {
                "original_text": r["original_text"],
                "metadata": clean_meta(json.loads(r["metadata"])),
            }
            for r in records
        ]
        with open(f"test_eval_datasets/{out_name}.json", "w", encoding="utf-8") as f:
            json.dump(export, f, ensure_ascii=False, indent=2)

    export_plain(eval_records, "eval")
    export_plain(test_a_records, "test_a")

    test_b_export = []
    for r in test_b_records:
        meta = clean_meta(json.loads(r["metadata"]))
        test_b_export.append({
            "original_text": r["original_text"],
            "text_with_missing": r["text_with_missing"],
            "metadata": meta
        })

    with open("test_eval_datasets/test_b.json", "w", encoding="utf-8") as f:
        json.dump(test_b_export, f, ensure_ascii=False, indent=2)

    # ---------------------------------------------------------
    # CREATE HF DATASET
    # ---------------------------------------------------------
    def strip_export_fields(recs):
        allowed = {"input_ids", "attention_mask", "labels", "date_labels", "region_labels", "original_text", "metadata", "text_with_missing"}
        return [{k: v for k, v in r.items() if k in allowed} for r in recs]

    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list(strip_export_fields(train_records)),
            "eval": Dataset.from_list(strip_export_fields(eval_records)),
            "test_a": Dataset.from_list(strip_export_fields(test_a_records)),
        }
    )

    if len(test_b_records) > 0:
        dataset_dict["test_b"] = Dataset.from_list(strip_export_fields(test_b_records))

    out_dir = 'prepared_datasets/hf_dataset'
    print(f"Saving to {out_dir}...")
    dataset_dict.save_to_disk(out_dir)
    print("Done!")


if __name__ == "__main__":
    main()
