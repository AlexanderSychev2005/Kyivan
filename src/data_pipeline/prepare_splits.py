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


def chunk_text(text, max_len=1024, stride=512):
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

    return masked_ids, labels, target_str


def main():
    with open("data/char_vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)

    print("Extracting Test B from birchbarks and epigraphica...")
    test_b_records = []
    test_b_texts = set()

    # Epigraphica Test B
    epigraphica_path = 'prepared_datasets/epigraphica_classes_prepared.jsonl'
    if os.path.exists(epigraphica_path):
        with open(epigraphica_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                if 'original' in doc:
                    res = process_test_b_line(doc['original'], vocab)
                    if res:
                        m_ids, labels, tgt_str = res
                        tgt_str_clean = re.sub(r'[\n\r]+', ' ', tgt_str)
                        tgt_str_clean = re.sub(r'\s{2,}', ' ', tgt_str_clean).strip()
                        test_b_texts.add(tgt_str_clean)
                        test_b_records.append({
                            'input_ids': m_ids,
                            'attention_mask': [1] * len(m_ids),
                            'labels': labels,
                            'date_labels': doc.get('date_target', [0.0]*20),
                            'region_labels': DIALECT_MAP['CS']
                        })

    # Birchbark Test B
    birchbark_path = "prepared_datasets/birchbark_classes_prepared.jsonl"
    if os.path.exists(birchbark_path):
        with open(birchbark_path, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                if "original" in doc:
                    res = process_test_b_line(doc["original"], vocab)
                    if res:
                        m_ids, labels, tgt_str = res
                        # Clean up target string similarly to how it was done for final_dataset
                        tgt_str_clean = re.sub(r"[\n\r]+", " ", tgt_str)
                        tgt_str_clean = re.sub(r"\s{2,}", " ", tgt_str_clean).strip()
                        test_b_texts.add(tgt_str_clean)

                        test_b_records.append(
                            {
                                "input_ids": m_ids,
                                "attention_mask": [1] * len(m_ids),
                                "labels": labels,
                                "date_labels": doc.get("date_target", [0.0] * 20),
                                "region_labels": DIALECT_MAP["NW"],
                            }
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
                    }
                )

    print(f"Excluded {overlap_count} documents due to Test B overlap.")
    print(f"Total chunks created for Train/Test A: {len(records)}")

    labels_strat = [r["region_labels"] for r in records]
    print("Performing stratified Train/Test A split (90/10)...")
    train_records, test_a_records = train_test_split(
        records, test_size=0.1, random_state=42, stratify=labels_strat
    )

    print(f"Train chunks: {len(train_records)}")
    print(f"Test A chunks: {len(test_a_records)}")

    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list(train_records),
            "test_a": Dataset.from_list(test_a_records),
        }
    )

    if len(test_b_records) > 0:
        dataset_dict["test_b"] = Dataset.from_list(test_b_records)

    out_dir = 'prepared_datasets/hf_dataset'
    print(f"Saving to {out_dir}...")
    dataset_dict.save_to_disk(out_dir)
    print("Done!")


if __name__ == "__main__":
    main()
