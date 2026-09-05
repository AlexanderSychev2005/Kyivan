"""Offline: compute and save one retrieval embedding (interpret.py's
document_embedding -- Aeneas-style 0.5*([SOS] + mean of the rest)) per
corpus document, for the web demo's similar-document lookup (src/web/app.py).
No retraining -- reuses the already fine-tuned checkpoint; a live query
embedding computed by app.py must come from the same checkpoint, or their
cosine similarity would be comparing two different embedding spaces.

train/eval/test_a store raw "text" (tokenized here the same way training
does); test_b stores pre-tokenized "input_ids"/"labels" with masked/gap
positions, so those are first resolved back to their ground-truth characters
(build_clean_sequence) -- the corpus embedding should reflect what the
document actually says, not the artificial gap placeholders test_b evaluates
against. Documents longer than --max_len are simply head-truncated (not
train's random crop): this is a one-time, deterministic precompute, not a
training signal, so reproducibility matters more than sampling coverage.

Run once (re-run only if the corpus or checkpoint changes):

    python src/model/compute_embeddings.py

Output: <out_dir>/doc_embeddings.npy ((N, hidden) float32) + doc_meta.json
(one {doc_id, split, source_dataset, macro_dialect, date_interval, text}
per row, same order) -- src/web/app.py loads both at startup.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.model.config import KyivanConfig
from src.model.evaluate_cer import build_clean_sequence
from src.model.interpret import batched_document_embedding
from src.model.model import Kyivan
from src.model.vocab_categories import tokenize_text

BASE_DIR = Path(__file__).resolve().parent.parent.parent
REGION_NAMES = {0: "OES", 1: "CS", 2: "NW", 3: "SW"}


def load_meta(row) -> dict:
    meta = row.get("metadata")
    return json.loads(meta) if isinstance(meta, str) else (meta or {})


def decode_ids(ids, id_to_char: dict) -> str:
    skip = {"[SOS]", "[PAD]"}
    out = []
    for i in ids:
        ch = id_to_char.get(int(i), "")
        if ch in skip:
            continue
        out.append("#" if ch == "[GAP]" or ch == "[#]" else ("_" if ch == "[-]" else ch))
    return "".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(BASE_DIR / "runs" / "kyivan_h224_mask018_300ep" / "final_model"))
    parser.add_argument("--vocab", default=str(BASE_DIR / "prepared_datasets" / "tokenizer" / "char_vocab.json"))
    parser.add_argument("--dataset_dir", default=str(BASE_DIR / "prepared_datasets" / "hf_dataset"))
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_dir", default=str(BASE_DIR / "runs" / "kyivan_h224_mask018_300ep" / "embeddings"))
    args = parser.parse_args()

    with open(args.vocab, encoding="utf-8") as f:
        vocab = json.load(f)
    id_to_char = {v: k for k, v in vocab.items()}
    pad_id = vocab["[PAD]"]
    sos_id = vocab["[SOS]"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.checkpoint} on {device}...")
    config = KyivanConfig.from_pretrained(args.checkpoint)
    model = Kyivan.from_pretrained(args.checkpoint, config=config).to(device)
    model.eval()

    print(f"Loading dataset from {args.dataset_dir}...")
    ds = load_from_disk(args.dataset_dir)

    doc_ids, records, batch_ids = [], [], []
    embeddings, meta = [], []

    def flush():
        if not batch_ids:
            return
        max_len = max(len(ids) for ids in batch_ids)
        input_ids = torch.full((len(batch_ids), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch_ids), max_len), dtype=torch.long)
        for i, ids in enumerate(batch_ids):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1
        embs = batched_document_embedding(model, input_ids.to(device), attention_mask.to(device))
        embeddings.extend(embs)
        meta.extend(records)
        batch_ids.clear()
        records.clear()

    for split in ["train", "eval", "test_a", "test_b"]:
        if split not in ds:
            continue
        rows = ds[split]
        for row in tqdm(rows, desc=split):
            if split == "test_b":
                clean = build_clean_sequence(row["input_ids"], row["labels"])
                ids = clean if clean and clean[0] == sos_id else [sos_id] + clean
                text_preview = decode_ids(ids, id_to_char)
                m = load_meta(row)
            else:
                ids = [sos_id] + tokenize_text(row["text"], vocab)
                text_preview = row["text"]
                m = load_meta(row)
            ids = ids[: args.max_len]

            date_interval = m.get("date_interval")
            batch_ids.append(ids)
            records.append({
                "doc_id": m.get("doc_id", f"{split}_{len(records)}"),
                "split": split,
                "source_dataset": m.get("source_dataset"),
                "macro_dialect": m.get("macro_dialect"),
                "date_interval": date_interval,
                "category": m.get("category"),
                "text": text_preview[:600],
            })
            if len(batch_ids) >= args.batch_size:
                flush()
        flush()

    os.makedirs(args.out_dir, exist_ok=True)
    arr = np.stack(embeddings).astype(np.float32)
    np.save(os.path.join(args.out_dir, "doc_embeddings.npy"), arr)
    with open(os.path.join(args.out_dir, "doc_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    print(f"Saved {arr.shape[0]} embeddings (dim={arr.shape[1]}) to {args.out_dir}")


if __name__ == "__main__":
    main()
