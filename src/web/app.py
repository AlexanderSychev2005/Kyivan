"""Web demo for the trained Kyivan checkpoint. FastAPI + a static vanilla-JS
page (no separate frontend build), same shape as akkadian/src/web -- a
single input box and a handful of result panels is plenty for one model.

Serves the final_model checkpoint from the completed h224/mask018/300-epoch
run (see runs/kyivan_h224_mask018_300ep/). Everything below is inference
only -- no training, no writes to the corpus.

Run:  python src/web/app.py   (serves http://127.0.0.1:8000)
"""
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.model.config import KyivanConfig
from src.model.model import Kyivan
from src.model.interpret import char_gradient_saliency, document_embedding, nearest_documents
from src.model.vocab_categories import is_maskable_char
from src.data_pipeline.normalization import normalize_historical_text

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="Kyivan Web")
app.mount(
    "/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
char_vocab = None
id_to_char = None
MASK_TOKEN_ID = None
UNK_MASK_TOKEN_ID = None
forbidden_restore_ids = None
doc_embeddings = None
doc_meta = None
doc_extra = None

# Constants
CHECKPOINT_DIR = BASE_DIR / "runs" / "kyivan_h224_mask018_300ep" / "final_model"
VOCAB_PATH = BASE_DIR / "prepared_datasets" / "tokenizer" / "char_vocab.json"
EMBEDDINGS_DIR = BASE_DIR / "runs" / "kyivan_h224_mask018_300ep" / "embeddings"

# Label order fixed by DIALECT_MAP in src/data_pipeline/prepare_splits.py --
# {"OES": 0, "CS": 1, "NW": 2, "SW": 3} -- must match exactly, or every
# predicted dialect displayed here is simply the wrong one.
REGION_NAMES = {
    0: "Древневосточнославянский (OES)",
    1: "Церковнославянский (CS)",
    2: "Новгородский (NW)",
    3: "Юго-Западный (SW)",
}


def bin_to_period(bin_idx: int) -> str:
    start = 800 + bin_idx * 50
    return f"{start}–{start + 50} гг."


class AnalyzeRequest(BaseModel):
    text: str
    temperature: float = 1.0
    iterative: bool = False
    # "full" restores '?'/'#' positions in addition to attribution; "attribution"
    # skips restoration entirely (no iterative fill, no per-position saliency/top-k)
    # and classifies date/region straight off the text as typed, gaps included.
    mode: str = "full"


def load_resources():
    global model, char_vocab, id_to_char, MASK_TOKEN_ID, UNK_MASK_TOKEN_ID, forbidden_restore_ids
    global doc_embeddings, doc_meta, doc_extra

    log.info(f"Loading vocabulary from {VOCAB_PATH}...")
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        char_vocab = json.load(f)
    id_to_char = {v: k for k, v in char_vocab.items()}
    MASK_TOKEN_ID = char_vocab.get("[-]")
    UNK_MASK_TOKEN_ID = char_vocab.get("[#]")
    # Special tokens and punctuation/digits are never a valid restoration --
    # only letters and spaces may fill a [-] (matches inference.py's own
    # KyivanRestorer.forbidden_restore_ids and vocab_categories.MASKABLE_CATEGORIES).
    special_ids = {v for k, v in char_vocab.items() if k.startswith("[") and k.endswith("]")}
    forbidden_restore_ids = special_ids | {
        v for k, v in char_vocab.items() if len(k) == 1 and not is_maskable_char(k)
    }

    log.info(f"Loading model from {CHECKPOINT_DIR} on {device}...")
    config = KyivanConfig.from_pretrained(CHECKPOINT_DIR)
    model_ = Kyivan.from_pretrained(CHECKPOINT_DIR, config=config)
    model_.to(device)
    model_.eval()
    model = model_
    log.info("Model loaded successfully!")

    emb_path = EMBEDDINGS_DIR / "doc_embeddings.npy"
    meta_path = EMBEDDINGS_DIR / "doc_meta.json"
    if emb_path.exists() and meta_path.exists():
        log.info(f"Loading precomputed corpus embeddings from {EMBEDDINGS_DIR}...")
        doc_embeddings = np.load(emb_path)
        with open(meta_path, encoding="utf-8") as f:
            doc_meta = json.load(f)
        log.info(f"  {len(doc_meta)} documents available for similarity search")
    else:
        log.warning(
            f"No precomputed embeddings at {EMBEDDINGS_DIR} -- run "
            "src/model/compute_embeddings.py first; 'similar documents' will be empty."
        )

    # Optional, source-specific extras (currently just birchbark's find-site
    # region/genre) -- a separate file so the core doc_meta.json schema stays
    # uniform across all 7 corpora; see build_doc_extra.py. Missing entirely
    # or missing for a given doc_id both just mean "nothing extra to show".
    extra_path = EMBEDDINGS_DIR / "doc_extra.json"
    if extra_path.exists():
        with open(extra_path, encoding="utf-8") as f:
            doc_extra = json.load(f)
    else:
        doc_extra = {}


load_resources()


@app.get("/")
def read_root():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


def _rebuild_query(chars, mask_positions_hint=None):
    """Turns raw user text into (tokens, input_ids): '?' -> [-] (predict
    this character), '#' -> [#] (predict a gap of unknown length), '-' ->
    [UNK] (a position the user already knows is unrecoverable -- matches
    the paper's own three-marker design, Section "Data"; the model never
    tries to restore this, same as a real editor's own unresolved mark in
    Test B), everything else -> itself (falling back to [UNK] if outside
    the current vocab)."""
    tokens = ["[SOS]"]
    for ch in chars:
        if ch == "?":
            tokens.append("[-]")
        elif ch == "#":
            tokens.append("[#]")
        elif ch == "-":
            tokens.append("[UNK]")
        else:
            tokens.append(ch)
    input_ids = [char_vocab.get(t, char_vocab.get("[UNK]")) for t in tokens]
    input_ids = [tid if tid < model.config.vocab_size else char_vocab["[UNK]"] for tid in input_ids]
    return tokens, input_ids


_MARKER_RE = re.compile(r"([?#\-])")


def _normalize_preserving_markers(text: str) -> str:
    """normalize_historical_text (keep_brackets=False, the default here)
    strips every literal '[' and ']', deletes literal '-' outright (residual
    OCR/PUA noise cleanup), and whitelists out '?'/'#' entirely (its
    char-level regex only keeps \\w/whitespace/a handful of punctuation) --
    so none of the three request markers can survive being run through it
    directly. Split on all three instead and normalize only the text
    between them, then splice the markers back in literally, so they never
    touch the normalizer at all."""
    parts = _MARKER_RE.split(text)
    return "".join(p if p in ("?", "#", "-") else normalize_historical_text(p) for p in parts)


@app.post("/api/analyze")
def analyze_text(req: AnalyzeRequest):
    text = _normalize_preserving_markers(req.text)
    tokens, input_ids = _rebuild_query(text)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_tensor)

    mask_positions = [i for i, tid in enumerate(input_ids) if tid == MASK_TOKEN_ID]
    unk_positions = [i for i, tid in enumerate(input_ids) if tid == UNK_MASK_TOKEN_ID]

    iterative_filled = {}
    if req.mode != "attribution" and req.iterative and mask_positions:
        # Repeatedly fill in the single most-confident remaining [-]
        # position, most-confident-first, until none are left. '#' is
        # deliberately left alone here: its length is a whole separate,
        # unreliable decision (see Results -- the gap-expansion head is
        # biased towards "expand" and underperforms a trivial baseline on
        # real damage), so the UI only ever reports its single/multi
        # probability rather than auto-expanding and guessing content for a
        # length it isn't confident about. A historian who wants that content
        # can add '?'s by hand, one at a time, and see the prediction update.
        current_ids = list(input_ids)
        remaining = list(mask_positions)
        with torch.no_grad():
            while remaining:
                t_input = torch.tensor([current_ids], dtype=torch.long, device=device)
                t_mask = torch.ones_like(t_input)
                out = model(input_ids=t_input, attention_mask=t_mask)
                logits_restore = out.logits_restore[0]

                best_idx, best_prob, best_char_id = -1, -1.0, -1
                for idx in remaining:
                    probs = F.softmax(logits_restore[idx], dim=0)
                    for sp_id in forbidden_restore_ids:
                        probs[sp_id] = 0.0
                    top_prob, top_char = torch.max(probs, dim=0)
                    if top_prob.item() > best_prob:
                        best_prob, best_idx, best_char_id = top_prob.item(), idx, top_char.item()

                current_ids[best_idx] = best_char_id
                remaining.remove(best_idx)
                iterative_filled[best_idx] = best_char_id

        input_tensor = torch.tensor([current_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_tensor)
        for idx, char_id in iterative_filled.items():
            tokens[idx] = id_to_char.get(char_id, "?")

    with torch.no_grad():
        out = model(input_ids=input_tensor, attention_mask=attention_mask)
    logits_restore = out.logits_restore[0]
    logits_unk = out.logits_unk[0]
    date_probs = F.softmax(out.logits_date[0], dim=0).tolist()
    region_probs = F.softmax(out.logits_region[0], dim=0).tolist()

    restorations = []
    if req.mode != "attribution":
        for idx in mask_positions:
            t = max(0.01, req.temperature)
            probs = F.softmax(logits_restore[idx] / t, dim=0)
            for sp_id in forbidden_restore_ids:
                probs[sp_id] = 0.0
            topk_probs, topk_indices = torch.topk(probs, min(5, probs.shape[0]))
            top_k = [
                {"char": id_to_char.get(cid.item(), "?"), "prob": p.item()}
                for p, cid in zip(topk_probs, topk_indices)
            ]

            if idx in iterative_filled:
                # Saliency for an iteratively-filled position is computed by
                # masking just that one position back to [-] in the final,
                # fully-resolved sequence (every other position keeps the
                # model's own best guess) -- the live-inference analogue of
                # evaluate_cer.py's "isolate one lacuna, fill the rest with the
                # best available reconstruction". Reading it straight off the
                # fully-filled sequence would let the position see its own
                # already-decided character through the embedding, collapsing
                # the gradient to a near-tautology.
                query = input_tensor.clone()
                query[0, idx] = MASK_TOKEN_ID
                saliency, _ = char_gradient_saliency(model, query, torch.ones_like(query), target="restore", position=idx)
            else:
                saliency, _ = char_gradient_saliency(model, input_tensor, attention_mask, target="restore", position=idx)

            restorations.append({
                "token_index": idx,
                "is_unk": False,
                "top_k": top_k,
                "saliency": saliency.tolist(),
                "iterative_filled_char": id_to_char.get(iterative_filled[idx], "?") if idx in iterative_filled else None,
            })

        for idx in unk_positions:
            probs = F.softmax(logits_unk[idx], dim=0)
            restorations.append({
                "token_index": idx,
                "is_unk": True,
                "prob_multi": probs[1].item(),
                "prob_single": probs[0].item(),
            })

    date_saliency, _ = char_gradient_saliency(model, input_tensor, attention_mask, target="date")
    region_saliency, _ = char_gradient_saliency(model, input_tensor, attention_mask, target="region")

    similar_documents = []
    if doc_embeddings is not None:
        query_emb = document_embedding(model, input_tensor, attention_mask)
        for doc_id, score in nearest_documents(query_emb, doc_embeddings, [m["doc_id"] for m in doc_meta], k=10):
            row = next((m for m in doc_meta if m["doc_id"] == doc_id), None)
            if row is None:
                continue
            similar_documents.append({
                "doc_id": doc_id,
                "score": score,
                "source_dataset": row.get("source_dataset"),
                "macro_dialect": row.get("macro_dialect"),
                "date_interval": row.get("date_interval"),
                "category": row.get("category"),
                "text": row.get("text"),
                **(doc_extra.get(doc_id) or {}),
            })

    return {
        "tokens": tokens,
        "date_probs": [{"period": bin_to_period(i), "prob": p} for i, p in enumerate(date_probs)],
        "region_probs": [{"region": REGION_NAMES.get(i, f"Region {i}"), "prob": p} for i, p in enumerate(region_probs)],
        "date_saliency": date_saliency.tolist(),
        "region_saliency": region_saliency.tolist(),
        "restorations": restorations,
        "similar_documents": similar_documents,
    }


if __name__ == "__main__":
    # reload=False on purpose: with it on, uvicorn's StatReload spawns a
    # separate watcher process that also imports this module, loading the
    # checkpoint twice and doubling startup time/GPU memory for a tool with
    # no frontend-only edit loop to speed up. Restart manually after
    # changing app.py (see akkadian/src/web/app.py for the same rationale).
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
