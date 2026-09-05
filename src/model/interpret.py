"""Post-hoc interpretability for an already-trained Kyivan checkpoint -- no
retraining, mirrors akkadian/src/analysis/interpret.py's approach for the
same reasons: gradient saliency (Simonyan et al. 2014's "vanilla gradient"),
not raw attention-weight visualization, matching what the paper commits to
(Methods, "Similar-document retrieval": Aeneas's authors found attention's
reliability as an explanation disputed, while gradient saliency was useful
to their historians).

document_embedding()/nearest_documents() implement the exact formula already
in papers/main.tex (Methods): 0.5*([SOS] + mean of the other real tokens),
Aeneas's own "historically enriched embedding" via the torso's own hidden
states, no separate retrieval model.
"""
from typing import Optional

import numpy as np
import torch
from torch import nn


def _encoder_hidden_states(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Replicates Kyivan.forward()'s embedding+encoder prefix (model.py) up
    to the shared torso output -- the one piece not exposed on KyivanOutput,
    needed here for document_embedding(). Kept as a few duplicated lines
    rather than changing model.py's return type, matching how the Akkadian
    reference calls model.backbone.bert(...) directly rather than editing
    the multi-task model's own forward()."""
    x = model.char_embeddings(input_ids)
    x = model.emb_norm(x)
    x = model.emb_dropout(x)
    ext_mask = model.get_extended_attention_mask(attention_mask, input_ids.shape)
    enc_out = model.encoder(x, attention_mask=ext_mask, return_dict=True)
    return enc_out.last_hidden_state


def char_gradient_saliency(
    model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    target: str, position: Optional[int] = None,
) -> tuple[np.ndarray, int]:
    """Per-character gradient-norm saliency for one scalar target logit.

    target="restore": saliency for the top-1 restored character at
    `position` (position required). target in ("date", "region"): saliency
    for that head's own top-1 predicted class, read from position 0
    ([SOS]) -- matching how those heads themselves are computed in model.py.

    Returns (scores, target_id): scores is a (seq_len,) float32 array in
    [0, 1] (gradient L2-norm on each position's own character embedding,
    max-normalized); target_id is the predicted class/character id the
    gradient was taken with respect to.
    """
    model.zero_grad(set_to_none=True)
    captured = {}

    def hook(_module, _inp, out):
        out.retain_grad()
        captured["emb"] = out

    handle = model.char_embeddings.register_forward_hook(hook)
    try:
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        if target == "restore":
            assert position is not None, "target='restore' needs a masked position"
            logits = out.logits_restore[0, position].clone()
        elif target == "date":
            logits = out.logits_date[0]
        elif target == "region":
            logits = out.logits_region[0]
        else:
            raise ValueError(f"Unknown target: {target}")
        target_id = int(logits.argmax().item())
        logits[target_id].backward()
        grad = captured["emb"].grad
        if grad is None:
            return np.zeros(input_ids.shape[1], dtype=np.float32), target_id
        scores = grad[0].norm(dim=-1).detach().cpu().numpy().astype(np.float32)
    finally:
        handle.remove()
    peak = scores.max()
    if peak > 0:
        scores = scores / peak
    return scores, target_id


def document_embedding(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
    """0.5*([SOS] + mean of the other real tokens) -- see module docstring."""
    with torch.no_grad():
        seq = _encoder_hidden_states(model, input_ids, attention_mask)[0]
        mask = attention_mask[0].bool()
        sos = seq[0]
        rest = seq[1:][mask[1:]]
        mean = rest.mean(dim=0) if rest.shape[0] > 0 else sos
        emb = 0.5 * (sos + mean)
    return emb.detach().cpu().numpy().astype(np.float32)


def batched_document_embedding(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
    """Vectorized over the batch dimension -- for compute_embeddings.py's
    corpus-wide pass, where looping one example at a time would leave the
    GPU mostly idle between tiny forward passes (same rationale as
    akkadian/src/analysis/compute_embeddings.py's own batched variant)."""
    with torch.no_grad():
        seq = _encoder_hidden_states(model, input_ids, attention_mask)
        mask = attention_mask.bool()
        sos = seq[:, 0]
        rest_mask = mask.clone()
        rest_mask[:, 0] = False
        rest_mask_f = rest_mask.unsqueeze(-1).float()
        counts = rest_mask_f.sum(dim=1)
        mean = (seq * rest_mask_f).sum(dim=1) / counts.clamp(min=1)
        mean = torch.where(counts > 0, mean, sos)
        emb = 0.5 * (sos + mean)
    return emb.detach().cpu().numpy().astype(np.float32)


def nearest_documents(
    query_emb: np.ndarray, doc_embeddings: np.ndarray, doc_ids: list, k: int = 10,
    exclude_id: Optional[str] = None,
) -> list[tuple[str, float]]:
    """Cosine similarity top-k against a precomputed (N, hidden) matrix (see
    compute_embeddings.py). Returns a list of (doc_id, score)."""
    q = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    d = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-8)
    sims = d @ q
    order = np.argsort(-sims)
    results: list[tuple[str, float]] = []
    for i in order:
        did = doc_ids[i]
        if did == exclude_id:
            continue
        results.append((did, float(sims[i])))
        if len(results) >= k:
            break
    return results
