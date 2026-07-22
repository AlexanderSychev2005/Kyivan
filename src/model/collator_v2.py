"""
Kyivan Physical Degradation Data Collator -- v2, aligned with DeepMind's
Aeneas dataloader (google-deepmind/predictingthepast, train/dataloader.py +
predictingthepast/util/text.py).

collator.py (v1) captured the *idea* of a compressed unknown-length gap
token but diverged from Aeneas in three structural ways, fixed here:

1. Per-example masking intensity. v1 masks each character independently at a
   fixed `mlm_prob`. Aeneas samples ONE rate per example from
   `Uniform(char_mask_rate_min, char_mask_rate_max)`, then masks *exactly*
   that count of positions (`np.random.choice(..., replace=False)`) -- every
   example gets a precisely controlled, but varying, masking intensity.

2. Non-compressing span masks. v1 collapses *every* multi-character masked
   span into a single `[#]`. In Aeneas, the `span_mask_ratio` budget
   (`util_text.random_mask_span`) masks a contiguous run but leaves it as
   *individual* single-character mask markers -- length stays fully visible
   and countable. The compressed "unknown-length gap" token is a SEPARATE,
   single-per-example mechanism (`util_text.inject_missing_unk`), and it
   carries no character-restoration target at all (Aeneas's `text_unmasked`
   is computed *after* the gap already collapsed the original characters
   away) -- only the binary "was it >1 char" label for the unk head.

3. Punctuation protection. Aeneas excludes punctuation from
   `non_missing_idx` and restricts `random_mask_span`'s search to
   `[a-zA-Z0α-ωΑ-Ω\\s]+` runs -- punctuation is never masked.

Also implements Aeneas's train/valid mode split: 'valid' skips the
random-rate budget entirely and instead masks exactly one span of random
size 1..span_mask_eval_len (`span_mask_eval_len`, `mode='valid'` in
generate_sample) -- a fixed, comparable evaluation difficulty instead of
train's noisier regime. Construct a second instance with mode='valid' for
eval_dataset if you want that behavior; Trainer's default single-collator
wiring uses one instance for both.

Not ported from Aeneas (out of scope for "masking"): random_word_swap/
_abbr/_delete, random_char_delete, random_sentence_swap, punctuation_delete,
and per-epoch random context-window resampling (we pre-chunk once in
prepare_splits.py's chunk_text instead).

Kept from v1, not present in this Aeneas dataloader file: edge_prob's
simulated physical tear at document start/end (birch-bark-specific). It
shares the same "one compressed gap per example" slot as inject_missing_unk
-- if an edge tear fires, it *is* that example's gap; inject_missing_unk is
only rolled when the edge tear didn't fire.

Real Aeneas hyperparameters (train/config_greek.py), used as defaults here:
char_mask_rate_min=0.0, char_mask_rate_max=0.75, span_mask_ratio=0.15,
span_mask_geometric_p=0.1, inject_missing_unk_p=0.25 (reused directly as
the geometric-distribution parameter for the gap's length, matching
inject_missing_unk's own signature), span_mask_eval_len=10.
"""

import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch

try:
    from .vocab_categories import maskable_ids as compute_maskable_ids  # package-style import
except ImportError:
    from vocab_categories import maskable_ids as compute_maskable_ids  # script-mode import


class KyivanPhysicalCollatorV2:
    def __init__(
        self,
        char_vocab: Dict[str, int],
        char_mask_rate_min: float = 0.0,
        char_mask_rate_max: float = 0.75,
        span_mask_ratio: float = 0.15,
        span_mask_geometric_p: float = 0.1,
        unk_geometric_p: float = 0.25,
        span_mask_eval_len: int = 10,
        edge_prob: float = 0.1,
        mode: str = "train",
        date_bins: int = 20,
    ) -> None:
        """
        Args:
            char_vocab: the character-level tokenizer vocabulary.
            char_mask_rate_min/max: per-example masking rate is sampled
                uniformly from this range (train mode only).
            span_mask_ratio: fraction of the per-example mask budget spent on
                contiguous (non-compressing) spans rather than lone chars.
            span_mask_geometric_p: geometric-distribution parameter for
                non-compressing span length (mean length = 1/p).
            unk_geometric_p: geometric-distribution parameter for the single
                compressed unknown-length gap's length (sampled as
                geometric(p) - 1, so it's a no-op for ~p of examples).
            span_mask_eval_len: valid mode masks exactly one span of size
                uniform(1, span_mask_eval_len).
            edge_prob: probability of a simulated physical edge tear, which
                (if it fires) takes the place of the compressed gap.
            mode: 'train' or 'valid' -- selects the masking regime.
            date_bins: width of the date-distribution placeholder used for
                examples with no date label (must match the model's num_date_bins).
        """
        self.vocab = char_vocab
        self.pad_id = char_vocab["[PAD]"]
        self.mask_id = char_vocab["[-]"]
        self.unk_id = char_vocab["[#]"]
        self.sos_id = char_vocab["[SOS]"]

        self.char_mask_rate_min = char_mask_rate_min
        self.char_mask_rate_max = char_mask_rate_max
        self.span_mask_ratio = span_mask_ratio
        self.span_mask_geometric_p = span_mask_geometric_p
        self.unk_geometric_p = unk_geometric_p
        self.span_mask_eval_len = span_mask_eval_len
        self.edge_prob = edge_prob
        self.mode = mode
        self.date_bins = date_bins

        # Special (bracket-wrapped) tokens are always protected.
        self.special_ids = {
            v for k, v in char_vocab.items() if k.startswith("[") and k.endswith("]")
        }
        # Only letters/spaces are maskable by default -- mirrors Aeneas
        # excluding alphabet.punctuation from non_missing_idx and
        # restricting random_mask_span's regex to letters + whitespace.
        # See vocab_categories.MASKABLE_CATEGORIES for the shared policy
        # (also used by train.py, inference.py, and prepare_splits.py).
        self.maskable_ids = compute_maskable_ids(char_vocab)

    def _maskable_runs(self, tokens: List[int], excluded: set) -> List[tuple]:
        """Maximal (start, end) runs of maskable, non-excluded positions --
        token-id-space equivalent of Aeneas's re.finditer over letters+space."""
        runs = []
        start = None
        for i, tok in enumerate(tokens):
            ok = tok in self.maskable_ids and i not in excluded
            if ok and start is None:
                start = i
            elif not ok and start is not None:
                runs.append((start, i))
                start = None
        if start is not None:
            runs.append((start, len(tokens)))
        return runs

    def _sample_span(
        self, tokens: List[int], excluded: set, span_len: int
    ) -> List[int]:
        """One random contiguous run of `span_len` maskable positions
        (mirrors util_text.random_mask_span)."""
        if span_len <= 0:
            return []
        runs = [r for r in self._maskable_runs(tokens, excluded) if r[1] - r[0] >= span_len]
        if not runs:
            return []
        start, end = random.choice(runs)
        span_start = random.randint(start, end - span_len)
        return list(range(span_start, span_start + span_len))

    def _pick_compressed_gap(
        self, tokens: List[int]
    ) -> Optional[tuple]:
        """The single compressed [#] gap for this example, if any: edge tear
        (our addition) or a random-position gap (Aeneas's
        inject_missing_unk). Returns (positions, is_multi_char) or None."""
        if random.random() < self.edge_prob and len(tokens) > 10:
            edge_len = random.randint(2, 5)
            if random.random() < 0.5:
                start = 1
                while start < len(tokens) and tokens[start] in self.special_ids:
                    start += 1
                end = min(start + edge_len, len(tokens))
            else:
                end = len(tokens)
                start = max(1, end - edge_len)
            positions = [i for i in range(start, end) if tokens[i] not in self.special_ids]
            if positions:
                return positions, len(positions) > 1

        # Aeneas: span_len = geometric(p) - 1; a no-op (~p probability) is
        # expected and intentional, not an error.
        span_len = int(np.random.geometric(self.unk_geometric_p)) - 1
        if span_len <= 0:
            return None
        # Unlike the non-compressing span budget below, Aeneas places this
        # gap uniformly anywhere in the raw string, not restricted to
        # letter-only runs -- only special tokens are off-limits here.
        candidates = [i for i in range(1, len(tokens)) if tokens[i] not in self.special_ids]
        if len(candidates) < span_len:
            return None
        # A random slice of `candidates` isn't guaranteed contiguous in the
        # original sequence (a special token may have been filtered out
        # in between) -- a real lacuna must be one unbroken physical gap.
        # Retry a bounded number of times instead of giving up on the first
        # non-contiguous draw, so edge_prob/unk_geometric_p aren't
        # under-realized on documents with mid-text special tokens.
        for _ in range(50):
            start_pos = random.randint(0, len(candidates) - span_len)
            positions = candidates[start_pos : start_pos + span_len]
            if positions[-1] - positions[0] == span_len - 1:
                return positions, span_len > 1
        return None

    def _process_one(self, tokens: List[int]) -> tuple:
        if tokens[0] != self.sos_id:
            tokens = [self.sos_id] + tokens

        gap = self._pick_compressed_gap(tokens)
        gap_positions = set(gap[0]) if gap else set()
        gap_is_multi = gap[1] if gap else False

        excluded = set(range(1)) | gap_positions  # position 0 (SOS) + the gap
        span_budget_idx: List[int] = []
        char_budget_idx: List[int] = []

        if self.mode == "valid":
            # Aeneas: exactly one span of random size 1..span_mask_eval_len,
            # non-compressing, no separate single-char budget.
            eval_len = random.randint(1, self.span_mask_eval_len)
            for _ in range(1000):
                span_budget_idx = self._sample_span(tokens, excluded, eval_len)
                if len(span_budget_idx) == eval_len:
                    break
        else:
            non_missing_idx = [
                i for i in range(1, len(tokens))
                if tokens[i] in self.maskable_ids and i not in excluded
            ]
            if non_missing_idx:
                rate = random.uniform(self.char_mask_rate_min, self.char_mask_rate_max)
                mask_num_total = int(rate * len(non_missing_idx))
                mask_num_span = int(mask_num_total * self.span_mask_ratio)
                mask_num_char = mask_num_total - mask_num_span

                # Spans drawn first so single-char picks don't overlap them.
                span_excluded = set(excluded)
                for _ in range(1000):
                    if len(span_budget_idx) >= mask_num_span:
                        break
                    span_len = int(np.random.geometric(self.span_mask_geometric_p))
                    span_len = min(span_len, mask_num_span - len(span_budget_idx))
                    picked = self._sample_span(tokens, span_excluded, span_len)
                    if not picked:
                        continue
                    span_budget_idx.extend(picked)
                    span_excluded.update(picked)

                # Any span budget that couldn't be placed (e.g. too few
                # long-enough maskable runs left) rolls over into the
                # single-char budget instead of silently masking this
                # example below the rate sampled above.
                span_shortfall = mask_num_span - len(span_budget_idx)
                if span_shortfall > 0:
                    mask_num_char += span_shortfall

                remaining_pool = [i for i in non_missing_idx if i not in span_excluded]
                mask_num_char = min(mask_num_char, len(remaining_pool))
                if mask_num_char > 0:
                    char_budget_idx = random.sample(remaining_pool, mask_num_char)

        new_tokens: List[int] = []
        new_labels_res: List[int] = []
        new_labels_unk: List[int] = []

        single_mask_idx = set(span_budget_idx) | set(char_budget_idx)
        i = 0
        while i < len(tokens):
            if i in gap_positions and i == min(gap_positions):
                new_tokens.append(self.unk_id)
                new_labels_res.append(-100)  # no target: Aeneas doesn't
                new_labels_unk.append(1 if gap_is_multi else 0)
                i = max(gap_positions) + 1
                continue
            if i in gap_positions:
                # Already emitted as part of the gap token above.
                i += 1
                continue

            if i in single_mask_idx:
                new_tokens.append(self.mask_id)
                new_labels_res.append(tokens[i])
                new_labels_unk.append(-100)
            else:
                new_tokens.append(tokens[i])
                new_labels_res.append(-100)
                new_labels_unk.append(-100)
            i += 1

        return new_tokens, new_labels_res, new_labels_unk

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        b_input_ids = []
        b_labels_res = []
        b_labels_unk = []

        for f in features:
            tokens, labels_res, labels_unk = self._process_one(list(f["input_ids"]))
            b_input_ids.append(tokens)
            b_labels_res.append(labels_res)
            b_labels_unk.append(labels_unk)

        max_len = max(len(seq) for seq in b_input_ids)

        t_input_ids = torch.full((len(features), max_len), self.pad_id, dtype=torch.long)
        t_labels_res = torch.full((len(features), max_len), -100, dtype=torch.long)
        t_labels_unk = torch.full((len(features), max_len), -100, dtype=torch.long)
        t_attention_mask = torch.zeros((len(features), max_len), dtype=torch.long)

        for b in range(len(features)):
            slen = len(b_input_ids[b])
            t_input_ids[b, :slen] = torch.tensor(b_input_ids[b])
            t_labels_res[b, :slen] = torch.tensor(b_labels_res[b])
            t_labels_unk[b, :slen] = torch.tensor(b_labels_unk[b])
            t_attention_mask[b, :slen] = 1

        batch = {
            "input_ids": t_input_ids,
            "attention_mask": t_attention_mask,
            "labels": t_labels_res,
            "unk_labels": t_labels_unk,
        }

        # date_labels may be None per-example (e.g. Ostrog Bible chunks with the
        # date tag deliberately withheld) -- unlike region_labels, KLDivLoss has
        # no built-in ignore_index, so missing entries get a zero placeholder
        # plus a companion date_labels_mask for compute_loss/compute_metrics to
        # exclude them explicitly.
        if "date_labels" in features[0]:
            date_vals, date_mask = [], []
            for f in features:
                d = f.get("date_labels")
                if d is None:
                    date_vals.append([0.0] * self.date_bins)
                    date_mask.append(0.0)
                else:
                    date_vals.append(d)
                    date_mask.append(1.0)
            batch["date_labels"] = torch.tensor(date_vals, dtype=torch.float)
            batch["date_labels_mask"] = torch.tensor(date_mask, dtype=torch.float)

        if "region_labels" in features[0]:
            regions = [
                f["region_labels"] if f["region_labels"] is not None else -100
                for f in features
            ]
            batch["region_labels"] = torch.tensor(regions, dtype=torch.long)

        return batch
