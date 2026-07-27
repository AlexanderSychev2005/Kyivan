"""
Kyivan Beam-Search Restorer.

Ports DeepMind Aeneas's non-sequential beam search decoding
(google-deepmind/predictingthepast, predictingthepast/util/eval.py::
beam_search_batch, called from predictingthepast/eval/inference.py::restore)
to Kyivan. Kept in its own file/class -- KyivanBeamRestorer subclasses
KyivanRestorer (inference.py) rather than modifying it; the greedy decoder
there is a separate, independently usable tool with its own CLI, not
touched by anything in this file.

Input syntax (see KyivanRestorer/_tokenize below): bare `?` for one missing
character, `#` for a lacuna of unknown length -- same convention as
inference.py, no brackets, no `[SOS]`.

Unlike KyivanRestorer.restore_text's greedy "fill the single most confident
mask, one at a time" loop, this explores many candidate restorations in
parallel: at each iteration every hypothesis in the beam branches over
EVERY still-open `[-]`/`[#]` position at once (all candidate characters for
`[-]`, both expand/stop for `[#]`), all resulting candidates are scored by
length-normalized log-probability, and only the top `beam_width` survive to
the next iteration. Returns the top-N full candidate restorations, ranked,
rather than a single greedy answer.

Also ports Aeneas's post-hoc saliency approach: KyivanBeamRestorer.
saliency_for_result() replays a single chosen result's winning trajectory
and extracts attention maps only for that path, mirroring Aeneas's
sequential_restoration_saliency() rather than tracking saliency for every
(mostly discarded) beam hypothesis during the search itself.

Not ported from Aeneas's restore() (out of scope for "beam search"):
sequential_decoding=True mode, nucleus sampling, vision conditioning --
none of those are specific to beam search itself.
"""

import argparse
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch

try:
    from .inference import KyivanRestorer  # package-style import
except ImportError:
    from inference import KyivanRestorer  # script-mode import

# inference.py's own module-level import already does
# sys.path.insert(0, <repo root>) as a side effect, so this resolves
# whichever way KyivanRestorer above was imported.
from src.data_pipeline.normalization import normalize_historical_text


@dataclass
class _BeamEntry:
    tokens: List[int]
    mask_positions: Set[int]  # indices in `tokens` still holding [-] or [#]
    pred_len: int  # number of [-] positions resolved to a real character
    unk_len: int  # number of [#] expand/stop decisions made
    logprob: float
    # Token-list snapshot after each resolved position, oldest first. Only
    # ever grows for entries that survive pruning (a plain list copy per
    # branch, not a saliency computation) -- the actual attention-map
    # extraction is deferred to saliency_for_result(), run once, only for
    # whichever single result the caller asks to explain.
    history: List[List[int]] = field(default_factory=list)


@dataclass
class BeamRestoration:
    """One ranked candidate restoration."""

    text: str
    logprob: float
    score: float  # exp(length-normalized logprob) -- for display/sorting
    history: List[List[int]] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"BeamRestoration(score={self.score:.4f}, text={self.text!r})"


class KyivanBeamRestorer(KyivanRestorer):
    """Adds Aeneas-style beam-search decoding on top of KyivanRestorer."""

    def _score(self, entry: _BeamEntry, a_penalty: float) -> float:
        # Length-normalized log-probability -- without this, beam search
        # trivially prefers the shortest completions (fewer factors in the
        # product of probabilities = higher raw logprob), regardless of
        # whether they're actually the most plausible text.
        return entry.logprob / (1 + entry.pred_len + entry.unk_len) ** a_penalty

    def restore_text_beam(
        self,
        text: str,
        beam_width: int = 30,
        a_penalty: float = 1.0,
        unk_restoration_max_len: int = 20,
        max_iterations: int = 100,
        skip_double_space: bool = True,
        top_n: Optional[int] = None,
    ) -> List[BeamRestoration]:
        """
        Restores a corrupted text sequence via non-sequential beam search,
        returning the top candidate restorations ranked by (length-normalized)
        probability, instead of a single greedy answer.

        Args:
            text: The corrupted input string, using bare `?` for one missing
                character and `#` for a lacuna of unknown length (e.g.,
                "а се покл#е" or "т?а").
            beam_width: Max hypotheses kept after each iteration (Aeneas
                default: 200 -- lowered here since this runs eagerly on
                whatever device the model is on, not batched/compiled Jax).
            a_penalty: Length-normalization exponent for ranking hypotheses.
                Higher favors shorter completions less/more depending on
                sign of the log-probabilities involved; 1.0 matches Aeneas's
                restore() default.
            unk_restoration_max_len: Combined cap on how many characters all
                `[#]` gaps in this text may expand into in total, relative
                to the input length (mirrors Aeneas's UNK_RESTORATION_MAX_LEN).
            max_iterations: Safety cap on beam-growth iterations. The search
                normally terminates on its own once every surviving
                hypothesis has no open `[-]`/`[#]` left.
            skip_double_space: Reject candidates that would place two spaces
                adjacent to each other (a word-boundary sanity check, not a
                real restoration).
            top_n: How many ranked results to return (defaults to beam_width).

        Returns:
            List[BeamRestoration]: candidates sorted best-first.
        """
        tokens = self._tokenize(text)
        num_unk_markers = tokens.count(self.unk_id)
        max_len = len(tokens) - num_unk_markers + unk_restoration_max_len

        initial_mask_positions = {
            i for i, t in enumerate(tokens) if t in (self.mask_id, self.unk_id)
        }
        if not initial_mask_positions:
            raise ValueError("At least one [-] or [#] must be present.")

        candidate_ids = (
            set(int(v) for v in self.char_vocab.values()) - self.forbidden_restore_ids
        )
        space_id = self.char_vocab[" "]

        beam: List[_BeamEntry] = [
            _BeamEntry(tokens, initial_mask_positions, 0, 0, 0.0, history=[tokens])
        ]
        finished: Dict[Tuple[int, ...], _BeamEntry] = {}

        iteration = 0
        while beam and iteration < max_iterations:
            iteration += 1

            max_batch_len = max(len(e.tokens) for e in beam)
            batch_input_ids = torch.full(
                (len(beam), max_batch_len), self.char_vocab["[PAD]"],
                dtype=torch.long, device=self.device,
            )
            batch_attention_mask = torch.zeros(
                (len(beam), max_batch_len), dtype=torch.long, device=self.device
            )
            for b, entry in enumerate(beam):
                slen = len(entry.tokens)
                batch_input_ids[b, :slen] = torch.tensor(entry.tokens, device=self.device)
                batch_attention_mask[b, :slen] = 1

            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch_input_ids, attention_mask=batch_attention_mask
                )
            # Move to CPU once here rather than doing a device sync on every
            # single-scalar .item() call below (thousands per iteration).
            log_probs_restore = torch.log_softmax(outputs.logits_restore, dim=-1).cpu()
            log_probs_unk = torch.log_softmax(outputs.logits_unk, dim=-1).cpu()

            next_round: List[_BeamEntry] = []

            for b, entry in enumerate(beam):
                for pos in sorted(entry.mask_positions):
                    tok = entry.tokens[pos]

                    if tok == self.unk_id:
                        # Branch 1: expand -- keep [#] open at `pos`, insert
                        # a fresh [-] right after it (matches Aeneas: new
                        # slot inserted at text_char_pos + 1, [#] itself
                        # never moves).
                        if len(entry.tokens) + 1 <= max_len:
                            new_tokens = (
                                entry.tokens[: pos + 1]
                                + [self.mask_id]
                                + entry.tokens[pos + 1 :]
                            )
                            new_mask_positions = {
                                (m + 1 if m > pos else m) for m in entry.mask_positions
                            }
                            new_mask_positions.add(pos + 1)
                            next_round.append(
                                _BeamEntry(
                                    new_tokens,
                                    new_mask_positions,
                                    entry.pred_len,
                                    entry.unk_len + 1,
                                    entry.logprob
                                    + float(log_probs_unk[b, pos, 1].item()),
                                    history=entry.history + [new_tokens],
                                )
                            )

                        # Branch 2: stop -- replace [#] in place with a
                        # regular [-] (same index, no length change).
                        new_tokens = list(entry.tokens)
                        new_tokens[pos] = self.mask_id
                        next_round.append(
                            _BeamEntry(
                                new_tokens,
                                set(entry.mask_positions),
                                entry.pred_len,
                                entry.unk_len + 1,
                                entry.logprob + float(log_probs_unk[b, pos, 0].item()),
                                history=entry.history + [new_tokens],
                            )
                        )
                        continue

                    # tok == self.mask_id: branch over every allowed character.
                    for char_id in candidate_ids:
                        if skip_double_space and char_id == space_id:
                            left = entry.tokens[pos - 1] if pos > 0 else None
                            right = (
                                entry.tokens[pos + 1]
                                if pos + 1 < len(entry.tokens)
                                else None
                            )
                            if left == space_id or right == space_id:
                                continue

                        new_tokens = list(entry.tokens)
                        new_tokens[pos] = char_id
                        new_logprob = entry.logprob + float(
                            log_probs_restore[b, pos, char_id].item()
                        )
                        new_mask_positions = entry.mask_positions - {pos}
                        new_entry = _BeamEntry(
                            new_tokens,
                            new_mask_positions,
                            entry.pred_len + 1,
                            entry.unk_len,
                            new_logprob,
                            history=entry.history + [new_tokens],
                        )

                        if not new_mask_positions:
                            key = tuple(new_tokens)
                            if key not in finished or finished[key].logprob < new_logprob:
                                finished[key] = new_entry
                        else:
                            next_round.append(new_entry)

            # Dedupe (keep best logprob per unique token sequence), then
            # keep only the top beam_width by length-normalized score.
            dedup: Dict[Tuple[int, ...], _BeamEntry] = {}
            for e in next_round:
                key = tuple(e.tokens)
                if key not in dedup or dedup[key].logprob < e.logprob:
                    dedup[key] = e
            beam = sorted(
                dedup.values(), key=lambda e: self._score(e, a_penalty), reverse=True
            )[:beam_width]

        results = sorted(
            finished.values(), key=lambda e: self._score(e, a_penalty), reverse=True
        )
        n = top_n if top_n is not None else beam_width
        return [
            BeamRestoration(
                text=self.decode(e.tokens),
                logprob=e.logprob,
                score=math.exp(self._score(e, a_penalty)),
                history=e.history,
            )
            for e in results[:n]
        ]

    def saliency_for_result(
        self, result: BeamRestoration, top_k: int = 5
    ) -> List[dict]:
        """
        Post-hoc interpretability for one beam-search result: replays its
        winning trajectory (result.history) and, for each step that resolved
        a `[-]` to a real character, runs one extra forward pass
        (output_attentions=True) to extract which context characters the
        model attended to most -- same mechanism as KyivanRestorer.
        restore_text's saliency maps, just computed after the fact for a
        single chosen path instead of inline during the search. `[#]`
        expand/stop steps aren't character predictions, so they're skipped.

        Only call this for a result you actually want to explain: it's
        O(len(history)) extra forward passes, but independent of
        beam_width -- it only ever replays the one path in result.history,
        never the discarded hypotheses.
        """
        steps = []
        for prev_tokens, curr_tokens in zip(result.history, result.history[1:]):
            if len(prev_tokens) != len(curr_tokens):
                continue  # a `[#]` expand step -- length changed, no char chosen

            changed_pos = next(
                (i for i, (p, c) in enumerate(zip(prev_tokens, curr_tokens)) if p != c),
                None,
            )
            if changed_pos is None or prev_tokens[changed_pos] != self.mask_id:
                continue  # `[#]` stop step (rewrites `#` -> `[-]`, no char yet)

            input_ids = torch.tensor(
                [curr_tokens], dtype=torch.long, device=self.device
            )
            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                )
            last_layer_attn = outputs.attentions[-1][0]
            mean_attn = last_layer_attn.mean(dim=0)
            focus_weights = mean_attn[changed_pos]

            top_weights, top_indices = torch.topk(
                focus_weights, k=min(top_k, focus_weights.shape[-1])
            )
            saliency = []
            for weight, token_idx in zip(top_weights, top_indices):
                tid = curr_tokens[token_idx.item()]
                if tid in self.special_ids:
                    continue
                saliency.append(
                    {
                        "char": self.id_to_char.get(tid, ""),
                        "position": int(token_idx.item()),
                        "weight": float(weight.item()),
                    }
                )

            steps.append(
                {
                    "position": changed_pos,
                    "predicted_char": self.id_to_char.get(curr_tokens[changed_pos], ""),
                    "text_so_far": self.decode(curr_tokens),
                    "saliency": saliency,
                }
            )

        return steps

    def _tokenize(self, text: str) -> List[int]:
        """Tokenizes user-facing input: bare `?` for one missing character,
        bare `#` for a lacuna of unknown length. No brackets, no `[SOS]` --
        `[SOS]`/`[-]`/`[#]` are the model's own internal vocab tokens, not
        something the caller should need to know about or type; `[SOS]` is
        always auto-prepended below regardless.

        `?` (not `-`) was chosen deliberately: `-` is a real character in
        this corpus (~4000 genuine hyphens, word-break marks), so a bare
        `-` would be ambiguous between "real hyphen" and "restore this".
        `?`/`#` never occur in the historical text itself -- in fact
        normalize_historical_text's own cleanup already treats a stray `?`
        as modern noise and deletes it (_DELETE_CHARS), which is exactly
        why markers are split out FIRST, before normalization ever sees the
        surrounding text: running normalize_historical_text on the whole
        string first would eat the very markers we're trying to recognize.
        """
        tokens: List[int] = []
        parts = re.split(r"([?#])", text)
        for part in parts:
            if not part:
                continue
            if part == "?":
                tokens.append(self.mask_id)
            elif part == "#":
                tokens.append(self.unk_id)
            else:
                for ch in normalize_historical_text(part):
                    tokens.append(
                        self.char_vocab.get(ch.lower(), self.char_vocab["[UNK]"])
                    )
        if not tokens or tokens[0] != self.sos_id:
            tokens.insert(0, self.sos_id)
        return tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kyivan Beam-Search Restorer")
    parser.add_argument(
        "--model_dir",
        default="outputs/kyivan_run5/final_model",
        help="Path to the trained model directory",
    )
    parser.add_argument(
        "--vocab",
        default="data/tokenizer/char_vocab.json",
        help="Path to the character vocabulary JSON",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Corrupted text sequence, using bare '?' for one missing "
        "character and '#' for a lacuna of unknown length. "
        "Example: 'а се покл#е'",
    )
    parser.add_argument(
        "--device", default="cpu", help="Compute device (e.g., cpu, cuda)"
    )
    parser.add_argument(
        "--beam_width", type=int, default=30, help="Max hypotheses kept per iteration"
    )
    parser.add_argument(
        "--top_n", type=int, default=5, help="How many ranked results to print"
    )
    parser.add_argument(
        "--a_penalty", type=float, default=1.0, help="Length-normalization exponent"
    )
    parser.add_argument(
        "--unk_restoration_max_len",
        type=int,
        default=20,
        help="Combined cap on how many characters all '#' gaps may expand into",
    )
    parser.add_argument(
        "--saliency",
        action="store_true",
        help="Also print the attention saliency map for the top result",
    )

    args = parser.parse_args()

    restorer = KyivanBeamRestorer(
        model_dir=args.model_dir, char_vocab_path=args.vocab, device=args.device
    )
    results = restorer.restore_text_beam(
        args.text,
        beam_width=args.beam_width,
        a_penalty=args.a_penalty,
        unk_restoration_max_len=args.unk_restoration_max_len,
        top_n=args.top_n,
    )

    print(f"\n--- TOP {len(results)} RESTORATIONS ---")
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r.score:.4f}] {r.text}")

    if args.saliency and results:
        print("\n--- SALIENCY FOR TOP RESULT ---")
        for step in restorer.saliency_for_result(results[0]):
            print(
                f"\nPosition {step['position']}: predicted '{step['predicted_char']}' "
                f"| text so far: {step['text_so_far']}"
            )
            for s in step["saliency"]:
                print(
                    f"    looked at '{s['char']}' (pos {s['position']}) "
                    f"- {s['weight'] * 100:.1f}%"
                )
