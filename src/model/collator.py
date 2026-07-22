"""
Kyivan Physical Degradation Data Collator.

This module provides a specialized data collator designed to train the model on
historical manuscript restoration. Instead of standard random masking, this collator
simulates real-world physical damage (e.g., torn edges of birch bark or parchment)
and implements the "Sequence Shrinking" technique from the Aeneas architecture.

Key Features:
1. Edge Degradation: Randomly simulates physical tearing at the left or right edges
   of the document, ignoring protected context tags.
2. Sequence Shrinking ([#]): Replaces continuous spans of missing characters with a
   single `[#]` token. This compresses the sequence length dynamically, teaching the
   model to handle lacunae of unknown lengths and reducing memory footprint.
3. Standard Masking ([-]): Uses the standard `[-]` token for single-character damage.
4. Unk Head Labels: Generates binary targets (`unk_labels`) to train the model's
   Unk Head to predict whether a `[#]` token represents a single missing character
   or a longer span that needs to be expanded during inference.
5. Metadata Handling: Safely routes regional dialects and date distributions if provided.
"""

import random
from typing import Any, Dict, List

import numpy as np
import torch


class KyivanPhysicalCollator:
    def __init__(
        self,
        char_vocab: Dict[str, int],
        mlm_prob: float = 0.15,
        span_mask_ratio: float = 0.2,
        span_geometric_p: float = 0.2,
        edge_prob: float = 0.1,
        date_bins: int = 20,
    ) -> None:
        """
        Initializes the physical degradation collator.

        Args:
            char_vocab (Dict[str, int]): The character-level tokenizer vocabulary.
            mlm_prob (float): The overall probability of masking a character.
            span_mask_ratio (float): The proportion of masks that should become span lacunae `[#]`.
            span_geometric_p (float): Parameter for the geometric distribution determining span length.
            edge_prob (float): Probability of simulating a physical tear at the document's edges.
            date_bins (int): Width of the date-distribution placeholder used for
                examples with no date label (must match the model's num_date_bins).

        Returns:
            None
        """
        self.vocab = char_vocab
        self.pad_id = char_vocab["[PAD]"]
        self.mask_id = char_vocab["[-]"]  # Standard single-character mask
        self.unk_id = char_vocab["[#]"]  # Lacuna of unknown length (sequence shrinking)
        self.sos_id = char_vocab["[SOS]"]  # Global sequence token

        self.mlm_prob = mlm_prob
        self.span_mask_ratio = span_mask_ratio
        self.span_geometric_p = span_geometric_p
        self.edge_prob = edge_prob
        self.date_bins = date_bins

        # Special tokens that must remain intact (prevents breaking dialect tags or [SOS])
        self.special_ids = {
            v for k, v in char_vocab.items() if k.startswith("[") and k.endswith("]")
        }

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Processes a batch of sequences, applying physical damage and span masking.

        Args:
            features (List[Dict[str, Any]]): A list of dictionaries, where each dict
                                             represents a single sequence's features.

        Returns:
            Dict[str, torch.Tensor]: A batched dictionary containing padded input_ids,
                                     attention masks, and necessary loss labels.
        """
        b_input_ids = []
        b_labels_res = []
        b_labels_unk = []

        for f in features:
            tokens = f["input_ids"]

            # 1. Enforce [SOS] token at the beginning for global multi-task heads
            if tokens[0] != self.sos_id:
                tokens = [self.sos_id] + tokens

            # EDGE DEGRADATION SIMULATION
            edge_mask_start = -1
            edge_mask_end = -1
            valid_len = len(tokens)

            # Trigger a physical document tear
            if random.random() < self.edge_prob and valid_len > 10:
                edge_len = random.randint(2, 5)
                if random.random() < 0.5:
                    # Left edge tear: skip [SOS] and any initial context tags
                    start_idx = 1
                    while (
                        start_idx < valid_len and tokens[start_idx] in self.special_ids
                    ):
                        start_idx += 1
                    edge_mask_start = start_idx
                    edge_mask_end = min(start_idx + edge_len, valid_len)
                else:
                    # Right edge tear
                    edge_mask_start = max(1, valid_len - edge_len)
                    edge_mask_end = valid_len

            new_tokens = []
            new_labels_res = []
            new_labels_unk = []

            i = 0
            while i < len(tokens):
                tok = tokens[i]

                # Skip and preserve special tokens
                if tok in self.special_ids:
                    new_tokens.append(tok)
                    new_labels_res.append(-100)
                    new_labels_unk.append(-100)
                    i += 1
                    continue

                # Check if the current position falls within the torn edge zone
                is_edge_damage = edge_mask_start <= i < edge_mask_end

                # Decide whether to mask the character (random MLM or physical edge tear)
                if is_edge_damage or random.random() < self.mlm_prob:
                    # Determine lacuna type: single mask `[-]` or long span `[#]`.
                    # Edge tears always default to long spans to simulate missing chunks.
                    is_span = is_edge_damage or (random.random() < self.span_mask_ratio)

                    if is_span:
                        # --- AENEAS LACUNA SPAN [#] ---
                        if is_edge_damage:
                            span_len = edge_mask_end - i
                        else:
                            span_len = np.random.geometric(self.span_geometric_p)
                            span_len = min(span_len, len(tokens) - i)

                        # Prevent the span from accidentally consuming special tokens
                        for j in range(1, span_len):
                            if tokens[i + j] in self.special_ids:
                                span_len = j
                                break

                        new_tokens.append(self.unk_id)

                        # Target is the first erased character of the span
                        new_labels_res.append(tokens[i])

                        # Target label for the Unk Head (1 if span length > 1, else 0)
                        new_labels_unk.append(1 if span_len > 1 else 0)

                        # SEQUENCE SHRINKING: skip the entire erased span to save memory
                        i += span_len
                    else:
                        # --- STANDARD MASK [-] ---
                        new_tokens.append(self.mask_id)
                        new_labels_res.append(tokens[i])
                        new_labels_unk.append(-100)
                        i += 1
                else:
                    new_tokens.append(tok)
                    new_labels_res.append(-100)
                    new_labels_unk.append(-100)
                    i += 1

            b_input_ids.append(new_tokens)
            b_labels_res.append(new_labels_res)
            b_labels_unk.append(new_labels_unk)

        # Dynamic padding to the maximum sequence length in the current batch
        max_len = max(len(seq) for seq in b_input_ids)

        t_input_ids = torch.full(
            (len(features), max_len), self.pad_id, dtype=torch.long
        )
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

        # Handle Metadata (Dates and Regions) gracefully if they exist in the dataset.
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
