"""
Kyivan Confidence-Based Restorer (Inference Module).
Implements the non-autoregressive decoding algorithm using a Confidence-Based Search approach.
"""

import argparse
import json
import re
from typing import List

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from src.model.config import KyivanConfig
from src.model.model import Kyivan
from src.model.vocab_categories import is_maskable_char
from src.data_pipeline.normalization import normalize_historical_text


class KyivanRestorer:
    """
    Handles the initialization of the Kyivan model and executes the iterative,
    confidence-based restoration algorithm on corrupted historical texts.
    """

    def __init__(
        self, model_dir: str, char_vocab_path: str, device: str = "cpu"
    ) -> None:
        """
        Initializes the restorer by loading the tokenizer vocabulary and the pre-trained model.

        Args:
            model_dir (str): Path to the directory containing the model weights and config.
            char_vocab_path (str): Path to the character vocabulary JSON file.
            device (str): Computation device to load the model onto ('cpu', 'cuda', or 'mps').

        Returns:
            None
        """
        self.device = torch.device(device)

        # Load the character-level vocabulary
        with open(char_vocab_path, "r", encoding="utf-8") as f:
            self.char_vocab = json.load(f)

        self.id_to_char = {int(v): k for k, v in self.char_vocab.items()}

        self.mask_id = self.char_vocab["[-]"]
        self.unk_id = self.char_vocab["[#]"]
        self.sos_id = self.char_vocab["[SOS]"]

        # Identify special tokens to prevent the model from predicting them during restoration
        self.special_ids = {
            v
            for k, v in self.char_vocab.items()
            if k.startswith("[") and k.endswith("]")
        }

        # Punctuation/digits are never valid restoration predictions either --
        # only letters and spaces may fill a `[-]` mask. Shares its policy
        # (vocab_categories.MASKABLE_CATEGORIES) with collator_v2's
        # maskable_ids and train.py's ALLOWED_PRED_IDS.
        self.forbidden_restore_ids = set(self.special_ids) | {
            v
            for k, v in self.char_vocab.items()
            if len(k) == 1 and not is_maskable_char(k)
        }

        # Load configuration and model weights
        config = KyivanConfig.from_pretrained(model_dir)
        self.model = Kyivan.from_pretrained(model_dir, config=config).to(self.device)
        self.model.eval()

    def decode(self, token_ids: List[int]) -> str:
        """
        Converts a list of token IDs back into a human-readable string, ignoring special tokens.

        Args:
            token_ids (List[int]): A list of token integer IDs.

        Returns:
            str: The decoded string.
        """
        return "".join(
            self.id_to_char.get(tid, "")
            for tid in token_ids
            if tid not in self.special_ids
        )

    def restore_text(self, text: str, max_steps: int = 50) -> str:
        """
        Iteratively restores a corrupted text sequence using confidence-based search and
        extracts attention weights for interpretability.

        Args:
            text (str): The corrupted input string, using bare `?` for one
                missing character and `#` for a lacuna of unknown length
                (e.g., "а се покл#е"). No brackets, no `[SOS]` -- those are
                the model's internal vocab tokens, not user-facing syntax;
                `[SOS]` is always prepended automatically.
            max_steps (int): A safety limit for the maximum number of decoding iterations.

        Returns:
            str: The fully restored text sequence without special tokens.
        """
        print(f"\n--- ORIGINAL TEXT: {text} ---")

        # Tokenize: split on `?`/`#` FIRST, then normalize only the
        # plain-text segments between them. normalize_historical_text is
        # designed for raw historical text, not this input syntax -- run on
        # the whole string first (as this used to do), it eats the markers:
        # `?` is treated as modern-punctuation noise and deleted
        # (_DELETE_CHARS), so a marker-then-normalize order loses it before
        # it can ever be recognized.
        tokens = []
        parts = re.split(r"([?#])", text)

        for part in parts:
            if not part:
                continue
            if part == "?":
                tokens.append(self.mask_id)
            elif part == "#":
                tokens.append(self.unk_id)
            else:
                # Vocab is lowercase-only (see build_char_tokenizer.py) --
                # fold here too, or any uppercase input character would
                # miss the vocab and silently become [UNK].
                for ch in normalize_historical_text(part):
                    tokens.append(
                        self.char_vocab.get(ch.lower(), self.char_vocab["[UNK]"])
                    )

        print(f"\n--- TOKENIZED (pre-[SOS]) LENGTH: {len(tokens)} ---")

        # Ensure the global [SOS] token is present for multi-task heads
        if not tokens or tokens[0] != self.sos_id:
            tokens.insert(0, self.sos_id)

        step = 0
        while step < max_steps:
            step += 1
            input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
            attention_mask = torch.ones_like(input_ids)

            with torch.no_grad():
                # Explicitly request attention weights for saliency map extraction
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                )

            # --- STEP 1: LACUNA EXPANSION ([#]) ---
            # Locate the first unknown-length lacuna and query the Unk Head
            if self.unk_id in tokens:
                unk_idx = tokens.index(self.unk_id)

                # Fetch length logits (0 = stop expansion, 1 = expand by one character)
                unk_logits = outputs.logits_unk[0, unk_idx]
                pred_action = torch.argmax(unk_logits).item()

                if pred_action == 1:
                    print(
                        f"Step {step} (Unk Head): Model decided to EXPAND the lacuna [#]"
                    )
                    # Insert an empty mask [-] just before the [#] token
                    tokens.insert(unk_idx, self.mask_id)
                else:
                    print(
                        f"Step {step} (Unk Head): Model HALTED the expansion of lacuna [#]"
                    )
                    # Terminate expansion by converting the [#] itself into a final [-] mask
                    tokens[unk_idx] = self.mask_id

                # Restart the loop with the updated sequence length
                continue

            # --- STEP 2: CONFIDENCE-BASED RESTORATION ([-]) ---
            # If no masks remain, the restoration is complete
            if self.mask_id not in tokens:
                break

            best_pos = -1
            best_prob = -1.0
            best_char_id = -1

            # Iterate over all currently empty masks to find the one with the highest confidence
            logits_restore = outputs.logits_restore[0]

            for i, tok_id in enumerate(tokens):
                if tok_id == self.mask_id:
                    # Clone logits to avoid modifying the original tensor
                    mask_logits = logits_restore[i].clone()

                    # Heavily penalize special tokens and punctuation to prevent the model
                    # from generating them mid-word as a restoration guess
                    for sp_id in self.forbidden_restore_ids:
                        mask_logits[sp_id] = -float("inf")

                    probs = torch.softmax(mask_logits, dim=-1)
                    max_prob, char_id = torch.max(probs, dim=-1)

                    if max_prob.item() > best_prob:
                        best_prob = max_prob.item()
                        best_pos = i
                        best_char_id = char_id.item()

            # Fill the single most confident mask
            if best_pos != -1:
                predicted_char = self.id_to_char[best_char_id]

                # --- STEP 3: SALIENCY MAP EXTRACTION ---
                # Retrieve the attention weights from the very last encoder layer
                # Shape: [Batch_size, Num_Heads, Seq_Len, Seq_Len]
                last_layer_attn = outputs.attentions[-1][0]

                # Average the attention weights across all heads to get a unified view
                # Resulting Shape: [Seq_Len, Seq_Len]
                mean_attn = last_layer_attn.mean(dim=0)

                # Extract the attention distribution specifically for the position being filled
                focus_weights = mean_attn[best_pos]

                print(
                    f"\n--- Saliency Map for predicting '{predicted_char}' at position {best_pos} ---"
                )

                # Identify the top 5 context characters the model focused on
                top_weights, top_indices = torch.topk(focus_weights, k=5)
                saliency_data_for_frontend = []

                for weight, token_idx in zip(top_weights, top_indices):
                    looked_at_id = tokens[token_idx.item()]
                    looked_at_char = self.id_to_char.get(looked_at_id, "")
                    # Compare the id against special_ids (a set of ids), not
                    # the decoded string against it -- a str is never `in` a
                    # set of ints, so this filter previously never excluded
                    # special tokens from the saliency output.
                    if looked_at_char and looked_at_id not in self.special_ids:
                        w_percent = weight.item() * 100
                        print(
                            f"Looked at: '{looked_at_char}' (Position: {token_idx.item()}) - Weight: {w_percent:.1f}%"
                        )

                        # Compile data intended for frontend visualization (e.g., Heatmap)
                        saliency_data_for_frontend.append(
                            {
                                "char": looked_at_char,
                                "position": token_idx.item(),
                                "weight": float(weight.item()),
                            }
                        )
                print("-" * 50)

                # Apply the prediction to the sequence
                tokens[best_pos] = best_char_id
                print(
                    f"Step {step} (Restore): Selected position {best_pos}. Character: '{predicted_char}' (Confidence: {best_prob * 100:.1f}%)"
                )

                # Print intermediate context state
                current_text = "".join(
                    self.id_to_char.get(t, "") for t in tokens if t != self.sos_id
                )
                print(f"Current sequence: {current_text}")

        print("\n--- FINAL RESTORED RESULT ---")
        final_str = "".join(
            self.id_to_char.get(t, "") for t in tokens if t not in self.special_ids
        )
        print(final_str)
        return final_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kyivan Inference Script")
    parser.add_argument(
        "--model_dir",
        default="novgorodets/artifacts/training_output/final_model",
        help="Path to the trained model directory",
    )
    parser.add_argument(
        "--vocab",
        default="novgorodets/artifacts/char_tokenizer/char_vocab.json",
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

    args = parser.parse_args()

    restorer = KyivanRestorer(
        model_dir=args.model_dir, char_vocab_path=args.vocab, device=args.device
    )
    restorer.restore_text(args.text)
