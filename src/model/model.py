"""
Kyivan Model Architecture.

This module defines a custom Transformer model based on the non-autoregressive
architecture, adapted for ancient Slavic texts.
It overrides the standard HuggingFace BERT encoder to introduce the following innovations:

1. Pure Character-Level Representation: Better handle highly inflected languages
and physically fragmented text (e.g., birch bark).
2. Rotary Position Embeddings (RoPE): Replaces absolute positional embeddings.
   RoPE rotates query and key vectors in the self-attention mechanism, allowing the model
   to natively understand relative distances between characters across variable-length gaps.
3. Multi-Task Learning (4 Heads):
   - Restore Head (Local): Standard Masked Language Modeling (MLM) to predict missing characters `[-]`.
   - Unk Head (Local): A binary classifier predicting whether a gap token `[#]` should be expanded.
   - Date Head (Global): Predicts the document's chronological distribution (reads from `[SOS]`).
   - Region Head (Global): Classifies the dialect/region of the document (reads from `[SOS]`).
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    from .config import KyivanConfig  # package-style import (e.g. inference.py)
except ImportError:
    from config import KyivanConfig  # script-mode import (e.g. train.py)
from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEncoder, BertSelfAttention
from transformers.utils.generic import ModelOutput


from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb
from transformers.models.llama.configuration_llama import LlamaConfig


# CUSTOM ATTENTION WITH RoPE
class BertSelfAttentionWithRoPE(BertSelfAttention):
    def __init__(self, config: KyivanConfig):
        super().__init__(config)
        # Some transformers versions' BertSelfAttention.__init__ doesn't keep
        # a `self.config` reference -- set it explicitly so forward() below
        # doesn't depend on that varying by version.
        self.config = config
        llama_config = LlamaConfig(
            hidden_size=self.attention_head_size * self.num_attention_heads,
            num_attention_heads=self.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
        )
        self.rotary_emb = LlamaRotaryEmbedding(
            config=llama_config,
        )

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for the custom Self-Attention layer injecting RoPE.

        Args:
            hidden_states (torch.Tensor): Input hidden states.
            attention_mask (Optional[torch.Tensor]): Mask to avoid performing attention on padding token indices.
            head_mask (Optional[torch.Tensor]): Mask to nullify selected heads of the self-attention modules.
            encoder_hidden_states (Optional[torch.Tensor]): Used in cross-attention (not applicable here).
            encoder_attention_mask (Optional[torch.Tensor]): Used in cross-attention (not applicable here).
            past_key_value (Optional[Tuple[torch.Tensor]]): Cached key and value states.
            output_attentions (bool): Whether or not to return the attentions tensors.
        """
        output_attentions = getattr(
            self.config, "output_attentions", output_attentions
        ) or kwargs.get("output_attentions", output_attentions)
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Apply RoPE (Rotary Position Embeddings) to Query and Key matrices
        seq_length = hidden_states.shape[1]
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=hidden_states.device
        )
        position_ids = position_ids.unsqueeze(0).expand(hidden_states.shape[0], -1)

        cos, sin = self.rotary_emb(value_layer, position_ids)
        query_layer, key_layer = apply_rotary_pos_emb(
            query_layer, key_layer, cos, sin
        )

        # Standard Dot-Product Attention mechanism follows
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        self._last_attention = attention_probs

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer, (attention_probs if output_attentions else None)


@dataclass
class KyivanOutput(ModelOutput):
    logits_restore: torch.FloatTensor = None
    logits_unk: torch.FloatTensor = None
    logits_date: Optional[torch.FloatTensor] = None
    logits_region: Optional[torch.FloatTensor] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# MAIN MODEL ARCHITECTURE
class Kyivan(BertPreTrainedModel):
    config_class = KyivanConfig

    def __init__(
        self, config: KyivanConfig, num_date_bins: int = 20, num_regions: int = 4
    ):
        super().__init__(config)
        # Disable absolute positional embeddings inherited from BERT
        config.position_embedding_type = "none"

        # 1. Pure Character Embeddings Layer (No Word IDs)
        self.char_embeddings = nn.Embedding(
            config.vocab_char_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.emb_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.emb_dropout = nn.Dropout(config.hidden_dropout_prob)

        # 2. Transformer Encoder with RoPE Injection
        self.encoder = BertEncoder(config)
        for layer in self.encoder.layer:
            # Overwrite the standard self-attention module with our custom RoPE implementation
            layer.attention.self = BertSelfAttentionWithRoPE(config)

        # 3. Multi-Task Output Heads
        # Local Head A: Restores missing characters (tied to char_embeddings weights)
        self.restore_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.restore_act = nn.GELU()
        self.restore_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.restore_bias = nn.Parameter(torch.zeros(config.vocab_char_size))

        # Local Head B: Predicts gap extension for the `[#]` token
        self.unk_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 2),
        )

        # Global Head A: Predicts historical date distribution (Reads from [SOS])
        self.date_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, num_date_bins),
        )

        # Global Head B: Classifies dialect/region (Reads from [SOS])
        self.region_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, num_regions),
        )
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        """
        Required by HuggingFace to support `resize_token_embeddings`.

        Returns:
            nn.Embedding: The embedding layer.
        """
        return self.char_embeddings

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """
        Required by HuggingFace to support `resize_token_embeddings`.

        Args:
            value (nn.Embedding): The new embedding layer.
        """
        self.char_embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        **kwargs,
    ) -> KyivanOutput:
        """
        Forward pass for the multi-task Transformer.

        Args:
            input_ids (Optional[torch.Tensor]): Indices of input sequence tokens in the vocabulary.
            attention_mask (Optional[torch.Tensor]): Mask to avoid performing attention on padding token indices.
            output_attentions (Optional[bool]): Whether or not to return the attentions tensors.
            **kwargs: Additional keyword arguments.

        Returns:
            KyivanOutput: A custom dataclass containing 4 sets of logits and optional attention weights.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )

        # Fast character embedding generation
        x = self.char_embeddings(input_ids)
        x = self.emb_norm(x)
        x = self.emb_dropout(x)

        ext_mask = self.get_extended_attention_mask(attention_mask, input_ids.shape)

        # Pass through the RoPE-enabled encoder torso
        enc_out = self.encoder(
            x,
            attention_mask=ext_mask,
            output_attentions=output_attentions,
            return_dict=True,
        )
        seq = enc_out.last_hidden_state  # Shape: [Batch, Seq_Len, Hidden_Dim]

        # Process local sequence heads (applied to all tokens)
        x_res = self.restore_norm(self.restore_act(self.restore_dense(seq)))
        # Tied-embedding logits: the embedding matrix isn't scaled for use as
        # an output projection, so raw hidden @ E^T grows with hidden_size,
        # over-sharpening the softmax. Rescale by sqrt(hidden_size) (the bias
        # is a per-class shift, unaffected by this variance growth, so it's
        # added after).
        logits_restore = (x_res @ self.char_embeddings.weight.T) / math.sqrt(
            self.char_embeddings.weight.shape[-1]
        ) + self.restore_bias
        logits_unk = self.unk_head(seq)

        # Process global heads (applied exclusively to the [SOS] token at index 0)
        sos_vectors = seq[:, 0, :]
        logits_date = self.date_head(sos_vectors)
        logits_region = self.region_head(sos_vectors)

        # Manually extract the attention from the last layer to bypass HF tuple dropping
        last_attn = self.encoder.layer[-1].attention.self._last_attention
        extracted_attentions = (last_attn,) if last_attn is not None else None

        return KyivanOutput(
            logits_restore=logits_restore,
            logits_unk=logits_unk,
            logits_date=logits_date,
            logits_region=logits_region,
            attentions=extracted_attentions,
        )
