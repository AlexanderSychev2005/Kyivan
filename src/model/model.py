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
from config import KyivanConfig
from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEncoder, BertSelfAttention
from transformers.utils.generic import ModelOutput


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Splits the last dimension of the tensor in half and applies a rotary transformation.
    Transformation: [x1, x2] -> [-x2, x1].

    Args:
        x (torch.Tensor): Input tensor of shape (..., dim).

    Returns:
        torch.Tensor: The transformed tensor with rotated hidden dimensions.
    """
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Position Embeddings (RoPE) to the query and key tensors.

    Args:
        q (torch.Tensor): The query tensor.
        k (torch.Tensor): The key tensor.
        cos (torch.Tensor): The cached cosine frequencies tensor.
        sin (torch.Tensor): The cached sine frequencies tensor.
        position_ids (torch.Tensor): Tensor containing the sequential position indices.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the rotated query and key tensors.
    """
    gather_indices = position_ids[:, None, :, None]
    gather_indices = gather_indices.expand(-1, cos.shape[1], -1, cos.shape[3])

    # Gather the appropriate cos/sin frequencies for the current sequence positions
    cos = torch.gather(cos.expand(position_ids.shape[0], -1, -1, -1), 2, gather_indices)
    sin = torch.gather(sin.expand(position_ids.shape[0], -1, -1, -1), 2, gather_indices)

    # Apply the rotary transformation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        # Calculate inverse frequencies for the rotational matrix
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embeddings

        # Pre-compute and cache cosine and sine values for the maximum expected sequence length
        t = torch.arange(
            self.max_seq_len_cached,
            device=self.inv_freq.device,
            dtype=self.inv_freq.dtype,
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :], persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :], persistent=False
        )

    def forward(
        self, x: torch.Tensor, seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the cached sine and cosine embeddings for the current sequence length.
        Dynamically computes them if the sequence exceeds the cached maximum.

        Args:
            x (torch.Tensor): Input tensor used for device and dtype reference.
            seq_len (Optional[int]): The current sequence length.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sliced cosine and sine tensors matching the sequence length.
        """
        if seq_len > self.max_seq_len_cached:
            # Dynamic extension if the sequence length exceeds the cached maximum
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...],
        )


# CUSTOM ATTENTION WITH RoPE
class BertSelfAttentionWithRoPE(BertSelfAttention):
    def __init__(self, config: KyivanConfig):
        super().__init__(config)
        self.rotary_emb = RotaryEmbedding(
            dim=self.attention_head_size,
            max_position_embeddings=config.max_position_embeddings,
        )

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
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

        Returns:
            Tuple[torch.Tensor, ...]: Tuple containing the context layer and optionally the attention probabilities.
        """
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

        cos, sin = self.rotary_emb(value_layer, seq_len=seq_length)
        query_layer, key_layer = apply_rotary_pos_emb(
            query_layer, key_layer, cos, sin, position_ids
        )

        # Standard Dot-Product Attention mechanism follows
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

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

        ext_mask = self.get_extended_attention_mask(
            attention_mask, input_ids.shape
        )

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
        logits_restore = x_res @ self.char_embeddings.weight.T + self.restore_bias
        logits_unk = self.unk_head(seq)

        # Process global heads (applied exclusively to the [SOS] token at index 0)
        sos_vectors = seq[:, 0, :]
        logits_date = self.date_head(sos_vectors)
        logits_region = self.region_head(sos_vectors)

        return KyivanOutput(
            logits_restore=logits_restore,
            logits_unk=logits_unk,
            logits_date=logits_date,
            logits_region=logits_region,
            attentions=enc_out.attentions,
        )
