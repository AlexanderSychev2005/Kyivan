"""
Kyivan Model Configuration Module.

This module defines the configuration class for the Kyivan model, inheriting
from HuggingFace's BertConfig. It encapsulates all operational hyper-parameters,
ensuring seamless integration with standard HuggingFace model-saving, serialization,
and loading mechanisms via the Hub.

Key Adaptations for Aeneas:
1. `position_embedding_type = "none"`: Explicitly deactivates absolute positional
   embeddings in the base BERT framework, clearing the way for the custom Rotary
   Position Embedding (RoPE) system to handle sequence coordinates internally.
2. Character-Level Focus: Introduces dedicated configurations optimized for a highly compact
   vocabulary (`vocab_char_size`) rather than expansive traditional sub-word tokens.
"""

from transformers import BertConfig


class KyivanConfig(BertConfig):
    model_type = "kyivan"

    def __init__(
        self,
        vocab_char_size: int = 256,
        hidden_size: int = 512,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        intermediate_size: int = 2048,
        max_position_embeddings: int = 2048,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        **kwargs,
    ) -> None:
        """
        Initializes the Kyivan model configuration class.

        Args:
            vocab_char_size (int): The total number of unique characters in the vocabulary.
            hidden_size (int): Dimensionality of the encoder layers and pooler layer.
            num_hidden_layers (int): Number of hidden layers in the Transformer encoder torso.
            num_attention_heads (int): Number of attention heads for each attention layer.
            intermediate_size (int): Dimensionality of the intermediate (feed-forward) layer.
            max_position_embeddings (int): Maximum sequence length that the model can handle.
            hidden_dropout_prob (float): The dropout probability for all fully connected layers.
            attention_probs_dropout_prob (float): The dropout ratio for the attention probabilities.
            **kwargs: Arbitrary keyword arguments forwarded directly to the parent BertConfig class.

        Returns:
            None
        """
        # Hardcode position embedding type to bypass traditional absolute positions
        kwargs["position_embedding_type"] = "none"

        super().__init__(
            vocab_size=vocab_char_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            **kwargs,
        )

        # Keep tracking character size as an explicit parameter attribute
        self.vocab_char_size = vocab_char_size
