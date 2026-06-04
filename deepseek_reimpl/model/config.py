"""Model configuration objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GPTConfig:
    """Configuration for a dense decoder-only GPT baseline."""

    vocab_size: int
    block_size: int
    n_layers: int
    n_heads: int
    d_model: int
    d_ff: int
    dropout: float = 0.0
    norm_type: str = "rmsnorm"
    positional_encoding: str = "rope"
    ffn_type: str = "swiglu"
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        """Validate architectural configuration values."""
        positive_int_fields = {
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_model": self.d_model,
            "d_ff": self.d_ff,
        }

        for field_name, field_value in positive_int_fields.items():
            if field_value <= 0:
                msg = f"{field_name} must be positive"
                raise ValueError(msg)

        if self.d_model % self.n_heads != 0:
            msg = "d_model must be divisible by n_heads"
            raise ValueError(msg)

        if not 0.0 <= self.dropout < 1.0:
            msg = "dropout must be in the interval [0, 1)"
            raise ValueError(msg)

        valid_norm_types = {"rmsnorm", "layernorm"}
        if self.norm_type not in valid_norm_types:
            msg = f"norm_type must be one of {sorted(valid_norm_types)}"
            raise ValueError(msg)

        valid_positional_encodings = {"rope", "learned"}
        if self.positional_encoding not in valid_positional_encodings:
            msg = "positional_encoding must be one of " f"{sorted(valid_positional_encodings)}"
            raise ValueError(msg)

        valid_ffn_types = {"swiglu", "gelu_mlp"}
        if self.ffn_type not in valid_ffn_types:
            msg = f"ffn_type must be one of {sorted(valid_ffn_types)}"
            raise ValueError(msg)

    @property
    def head_dim(self) -> int:
        """Per-head attention dimension."""
        return self.d_model // self.n_heads

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> GPTConfig:
        """Build GPTConfig from a loaded YAML model config dictionary."""
        model_config = dict(config.get("model", config))
        model_config.pop("name", None)
        return cls(**model_config)
