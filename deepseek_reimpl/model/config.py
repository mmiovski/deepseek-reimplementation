"""Model configuration objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GPTConfig:
    """Configuration for decoder-only GPT-style models.

    The default configuration remains the dense GPT baseline. MLA fields are only
    active when attention_type == "mla".
    """

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
    attention_type: str = "dense"
    tie_embeddings: bool = True
    mla_kv_latent_dim: int | None = None
    mla_q_rope_dim: int | None = None
    n_routed_experts: int | None = None
    n_shared_experts: int = 0
    moe_top_k: int | None = None
    moe_expert_d_ff: int | None = None
    moe_router_score: str = "softmax"
    moe_normalize_top_k_weights: bool = True
    moe_aux_loss_weight: float = 0.0
    moe_drop_tokens: bool = False

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

        valid_ffn_types = {"swiglu", "gelu_mlp", "moe"}
        if self.ffn_type not in valid_ffn_types:
            msg = f"ffn_type must be one of {sorted(valid_ffn_types)}"
            raise ValueError(msg)

        valid_attention_types = {"dense", "mla"}
        if self.attention_type not in valid_attention_types:
            msg = f"attention_type must be one of {sorted(valid_attention_types)}"
            raise ValueError(msg)

        if self.attention_type == "mla":
            self._validate_mla_config()

        if self.ffn_type == "moe":
            self._validate_moe_config()

    def _validate_mla_config(self) -> None:
        """Validate MLA-specific architecture fields."""
        if self.positional_encoding != "rope":
            msg = "MLA attention currently requires positional_encoding='rope'"
            raise ValueError(msg)

        if self.mla_kv_latent_dim is None:
            msg = "mla_kv_latent_dim must be set when attention_type='mla'"
            raise ValueError(msg)

        if self.mla_q_rope_dim is None:
            msg = "mla_q_rope_dim must be set when attention_type='mla'"
            raise ValueError(msg)

        if self.mla_kv_latent_dim <= 0:
            msg = "mla_kv_latent_dim must be positive"
            raise ValueError(msg)

        if self.mla_kv_latent_dim >= self.d_model:
            msg = "mla_kv_latent_dim must be smaller than d_model"
            raise ValueError(msg)

        if self.mla_q_rope_dim <= 0:
            msg = "mla_q_rope_dim must be positive"
            raise ValueError(msg)

        if self.mla_q_rope_dim >= self.head_dim:
            msg = "mla_q_rope_dim must be smaller than head_dim"
            raise ValueError(msg)

        if self.mla_q_rope_dim % 2 != 0:
            msg = "mla_q_rope_dim must be even for rotary embeddings"
            raise ValueError(msg)

    def _validate_moe_config(self) -> None:
        """Validate MoE-specific architecture fields."""
        if self.n_routed_experts is None:
            msg = "n_routed_experts must be set when ffn_type='moe'"
            raise ValueError(msg)

        if self.moe_top_k is None:
            msg = "moe_top_k must be set when ffn_type='moe'"
            raise ValueError(msg)

        if self.moe_expert_d_ff is None:
            msg = "moe_expert_d_ff must be set when ffn_type='moe'"
            raise ValueError(msg)

        if self.n_routed_experts <= 0:
            msg = "n_routed_experts must be positive"
            raise ValueError(msg)

        if self.n_shared_experts < 0:
            msg = "n_shared_experts must be nonnegative"
            raise ValueError(msg)

        if self.moe_top_k <= 0:
            msg = "moe_top_k must be positive"
            raise ValueError(msg)

        if self.moe_top_k > self.n_routed_experts:
            msg = "moe_top_k must be less than or equal to n_routed_experts"
            raise ValueError(msg)

        if self.moe_expert_d_ff <= 0:
            msg = "moe_expert_d_ff must be positive"
            raise ValueError(msg)

        if self.moe_router_score != "softmax":
            msg = "moe_router_score currently supports only 'softmax'"
            raise ValueError(msg)

        if self.moe_aux_loss_weight < 0.0:
            msg = "moe_aux_loss_weight must be nonnegative"
            raise ValueError(msg)

        if self.moe_drop_tokens:
            msg = "moe_drop_tokens is reserved for a later capacity-limited MoE phase"
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
