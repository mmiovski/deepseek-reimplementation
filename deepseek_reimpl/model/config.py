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
    mla_qk_nope_head_dim: int | None = None
    mla_v_head_dim: int | None = None
    n_routed_experts: int | None = None
    n_shared_experts: int = 0
    moe_top_k: int | None = None
    moe_expert_d_ff: int | None = None
    moe_router_score: str = "softmax"
    moe_normalize_top_k_weights: bool = True
    moe_aux_loss_weight: float = 0.0
    moe_drop_tokens: bool = False
    moe_routing_mode: str = "aux_loss"
    moe_use_expert_bias: bool = False
    moe_expert_bias_update_rate: float = 0.0
    moe_expert_bias_update_interval: int = 1
    moe_expert_bias_min: float = -1.0
    moe_expert_bias_max: float = 1.0
    mtp_enabled: bool = False
    mtp_num_future_tokens: int = 0
    mtp_loss_weight: float = 0.0
    mtp_share_lm_head: bool = True

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

        if self.mtp_enabled:
            self._validate_mtp_config()
        else:
            if self.mtp_num_future_tokens != 0:
                msg = "mtp_num_future_tokens must be 0 when mtp_enabled is false"
                raise ValueError(msg)
            if self.mtp_loss_weight != 0.0:
                msg = "mtp_loss_weight must be 0.0 when mtp_enabled is false"
                raise ValueError(msg)

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

        if self.mla_q_rope_dim % 2 != 0:
            msg = "mla_q_rope_dim must be even for rotary embeddings"
            raise ValueError(msg)

        if self.mla_qk_nope_dim <= 0:
            msg = "mla_qk_nope_head_dim must be positive"
            raise ValueError(msg)

        if self.mla_v_dim <= 0:
            msg = "mla_v_head_dim must be positive"
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

        valid_routing_modes = {"aux_loss", "aux_loss_free_bias"}
        if self.moe_routing_mode not in valid_routing_modes:
            msg = f"moe_routing_mode must be one of {sorted(valid_routing_modes)}"
            raise ValueError(msg)

        if self.moe_expert_bias_update_rate < 0.0:
            msg = "moe_expert_bias_update_rate must be nonnegative"
            raise ValueError(msg)

        if self.moe_expert_bias_update_interval <= 0:
            msg = "moe_expert_bias_update_interval must be positive"
            raise ValueError(msg)

        if self.moe_expert_bias_min >= self.moe_expert_bias_max:
            msg = "moe_expert_bias_min must be less than moe_expert_bias_max"
            raise ValueError(msg)

        if self.moe_routing_mode == "aux_loss":
            if self.moe_use_expert_bias:
                msg = "moe_use_expert_bias must be false when moe_routing_mode='aux_loss'"
                raise ValueError(msg)
            if self.moe_expert_bias_update_rate != 0.0:
                msg = "moe_expert_bias_update_rate must be 0.0 when " "moe_routing_mode='aux_loss'"
                raise ValueError(msg)

        if self.moe_routing_mode == "aux_loss_free_bias":
            if not self.moe_use_expert_bias:
                msg = (
                    "moe_use_expert_bias must be true when " "moe_routing_mode='aux_loss_free_bias'"
                )
                raise ValueError(msg)
            if self.moe_aux_loss_weight != 0.0:
                msg = (
                    "moe_aux_loss_weight must be 0.0 when " "moe_routing_mode='aux_loss_free_bias'"
                )
                raise ValueError(msg)
            if self.moe_expert_bias_update_rate <= 0.0:
                msg = (
                    "moe_expert_bias_update_rate must be positive when "
                    "moe_routing_mode='aux_loss_free_bias'"
                )
                raise ValueError(msg)

    def _validate_mtp_config(self) -> None:
        """Validate multi-token-prediction configuration fields."""
        if self.mtp_num_future_tokens <= 0:
            msg = "mtp_num_future_tokens must be positive when mtp_enabled is true"
            raise ValueError(msg)

        if self.mtp_num_future_tokens >= self.block_size:
            msg = "mtp_num_future_tokens must be smaller than block_size"
            raise ValueError(msg)

        if self.mtp_loss_weight <= 0.0:
            msg = "mtp_loss_weight must be positive when mtp_enabled is true"
            raise ValueError(msg)

        if self.mtp_share_lm_head:
            msg = "mtp_share_lm_head=True is reserved for a later shared-head MTP variant"
            raise ValueError(msg)

    @property
    def head_dim(self) -> int:
        """Per-head attention dimension."""
        return self.d_model // self.n_heads

    @property
    def mla_qk_nope_dim(self) -> int:
        """Return MLA non-rotary query/key head dimension."""
        if self.mla_qk_nope_head_dim is not None:
            return self.mla_qk_nope_head_dim

        if self.mla_q_rope_dim is None:
            msg = "mla_q_rope_dim must be set before deriving mla_qk_nope_dim"
            raise ValueError(msg)

        return self.head_dim - self.mla_q_rope_dim

    @property
    def mla_v_dim(self) -> int:
        """Return MLA value head dimension."""
        if self.mla_v_head_dim is not None:
            return self.mla_v_head_dim

        return self.head_dim

    @property
    def mla_qk_head_dim(self) -> int:
        """Return total MLA query/key head dimension."""
        if self.mla_q_rope_dim is None:
            msg = "mla_q_rope_dim must be set before deriving mla_qk_head_dim"
            raise ValueError(msg)

        return self.mla_qk_nope_dim + self.mla_q_rope_dim

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> GPTConfig:
        """Build GPTConfig from a loaded YAML model config dictionary."""
        model_config = dict(config.get("model", config))
        model_config.pop("name", None)
        return cls(**model_config)
