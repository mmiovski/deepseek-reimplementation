"""Configurable decoder-only model for controlled architecture variants."""

from __future__ import annotations

import torch
from torch import nn

from deepseek_reimpl.layers.mtp import MTPOutput, MultiTokenPredictionHead
from deepseek_reimpl.model.config import GPTConfig
from deepseek_reimpl.model.decoder_block import DecoderBlock, build_norm


class BaselineGPT(nn.Module):
    """Shared decoder implementation for dense and DeepSeek-inspired variants."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = (
            nn.Embedding(config.block_size, config.d_model)
            if config.positional_encoding == "learned"
            else None
        )
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
        self.final_norm = build_norm(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.mtp_head = (
            MultiTokenPredictionHead(
                d_model=config.d_model,
                vocab_size=config.vocab_size,
                num_future_tokens=config.mtp_num_future_tokens,
            )
            if config.mtp_enabled
            else None
        )

        self.apply(self._init_weights)

        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize trainable weights with GPT-style small normal scales."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _forward_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return final decoder hidden states for input token IDs."""
        _, seq_len = input_ids.shape

        if seq_len > self.config.block_size:
            msg = "sequence length exceeds configured block_size"
            raise ValueError(msg)

        hidden_states = self.token_embedding(input_ids)

        if self.position_embedding is not None:
            positions = torch.arange(seq_len, device=input_ids.device)
            hidden_states = hidden_states + self.position_embedding(positions)[None, :, :]

        hidden_states = self.dropout(hidden_states)

        for block in self.blocks:
            hidden_states = block(hidden_states)

        hidden_states = self.final_norm(hidden_states)
        output: torch.Tensor = hidden_states
        return output

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return next-token logits for input token IDs."""
        hidden_states = self._forward_hidden_states(input_ids)
        logits: torch.Tensor = self.lm_head(hidden_states)
        return logits

    def forward_mtp(self, input_ids: torch.Tensor) -> MTPOutput:
        """Return next-token and auxiliary future-token logits."""
        if self.mtp_head is None:
            msg = "forward_mtp requires mtp_enabled=True"
            raise RuntimeError(msg)

        hidden_states = self._forward_hidden_states(input_ids)
        next_token_logits = self.lm_head(hidden_states)
        future_token_logits = self.mtp_head(hidden_states)

        return MTPOutput(
            next_token_logits=next_token_logits,
            future_token_logits=future_token_logits,
        )

    def auxiliary_loss(self) -> torch.Tensor | None:
        """Return summed auxiliary losses from modules that expose them.

        Dense and MLA-only models return None. MoE models return the sum of the
        latest per-layer auxiliary load-balancing losses after a forward pass.
        """
        aux_losses: list[torch.Tensor] = []

        for module in self.modules():
            aux_loss = getattr(module, "last_aux_loss", None)
            if isinstance(aux_loss, torch.Tensor):
                aux_losses.append(aux_loss)

        if not aux_losses:
            return None

        return torch.stack(aux_losses).sum()
