from __future__ import annotations

from unittest.mock import patch

import torch

from deepseek_reimpl.layers.mla import MLAAttention
from deepseek_reimpl.layers.rope import RotaryEmbedding
from deepseek_reimpl.model.config import GPTConfig


def test_rotary_embedding_uses_one_frequency_per_adjacent_pair() -> None:
    rope = RotaryEmbedding(head_dim=4, base=100.0)
    q = torch.tensor([[[[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]]]])

    q_rot, _ = rope.apply(q, q)

    expected = torch.tensor(
        [
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [
                        torch.cos(torch.tensor(1.0)),
                        torch.sin(torch.tensor(1.0)),
                        torch.cos(torch.tensor(0.1)),
                        torch.sin(torch.tensor(0.1)),
                    ],
                ]
            ]
        ]
    )
    assert torch.allclose(q_rot, expected, atol=1e-6)


def test_rotary_embedding_preserves_each_adjacent_pair_norm() -> None:
    torch.manual_seed(0)
    rope = RotaryEmbedding(head_dim=8)
    q = torch.randn(2, 3, 5, 8)

    q_rot, _ = rope.apply(q, q)

    pair_norms = q.view(*q.shape[:-1], -1, 2).norm(dim=-1)
    rotated_pair_norms = q_rot.view(*q_rot.shape[:-1], -1, 2).norm(dim=-1)
    assert torch.allclose(rotated_pair_norms, pair_norms, atol=1e-6)


def test_rotary_embedding_is_head_invariant_and_position_dependent() -> None:
    rope = RotaryEmbedding(head_dim=4, base=100.0)
    per_position = torch.tensor(
        [[[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]]],
        dtype=torch.float64,
    )
    query = per_position.unsqueeze(1).expand(1, 3, 2, 4).clone()

    rotated, _ = rope.apply(query, query)

    assert torch.equal(rotated[:, 0], rotated[:, 1])
    assert torch.equal(rotated[:, 1], rotated[:, 2])
    assert not torch.equal(rotated[:, :, 0], rotated[:, :, 1])


def test_rotary_embedding_passes_double_precision_gradient_check() -> None:
    rope = RotaryEmbedding(head_dim=4, base=100.0)
    query = torch.randn(1, 2, 3, 4, dtype=torch.float64, requires_grad=True)
    key = torch.randn(1, 2, 3, 4, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(rope.apply, (query, key))


def test_mla_sends_batch_head_sequence_layout_to_rotary_embedding() -> None:
    config = GPTConfig(
        vocab_size=32,
        block_size=8,
        n_layers=1,
        n_heads=2,
        d_model=8,
        d_ff=32,
        dropout=0.0,
        positional_encoding="rope",
        attention_type="mla",
        mla_kv_latent_dim=4,
        mla_q_rope_dim=2,
    )
    attention = MLAAttention(config)
    x = torch.randn(3, 5, config.d_model)

    with patch.object(attention.rope, "apply", wraps=attention.rope.apply) as apply_rope:
        attention(x)

    q_rope, k_rope = apply_rope.call_args.args
    assert q_rope.shape == (3, config.n_heads, 5, config.mla_q_rope_dim)
    assert k_rope.shape == q_rope.shape
