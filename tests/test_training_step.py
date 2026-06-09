from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from deepseek_reimpl.eval.language_model_eval import evaluate_language_model
from deepseek_reimpl.train.losses import next_token_cross_entropy
from deepseek_reimpl.train.optim import build_adamw, build_optimizer
from deepseek_reimpl.train.train_utils import (
    count_batch_tokens,
    move_batch_to_device,
    resolve_device,
    set_seed,
    unpack_lm_batch,
)
from deepseek_reimpl.train.trainer import TrainingLoopConfig, train_loop, train_step


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size: int = 8, d_model: int = 6) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.output(self.embedding(input_ids))


def test_next_token_cross_entropy_returns_scalar_loss() -> None:
    logits = torch.randn(2, 3, 5)
    targets = torch.tensor([[0, 1, 2], [2, 3, 4]])

    loss = next_token_cross_entropy(logits, targets)

    assert loss.ndim == 0


def test_next_token_cross_entropy_is_finite() -> None:
    logits = torch.randn(2, 3, 5)
    targets = torch.tensor([[0, 1, 2], [2, 3, 4]])

    loss = next_token_cross_entropy(logits, targets)

    assert torch.isfinite(loss)


def test_next_token_cross_entropy_backward_creates_gradients() -> None:
    logits = torch.randn(2, 3, 5, requires_grad=True)
    targets = torch.tensor([[0, 1, 2], [2, 3, 4]])

    loss = next_token_cross_entropy(logits, targets)
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_next_token_cross_entropy_rejects_invalid_logits_rank() -> None:
    logits = torch.randn(2, 3)
    targets = torch.tensor([[0, 1, 2], [2, 3, 4]])

    with pytest.raises(ValueError, match="logits must have shape"):
        next_token_cross_entropy(logits, targets)


def test_next_token_cross_entropy_rejects_invalid_target_rank() -> None:
    logits = torch.randn(2, 3, 5)
    targets = torch.tensor([0, 1, 2])

    with pytest.raises(ValueError, match="targets must have shape"):
        next_token_cross_entropy(logits, targets)


def test_next_token_cross_entropy_rejects_mismatched_batch_sequence_dims() -> None:
    logits = torch.randn(2, 3, 5)
    targets = torch.tensor([[0, 1], [2, 3]])

    with pytest.raises(ValueError, match="batch/sequence dimensions"):
        next_token_cross_entropy(logits, targets)


def test_build_adamw_returns_adamw_optimizer() -> None:
    model = nn.Linear(3, 2)

    optimizer = build_adamw(
        model,
        learning_rate=0.001,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )

    assert isinstance(optimizer, torch.optim.AdamW)


def test_build_adamw_uses_configured_learning_rate() -> None:
    model = nn.Linear(3, 2)

    optimizer = build_adamw(
        model,
        learning_rate=0.002,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )

    assert optimizer.param_groups[0]["lr"] == 0.002


def test_build_adamw_can_step_after_backward() -> None:
    model = nn.Linear(3, 2)
    optimizer = build_adamw(
        model,
        learning_rate=0.001,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    inputs = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 0, 1])

    logits = model(inputs)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)


def test_build_adamw_rejects_invalid_learning_rate() -> None:
    model = nn.Linear(3, 2)

    with pytest.raises(ValueError, match="learning_rate must be positive"):
        build_adamw(
            model,
            learning_rate=0.0,
            weight_decay=0.1,
            betas=(0.9, 0.95),
        )


def test_build_optimizer_from_train_config() -> None:
    model = nn.Linear(3, 2)
    optimizer = build_optimizer(
        model,
        {
            "learning_rate": 0.003,
            "weight_decay": 0.2,
            "betas": [0.8, 0.9],
        },
    )

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 0.003
    assert optimizer.param_groups[0]["weight_decay"] == 0.2
    assert optimizer.param_groups[0]["betas"] == (0.8, 0.9)


def test_evaluate_language_model_returns_finite_metrics() -> None:
    model = TinyLanguageModel()
    inputs = torch.tensor([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]])
    targets = torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
    dataloader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

    metrics = evaluate_language_model(model, dataloader, device=torch.device("cpu"))

    assert metrics.num_batches == 2
    assert metrics.num_tokens == 12
    assert torch.isfinite(torch.tensor(metrics.loss))
    assert torch.isfinite(torch.tensor(metrics.perplexity))


def test_evaluate_language_model_respects_max_batches() -> None:
    model = TinyLanguageModel()
    inputs = torch.tensor([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]])
    targets = torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
    dataloader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

    metrics = evaluate_language_model(
        model,
        dataloader,
        device=torch.device("cpu"),
        max_batches=1,
    )

    assert metrics.num_batches == 1
    assert metrics.num_tokens == 6


def test_evaluate_language_model_restores_training_mode() -> None:
    model = TinyLanguageModel()
    model.train()

    inputs = torch.tensor([[0, 1, 2], [1, 2, 3]])
    targets = torch.tensor([[1, 2, 3], [2, 3, 4]])
    dataloader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

    evaluate_language_model(model, dataloader, device=torch.device("cpu"))

    assert model.training


def test_evaluate_language_model_rejects_empty_dataloader() -> None:
    model = TinyLanguageModel()
    inputs = torch.empty((0, 3), dtype=torch.long)
    targets = torch.empty((0, 3), dtype=torch.long)
    dataloader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

    with pytest.raises(ValueError, match="no batches"):
        evaluate_language_model(model, dataloader, device=torch.device("cpu"))


def test_resolve_device_cpu() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_auto_returns_valid_device() -> None:
    device = resolve_device("auto")

    assert device.type in {"cpu", "cuda"}


def test_resolve_device_rejects_invalid_device_name() -> None:
    with pytest.raises(ValueError, match="device must be one of"):
        resolve_device("mps")


def test_resolve_device_cuda_unavailable_behavior_is_clear() -> None:
    if torch.cuda.is_available():
        assert resolve_device("cuda").type == "cuda"
    else:
        with pytest.raises(RuntimeError, match="CUDA was requested"):
            resolve_device("cuda")


def test_set_seed_makes_torch_randomness_repeatable() -> None:
    set_seed(123)
    first = torch.randn(3)

    set_seed(123)
    second = torch.randn(3)

    assert torch.equal(first, second)


def test_unpack_lm_batch_supports_tuple_batch() -> None:
    input_ids = torch.tensor([[1, 2, 3]])
    targets = torch.tensor([[2, 3, 4]])

    unpacked_inputs, unpacked_targets = unpack_lm_batch((input_ids, targets))

    assert torch.equal(unpacked_inputs, input_ids)
    assert torch.equal(unpacked_targets, targets)


def test_unpack_lm_batch_supports_mapping_batch() -> None:
    input_ids = torch.tensor([[1, 2, 3]])
    targets = torch.tensor([[2, 3, 4]])

    unpacked_inputs, unpacked_targets = unpack_lm_batch(
        {"input_ids": input_ids, "target_ids": targets}
    )

    assert torch.equal(unpacked_inputs, input_ids)
    assert torch.equal(unpacked_targets, targets)


def test_move_batch_to_device_moves_tensors() -> None:
    batch = (torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]]))

    input_ids, targets = move_batch_to_device(batch, torch.device("cpu"))

    assert input_ids.device.type == "cpu"
    assert targets.device.type == "cpu"


def test_count_batch_tokens_uses_input_ids_numel() -> None:
    batch = (torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([[2, 3, 4], [5, 6, 7]]))

    assert count_batch_tokens(batch) == 6


def test_train_step_returns_finite_loss_and_token_count() -> None:
    model = TinyLanguageModel()
    optimizer = build_adamw(
        model,
        learning_rate=0.001,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    batch = (
        torch.tensor([[0, 1, 2], [1, 2, 3]]),
        torch.tensor([[1, 2, 3], [2, 3, 4]]),
    )

    metrics = train_step(model, batch, optimizer, device=torch.device("cpu"))

    assert torch.isfinite(torch.tensor(metrics.loss))
    assert metrics.num_tokens == 6
    assert metrics.grad_norm is None


def test_train_step_updates_parameters() -> None:
    model = TinyLanguageModel()
    optimizer = build_adamw(
        model,
        learning_rate=0.001,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    before = [parameter.detach().clone() for parameter in model.parameters()]
    batch = (
        torch.tensor([[0, 1, 2], [1, 2, 3]]),
        torch.tensor([[1, 2, 3], [2, 3, 4]]),
    )

    train_step(model, batch, optimizer, device=torch.device("cpu"))

    after = list(model.parameters())
    assert any(
        not torch.equal(before_param, after_param)
        for before_param, after_param in zip(before, after, strict=True)
    )


def test_train_step_grad_clip_path_returns_grad_norm() -> None:
    model = TinyLanguageModel()
    optimizer = build_adamw(
        model,
        learning_rate=0.001,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    batch = (
        torch.tensor([[0, 1, 2], [1, 2, 3]]),
        torch.tensor([[1, 2, 3], [2, 3, 4]]),
    )

    metrics = train_step(
        model,
        batch,
        optimizer,
        device=torch.device("cpu"),
        grad_clip=1.0,
    )

    assert metrics.grad_norm is not None
    assert metrics.grad_norm >= 0.0


def test_train_step_rejects_invalid_grad_clip() -> None:
    model = TinyLanguageModel()
    optimizer = build_adamw(
        model,
        learning_rate=0.001,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    batch = (
        torch.tensor([[0, 1, 2], [1, 2, 3]]),
        torch.tensor([[1, 2, 3], [2, 3, 4]]),
    )

    with pytest.raises(ValueError, match="grad_clip must be positive"):
        train_step(
            model,
            batch,
            optimizer,
            device=torch.device("cpu"),
            grad_clip=0.0,
        )


def test_train_loop_stops_at_max_steps() -> None:
    model = TinyLanguageModel()
    optimizer = build_adamw(
        model,
        learning_rate=0.001,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    inputs = torch.tensor([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]])
    targets = torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
    dataloader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

    summary = train_loop(
        model,
        dataloader,
        optimizer,
        device=torch.device("cpu"),
        config=TrainingLoopConfig(
            max_steps=2,
            max_tokens=None,
            eval_interval=None,
            log_interval=1,
            eval_batches=1,
        ),
    )

    assert summary.steps == 2
    assert summary.train_tokens == 12
    assert torch.isfinite(torch.tensor(summary.final_train_loss))
    assert summary.peak_memory_bytes is None


def test_train_loop_stops_at_max_tokens() -> None:
    model = TinyLanguageModel()
    optimizer = build_adamw(
        model,
        learning_rate=0.001,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    inputs = torch.tensor([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]])
    targets = torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
    dataloader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

    summary = train_loop(
        model,
        dataloader,
        optimizer,
        device=torch.device("cpu"),
        config=TrainingLoopConfig(
            max_steps=None,
            max_tokens=6,
            eval_interval=None,
            log_interval=1,
            eval_batches=1,
        ),
    )

    assert summary.steps == 1
    assert summary.train_tokens == 6


def test_train_loop_validation_eval_path_returns_metrics() -> None:
    model = TinyLanguageModel()
    optimizer = build_adamw(
        model,
        learning_rate=0.001,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    inputs = torch.tensor([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]])
    targets = torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
    train_dataloader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    validation_dataloader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

    summary = train_loop(
        model,
        train_dataloader,
        optimizer,
        device=torch.device("cpu"),
        config=TrainingLoopConfig(
            max_steps=1,
            max_tokens=None,
            eval_interval=1,
            log_interval=1,
            eval_batches=1,
        ),
        validation_dataloader=validation_dataloader,
    )

    assert summary.validation_loss is not None
    assert summary.validation_perplexity is not None
    assert torch.isfinite(torch.tensor(summary.validation_loss))
    assert torch.isfinite(torch.tensor(summary.validation_perplexity))


def test_train_loop_test_eval_path_returns_metrics() -> None:
    model = TinyLanguageModel()
    optimizer = build_adamw(
        model,
        learning_rate=0.001,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    inputs = torch.tensor([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]])
    targets = torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
    train_dataloader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    test_dataloader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

    summary = train_loop(
        model,
        train_dataloader,
        optimizer,
        device=torch.device("cpu"),
        config=TrainingLoopConfig(
            max_steps=1,
            max_tokens=None,
            eval_interval=None,
            log_interval=1,
            eval_batches=1,
        ),
        test_dataloader=test_dataloader,
    )

    assert summary.test_loss is not None
    assert summary.test_perplexity is not None


def test_train_loop_log_callback_receives_records() -> None:
    model = TinyLanguageModel()
    optimizer = build_adamw(
        model,
        learning_rate=0.001,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    records: list[dict[str, object]] = []
    inputs = torch.tensor([[0, 1, 2], [1, 2, 3]])
    targets = torch.tensor([[1, 2, 3], [2, 3, 4]])
    dataloader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

    train_loop(
        model,
        dataloader,
        optimizer,
        device=torch.device("cpu"),
        config=TrainingLoopConfig(
            max_steps=1,
            max_tokens=None,
            eval_interval=None,
            log_interval=1,
            eval_batches=1,
        ),
        log_callback=records.append,
    )

    assert len(records) == 1
    assert records[0]["step"] == 1


def test_train_loop_requires_step_or_token_budget() -> None:
    model = TinyLanguageModel()
    optimizer = build_adamw(
        model,
        learning_rate=0.001,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    inputs = torch.tensor([[0, 1, 2], [1, 2, 3]])
    targets = torch.tensor([[1, 2, 3], [2, 3, 4]])
    dataloader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

    with pytest.raises(ValueError, match="max_steps or max_tokens"):
        train_loop(
            model,
            dataloader,
            optimizer,
            device=torch.device("cpu"),
            config=TrainingLoopConfig(
                max_steps=None,
                max_tokens=None,
                eval_interval=None,
                log_interval=1,
                eval_batches=1,
            ),
        )


def test_train_step_includes_moe_auxiliary_loss() -> None:
    from deepseek_reimpl.model.baseline_gpt import BaselineGPT
    from deepseek_reimpl.model.config import GPTConfig

    config = GPTConfig(
        vocab_size=32,
        block_size=8,
        n_layers=1,
        n_heads=2,
        d_model=16,
        d_ff=64,
        dropout=0.0,
        ffn_type="moe",
        n_routed_experts=4,
        n_shared_experts=1,
        moe_top_k=2,
        moe_expert_d_ff=32,
        moe_aux_loss_weight=0.01,
    )
    model = BaselineGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    batch = (
        torch.randint(0, config.vocab_size, (2, 4)),
        torch.randint(0, config.vocab_size, (2, 4)),
    )

    metrics = train_step(model, batch, optimizer, device=torch.device("cpu"))

    assert metrics.aux_loss is not None
    assert metrics.aux_loss > 0.0
    assert metrics.loss > metrics.lm_loss


def test_dense_train_step_reports_no_auxiliary_loss() -> None:
    model = TinyLanguageModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    batch = (
        torch.randint(0, 8, (2, 4)),
        torch.randint(0, 8, (2, 4)),
    )

    metrics = train_step(model, batch, optimizer, device=torch.device("cpu"))

    assert metrics.aux_loss is None
    assert metrics.loss == metrics.lm_loss
