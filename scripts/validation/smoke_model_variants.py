"""Exercise every architecture under strict CUDA determinism before primary launch."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deepseek_reimpl.model.model_factory import build_model_from_config  # noqa: E402
from deepseek_reimpl.train.checkpointing import (  # noqa: E402
    capture_rng_state,
    restore_rng_state,
)
from deepseek_reimpl.train.optim import build_optimizer  # noqa: E402
from deepseek_reimpl.train.train_utils import configure_determinism, set_seed  # noqa: E402
from deepseek_reimpl.train.trainer import train_step  # noqa: E402
from deepseek_reimpl.utils.config import load_yaml_config  # noqa: E402

MODEL_CONFIGS = [
    Path("configs/model/dense_121m.yaml"),
    Path("configs/model/mla_121m.yaml"),
    Path("configs/model/mtp_121m.yaml"),
    Path("configs/model/moe_220m.yaml"),
    Path("configs/model/mla_moe_220m.yaml"),
    Path("configs/model/v3_routing_220m.yaml"),
]
OPTIMIZER_CONFIG = {
    "learning_rate": 0.0003,
    "weight_decay": 0.1,
    "betas": [0.9, 0.95],
}


def _scaled_config(path: Path) -> dict[str, Any]:
    wrapper = load_yaml_config(PROJECT_ROOT / path)
    model = dict(wrapper["model"])
    model.update(
        vocab_size=128,
        block_size=8,
        n_layers=1,
        n_heads=4,
        d_model=32,
        d_ff=64,
        dropout=0.1,
    )
    if model["attention_type"] == "mla":
        model.update(
            mla_kv_latent_dim=16,
            mla_q_rope_dim=8,
            mla_qk_nope_head_dim=8,
            mla_v_head_dim=8,
        )
    if model["ffn_type"] == "moe":
        model.update(
            n_routed_experts=4,
            n_shared_experts=1,
            moe_top_k=2,
            moe_expert_d_ff=32,
        )
    return {"model": model}


def _batches() -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(909)
    stream = torch.randint(0, 128, (4, 2, 9), generator=generator)
    return [(row[:, :-1], row[:, 1:]) for row in stream]


def _new_training_objects(config: dict[str, Any], device: torch.device):
    set_seed(12345)
    model = build_model_from_config(config).to(device)
    optimizer = build_optimizer(model, OPTIMIZER_CONFIG)
    return model, optimizer


def _state_equal(left: torch.nn.Module, right: torch.nn.Module) -> bool:
    return all(
        torch.equal(tensor, right.state_dict()[name]) for name, tensor in left.state_dict().items()
    )


def _resume_smoke(path: Path, device: torch.device) -> dict[str, Any]:
    config = _scaled_config(path)
    batches = _batches()
    reference, reference_optimizer = _new_training_objects(config, device)
    reference_losses = [
        train_step(reference, batch, reference_optimizer, device=device, grad_clip=1.0).loss
        for batch in batches
    ]

    interrupted, interrupted_optimizer = _new_training_objects(config, device)
    for batch in batches[:2]:
        train_step(interrupted, batch, interrupted_optimizer, device=device, grad_clip=1.0)
    checkpoint = {
        "model": interrupted.state_dict(),
        "optimizer": interrupted_optimizer.state_dict(),
        "rng": capture_rng_state(),
    }
    resumed, resumed_optimizer = _new_training_objects(config, device)
    resumed.load_state_dict(checkpoint["model"])
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    restore_rng_state(checkpoint["rng"])
    resumed_losses = [
        train_step(resumed, batch, resumed_optimizer, device=device, grad_clip=1.0).loss
        for batch in batches[2:]
    ]
    if not _state_equal(reference, resumed):
        raise RuntimeError(f"Resume mismatch for {path.stem}")
    if reference_losses[2:] != resumed_losses:
        raise RuntimeError(f"Scientific metric mismatch after resume for {path.stem}")
    return {
        "variant": path.stem,
        "scaled_resume_bitwise_equal": True,
        "final_loss": reference_losses[-1],
    }


def _full_model_step(path: Path, device: torch.device, *, primary_shape: bool) -> dict[str, Any]:
    config = load_yaml_config(PROJECT_ROOT / path)
    model, optimizer = _new_training_objects(config, device)
    generator = torch.Generator().manual_seed(707)
    batch_size, stream_length = (4, 257) if primary_shape else (1, 9)
    tokens = torch.randint(0, 10_000, (batch_size, stream_length), generator=generator)
    batch = (tokens[:, :-1], tokens[:, 1:])
    train_step(
        model,
        batch,
        optimizer,
        device=device,
        grad_clip=1.0,
    )
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    metrics = train_step(model, batch, optimizer, device=device, grad_clip=1.0)
    torch.cuda.synchronize(device)
    step_seconds = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated(device))
    return {
        "variant": path.stem,
        "full_model_optimizer_step": True,
        "loss": metrics.loss,
        "peak_memory_bytes": peak,
        "batch_size": batch_size,
        "sequence_length": stream_length - 1,
        "steady_step_seconds": step_seconds,
        "steady_tokens_per_second": batch_size * (stream_length - 1) / step_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-model-step", action="store_true")
    parser.add_argument("--primary-shape", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the six-variant determinism smoke")
    configure_determinism(enabled=True)
    device = torch.device("cuda")
    records = []
    for path in MODEL_CONFIGS:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        record = (
            _full_model_step(path, device, primary_shape=args.primary_shape)
            if args.full_model_step
            else _resume_smoke(path, device)
        )
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    print(json.dumps({"passed": True, "variants": records}, indent=2))


if __name__ == "__main__":
    main()
