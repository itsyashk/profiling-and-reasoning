from __future__ import annotations

import argparse
import statistics
import timeit
from dataclasses import dataclass
from typing import Iterable

import torch

import basics.model as basics_model


@dataclass(frozen=True)
class AttentionBenchmarkConfig:
    head_dims: tuple[int, ...] = (16, 32, 64, 128)
    sequence_lengths: tuple[int, ...] = (64, 128, 256, 512, 1024)
    batch_size: int = 8
    forward_passes: int = 100
    backward_passes: int = 100
    compile_attention: bool = False


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark attention implementations.")
    parser.add_argument("--compile-attention", action="store_true")
    return parser


def iter_benchmark_shapes(config: AttentionBenchmarkConfig) -> Iterable[tuple[int, int]]:
    for head_dim in config.head_dims:
        for sequence_length in config.sequence_lengths:
            yield head_dim, sequence_length


def make_qkv(batch_size: int, sequence_length: int, head_dim: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Create random Q, K, and V tensors for the attention benchmark."""
    shape = (batch_size, sequence_length, head_dim)
    q = torch.randn(*shape, device=device, requires_grad=True)
    k = torch.randn(*shape, device=device, requires_grad=True)
    v = torch.randn(*shape, device=device, requires_grad=True)
    return q, k, v


def benchmark_attention_once(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_fn,
    num_forward: int,
    num_backward: int,
) -> dict[str, float]:
    """Time the forward and backward pass for a single attention configuration."""
    device = q.device

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            attention_fn(q, k, v)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # Time forward passes
    fwd_times = []
    for _ in range(num_forward):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = timeit.default_timer()
        with torch.no_grad():
            attention_fn(q, k, v)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        fwd_times.append(timeit.default_timer() - t0)

    # Measure peak memory allocated just before backward (includes saved activations)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    # Run one forward to fill memory with activations
    out = attention_fn(q, k, v)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    memory_before_bwd_mb = torch.cuda.memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
    # Free this graph so we can do fresh ones below
    del out

    # Warmup backward
    for _ in range(3):
        if q.grad is not None:
            q.grad = None
        if k.grad is not None:
            k.grad = None
        if v.grad is not None:
            v.grad = None
        out = attention_fn(q, k, v)
        loss = out.sum()
        loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # Time backward passes
    bwd_times = []
    for _ in range(num_backward):
        for tensor in (q, k, v):
            if tensor.grad is not None:
                tensor.grad = None
        out = attention_fn(q, k, v)
        loss = out.sum()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = timeit.default_timer()
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        bwd_times.append(timeit.default_timer() - t0)

    return {
        "fwd_mean_ms": statistics.fmean(fwd_times) * 1000,
        "fwd_std_ms": (statistics.stdev(fwd_times) if len(fwd_times) > 1 else 0.0) * 1000,
        "bwd_mean_ms": statistics.fmean(bwd_times) * 1000,
        "bwd_std_ms": (statistics.stdev(bwd_times) if len(bwd_times) > 1 else 0.0) * 1000,
        "memory_before_bwd_mb": memory_before_bwd_mb,
    }


def benchmark_attention_grid(config: AttentionBenchmarkConfig) -> list[dict[str, float | int | str]]:
    """Run the attention benchmark over the Section 2.7 Cartesian product of scales."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    attention_fn = basics_model.scaled_dot_product_attention
    if config.compile_attention:
        # Triton (inductor backend) is unavailable on Windows; use aot_eager instead
        attention_fn = torch.compile(attention_fn, backend="aot_eager")

    results = []
    header = f"{'d_model':>8} {'seq_len':>8} {'fwd_ms':>10} {'bwd_ms':>10} {'mem_mb':>10}"
    print(header)
    print("-" * len(header))

    for head_dim, seq_len in iter_benchmark_shapes(config):
        try:
            q, k, v = make_qkv(config.batch_size, seq_len, head_dim, device)
            row = benchmark_attention_once(
                q, k, v, attention_fn, config.forward_passes, config.backward_passes
            )
            row["head_dim"] = head_dim
            row["seq_len"] = seq_len
            row["oom"] = False
            print(
                f"{head_dim:>8} {seq_len:>8} "
                f"{row['fwd_mean_ms']:>9.3f}ms "
                f"{row['bwd_mean_ms']:>9.3f}ms "
                f"{row['memory_before_bwd_mb']:>9.1f}MB"
            )
        except torch.cuda.OutOfMemoryError:
            row = {"head_dim": head_dim, "seq_len": seq_len, "oom": True}
            print(f"{head_dim:>8} {seq_len:>8} {'OOM':>10} {'OOM':>10} {'OOM':>10}")
            torch.cuda.empty_cache()
        finally:
            del q, k, v
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results.append(row)

    return results


def main() -> None:
    args = build_argparser().parse_args()
    config = AttentionBenchmarkConfig(compile_attention=args.compile_attention)
    benchmark_attention_grid(config)


if __name__ == "__main__":
    main()
