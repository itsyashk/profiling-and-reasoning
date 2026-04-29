from __future__ import annotations

import argparse
import math
import statistics
import timeit
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import basics.model as basics_model
from basics.model import BasicsTransformerLM
from basics.optimizer import AdamW

try:
    import torch.cuda.nvtx as nvtx
except ImportError:
    nvtx = None

_NVTX_AVAILABLE = True


@dataclass(frozen=True)
class ModelSpec:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


MODEL_SPECS: dict[str, ModelSpec] = {
    "small": ModelSpec(d_model=512, d_ff=2048, num_layers=8, num_heads=8),
    "medium": ModelSpec(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "large": ModelSpec(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
}


@dataclass(frozen=True)
class BenchmarkConfig:
    model_size: str
    context_length: int = 128
    batch_size: int = 4
    vocab_size: int = 10_000
    warmup_steps: int = 5
    measure_steps: int = 10
    mode: Literal["forward", "backward", "forward-backward", "train-step"] = "forward"
    use_bf16: bool = False
    use_memory_profiler: bool = False
    compile_model: bool = False
    annotate_attention: bool = False
    output_dir: Path = Path("artifacts")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark and profile the Basics transformer.")
    parser.add_argument("--model-size", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--mode", choices=["forward", "backward", "forward-backward", "train-step"], default="forward")
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--use-memory-profiler", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--annotate-attention", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser


@contextmanager
def nvtx_range(message: str):
    global _NVTX_AVAILABLE
    if nvtx is None or not _NVTX_AVAILABLE:
        yield
        return

    try:
        nvtx.range_push(message)
    except RuntimeError:
        _NVTX_AVAILABLE = False
        yield
        return

    try:
        yield
    finally:
        nvtx.range_pop()


def build_model(config: BenchmarkConfig) -> torch.nn.Module:
    """Instantiate the staff Basics transformer for the requested model size."""
    spec = MODEL_SPECS[config.model_size]
    return BasicsTransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=spec.d_model,
        num_layers=spec.num_layers,
        num_heads=spec.num_heads,
        d_ff=spec.d_ff,
        rope_theta=10_000.0,
    )


def make_random_batch(config: BenchmarkConfig, device: torch.device) -> torch.Tensor:
    """Construct a random token batch for benchmarking and profiling."""
    return torch.randint(
        low=0,
        high=config.vocab_size,
        size=(config.batch_size, config.context_length),
        device=device,
        dtype=torch.long,
    )


def run_single_step(
    model: torch.nn.Module,
    batch: torch.Tensor,
    mode: Literal["forward", "backward", "forward-backward", "train-step"],
    autocast_context,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Execute one benchmark step and synchronize CUDA before returning."""
    if mode == "forward":
        with torch.no_grad(), nvtx_range("forward pass"), autocast_context:
            model(batch)
    elif mode == "forward-backward":
        model.zero_grad(set_to_none=True)
        with nvtx_range("forward pass"), autocast_context:
            logits = model(batch)
        with nvtx_range("compute loss"):
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                batch.reshape(-1),
            )
        with nvtx_range("backward pass"):
            loss.backward()
    elif mode == "train-step":
        if optimizer is None:
            raise ValueError("train-step mode requires an optimizer")

        optimizer.zero_grad(set_to_none=True)
        with nvtx_range("forward pass"), autocast_context:
            logits = model(batch)
        with nvtx_range("compute loss"):
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                batch.reshape(-1),
            )
        with nvtx_range("backward pass"):
            loss.backward()
        with nvtx_range("optimizer step"):
            optimizer.step()
    else:
        raise NotImplementedError(f"{mode!r} mode is not supported by run_single_step.")

    if batch.is_cuda:
        torch.cuda.synchronize(batch.device)


def make_loss(model: torch.nn.Module, batch: torch.Tensor, autocast_context) -> torch.Tensor:
    with nvtx_range("forward pass"), autocast_context:
        logits = model(batch)
    with nvtx_range("compute loss"):
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch.reshape(-1),
        )


def run_backward_step(
    model: torch.nn.Module,
    batch: torch.Tensor,
    autocast_context,
) -> None:
    """Time only the backward pass after building the graph with an untimed forward pass."""
    model.zero_grad(set_to_none=True)
    loss = make_loss(model, batch, autocast_context)
    if batch.is_cuda:
        torch.cuda.synchronize(batch.device)
    with nvtx_range("backward pass"):
        loss.backward()
    if batch.is_cuda:
        torch.cuda.synchronize(batch.device)


def benchmark_model(config: BenchmarkConfig) -> dict[str, float]:
    """Run warmup steps followed by timed measurement steps."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.annotate_attention:
        basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

    model = build_model(config).to(device)
    model.train(config.mode in {"backward", "forward-backward", "train-step"})
    if config.compile_model:
        model = torch.compile(model)

    optimizer = AdamW(model.parameters(), lr=1e-3) if config.mode == "train-step" else None
    batch = make_random_batch(config, device)
    autocast_context = make_autocast_context(config.use_bf16)

    with nvtx_range("warmup"):
        for step in range(config.warmup_steps):
            with nvtx_range(f"warmup step {step}"):
                if config.mode == "backward":
                    run_backward_step(model, batch, autocast_context)
                else:
                    run_single_step(model, batch, config.mode, autocast_context, optimizer)

    step_times = []
    with nvtx_range("measurement"):
        for step in range(config.measure_steps):
            with nvtx_range(f"measurement step {step}"):
                if config.mode == "backward":
                    model.zero_grad(set_to_none=True)
                    loss = make_loss(model, batch, autocast_context)
                    if batch.is_cuda:
                        torch.cuda.synchronize(batch.device)
                    start = timeit.default_timer()
                    with nvtx_range("timed backward pass"):
                        loss.backward()
                    if batch.is_cuda:
                        torch.cuda.synchronize(batch.device)
                else:
                    start = timeit.default_timer()
                    run_single_step(model, batch, config.mode, autocast_context, optimizer)
                step_times.append(timeit.default_timer() - start)

    total_time_seconds = sum(step_times)
    average_step_time_seconds = statistics.fmean(step_times)
    std_step_time_seconds = statistics.stdev(step_times) if len(step_times) > 1 else 0.0
    results = {
        "total_time_seconds": total_time_seconds,
        "average_step_time_seconds": average_step_time_seconds,
        "std_step_time_seconds": std_step_time_seconds,
        "steps_per_second": 1.0 / average_step_time_seconds,
    }
    print(results)
    return results


def annotated_scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Optional NVTX-annotated attention path for Nsight Systems profiling."""
    with nvtx_range("scaled dot product attention"):
        d_k = K.shape[-1]
        with nvtx_range("computing attention scores"):
            attention_scores = basics_model.einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

        if mask is not None:
            with nvtx_range("applying attention mask"):
                attention_scores = torch.where(mask, attention_scores, float("-inf"))

        with nvtx_range("computing softmax"):
            attention_weights = basics_model.softmax(attention_scores, dim=-1)

        with nvtx_range("final matmul"):
            return basics_model.einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")


def maybe_start_memory_history(enabled: bool) -> None:
    if enabled:
        raise NotImplementedError


def maybe_dump_memory_snapshot(enabled: bool, output_path: Path) -> None:
    if enabled:
        raise NotImplementedError


def make_autocast_context(use_bf16: bool):
    if use_bf16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def main() -> None:
    args = build_argparser().parse_args()
    config = BenchmarkConfig(
        model_size=args.model_size,
        context_length=args.context_length,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        mode=args.mode,
        use_bf16=args.use_bf16,
        use_memory_profiler=args.use_memory_profiler,
        compile_model=args.compile_model,
        annotate_attention=args.annotate_attention,
        output_dir=args.output_dir,
    )
    benchmark_model(config)


if __name__ == "__main__":
    main()
