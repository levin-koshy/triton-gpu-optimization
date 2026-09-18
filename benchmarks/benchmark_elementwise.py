"""Benchmark the Project 1 PyTorch and Triton implementations.

We will complete this file after correctness has been established.
"""
import torch

try:
    import triton
    from kernels.fused_elementwise import fused_launcher
except ModuleNotFoundError as error:
    if error.name != "triton":
        raise
    triton = None
    fused_launcher = None


def require_gpu_environment():
    if triton is None:
        raise SystemExit(
            "This benchmark requires Triton on Linux. "
            "Run it on the Linux NVIDIA GPU environment, not macOS."
        )
    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires an available NVIDIA CUDA GPU.")


def pytorch_baseline(x, bias):
    return torch.relu(x * 2 + bias)


def triton_implementation(x, bias, block_size=128):
    y = torch.empty_like(x)
    fused_launcher(x, bias, y, x.numel(), block_size)
    return y


def benchmark_once(n=1_000_000):
    require_gpu_environment()

    x = torch.randn(n, device="cuda", dtype=torch.float32)
    bias = torch.randn(n, device="cuda", dtype=torch.float32)

    pytorch_ms = triton.testing.do_bench(
        lambda: pytorch_baseline(x, bias),
        warmup=25,
        rep=100,
        return_mode="median",
    )

    triton_ms = triton.testing.do_bench(
        lambda: triton_implementation(x, bias),
        warmup=25,
        rep=100,
        return_mode="median",
    )

    speedup = pytorch_ms / triton_ms

    print(f"N: {n:,}")
    print(f"PyTorch: {pytorch_ms:.4f} ms")
    print(f"Triton:  {triton_ms:.4f} ms")
    print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    benchmark_once()
