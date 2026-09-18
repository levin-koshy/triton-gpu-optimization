"""Benchmark the Project 1 PyTorch and Triton implementations."""
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
    """Run the reference operation on input tensor x and same-shaped bias."""
    return torch.relu(x * 2 + bias)


def triton_implementation(x, bias, block_size=128):
    """Run the fused kernel, processing block_size elements per Triton program."""
    y = torch.empty_like(x)
    # x, bias: input tensors; y: output tensor; x.numel(): total element count.
    # block_size: number of elements assigned to each Triton program instance.
    fused_launcher(x, bias, y, x.numel(), block_size)
    return y


def benchmark_once(n=1_000_000):
    """Benchmark both implementations using n float32 elements."""
    require_gpu_environment()

    x = torch.randn(n, device="cuda", dtype=torch.float32)
    bias = torch.randn(n, device="cuda", dtype=torch.float32)

    pytorch_ms = triton.testing.do_bench(
        # fn: zero-argument callable containing the operation being measured.
        lambda: pytorch_baseline(x, bias),
        # warmup: milliseconds spent warming caches and compiling before timing.
        warmup=25,
        # rep: milliseconds spent collecting timed repetitions.
        rep=100,
        # return_mode: statistic returned from the collected timings.
        return_mode="median",
    )

    triton_ms = triton.testing.do_bench(
        # Use the same timing parameters so the comparison is fair.
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
