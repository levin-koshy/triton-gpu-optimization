"""Correctness tests for Project 1.

We will complete this file only after the kernel and launcher work.
"""
import torch
from kernels.fused_elementwise import fused_launcher

def pytorch_baseline(x, bias):
    return torch.relu(x * 2 + bias)

TEST_SIZES = [
    1,
    31,
    32,
    33,
    127,
    128,
    129,
    1000,
    1023,
    1024,
    1025,
    10_000,
]


# TODO(Stage 6): compare the Triton result with the PyTorch baseline for each
# size above, including sizes that are not multiples of BLOCK_SIZE.
def test_fused_elementwise():
    BLOCK_SIZE = 128
    for N in TEST_SIZES:
        x = torch.randn(N, device='cuda', dtype=torch.float32)
        bias = torch.randn(N, device='cuda', dtype=torch.float32)
        y_triton = torch.empty_like(x)
        fused_launcher(x, bias, y_triton, N, BLOCK_SIZE)
        y_torch = pytorch_baseline(x, bias)
        assert torch.allclose(y_triton, y_torch, atol=1e-6), f"Mismatch for N={N}"
