"""Correctness tests for Project 1.

We will complete this file only after the kernel and launcher work.
"""
import importlib.util

import pytest
import torch

TRITON_AVAILABLE = importlib.util.find_spec("triton") is not None
CUDA_AVAILABLE = torch.cuda.is_available()

if TRITON_AVAILABLE:
    from kernels.fused_elementwise import fused_launcher


def pytorch_baseline(x, bias):
    return torch.relu(x * 2 + bias)


def test_pytorch_baseline():
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0])
    bias = torch.tensor([1.0, 0.5, -1.0, 2.0])

    actual = pytorch_baseline(x, bias)
    expected = torch.tensor([0.0, 0.0, 0.0, 4.0])

    torch.testing.assert_close(actual, expected)

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
@pytest.mark.skipif(
    not (TRITON_AVAILABLE and CUDA_AVAILABLE),
    reason="Triton tests require Triton and a CUDA GPU",
)
def test_fused_elementwise():
    BLOCK_SIZE = 128
    for N in TEST_SIZES:
        x = torch.randn(N, device='cuda', dtype=torch.float32)
        bias = torch.randn(N, device='cuda', dtype=torch.float32)
        y_triton = torch.empty_like(x)
        fused_launcher(x, bias, y_triton, N, BLOCK_SIZE)
        y_torch = pytorch_baseline(x, bias)
        torch.testing.assert_close(y_triton, y_torch)
