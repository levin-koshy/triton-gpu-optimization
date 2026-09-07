"""Project 1: fused elementwise Triton kernel.

Complete this file one stage at a time. The first task is program mapping.
"""

import triton
import triton.language as tl


@triton.jit
def fused_kernel(
    x_ptr,
    bias_ptr,
    y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # Stage 1: map this Triton program to a block of element indices.
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Stage 2: load x and bias. We will add this after reviewing Stage 1.
    x = tl.load(x_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets, mask=mask)

    # Stage 3: compute x * 2 + bias and apply ReLU.
    values = x * 2.0 + bias
    y = tl.maximum(values, 0.0)

    # Stage 4: store the result using the mask.
    tl.store(y_ptr + offsets, y, mask=mask)


# Stage 5: add the Python launcher after the kernel is complete.
def fused_launcher(x, bias, y, N, BLOCK_SIZE):
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    fused_kernel[grid](x, bias, y, N, BLOCK_SIZE=BLOCK_SIZE)
    return y
