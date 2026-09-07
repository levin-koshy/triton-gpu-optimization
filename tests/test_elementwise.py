"""Correctness tests for Project 1.

We will complete this file only after the kernel and launcher work.
"""

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
