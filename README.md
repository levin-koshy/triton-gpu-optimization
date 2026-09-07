# Triton GPU Kernel Optimization

A benchmark-driven project for learning GPU kernel optimization with PyTorch
and Triton. Each kernel follows the same workflow: implement, prove
correctness, benchmark, profile, optimize, and document the results.

## Project 1: fused elementwise kernel

The first kernel fuses this PyTorch expression:

```python
torch.relu(x * 2 + bias)
```

into one Triton kernel. The current implementation includes:

- one-dimensional program mapping;
- masking for tensor sizes that are not multiples of the block size;
- fused loads, arithmetic, ReLU, and output store;
- a minimal Python launcher.

Correctness testing and performance measurements are the next milestones.
Benchmark results will be added only after running on a supported GPU.

## Repository structure

```text
benchmarks/                    Benchmark scripts
kernels/                       Triton kernels and launchers
tests/                         Correctness tests
triton_gpu_optimization_project.md
```

## Hardware

Triton kernel execution and benchmarking require a supported GPU environment.
The source can be reviewed on macOS, but the planned measurements require an
appropriate NVIDIA GPU environment.

## Status

Work in progress. Project 1 is currently entering the correctness-testing
stage.
