# Triton GPU Kernel Optimization Project

## Project Goal

Build a real, benchmark-driven GPU kernel optimization project using **PyTorch + Triton**.

The purpose is not just to learn Triton syntax. The goal is to develop the practical skills needed to:

- Understand how GPU kernels execute.
- Write Triton kernels from scratch.
- Reason about memory access, coalescing, tiling, registers, warps, and occupancy.
- Profile GPU workloads and identify bottlenecks.
- Optimize kernels based on evidence rather than intuition.
- Use benchmarking and correctness testing rigorously.
- Autotune kernel configurations.
- Produce a polished GitHub project that demonstrates GPU/ML systems engineering ability.
- Eventually progress toward more sophisticated ML kernels and open-source contributions.

This project should be treated like a small real-world performance engineering project.

---

# Overall Roadmap

```text
PyTorch baseline
       ↓
Naive Triton kernel
       ↓
Correctness tests
       ↓
Benchmark
       ↓
Profile
       ↓
Identify bottleneck
       ↓
Optimize
       ↓
Benchmark again
       ↓
Autotune
       ↓
Document results
       ↓
GitHub portfolio project
       ↓
More advanced ML kernels
       ↓
Potential open-source contribution
```

---

# Phase 1 — Fused Elementwise Kernel

## Baseline

Start with this PyTorch operation:

```python
def baseline(x, bias):
    return torch.relu(x * 2 + bias)
```

Conceptually, the unfused computation may involve multiple operations:

```text
x
 ↓
multiply by 2
 ↓
temporary
 ↓
add bias
 ↓
temporary
 ↓
ReLU
 ↓
y
```

The Triton version should fuse the operations into one GPU kernel:

```text
        ┌───────────────┐
x ─────→│               │
bias ──→│ Fused Kernel  │──→ y
        │               │
        │ x * 2 + bias  │
        │ ReLU          │
        └───────────────┘
```

The objective is to reduce:

- Kernel launch overhead.
- Intermediate global-memory traffic.
- Unnecessary reads/writes of temporary tensors.

---

# Phase 2 — Understand Triton Program Mapping

For:

```text
N = 1000
BLOCK_SIZE = 128
```

The number of programs is:

```text
ceil(1000 / 128) = 8
```

Each normal program handles 128 elements.

The final program handles:

```text
1000 - 7 × 128 = 104 valid elements
```

and has:

```text
128 - 104 = 24 invalid lanes
```

The kernel therefore needs masking.

Basic mapping:

```python
pid = tl.program_id(0)

offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

mask = offsets < N
```

The mask protects against out-of-bounds memory accesses.

---

# Phase 3 — Write the Triton Kernel

Target structure:

```python
@triton.jit
def fused_kernel(
    x_ptr,
    bias_ptr,
    y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets, mask=mask)

    y = x * 2 + bias
    y = tl.maximum(y, 0)

    tl.store(y_ptr + offsets, y, mask=mask)
```

The launch configuration should be determined separately.

For example:

```python
grid = lambda meta: (
    triton.cdiv(N, meta["BLOCK_SIZE"]),
)
```

---

# Phase 4 — Correctness

Before optimizing anything, prove that the Triton kernel produces the same result as PyTorch.

Test multiple tensor sizes, including sizes that are not multiples of the block size.

Examples:

```text
1
31
32
33
127
128
129
1000
1023
1024
1025
10000
```

Compare:

```python
torch.testing.assert_close(
    triton_output,
    pytorch_output,
)
```

Test multiple dtypes where appropriate.

Correctness comes before performance.

---

# Phase 5 — Benchmarking

Create a reliable benchmark comparing:

```text
PyTorch baseline
vs
Triton implementation
```

Important principles:

1. Warm up the kernels.
2. Account for CUDA's asynchronous execution.
3. Use GPU-aware timing.
4. Run multiple iterations.
5. Report useful statistics such as median/minimum time.
6. Test multiple tensor sizes.
7. Verify correctness before reporting performance.

Measure:

```text
Latency
Throughput
Speedup
```

Speedup:

```text
baseline_time / optimized_time
```

Example:

```text
PyTorch:  20 µs
Triton:   12 µs

Speedup = 20 / 12 = 1.67×
```

Do not optimize based on a single timing measurement.

---

# Phase 6 — Profile

Once the baseline Triton kernel works, profile it.

The objective is to answer:

> What is actually limiting performance?

Potential bottlenecks:

### Memory-bound

Symptoms may include:

- High memory bandwidth utilization.
- Relatively low compute utilization.
- Performance dominated by global-memory traffic.

Investigate:

- Coalescing.
- Unnecessary loads/stores.
- Data reuse.
- Kernel fusion.
- Block size.

### Compute-bound

Symptoms may include:

- High compute utilization.
- Lower memory bandwidth utilization.
- Arithmetic dominates execution time.

Investigate:

- Instruction efficiency.
- Warps.
- Tile sizes.
- Specialized hardware utilization.

Do not assume the bottleneck.

Measure it.

---

# Phase 7 — Optimization Experiments

Run controlled experiments.

Potential parameters:

```text
BLOCK_SIZE
num_warps
num_stages
```

For each experiment:

```text
Change ONE important variable
        ↓
Benchmark
        ↓
Record result
        ↓
Explain why performance changed
```

Create a table such as:

| Configuration | Latency | Speedup |
|---|---:|---:|
| PyTorch | ... | 1.00× |
| Triton BLOCK_SIZE=64 | ... | ... |
| Triton BLOCK_SIZE=128 | ... | ... |
| Triton BLOCK_SIZE=256 | ... | ... |
| Triton BLOCK_SIZE=512 | ... | ... |

The goal is to connect the performance result to GPU architecture.

---

# Phase 8 — Kernel Fusion Analysis

Explain why fusion can improve performance.

Unfused:

```text
Kernel 1:
x → x*2 → temporary

Kernel 2:
temporary + bias → temporary

Kernel 3:
ReLU → output
```

Fused:

```text
One kernel:
load x
load bias
compute x*2 + bias
compute ReLU
store output
```

The fused version can reduce intermediate global-memory traffic and kernel launches.

But fusion is not automatically better.

Too much fusion can increase:

- Register pressure.
- Resource usage.
- Kernel complexity.

The project should demonstrate both the benefit and the trade-off.

---

# Phase 9 — Autotuning

After manually understanding the important parameters, introduce Triton autotuning.

Example concept:

```python
configs = [
    triton.Config(
        {"BLOCK_SIZE": 64},
        num_warps=2,
    ),
    triton.Config(
        {"BLOCK_SIZE": 128},
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_SIZE": 256},
        num_warps=4,
    ),
]
```

Use autotuning to determine which configuration performs best for different workload sizes.

Important principle:

> Autotuning should be informed by understanding, not used as a substitute for understanding.

---

# Phase 10 — Move to More Difficult Kernels

After the fused elementwise kernel, progressively increase difficulty.

Recommended progression:

```text
1. Fused elementwise
       ↓
2. Reduction
       ↓
3. Softmax
       ↓
4. Matrix multiplication
       ↓
5. Optimized + autotuned matmul
       ↓
6. RMSNorm / LayerNorm
       ↓
7. SwiGLU / Transformer-style fused operation
       ↓
8. More realistic ML inference/training kernels
```

For each kernel, repeat the same engineering cycle:

```text
Understand
   ↓
PyTorch baseline
   ↓
Triton implementation
   ↓
Correctness
   ↓
Benchmark
   ↓
Profile
   ↓
Optimize
   ↓
Autotune
   ↓
Document
```

---

# Concepts We Should Continue Reinforcing

The project should continuously connect code to these concepts:

## GPU execution

- GPU
- SM
- Warp
- Thread
- Triton program
- Program ID
- `num_warps`

## Memory hierarchy

- Registers
- Shared memory
- L1 cache
- L2 cache
- Global memory / VRAM

## Memory performance

- Memory bandwidth
- Memory latency
- Coalescing
- Strides
- Data reuse
- Global-memory traffic

## Kernel design

- Tiling
- Blocking
- Masking
- Kernel fusion
- Reductions
- Pointer arithmetic

## Compute

- CUDA cores
- Tensor Cores
- `tl.dot`
- FP32
- FP16
- BF16
- TF32 where relevant

## Performance

- Occupancy
- Register pressure
- Warp count
- Pipeline stages
- Arithmetic intensity
- Roofline model

## Engineering

- Correctness
- Benchmark methodology
- Profiling
- Autotuning
- Regression testing
- Reproducibility

---

# Final Portfolio Goal

The final repository should not look like a collection of random Triton exercises.

It should tell a performance-engineering story.

Suggested structure:

```text
triton-gpu-optimization/
│
├── README.md
├── benchmarks/
│   ├── benchmark_elementwise.py
│   ├── benchmark_softmax.py
│   └── benchmark_matmul.py
│
├── kernels/
│   ├── fused_elementwise.py
│   ├── softmax.py
│   ├── matmul.py
│   └── ...
│
├── tests/
│   ├── test_elementwise.py
│   ├── test_softmax.py
│   └── test_matmul.py
│
├── profiling/
│   └── ...
│
└── docs/
    ├── elementwise.md
    ├── softmax.md
    └── matmul.md
```

Each optimization should document:

```text
Problem
↓
Baseline
↓
Initial implementation
↓
Profiling result
↓
Bottleneck
↓
Optimization
↓
Benchmark
↓
Performance improvement
↓
Trade-offs
```

---

# Definition of Success

This project is successful if, by the end, we can demonstrate that we can:

1. Read a PyTorch operation and identify opportunities for kernel optimization.
2. Write a Triton kernel without relying entirely on generated code.
3. Reason about program IDs, blocks, warps, strides, and memory access.
4. Correctly handle arbitrary tensor sizes with masking.
5. Benchmark GPU kernels correctly.
6. Profile kernels and identify whether they are memory- or compute-bound.
7. Optimize based on measured bottlenecks.
8. Understand how tile size, warps, registers, and pipeline stages interact.
9. Use Triton autotuning effectively.
10. Explain performance improvements technically and quantitatively.
11. Produce a polished GitHub repository demonstrating GPU/ML systems engineering ability.

---

# Current Starting Point

We are currently at:

**Project 1 — Fused Elementwise Kernel**

PyTorch:

```python
def baseline(x, bias):
    return torch.relu(x * 2 + bias)
```

The immediate next steps are:

```text
1. Implement the Triton kernel
2. Build the launcher
3. Test correctness
4. Benchmark against PyTorch
5. Profile
6. Optimize BLOCK_SIZE / warps
7. Record results
8. Understand why the optimization worked
```

Do not skip directly to optimization.

The objective is to learn the complete GPU optimization workflow, not merely produce a faster kernel.

---

# Environment Note

Actual Triton GPU benchmarking requires a supported GPU environment.

If development is being done from a Mac, the code can still be written and reviewed locally, but actual NVIDIA Triton performance measurements require access to an appropriate NVIDIA GPU, such as a remote workstation or cloud GPU.

---

# Working Style for This Project

Proceed interactively.

For each stage:

1. Explain the concept briefly.
2. Give the smallest useful coding task.
3. Let me implement or answer.
4. Review my answer/code.
5. Correct misunderstandings.
6. Run tests/benchmarks when a GPU environment is available.
7. Only then move to the next optimization.

Avoid jumping ahead.

The goal is to build genuine intuition about GPU performance while producing a real portfolio-quality project.
