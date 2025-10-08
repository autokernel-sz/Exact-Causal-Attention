
# Fast-Causal-Attention
This repository contains CUDA C kernels for Fast Causal Attention (FCA) from the paper "Exact Causal Attention with 10% Fewer Operations" by autokernel.ai, in collaboration with Shenzhen Research Institute of Big Data (SRIBD). This codebase verifies speed, accuracy, and correctness of the proposed algorithms.

# Quickstart
Requirements:
- torch2.7.0+
- cu126+
- NVidia GPU (e.g. RTX4090)

The file bench_qk.py contains CUDA code kernel_qk and benchmarking function evaluator_qk. Running the code (preferably on RTX4090) verifies that:
1. the output of the algorithm is the same as Causal Attention scores Mask(QK^T)
2. the runtime is 20% faster than compiled torch.matmul(Q, K.t()).tril_() for shapes 8192 x 8192
```bash
    python3 bench_qk.py
```

The file bench_pv.py contains CUDA code kernel_pv and benchmarking function evaluator_pv. Running the code (preferably on RTX4090) verifies that
1. the output of the algorithm is the same as multiplication of lower-triangular matrix P by matrix V
2. the runtime matches compiled torch.matmul(P, V) for shapes 8192 x 8192
```bash
    python3 bench_pv.py
```

# FLOP Count, Precision, Memory Traffic
For algorithm analysis consult the original paper: [https://arxiv.org/abs/2510.05175](https://arxiv.org/abs/2510.05175). Kernels were designed in an automated way without human involvement. All experiments are reported for RTX4090 GPU with FP32.


# License
Copyright (c) 2025 autokernel.ai All rights reserved.
No permission is granted to use, copy, modify, sublicense, or distribute this software or any portion thereof.
