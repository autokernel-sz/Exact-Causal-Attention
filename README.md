# Fast-Causal-Attention
This repository contains CUDA C kernels for Fast Causal Attention (FCA) from the paper "Exact Attention Using 10% Fewer Operations" by autokernel.ai, in collaboration with Shenzhen Research Institute of Big Data (SRIBD), Moonshot.ai. This codebase verifies speed, accuracy, and correctness of the proposed algorithms.

# Quickstart
Requirements:
- PyTorch with CUDA
- NVidia GPU (e.g. RTX4090)

The file bench_qk.py contains CUDA code kernel_qk and benchmarking function evaluator_qk. Running the code (preferably on RTX4090) verifies that
1. the output of the algorithm is correct
2. the runtime is 20% faster than torch.matmul(Q, K.t()).tril_() for shapes 8192 x 8192
```bash
    python3 bench_qk.py
```

The file bench_pv.py contains CUDA code kernel_pv and benchmarking function evaluator_pv. Running the code (preferably on RTX4090) verifies that
1. the output of the algorithm is correct
2. the runtime is <TBD> faster than torch.matmul(P, V) for shapes 8192 x 8192
```bash
    python3 bench_pv.py
```

# FLOP Count, Precision, Memory Traffic
For algorithm analysis consult the original paper: <arxiv link here>. Kernels were designed in an automated way without human involvement. All experiments are reported for RTX4090 GPU with FP32.


# License
Copyright (c) 2025 autokernel.ai All rights reserved.
No permission is granted to use, copy, modify, sublicense, or distribute this software or any portion thereof.
