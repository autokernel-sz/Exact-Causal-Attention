import torch.cuda
from torch.utils.cpp_extension import load_inline

torch.cuda.set_device(0)

def evaluator_qk(code):
    """
    Report measured runtime of the python inline compiled cuda c kernel
    The runtime is measured on an RTX4090 GPU with L x d matrices, average over 100 runs.
    """

    import time
    import torch

    
    qk_cpp_wrapper = r"""
    #include <torch/extension.h>

    torch::Tensor qk(torch::Tensor Q, torch::Tensor K);
    """

    arch = f"sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}"

    try:
        qk_ops = load_inline(
            name="qk_ops",
            cpp_sources=qk_cpp_wrapper,
            cuda_sources=code,
            functions=["qk"],
            verbose=True,
            extra_cuda_cflags=[f"-arch={arch}", "--use_fast_math", "-std=c++17"]
        )
    except Exception as e:
        print(f"Error loading inline CUDA C code: {e}")
        return 10**6, False

    def fca_qk(Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """Exact causal QK^T for Lxd matrices (lower‑triangular result)."""
        assert Q.device.type == "cuda" and K.device == Q.device, "tensors must be on same CUDA device"
        assert Q.dtype == torch.float32 and K.dtype == torch.float32, "dtype must be float32"
        return qk_ops.qk(Q.contiguous(), K.contiguous())
    
    try:
        L = 8192
        d = 8192
        num_runs = 100
        torch.manual_seed(42)
        # Clear GPU memory once at start
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Create tensors once
        Q = torch.randn(L, d, dtype=torch.float32, device="cuda")
        K = torch.randn(L, d, dtype=torch.float32, device="cuda")

        compiled_matmul = torch.compile(
            lambda Q, K: torch.matmul(Q, K.t()).tril_(),
            backend="inductor",
            mode="max-autotune-no-cudagraphs",
            fullgraph=False,
            dynamic=False,
        )

        # VERIFY CORRECTNESS
        S_true = compiled_matmul(Q, K)
        S_fca = fca_qk(Q, K)

        # if results are wrong, end benchmarking
        if S_true.shape != S_fca.shape:
            print(f"Shape mismatch: expected {S_true.shape}, got {S_fca.shape}")
            return 10**6, False
        
        if (S_true - S_fca).abs().max().item() > 5e-3:
            print(f"Results mismatch: max error {((S_true - S_fca).abs()).max().item()}")
            print("Results are not equal, ending benchmarking.")
            return 10**6, False
    
        # otherwise continue benchmarking
        # warmup
        print("Warming up...")
        for _ in range(10):
            compiled_matmul(Q, K)
            fca_qk(Q, K)
        torch.cuda.synchronize()
        
        # Benchmark functions
        print("Benchmarking...")
        results = {}
        for name, func in [
            ("torch.matmul", lambda: compiled_matmul(Q, K)),
            ("fca_qk", lambda: fca_qk(Q, K)),
        ]:
            times = []
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            for _ in range(num_runs):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                # Ensure GPU is ready
                torch.cuda.synchronize()
                
                start_event.record()
                func()
                end_event.record()
                
                # Wait for GPU
                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event))
        
            results[name] = times

        for name, times in results.items():
            mean_time = sum(times) / len(times)
            print(f"{name:20s}{mean_time:12.3f}{min(times):12.3f}{max(times):12.3f}")

        torch.cuda.synchronize()
        
        return sum(results['fca_qk']) / len(results['fca_qk']), True
    
    except Exception as e:
        print(f"Error during correctness check: {e}")
        return 10**6, False


torch.cuda.empty_cache()

kernel_qk = r"""
#include <torch/extension.h>
#include <vector>

// -----------------------------------------------------------------------------
// slice a (BR × BC) tile out of an (L' × d') matrix, using a 4×4 grid
// -----------------------------------------------------------------------------
static inline torch::Tensor get_block(const torch::Tensor& M,
                                      int row_block,
                                      int col_block,
                                      const int64_t BR,
                                      const int64_t BC) {
    const int64_t rs = row_block * BR;
    const int64_t cs = col_block * BC;
    return M.slice(/*dim=*/0, /*start=*/rs, /*end=*/rs + BR)
            .slice(/*dim=*/1, /*start=*/cs, /*end=*/cs + BC);
}

static inline void causal_mask_inplace(torch::Tensor& T) { T.tril_(); }

// build a linear combination of tiles directly into `dst`
static inline void write_combo(
        torch::Tensor dst,
        std::initializer_list<std::pair<torch::Tensor, float>> terms) {
    dst.zero_();
    for (const auto& p : terms) { dst.add_(p.first, p.second); }
}

// -----------------------------------------------------------------------------
//                                MAIN KERNEL
// -----------------------------------------------------------------------------
// Computes S = tril(Q @ K^T) for any L×d using the same 4×4 schedule.
// If L or d are not divisible by 4, zero-pad to L'=ceil(L/4)*4, d'=ceil(d/4)*4.
// Returns the top-left L×L crop of the padded result.
torch::Tensor qk(torch::Tensor Q, torch::Tensor K) {
    // -------- sanity checks ---------------------------------------------------
    TORCH_CHECK(Q.device().is_cuda(), "Q must be CUDA");
    TORCH_CHECK(K.device().is_cuda(), "K must be CUDA");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32 &&
                K.scalar_type() == torch::kFloat32, "dtype must be FP32");
    TORCH_CHECK(Q.sizes() == K.sizes(), "Q and K must have same shape");
    TORCH_CHECK(Q.dim() == 2, "Q and K must be 2D (L x d)");

    constexpr int NBATCH_TOTAL = 34;  // (m1..m24) + (r2,r3) + (r1 split=4) + (r4 split=4)

    const auto opts = Q.options();
    const int64_t L  = Q.size(0);
    const int64_t d  = Q.size(1);
    const int64_t Lp = ((L + 3) / 4) * 4;   // pad to multiple of 4
    const int64_t dp = ((d + 3) / 4) * 4;   // pad to multiple of 4
    const int64_t BR = Lp / 4;              // rows per tile
    const int64_t BC = dp / 4;              // cols per tile (shared "k" for block GEMM)

    // zero-pad Q, K if needed so each tile is uniform (BR × BC)
    torch::Tensor Qw, Kw;
    if (Lp != L || dp != d) {
        Qw = torch::zeros({Lp, dp}, opts);
        Kw = torch::zeros({Lp, dp}, opts);
        Qw.slice(0, 0, L).slice(1, 0, d).copy_(Q);
        Kw.slice(0, 0, L).slice(1, 0, d).copy_(K);
    } else {
        Qw = Q;
        Kw = K;
    }

    // -------- slice Q & K into 16 tiles (4×4 grid) ----------------------------
    torch::Tensor Qb[4][4], Kb[4][4];
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) {
            Qb[r][c] = get_block(Qw, r, c, BR, BC);  // (BR × BC)
            Kb[r][c] = get_block(Kw, r, c, BR, BC);  // (BR × BC)
        }
#define Q_(idx) Qb[(idx - 1) / 4][(idx - 1) % 4]
#define K_(idx) Kb[(idx - 1) / 4][(idx - 1) % 4]

    // -------- workspaces ------------------------------------------------------
    auto Abatch = torch::empty({NBATCH_TOTAL, BR, BC}, opts);  // (N, BR, BC)
    auto Bbatch = torch::empty({NBATCH_TOTAL, BR, BC}, opts);  // (N, BR, BC)

    int s = 0;   // cursor for batch

    // -------------------------- m₁ … m₂₄ -------------------------------------
    {   // m1
        auto Q8 = Q_(8), Q11 = Q_(11);
        auto K3 = K_(3), K2 = K_(2), K4 = K_(4), K8 = K_(8);
        write_combo(Abatch[s], {{Q8, 1.f}, {Q11, 1.f}});
        write_combo(Bbatch[s], {{K3, 1.f}, {K2, -1.f}, {K4, -1.f}, {K8, 1.f}}); ++s;
    }
    {   // m2 
        auto Q15 = Q_(15), Q5 = Q_(5);
        auto K1 = K_(1), K5 = K_(5), K6 = K_(6), K7 = K_(7);
        write_combo(Abatch[s], {{Q15, 1.f}, {Q5, 1.f}});
        write_combo(Bbatch[s], {{K1, 1.f}, {K5, -1.f}, {K6, -1.f}, {K7, 1.f}}); ++s;
    }
    {   // m3
        auto Q10 = Q_(10), Q16 = Q_(16), Q12 = Q_(12);
        auto K12 = K_(12), K2 = K_(2);
        write_combo(Abatch[s], {{Q10, -1.f}, {Q16, 1.f}, {Q12, 1.f}});
        write_combo(Bbatch[s], {{K12, 1.f}, {K2, -1.f}}); ++s;
    }
    {   // m4
        auto Q13 = Q_(13), Q9 = Q_(9), Q14 = Q_(14);
        auto K9 = K_(9), K6 = K_(6);
        write_combo(Abatch[s], {{Q13, 1.f}, {Q9, 1.f}, {Q14, -1.f}});
        write_combo(Bbatch[s], {{K9, 1.f}, {K6, -1.f}}); ++s;
    }
    {   // m5
        auto Q6 = Q_(6), Q15 = Q_(15), Q7 = Q_(7);
        auto K2 = K_(2), K11 = K_(11);
        write_combo(Abatch[s], {{Q6, -1.f}, {Q15, 1.f}, {Q7, -1.f}});
        write_combo(Bbatch[s], {{K2, 1.f}, {K11, 1.f}}); ++s;
    }
    {   // m6
        auto Q6 = Q_(6), Q7 = Q_(7), Q11 = Q_(11);
        auto K6 = K_(6), K11 = K_(11);
        write_combo(Abatch[s], {{Q6, 1.f}, {Q7, 1.f}, {Q11, -1.f}});
        write_combo(Bbatch[s], {{K6, 1.f}, {K11, 1.f}}); ++s;
    }
    {   // m7
        auto Q6 = Q_(6), Q7 = Q_(7);
        auto K11 = K_(11);
        write_combo(Abatch[s], {{Q6, 1.f}, {Q7, 1.f}});
        write_combo(Bbatch[s], {{K11, 1.f}}); ++s;
    }
    {   // m8 
        auto Q14 = Q_(14), Q10 = Q_(10), Q6 = Q_(6), Q15 = Q_(15);
        auto Q7 = Q_(7), Q16 = Q_(16), Q12 = Q_(12);
        auto K2 = K_(2);
        write_combo(Abatch[s], {{Q14, -1.f}, {Q10, -1.f}, {Q6, 1.f}, 
                                    {Q15, -1.f}, {Q7, 1.f}, {Q16, 1.f}, {Q12, 1.f}});
        write_combo(Bbatch[s], {{K2, 1.f}}); ++s;
    }
    {   // m9
        auto Q13 = Q_(13), Q9 = Q_(9), Q14 = Q_(14);
        auto Q10 = Q_(10), Q6 = Q_(6), Q7 = Q_(7), Q11 = Q_(11);
        auto K6 = K_(6);
        write_combo(Abatch[s], {{Q13, 1.f}, {Q9, 1.f}, {Q14, -1.f}, 
                                    {Q10, -1.f}, {Q6, 1.f}, {Q7, 1.f}, {Q11, -1.f}});
        write_combo(Bbatch[s], {{K6, 1.f}}); ++s;
    }
    {   // m10
        auto Q11 = Q_(11);
        auto K2 = K_(2), K3 = K_(3), K7 = K_(7);
        auto K11 = K_(11), K4 = K_(4), K8 = K_(8);
        write_combo(Abatch[s], {{Q11, 1.f}});
        write_combo(Bbatch[s], {{K2, 1.f}, {K3, -1.f}, {K7, 1.f}, 
                                    {K11, 1.f}, {K4, 1.f}, {K8, -1.f}}); ++s;
    }
    {   // m11
        auto Q5 = Q_(5);
        auto K5 = K_(5), K6 = K_(6), K7 = K_(7);
        write_combo(Abatch[s], {{Q5, 1.f}});
        write_combo(Bbatch[s], {{K5, 1.f}, {K6, 1.f}, {K7, -1.f}}); ++s;
    }
    {   // m12
        auto Q8 = Q_(8);
        auto K2 = K_(2), K3 = K_(3), K4 = K_(4);
        write_combo(Abatch[s], {{Q8, 1.f}});
        write_combo(Bbatch[s], {{K2, 1.f}, {K3, -1.f}, {K4, 1.f}}); ++s;
    }
    {   // m13
        auto Q15 = Q_(15);
        auto K1 = K_(1), K5 = K_(5), K6 = K_(6);
        auto K3 = K_(3), K7 = K_(7), K11 = K_(11);
        write_combo(Abatch[s], {{Q15, 1.f}});
        write_combo(Bbatch[s], {{K1, -1.f}, {K5, 1.f}, {K6, 1.f},
                                    {K3, 1.f}, {K7, -1.f}, {K11, 1.f}}); ++s;
    }
    {   // m14
        auto Q13 = Q_(13), Q9 = Q_(9), Q15 = Q_(15);
        auto K1 = K_(1), K5 = K_(5), K6 = K_(6);
        write_combo(Abatch[s], {{Q13, 1.f}, {Q9, 1.f}, {Q15, 1.f}});
        write_combo(Bbatch[s], {{K1, -1.f}, {K5, 1.f}, {K6, 1.f}}); ++s;
    }
    {   // m15
        auto Q11 = Q_(11), Q16 = Q_(16), Q12 = Q_(12);
        auto K2 = K_(2), K4 = K_(4), K8 = K_(8);
        write_combo(Abatch[s], {{Q11, 1.f}, {Q16, 1.f}, {Q12, 1.f}});
        write_combo(Bbatch[s], {{K2, 1.f}, {K4, 1.f}, {K8, -1.f}}); ++s;
    }
    {   // m16
        auto Q9 = Q_(9), Q16 = Q_(16);
        auto K1 = K_(1), K8 = K_(8);
        write_combo(Abatch[s], {{Q9, 1.f}, {Q16, -1.f}});
        write_combo(Bbatch[s], {{K1, 1.f}, {K8, -1.f}}); ++s;
    }
    {   // m17
        auto Q10 = Q_(10), Q12 = Q_(12);
        auto K12 = K_(12);
        write_combo(Abatch[s], {{Q10, 1.f}, {Q12, -1.f}});
        write_combo(Bbatch[s], {{K12, 1.f}}); ++s;
    }
    {   // m18
        auto Q13 = Q_(13), Q14 = Q_(14);
        auto K9 = K_(9);
        write_combo(Abatch[s], {{Q13, 1.f}, {Q14, -1.f}});
        write_combo(Bbatch[s], {{K9, 1.f}}); ++s;
    }
    {   // m19
        auto Q15 = Q_(15), Q7 = Q_(7), Q8 = Q_(8);
        auto K3 = K_(3), K2 = K_(2);
        write_combo(Abatch[s], {{Q15, -1.f}, {Q7, 1.f}, {Q8, 1.f}});
        write_combo(Bbatch[s], {{K3, 1.f}, {K2, -1.f}}); ++s;
    }
    {   // m20
        auto Q9 = Q_(9);
        auto K5 = K_(5), K9 = K_(9), K8 = K_(8);
        write_combo(Abatch[s], {{Q9, 1.f}});
        write_combo(Bbatch[s], {{K5, 1.f}, {K9, 1.f}, {K8, -1.f}}); ++s;
    }
    {   // m21
        auto Q9 = Q_(9), Q8 = Q_(8), Q12 = Q_(12);
        auto K8 = K_(8);
        write_combo(Abatch[s], {{Q9, 1.f}, {Q8, -1.f}, {Q12, 1.f}});
        write_combo(Bbatch[s], {{K8, 1.f}}); ++s;
    }
    {   // m22
        auto Q13 = Q_(13), Q5 = Q_(5), Q16 = Q_(16);
        auto K1 = K_(1);
        write_combo(Abatch[s], {{Q13, 1.f}, {Q5, -1.f}, {Q16, 1.f}});
        write_combo(Bbatch[s], {{K1, 1.f}}); ++s;
    }
    {   // m23
        auto Q16 = Q_(16);
        auto K1 = K_(1), K4 = K_(4), K12 = K_(12);
        write_combo(Abatch[s], {{Q16, 1.f}});
        write_combo(Bbatch[s], {{K1, -1.f}, {K4, 1.f}, {K12, 1.f}}); ++s;
    }
    {   // m24
        auto Q14 = Q_(14);
        auto K9 = K_(9), K2 = K_(2), K10 = K_(10);
        write_combo(Abatch[s], {{Q14, 1.f}});
        write_combo(Bbatch[s], {{K9, 1.f}, {K2, 1.f}, {K10, 1.f}}); ++s;
    }

    TORCH_INTERNAL_ASSERT(s == 24, "m1–m24 batch index mismatch");

    // -------------------------- r₂, r₃ (still k = BC) -------------------------
    {   // r2
        auto Q5 = Q_(5), Q7 = Q_(7), Q11 = Q_(11);
        auto K6 = K_(6), K7 = K_(7);
        write_combo(Abatch[s], {{Q5, 1.f}, {Q7, 1.f}, {Q11, -1.f}});
        write_combo(Bbatch[s], {{K6, -1.f}, {K7, 1.f}}); ++s;
    }
    {   // r3
        auto Q10 = Q_(10);
        auto K6 = K_(6), K10 = K_(10), K12 = K_(12);
        write_combo(Abatch[s], {{Q10, 1.f}});
        write_combo(Bbatch[s], {{K6, 1.f}, {K10, 1.f}, {K12, 1.f}}); ++s;
    }

    TORCH_INTERNAL_ASSERT(s == 26, "small-batch cursor mismatch");

    // -------------------------- r₁, r₄ (expand to small batches) -------------
    // r1: Expand block (0,0) to 4 small batches
    {   // r1_1
        auto Q1 = Q_(1);  auto K1 = K_(1);
        write_combo(Abatch[s], {{Q1, 1.f}});
        write_combo(Bbatch[s], {{K1, 1.f}}); ++s;
    }
    {   // r1_2
        auto Q2 = Q_(2);  auto K2 = K_(2);
        write_combo(Abatch[s], {{Q2, 1.f}});
        write_combo(Bbatch[s], {{K2, 1.f}}); ++s;
    }
    {   // r1_3
        auto Q3 = Q_(3);  auto K3 = K_(3);
        write_combo(Abatch[s], {{Q3, 1.f}});
        write_combo(Bbatch[s], {{K3, 1.f}}); ++s;
    }
    {   // r1_4
        auto Q4 = Q_(4);  auto K4 = K_(4);
        write_combo(Abatch[s], {{Q4, 1.f}});
        write_combo(Bbatch[s], {{K4, 1.f}}); ++s;
    }

    // r4: Expand block (3,3) to 4 small batches
    {   // r4_1
        auto Q13 = Q_(13); auto K13 = K_(13);
        write_combo(Abatch[s], {{Q13, 1.f}});
        write_combo(Bbatch[s], {{K13, 1.f}}); ++s;
    }
    {   // r4_2
        auto Q14 = Q_(14); auto K14 = K_(14);
        write_combo(Abatch[s], {{Q14, 1.f}});
        write_combo(Bbatch[s], {{K14, 1.f}}); ++s;
    }
    {   // r4_3
        auto Q15 = Q_(15); auto K15 = K_(15);
        write_combo(Abatch[s], {{Q15, 1.f}});
        write_combo(Bbatch[s], {{K15, 1.f}}); ++s;
    }
    {   // r4_4
        auto Q16 = Q_(16); auto K16 = K_(16);
        write_combo(Abatch[s], {{Q16, 1.f}});
        write_combo(Bbatch[s], {{K16, 1.f}}); ++s;
    }

    TORCH_INTERNAL_ASSERT(s == NBATCH_TOTAL, "Total batch size mismatch");

    // -------- batched GEMMs (PyTorch maps to cuBLAS) -------------------------
    // Abatch: (N, BR, BC), Bbatch^T: (N, BC, BR) -> C: (N, BR, BR)
    auto C = torch::bmm(Abatch, Bbatch.transpose(1, 2));

    // -------- unpack ----------------------------------------------------------
    torch::Tensor Sblk[4][4];

    // r1 (split into 4)
    Sblk[0][0] = torch::zeros({BR, BR}, opts);
    Sblk[0][0].add_(C[26], 1.0f);
    Sblk[0][0].add_(C[27], 1.0f);
    Sblk[0][0].add_(C[28], 1.0f);
    Sblk[0][0].add_(C[29], 1.0f);
    causal_mask_inplace(Sblk[0][0]);

    // m1..m24 indexes: m1=C[0], …, m24=C[23]; r2=C[24], r3=C[25];
    // r1 parts C[26..29], r4 parts C[30..33].

    // Sblk[1][0]
    Sblk[1][0] = C[1].clone();
    Sblk[1][0].add_(C[4], -1.0f);
    Sblk[1][0].add_(C[6], -1.0f);
    Sblk[1][0].add_(C[10], 1.0f);
    Sblk[1][0].add_(C[11], 1.0f);
    Sblk[1][0].add_(C[12], 1.0f);
    Sblk[1][0].add_(C[18], 1.0f);

    // Sblk[1][1]
    Sblk[1][1] = C[0].clone();
    Sblk[1][1].add_(C[5], 1.0f);
    Sblk[1][1].add_(C[6], -1.0f);
    Sblk[1][1].add_(C[9], 1.0f);
    Sblk[1][1].add_(C[10], 1.0f);
    Sblk[1][1].add_(C[11], 1.0f);
    Sblk[1][1].add_(C[24], 1.0f);
    causal_mask_inplace(Sblk[1][1]);

    // Sblk[2][0]
    Sblk[2][0] = C[0].clone();
    Sblk[2][0].add_(C[2], 1.0f);
    Sblk[2][0].add_(C[11], 1.0f);
    Sblk[2][0].add_(C[14], 1.0f);
    Sblk[2][0].add_(C[15], 1.0f);
    Sblk[2][0].add_(C[16], 1.0f);
    Sblk[2][0].add_(C[20], 1.0f);
    Sblk[2][0].add_(C[22], -1.0f);

    // Sblk[2][1]
    Sblk[2][1] = C[0].clone();
    Sblk[2][1].add_(C[3], -1.0f);
    Sblk[2][1].add_(C[5], 1.0f);
    Sblk[2][1].add_(C[6], -1.0f);
    Sblk[2][1].add_(C[8], -1.0f);
    Sblk[2][1].add_(C[9], 1.0f);
    Sblk[2][1].add_(C[11], 1.0f);
    Sblk[2][1].add_(C[17], 1.0f);
    Sblk[2][1].add_(C[19], 1.0f);
    Sblk[2][1].add_(C[20], 1.0f);

    // Sblk[2][2]
    Sblk[2][2] = C[3].clone();
    Sblk[2][2].add_(C[5], -1.0f);
    Sblk[2][2].add_(C[6], 1.0f);
    Sblk[2][2].add_(C[8], 1.0f);
    Sblk[2][2].add_(C[16], -1.0f);
    Sblk[2][2].add_(C[17], -1.0f);
    Sblk[2][2].add_(C[25], 1.0f);
    causal_mask_inplace(Sblk[2][2]);

    // Sblk[3][0]
    Sblk[3][0] = C[1].clone();
    Sblk[3][0].add_(C[2], -1.0f);
    Sblk[3][0].add_(C[4], -1.0f);
    Sblk[3][0].add_(C[6], -1.0f);
    Sblk[3][0].add_(C[7], -1.0f);
    Sblk[3][0].add_(C[10],  1.0f);
    Sblk[3][0].add_(C[12],  1.0f);
    Sblk[3][0].add_(C[16], -1.0f);
    Sblk[3][0].add_(C[21],  1.0f);
    Sblk[3][0].add_(C[22],  1.0f);

    // Sblk[3][1]
    Sblk[3][1] = C[1].clone();
    Sblk[3][1].add_(C[3],  1.0f);
    Sblk[3][1].add_(C[10], 1.0f);
    Sblk[3][1].add_(C[13], 1.0f);
    Sblk[3][1].add_(C[15], 1.0f);
    Sblk[3][1].add_(C[17], -1.0f);
    Sblk[3][1].add_(C[19], -1.0f);
    Sblk[3][1].add_(C[21], 1.0f);

    // Sblk[3][2]
    Sblk[3][2] = C[2].clone();
    Sblk[3][2].add_(C[4],  1.0f);
    Sblk[3][2].add_(C[6],  1.0f);
    Sblk[3][2].add_(C[7],  1.0f);
    Sblk[3][2].add_(C[16], 1.0f);
    Sblk[3][2].add_(C[17], 1.0f);
    Sblk[3][2].add_(C[23], 1.0f);

    // r4 (split into 4)
    Sblk[3][3] = torch::zeros({BR, BR}, opts);
    Sblk[3][3].add_(C[30], 1.0f);
    Sblk[3][3].add_(C[31], 1.0f);
    Sblk[3][3].add_(C[32], 1.0f);
    Sblk[3][3].add_(C[33], 1.0f);
    causal_mask_inplace(Sblk[3][3]);

    // -------- stitch into L'×L' output and crop to L×L -----------------------
    auto Sfull = torch::zeros({Lp, Lp}, opts);
    for (int br = 0; br < 4; ++br)
        for (int bc = 0; bc <= br; ++bc)
            Sfull.slice(0, br*BR, (br+1)*BR)
                 .slice(1, bc*BR, (bc+1)*BR)
                 .copy_(Sblk[br][bc]);

    // Crop away any padding in sequence dimension
    if (Lp != L) {
        return Sfull.slice(0, 0, L).slice(1, 0, L);
    }
    return Sfull;
}
"""

# python3 bench_qk.py
metric, _ = evaluator_qk(kernel_qk)