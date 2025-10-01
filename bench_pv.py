import torch.cuda
from torch.utils.cpp_extension import load_inline

torch.cuda.set_device(0)

def evaluator_pv(code):
    """
    Report measured runtime of the python inline compiled cuda c kernel
    The runtime is measured on an RTX4090 GPU with L x L matrix P and L x d matrix V, average over 100 runs.
    """

    import time
    import torch

    
    pv_cpp_wrapper = r"""
    #include <torch/extension.h>

    torch::Tensor pv(torch::Tensor P, torch::Tensor V);
    """

    arch = f"sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}"

    try:
        pv_ops = load_inline(
            name="pv_ops",
            cpp_sources=pv_cpp_wrapper,
            cuda_sources=code,
            functions=["pv"],
            verbose=True,
            extra_cuda_cflags=[f"-arch={arch}", "--use_fast_math", "-std=c++17"]
        )
    except Exception as e:
        print(f"Error loading inline CUDA C code: {e}")
        return 10**6, False

    def fca_pv(P: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Exact PV for LxL and Lxd matrices (lower-triangular P)."""
        assert P.device.type == "cuda" and V.device == P.device, "tensors must be on same CUDA device"
        assert V.dtype == torch.float32 and V.dtype == torch.float32, "dtype must be float32"
        return pv_ops.pv(P.contiguous(), V.contiguous())
    
    try:
        L = 8192
        d = 8192
        num_runs = 100
        torch.manual_seed(42)
        # Clear GPU memory once at start
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Create tensors once
        P = torch.randn(L, L, dtype=torch.float32, device="cuda")
        P.tril_()  # make P lower-triangular
        V = torch.randn(L, d, dtype=torch.float32, device="cuda")

        compiled_matmul = torch.compile(
            lambda P, V: torch.matmul(P, V),
            backend="inductor",
            mode="max-autotune-no-cudagraphs",
            fullgraph=False,
            dynamic=False,
        )

        # VERIFY CORRECTNESS
        O_true = compiled_matmul(P, V)
        O_fca = fca_pv(P, V)

        # if results are wrong, end benchmarking
        if O_true.shape != O_fca.shape:
            print(f"Shape mismatch: expected {O_true.shape}, got {O_fca.shape}")
            return 10**6, False
        
        if (O_true - O_fca).abs().max().item() > 5e-3:
            print(f"Results mismatch: max error {((O_true - O_fca).abs()).max().item()}")
            print("Results are not equal, ending benchmarking.")
            return 10**6, False
    
        # otherwise continue benchmarking
        # warmup
        print("Warming up...")
        for _ in range(10):
            compiled_matmul(P, V)
            fca_pv(P, V)
        torch.cuda.synchronize()
        
        # Benchmark functions
        print("Benchmarking...")
        results = {}
        for name, func in [
            ("torch.matmul", lambda: compiled_matmul(P, V)),
            ("fca_pv", lambda: fca_pv(P, V)),
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
        
        return sum(results['fca_pv']) / len(results['fca_pv']), True
    
    except Exception as e:
        print(f"Error during correctness check: {e}")
        return 10**6, False


torch.cuda.empty_cache()

baseline = r"""
#include <torch/extension.h>
#include <vector>

// get_block + write_combo unchanged
static inline torch::Tensor get_block(const torch::Tensor& M,
                                      int row_block, int col_block,
                                      const int64_t BR, const int64_t BC) {
    const int64_t rs = row_block * BR, cs = col_block * BC;
    return M.slice(0, rs, rs + BR).slice(1, cs, cs + BC);
}
static inline void write_combo(
        torch::Tensor dst,
        std::initializer_list<std::pair<torch::Tensor, float>> terms) {
    dst.zero_();
    for (const auto& p : terms) { dst.add_(p.first, p.second); }
}

// Tiny helper: compute combos then O += A @ B
static inline void addmm_combo(
        torch::Tensor O, torch::Tensor A_scratch, torch::Tensor B_scratch,
        std::initializer_list<std::pair<torch::Tensor, float>> A_terms,
        std::initializer_list<std::pair<torch::Tensor, float>> B_terms) {
    write_combo(A_scratch, A_terms);
    write_combo(B_scratch, B_terms);
    // O = 1*O + 1*(A @ B)
    O.addmm_(A_scratch, B_scratch, /*beta=*/1.0f, /*alpha=*/1.0f);
}

// -----------------------------------------------------------------------------
//                                   PV KERNEL
// -----------------------------------------------------------------------------
torch::Tensor pv(torch::Tensor P, torch::Tensor V) {
    TORCH_CHECK(P.device().is_cuda() && V.device().is_cuda(), "must be CUDA");
    TORCH_CHECK(P.scalar_type() == torch::kFloat32 && V.scalar_type() == torch::kFloat32, "dtype must be FP32");
    TORCH_CHECK(P.dim() == 2 && V.dim() == 2, "P,V must be 2D");
    TORCH_CHECK(P.size(0) == P.size(1), "P must be square");
    TORCH_CHECK(P.size(0) == V.size(0), "P,V row match");


    const auto opts = P.options();
    const int64_t L  = P.size(0);
    const int64_t d  = V.size(1);
    const int64_t Lp = ((L + 3) / 4) * 4;
    const int64_t dp = ((d + 3) / 4) * 4;
    const int64_t BR = Lp / 4;
    const int64_t BC = dp / 4;

    // Zero-pad if needed
    torch::Tensor Pw, Vw;
    if (Lp != L || dp != d) {
        Pw = torch::zeros({Lp, Lp}, opts);
        Vw = torch::zeros({Lp, dp}, opts);
        Pw.slice(0, 0, L).slice(1, 0, L).copy_(P);
        Vw.slice(0, 0, L).slice(1, 0, d).copy_(V);
    } else {
        Pw = P; Vw = V;
    }

    // 4×4 tiling
    torch::Tensor Pblk[4][4], Vblk[4][4];
    for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) {
        Pblk[r][c] = get_block(Pw, r, c, BR, BR);
        Vblk[r][c] = get_block(Vw, r, c, BR, BC);
    }

    // P1..P10 (lower-tri) + V1..V16 convenience aliases
    torch::Tensor P1 = Pblk[0][0];
    torch::Tensor P2 = Pblk[1][0], P3 = Pblk[1][1];
    torch::Tensor P4 = Pblk[2][0], P5 = Pblk[2][1], P6 = Pblk[2][2];
    torch::Tensor P7 = Pblk[3][0], P8 = Pblk[3][1], P9 = Pblk[3][2], P10 = Pblk[3][3];

    torch::Tensor V1 = Vblk[0][0], V2 = Vblk[0][1], V3 = Vblk[0][2], V4 = Vblk[0][3];
    torch::Tensor V5 = Vblk[1][0], V6 = Vblk[1][1], V7 = Vblk[1][2], V8 = Vblk[1][3];
    torch::Tensor V9 = Vblk[2][0], V10= Vblk[2][1], V11= Vblk[2][2], V12= Vblk[2][3];
    torch::Tensor V13= Vblk[3][0], V14= Vblk[3][1], V15= Vblk[3][2], V16= Vblk[3][3];

    // Output storage and views
    auto Ofull = torch::zeros({Lp, dp}, opts);
    auto O = [&](int r, int c)->torch::Tensor {
        return Ofull.slice(0, r*BR, (r+1)*BR).slice(1, c*BC, (c+1)*BC);
    };

    // Reusable scratch for direct addmm path (avoids many small allocs)
    auto Atemp = torch::empty({BR, BR}, opts);
    auto Btemp = torch::empty({BR, BC}, opts);

    // ---------------------- DIRECT WRITES (single-use) ------------------------
    // h3..h6 -> O[0][0..3]
    addmm_combo(O(0,0), Atemp, Btemp, {{P1,1.f}}, {{V1,1.f}}); // h3
    addmm_combo(O(0,1), Atemp, Btemp, {{P1,1.f}}, {{V2,1.f}}); // h4
    addmm_combo(O(0,2), Atemp, Btemp, {{P1,1.f}}, {{V3,1.f}}); // h5
    addmm_combo(O(0,3), Atemp, Btemp, {{P1,1.f}}, {{V4,1.f}}); // h6

    // h7..h10
    addmm_combo(O(3,0), Atemp, Btemp, {{P10,1.f}}, {{V13,1.f}}); // h7
    addmm_combo(O(3,1), Atemp, Btemp, {{P10,1.f}}, {{V14,1.f}}); // h8
    addmm_combo(O(3,2), Atemp, Btemp, {{P10,1.f}}, {{V15,1.f}}); // h9
    addmm_combo(O(3,3), Atemp, Btemp, {{P10,1.f}}, {{V16,1.f}}); // h10

    // h2
    addmm_combo(O(2,1), Atemp, Btemp, {{P6,1.f}}, {{V6,1.f},{V10,1.f},{V12,1.f}}); // h2

    // m10,11,12,13,20,23,24
    addmm_combo(O(2,2), Atemp, Btemp, {{P3,1.f},{P5,1.f}},
                {{V2,1.f},{V3,-1.f},{V7,1.f},{V11,1.f},{V4,1.f},{V8,-1.f}});       // m10 -> O11
    addmm_combo(O(1,0), Atemp, Btemp, {{P2,1.f},{P3,1.f},{P7,1.f},{P8,1.f}},
                {{V5,1.f},{V6,1.f},{V7,-1.f}});                                     // m11 -> O5
    addmm_combo(O(1,3), Atemp, Btemp, {{P2,1.f},{P3,1.f},{P4,1.f},{P5,1.f}},
                {{V2,1.f},{V3,-1.f},{V4,1.f}});                                      // m12 -> O8
    addmm_combo(O(3,2), Atemp, Btemp, {{P2,1.f},{P7,1.f}},
                {{V1,-1.f},{V5,1.f},{V6,1.f},{V3,1.f},{V7,-1.f},{V11,1.f}});        // m13 -> O15
    addmm_combo(O(2,0), Atemp, Btemp, {{P5,1.f},{P8,-1.f}},
                {{V5,1.f},{V9,1.f},{V8,-1.f}});                                      // m20 -> O9
    addmm_combo(O(3,3), Atemp, Btemp, {{P4,-1.f},{P7,1.f}},
                {{V1,-1.f},{V4,1.f},{V12,1.f}});                                     // m23 -> O16
    addmm_combo(O(3,1), Atemp, Btemp, {{P9,1.f}},
                {{V9,1.f},{V2,1.f},{V10,1.f}});                                      // m24 -> O14

    // ------------------ REUSED TERMS: COMPACT BATCHED GEMM -------------------
    // Keep only reused m's + h1
    // Order we push: [m1,m2,m3,m4,m5,m6,m7,m8,m9,m14,m15,m16,m17,m18,m19,m21,m22,h1]
    std::vector<int> map_m(25, -1); // 1..24
    int idx = 0;

    auto NR = 18; // 17 m's + h1
    auto Abatch = torch::empty({NR, BR, BR}, opts);
    auto Bbatch = torch::empty({NR, BR, BC}, opts);

    auto push = [&](int m_id,
                    std::initializer_list<std::pair<torch::Tensor,float>> A,
                    std::initializer_list<std::pair<torch::Tensor,float>> B){
        write_combo(Abatch[idx], A);
        write_combo(Bbatch[idx], B);
        map_m[m_id] = idx; ++idx;
    };

    // m1..m9 (excluding 10..13)
    push(1, {{P3,1},{P4,1},{P5,1}}, {{V2,-1},{V3,1},{V4,-1},{V8,1}});
    push(2, {{P2,1},{P7,1},{P8,1}}, {{V1,1},{V5,-1},{V6,-1},{V7,1}});
    push(3, {{P4,1},{P7,-1},{P9,1}}, {{V2,-1},{V12,1}});
    push(4, {{P5,-1},{P6,1},{P8,1}}, {{V9,1},{V6,-1}});
    push(5, {{P2,-1},{P7,-1},{P9,1}}, {{V2,1},{V11,1}});
    push(6, {{P3,1},{P5,1},{P6,-1}}, {{V6,1},{V11,1}});
    push(7, {{P2,-1},{P3,-1},{P5,-1},{P6,1},{P7,-1},{P9,1}}, {{V11,1}});
    push(8, {{P7,-1},{P9,1}}, {{V2,1}});
    push(9, {{P5,-1},{P6,1}}, {{V6,1}});

    // m14..m22 (skip single-use 20)
    push(14, {{P8,1}}, {{V1,-1},{V5,1},{V6,1}});
    push(15, {{P4,1}}, {{V2,1},{V4,1},{V8,-1}});
    push(16, {{P4,1},{P8,1}}, {{V1,1},{V8,-1}});
    push(17, {{P4,1},{P6,-1},{P7,-1},{P9,1}}, {{V12,1}});
    push(18, {{P5,1},{P6,-1},{P8,-1},{P9,1}}, {{V9,1}});
    push(19, {{P2,1}}, {{V2,-1},{V3,1}});
    push(21, {{P4,1},{P5,1}}, {{V8,1}});
    push(22, {{P7,1},{P8,1}}, {{V1,1}});

    // h1 at the end; give it a synthetic "m id" 0 so we fetch via h1_idx
    int h1_idx = idx;
    write_combo(Abatch[idx], {{P3,1.f}});
    write_combo(Bbatch[idx], {{V6,-1.f},{V7,1.f}});
    ++idx;

    TORCH_INTERNAL_ASSERT(idx == NR, "batch fill mismatch");

    auto C = torch::bmm(Abatch, Bbatch);
    auto m = [&](int i)->torch::Tensor{ int k = map_m[i]; TORCH_INTERNAL_ASSERT(k>=0); return C[k]; };
    auto h1 = [&]()->torch::Tensor{ return C[h1_idx]; };

    // -------------------------- ACCUMULATE REUSED -----------------------------
    // Row 1: O5..O8
    O(1,0).add_( m(2),  1.f).add_( m(22), -1.f).add_( h1(), 1.f);  // O5 (m11 already added)
    O(1,1).add_( m(5), -1.f).add_( m(6), 1.f).add_( m(7), 1.f).add_( m(8), 1.f).add_( m(9), 1.f); // O6
    O(1,2).add_( m(5), -1.f).add_( m(6), 1.f).add_( m(7), 1.f).add_( m(8), 1.f)
           .add_( m(9), 1.f).add_( m(19), 1.f).add_( h1(), 1.f);                                         // O7
    O(1,3).add_( m(1), 1.f).add_( m(19), 1.f).add_( m(21), -1.f);                                        // O8 (m12 added)

    // Row 2: O9..O12
    O(2,0).add_( m(4), 1.f).add_( m(9), 1.f).add_( m(14), 1.f).add_( m(16), 1.f).add_( m(21), 1.f);      // O9 (m20 added)
    O(2,1).add_( m(3), -1.f).add_( m(8), -1.f).add_( m(9), -1.f).add_( m(17), 1.f);                      // O10 (h2 added)
    O(2,2).add_( m(1), 1.f).add_( m(6), -1.f).add_( m(9), -1.f).add_( m(15), 1.f).add_( h1(), -1.f);     // O11 (m10 added)
    O(2,3).add_( m(3), 1.f).add_( m(8), 1.f).add_( m(15), 1.f).add_( m(17), -1.f).add_( m(21), 1.f);     // O12

    // Row 3: O13..O16
    O(3,0).add_( m(4), 1.f).add_( m(9), 1.f).add_( m(14), 1.f).add_( m(18), 1.f).add_( m(22), 1.f);      // O13 (h7 added)
    O(3,1).add_( m(4), -1.f).add_( m(8), -1.f).add_( m(9), -1.f).add_( m(18), -1.f);                     // O14 (m24,h8 added)
    O(3,2).add_( m(2), 1.f).add_( m(5), 1.f).add_( m(8), -1.f).add_( m(14), 1.f).add_( m(19), -1.f);     // O15 (m13,h9 added)
    O(3,3).add_( m(3), 1.f).add_( m(8), 1.f).add_( m(15), 1.f).add_( m(16), -1.f).add_( m(22), 1.f);     // O16 (m23,h10 added)

    // Crop padding
    if (Lp != L || dp != d) {
        return Ofull.slice(0, 0, L).slice(1, 0, d);
    }
    return Ofull;
}
"""

# python3 bench_pv.py
metric, _ = evaluator_pv(baseline)