//
// Verifies the fused FlashAttention kernel against the CPU reference (AttentionCPU.h)
// AND the naive GPU baseline, and times naive vs flash (the FlashAttention speedup).
// Same FP32 / [H,N,d] / causal / GQA setup as the naive baseline (apples-to-apples).
//
#include "../CudaBuffer.cuh"
#include "../cpu_ref/AttentionCPU.h"
#include "../gpu_ops/NaivePrefillAttention.cuh"
#include "../gpu_ops/FlashAttention.cuh"
#include <cstdio>
#include <random>
#include <vector>
#include <cmath>
#include <algorithm>

namespace {
int g_failures = 0;

// Gate on BOTH the per-element allclose (catches localized errors) AND a tight max_abs
// bound (catches a small systematic bias the loose per-element rtol would let through).
void report(const char* name, int N, const std::vector<float>& a, const std::vector<float>& b,
            float rtol, float atol, double max_abs_tol) {
    double max_abs = 0.0; int n_bad = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = std::fabs((double)a[i] - (double)b[i]);
        max_abs = std::max(max_abs, diff);
        if (diff > atol + (double)rtol * std::fabs((double)b[i])) n_bad++;
    }
    bool ok = (n_bad == 0) && (max_abs <= max_abs_tol);
    std::printf("  [%s] %-28s N=%-4d max_abs=%.3e (%d/%zu bad, max_abs_tol=%.0e)\n",
                ok ? "PASS" : "FAIL", name, N, max_abs, n_bad, a.size(), max_abs_tol);
    if (!ok) g_failures++;
}
}  // namespace

int main() {
    std::printf("=== FlashAttention vs CPU reference and naive GPU (FP32, GQA, causal) ===\n");
    constexpr int H = 14, Hkv = 2, d = 64, MAX_N = 4096;
    NaivePrefillAttention naive(H, Hkv, d, MAX_N);
    FlashAttention flash(H, Hkv, d, MAX_N);
    std::mt19937 gen{123};
    std::normal_distribution<float> dist(0.0f, 1.0f);
    cudaEvent_t e0, e1; checkCuda(cudaEventCreate(&e0)); checkCuda(cudaEventCreate(&e1));

    // ---- correctness: flash vs CPU reference, and flash vs naive GPU ----
    for (int N : {1, 2, 8, 64, 128, 256, 512, 1024}) {
        AttentionDims dims{H, Hkv, d, N};
        CudaBuffer Q{(size_t)H*N*d*sizeof(float)}, K{(size_t)Hkv*N*d*sizeof(float)},
                   V{(size_t)Hkv*N*d*sizeof(float)}, Of{(size_t)H*N*d*sizeof(float)}, On{(size_t)H*N*d*sizeof(float)};
        auto *Qp=(float*)Q.data, *Kp=(float*)K.data, *Vp=(float*)V.data;
        for (size_t i=0;i<(size_t)H*N*d;++i) Qp[i]=dist(gen);
        for (size_t i=0;i<(size_t)Hkv*N*d;++i){ Kp[i]=dist(gen); Vp[i]=dist(gen); }

        std::vector<float> O_cpu((size_t)H*N*d);
        attention_materialized(Qp, Kp, Vp, O_cpu.data(), dims);

        flash.forward(Qp, Kp, Vp, (float*)Of.data, N, cudaStreamPerThread);
        naive.forward(Qp, Kp, Vp, (float*)On.data, N, cudaStreamPerThread);
        checkCuda(cudaStreamSynchronize(cudaStreamPerThread));

        std::vector<float> O_flash((float*)Of.data, (float*)Of.data + (size_t)H*N*d);
        std::vector<float> O_naive((float*)On.data, (float*)On.data + (size_t)H*N*d);
        report("flash vs CPU ref", N, O_flash, O_cpu, 1e-4f, 2e-6f, /*max_abs_tol=*/5e-6);
        report("flash vs naive GPU", N, O_flash, O_naive, 1e-4f, 2e-6f, /*max_abs_tol=*/5e-6);
    }

    // ---- speed: naive vs flash (GPU kernel time, cudaEvent best of 20) ----
    std::printf("\n  naive vs FlashAttention (GPU kernel time, best of 20):\n");
    auto time_fwd = [&](auto& impl, float* Qp, float* Kp, float* Vp, float* Op, int N) {
        impl.forward(Qp, Kp, Vp, Op, N, cudaStreamPerThread);  // warmup
        checkCuda(cudaStreamSynchronize(cudaStreamPerThread));
        float best = 1e30f;
        for (int r = 0; r < 20; ++r) {
            checkCuda(cudaEventRecord(e0, cudaStreamPerThread));
            impl.forward(Qp, Kp, Vp, Op, N, cudaStreamPerThread);
            checkCuda(cudaEventRecord(e1, cudaStreamPerThread));
            checkCuda(cudaEventSynchronize(e1));
            float t; checkCuda(cudaEventElapsedTime(&t, e0, e1)); best = std::min(best, t);
        }
        return (double)best;
    };
    for (int N : {512, 1024, 2048, 4096}) {
        CudaBuffer Q{(size_t)H*N*d*sizeof(float)}, K{(size_t)Hkv*N*d*sizeof(float)},
                   V{(size_t)Hkv*N*d*sizeof(float)}, O{(size_t)H*N*d*sizeof(float)};
        auto *Qp=(float*)Q.data, *Kp=(float*)K.data, *Vp=(float*)V.data;
        for (size_t i=0;i<(size_t)H*N*d;++i) Qp[i]=dist(gen);
        for (size_t i=0;i<(size_t)Hkv*N*d;++i){ Kp[i]=dist(gen); Vp[i]=dist(gen); }
        double tn = time_fwd(naive, Qp, Kp, Vp, (float*)O.data, N);
        double tf = time_fwd(flash, Qp, Kp, Vp, (float*)O.data, N);
        std::printf("    N=%-4d  naive %8.3f ms   flash %7.3f ms   (%.1fx faster)\n", N, tn, tf, tn/tf);
    }
    checkCuda(cudaEventDestroy(e0)); checkCuda(cudaEventDestroy(e1));

    std::printf("========================================================================\n");
    if (g_failures == 0) { std::printf("ALL CHECKS PASSED\n"); return 0; }
    std::printf("%d CHECK(S) FAILED\n", g_failures);
    return 1;
}
