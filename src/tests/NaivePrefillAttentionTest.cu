//
// Verifies the naive CUDA prefill-attention baseline (3 kernels, FP32, causal, GQA)
// against the CPU reference src/cpu_ref/AttentionCPU.h (attention_materialized).
//
// Same Q,K,V layout as the CPU reference (Q:[H,N,d], K,V:[Hkv,N,d], O:[H,N,d]); random
// N(0,1) inputs across N=1..1024. Tolerance rtol=1e-4/atol=1e-5 (GPU vs CPU FP32 differ
// only by reduction order); max_abs is the headline metric.
//
#include "../CudaBuffer.cuh"
#include "../cpu_ref/AttentionCPU.h"
#include "../gpu_ops/NaivePrefillAttention.cuh"
#include <cstdio>
#include <random>
#include <vector>
#include <chrono>

namespace {
using Clock = std::chrono::high_resolution_clock;
double msec(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

int g_failures = 0;

void report(int N, const std::vector<float>& gpu, const std::vector<float>& cpu, float rtol, float atol) {
    double max_abs = 0.0, max_rel = 0.0;
    int n_bad = 0;
    for (size_t i = 0; i < gpu.size(); ++i) {
        double diff = std::fabs((double)gpu[i] - (double)cpu[i]);
        max_abs = std::max(max_abs, diff);
        max_rel = std::max(max_rel, diff / (std::fabs((double)cpu[i]) + 1e-30));
        if (diff > atol + (double)rtol * std::fabs((double)cpu[i])) n_bad++;
    }
    bool ok = (n_bad == 0);
    std::printf("  [%s] N=%-4d  max_abs=%.3e  max_rel=%.3e  (rtol=%.0e atol=%.0e, %d/%zu bad)\n",
                ok ? "PASS" : "FAIL", N, max_abs, max_rel, rtol, atol, n_bad, gpu.size());
    if (!ok) g_failures++;
}

}  // namespace

int main() {
    std::printf("=== Naive CUDA prefill attention vs CPU reference (FP32, GQA, causal) ===\n");
    constexpr int H = 14, Hkv = 2, d = 64, MAX_N = 1024;
    const int Ns[] = {1, 2, 8, 64, 128, 256, 512, 1024};

    NaivePrefillAttention attn(H, Hkv, d, MAX_N);
    std::mt19937 gen{123};
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int N : Ns) {
        AttentionDims dims{H, Hkv, d, N};
        CudaBuffer Q{(size_t)H * N * d * sizeof(float)};
        CudaBuffer K{(size_t)Hkv * N * d * sizeof(float)};
        CudaBuffer V{(size_t)Hkv * N * d * sizeof(float)};
        CudaBuffer O{(size_t)H * N * d * sizeof(float)};
        auto *Qp = static_cast<float*>(Q.data);
        auto *Kp = static_cast<float*>(K.data);
        auto *Vp = static_cast<float*>(V.data);
        for (size_t i = 0; i < (size_t)H * N * d; ++i) Qp[i] = dist(gen);
        for (size_t i = 0; i < (size_t)Hkv * N * d; ++i) Kp[i] = dist(gen);
        for (size_t i = 0; i < (size_t)Hkv * N * d; ++i) Vp[i] = dist(gen);

        std::vector<float> O_cpu((size_t)H * N * d);
        attention_materialized(Qp, Kp, Vp, O_cpu.data(), dims);

        // run twice to confirm scratch is reused correctly across calls
        for (int run = 0; run < 2; ++run) {
            attn.forward(Qp, Kp, Vp, static_cast<float*>(O.data), N, cudaStreamPerThread);
            // checkCuda on the sync catches ASYNC kernel faults (e.g. an OOB access from a
            // future indexing regression); forward() itself stays sync-free for benchmarking.
            checkCuda(cudaStreamSynchronize(cudaStreamPerThread));
        }

        std::vector<float> O_gpu(static_cast<float*>(O.data), static_cast<float*>(O.data) + (size_t)H * N * d);
        // atol tightened to ~2x the observed GPU-vs-CPU FP32 reduction-order error so a
        // systematic bias >~1e-6 is caught; test is deterministic (fixed seed, no atomics).
        report(N, O_gpu, O_cpu, /*rtol=*/1e-4f, /*atol=*/2e-6f);
    }

    // ---- CPU vs GPU timing for the same N-query attention (identical FP32 inputs) ----
    // CPU: single-thread wall time (chrono). GPU: kernel execution time (cudaEvent, best of 20)
    // -- the same metric the FlashAttention comparison and ncu profiling will use.
    std::printf("\n  CPU (attention_materialized, 1 thread) vs GPU (naive attention kernels), same inputs:\n");
    cudaEvent_t e0, e1; checkCuda(cudaEventCreate(&e0)); checkCuda(cudaEventCreate(&e1));
    for (int N : {128, 256, 512, 1024}) {
        AttentionDims dims{H, Hkv, d, N};
        CudaBuffer Q{(size_t)H * N * d * sizeof(float)}, K{(size_t)Hkv * N * d * sizeof(float)},
                   V{(size_t)Hkv * N * d * sizeof(float)}, O{(size_t)H * N * d * sizeof(float)};
        auto *Qp = (float*)Q.data, *Kp = (float*)K.data, *Vp = (float*)V.data;
        for (size_t i = 0; i < (size_t)H * N * d; ++i) Qp[i] = dist(gen);
        for (size_t i = 0; i < (size_t)Hkv * N * d; ++i) { Kp[i] = dist(gen); Vp[i] = dist(gen); }

        std::vector<float> Ocpu((size_t)H * N * d);
        auto c0 = Clock::now(); attention_materialized(Qp, Kp, Vp, Ocpu.data(), dims); auto c1 = Clock::now();
        double cpu_ms = msec(c0, c1);

        attn.forward(Qp, Kp, Vp, (float*)O.data, N, cudaStreamPerThread);   // warmup
        checkCuda(cudaStreamSynchronize(cudaStreamPerThread));
        float gpu_ms = 1e30f;
        for (int r = 0; r < 20; ++r) {
            checkCuda(cudaEventRecord(e0, cudaStreamPerThread));
            attn.forward(Qp, Kp, Vp, (float*)O.data, N, cudaStreamPerThread);
            checkCuda(cudaEventRecord(e1, cudaStreamPerThread));
            checkCuda(cudaEventSynchronize(e1));
            float t; checkCuda(cudaEventElapsedTime(&t, e0, e1)); gpu_ms = std::min(gpu_ms, t);
        }
        std::printf("    N=%-4d  CPU %9.2f ms   GPU(kernels) %7.3f ms   (%.0fx faster on GPU)\n",
                    N, cpu_ms, (double)gpu_ms, cpu_ms / gpu_ms);
    }
    checkCuda(cudaEventDestroy(e0)); checkCuda(cudaEventDestroy(e1));

    std::printf("========================================================================\n");
    if (g_failures == 0) { std::printf("ALL CHECKS PASSED\n"); return 0; }
    std::printf("%d CHECK(S) FAILED\n", g_failures);
    return 1;
}
