//
// Minimal runner to ncu-profile the naive prefill-attention baseline kernels in
// isolation (naive_qk_scores / naive_softmax_rows / naive_pv). One forward() launches
// each kernel exactly once, so an ncu run captures the three kernels cleanly.
//
// Usage:  profile_naive_attention [N]      (default N=512)
// Profile (full report incl. roofline):
//   ncu --set full -f -o profiling/naive_attention_N512 ./profile_naive_attention 512
// Profile (key metrics as text):
//   ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum,\
//     dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,\
//     sm__warps_active.avg.pct_of_peak_sustained_active,\
//     sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum \
//     ./profile_naive_attention 512
//
// FP32, Qwen2-0.5B head config (H=14, Hkv=2, d=64), causal, GQA -- IDENTICAL inputs/
// layout/precision to what the FlashAttention kernel must be profiled against, so the
// naive-vs-flash comparison stays apples-to-apples.
//
#include "CudaBuffer.cuh"
#include "ErrorCheck.h"
#include "gpu_ops/NaivePrefillAttention.cuh"
#include <random>
#include <cstdlib>
#include <cstdio>

int main(int argc, char** argv) {
    constexpr int H = 14, Hkv = 2, d = 64;
    int N = (argc > 1) ? std::atoi(argv[1]) : 512;

    NaivePrefillAttention attn(H, Hkv, d, N);
    CudaBuffer Q{(size_t)H * N * d * sizeof(float)}, K{(size_t)Hkv * N * d * sizeof(float)},
               V{(size_t)Hkv * N * d * sizeof(float)}, O{(size_t)H * N * d * sizeof(float)};
    std::mt19937 gen(7);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < (size_t)H * N * d; ++i) ((float*)Q.data)[i] = dist(gen);
    for (size_t i = 0; i < (size_t)Hkv * N * d; ++i) { ((float*)K.data)[i] = dist(gen); ((float*)V.data)[i] = dist(gen); }

    // warmup (page migration + JIT) so a steady-state launch is what gets profiled
    attn.forward((float*)Q.data, (float*)K.data, (float*)V.data, (float*)O.data, N, cudaStreamPerThread);
    checkCuda(cudaStreamSynchronize(cudaStreamPerThread));

    // the launch ncu profiles
    attn.forward((float*)Q.data, (float*)K.data, (float*)V.data, (float*)O.data, N, cudaStreamPerThread);
    checkCuda(cudaStreamSynchronize(cudaStreamPerThread));

    std::printf("naive attention profiled: N=%d (H=%d Hkv=%d d=%d)\n", N, H, Hkv, d);
    return 0;
}
