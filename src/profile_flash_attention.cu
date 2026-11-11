//
// ncu profiling runner for the FlashAttention kernel -- mirrors src/profile_naive_attention.cu
// EXACTLY (same FP32 inputs, layout, seed, H/Hkv/d, causal/GQA, warmup+profiled launch) so
// the naive-vs-flash ncu comparison is apples-to-apples. Profile identically, e.g.:
//   ncu --set full --launch-skip 1 --launch-count 1 -f -o profiling/flash_attention_N512 \
//       ./profile_flash_attention 512
// (FlashAttention is a single kernel, so 1 launch to skip for warmup, 1 to profile.)
//
#include "CudaBuffer.cuh"
#include "ErrorCheck.h"
#include "gpu_ops/FlashAttention.cuh"
#include <random>
#include <cstdlib>
#include <cstdio>

int main(int argc, char** argv) {
    constexpr int H = 14, Hkv = 2, d = 64;
    int N = (argc > 1) ? std::atoi(argv[1]) : 512;

    FlashAttention attn(H, Hkv, d, N);
    CudaBuffer Q{(size_t)H * N * d * sizeof(float)}, K{(size_t)Hkv * N * d * sizeof(float)},
               V{(size_t)Hkv * N * d * sizeof(float)}, O{(size_t)H * N * d * sizeof(float)};
    std::mt19937 gen(7);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < (size_t)H * N * d; ++i) ((float*)Q.data)[i] = dist(gen);
    for (size_t i = 0; i < (size_t)Hkv * N * d; ++i) { ((float*)K.data)[i] = dist(gen); ((float*)V.data)[i] = dist(gen); }

    attn.forward((float*)Q.data, (float*)K.data, (float*)V.data, (float*)O.data, N, cudaStreamPerThread);  // warmup
    checkCuda(cudaStreamSynchronize(cudaStreamPerThread));
    attn.forward((float*)Q.data, (float*)K.data, (float*)V.data, (float*)O.data, N, cudaStreamPerThread);  // profiled
    checkCuda(cudaStreamSynchronize(cudaStreamPerThread));

    std::printf("flash attention profiled: N=%d (H=%d Hkv=%d d=%d)\n", N, H, Hkv, d);
    return 0;
}
