//
// Tile-size sweep for the FlashAttention kernel: time flash_attention_kernel<RPB,BC,64>
// across (RPB = query rows/warps per block, BC = key-tile width) and verify each against
// the CPU reference. Picks the fastest valid config. RPB*32 = threads/block (<=1024);
// shared mem = 2*BC*64*4 bytes (<=48KB default -> BC<=64).
//
#include "../CudaBuffer.cuh"
#include "../cpu_ref/AttentionCPU.h"
#include "../gpu_ops/FlashAttention.cuh"
#include <cstdio>
#include <random>
#include <vector>
#include <cmath>
#include <algorithm>

namespace {
constexpr int H = 14, Hkv = 2, d = 64;
cudaEvent_t e0, e1;

template <int RPB, int BC>
void run(const float* Q, const float* K, const float* V, float* O, int N, const std::vector<float>& O_cpu) {
    dim3 grid(H, (N + RPB - 1) / RPB);
    float scale = 1.0f / sqrtf((float)d);
    flash_attention_kernel<RPB, BC, 64><<<grid, RPB * 32>>>(Q, K, V, O, H, Hkv, N, scale);  // warmup
    cudaError_t err = cudaGetLastError();
    cudaDeviceSynchronize();
    if (err != cudaSuccess) { std::printf("    RPB=%-2d BC=%-3d : launch error (%s)\n", RPB, BC, cudaGetErrorString(err)); return; }

    double max_abs = 0.0;
    for (size_t i = 0; i < O_cpu.size(); ++i) max_abs = std::max(max_abs, std::fabs((double)O[i] - (double)O_cpu[i]));

    float best = 1e30f;
    for (int r = 0; r < 20; ++r) {
        cudaEventRecord(e0);
        flash_attention_kernel<RPB, BC, 64><<<grid, RPB * 32>>>(Q, K, V, O, H, Hkv, N, scale);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float t; cudaEventElapsedTime(&t, e0, e1); best = std::min(best, t);
    }
    std::printf("    RPB=%-2d BC=%-3d (smem %2dKB, %4d thr) : %8.3f ms   max_abs=%.2e %s\n",
                RPB, BC, (2 * BC * 64 * 4) / 1024, RPB * 32, best, max_abs, max_abs < 5e-6 ? "" : "  <-- MISMATCH");
}
}  // namespace

int main() {
    std::printf("=== FlashAttention tile-size sweep (FP32, H=14 Hkv=2 d=64) ===\n");
    checkCuda(cudaEventCreate(&e0)); checkCuda(cudaEventCreate(&e1));
    std::mt19937 gen{123};
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int N : {1024, 4096}) {
        std::printf("\n  N=%d  (current default: RPB=16 BC=32)\n", N);
        AttentionDims dims{H, Hkv, d, N};
        CudaBuffer Q{(size_t)H*N*d*sizeof(float)}, K{(size_t)Hkv*N*d*sizeof(float)},
                   V{(size_t)Hkv*N*d*sizeof(float)}, O{(size_t)H*N*d*sizeof(float)};
        auto *Qp=(float*)Q.data, *Kp=(float*)K.data, *Vp=(float*)V.data;
        for (size_t i=0;i<(size_t)H*N*d;++i) Qp[i]=dist(gen);
        for (size_t i=0;i<(size_t)Hkv*N*d;++i){ Kp[i]=dist(gen); Vp[i]=dist(gen); }
        std::vector<float> O_cpu((size_t)H*N*d);
        attention_materialized(Qp, Kp, Vp, O_cpu.data(), dims);

        #define RUN(rpb, bc) run<rpb, bc>(Qp, Kp, Vp, (float*)O.data, N, O_cpu);
        RUN(4,16)  RUN(4,32)  RUN(4,64)
        RUN(8,16)  RUN(8,32)  RUN(8,64)
        RUN(16,16) RUN(16,32) RUN(16,64)
        RUN(32,16) RUN(32,32) RUN(32,64)
        #undef RUN
    }
    checkCuda(cudaEventDestroy(e0)); checkCuda(cudaEventDestroy(e1));
    return 0;
}
