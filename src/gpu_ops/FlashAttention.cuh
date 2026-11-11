#pragma once
//
// Fused, tiled FlashAttention forward kernel, FP32, causal, GQA.
//
// The optimized counterpart to NaivePrefillAttention: it computes EXACT attention but
// never materializes the N x N S/P matrices in HBM. K/V are streamed in tiles into shared
// memory and reused across a block of query rows; an online softmax (Milakov-Gimelshein:
// running max m, running denom l, rescaled output accumulator) folds each tile into the
// result. This removes the ~4*H*N^2*4 bytes of S/P HBM traffic the naive baseline pays.
//
// Same interface/layout/precision as NaivePrefillAttention so the two are a drop-in,
// apples-to-apples pair for correctness (vs the CPU reference) and ncu profiling:
//   Q : [H, N, d]   K,V : [Hkv, N, d]   O : [H, N, d]   GQA kv(h)=h*Hkv/H   scale 1/sqrt(d)
//
// Thread mapping (assumes head_dim D = 64): one WARP per query row, each lane owns 2 head
// dims (lane, lane+32) -- the same warp-shuffle dot as the decode kernel, now with
// K/V tiles in shared memory. RPB warps (= query rows) per block share each loaded K/V tile.
//
#include <cuda_bf16.h>
#include "../ErrorCheck.h"

// RPB = query rows (warps) per block; BC = key-tile width; D = head dim (64).
template <int RPB, int BC, int D>
static __global__ void flash_attention_kernel(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
    float* __restrict__ O, int H, int Hkv, int N, float scale
) {
    static_assert(D == 64, "warp-per-row mapping assumes head_dim 64 (D/2 == warpSize)");
    const int h = blockIdx.x;
    const int row_base = blockIdx.y * RPB;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = row_base + warp;            // this warp's query row (uniform across the warp)
    const int g = h * Hkv / H;

    __shared__ float Ksh[BC * D];
    __shared__ float Vsh[BC * D];

    const bool active = (row < N);
    float q0 = 0.0f, q1 = 0.0f;
    if (active) {
        const float* q = Q + ((size_t)h * N + row) * D;
        q0 = q[lane];
        q1 = q[lane + 32];
    }
    float m = -INFINITY, l = 0.0f, acc0 = 0.0f, acc1 = 0.0f;  // online-softmax state (registers)

    // Causal: keys only up to the block's last query row -- tiles beyond are fully masked
    // and never even loaded (the "skip fully-masked tiles" optimization).
    const int block_max_row = min(row_base + RPB - 1, N - 1);
    for (int kt = 0; kt <= block_max_row; kt += BC) {
        // cooperatively load this K/V tile into shared memory (out-of-range rows -> 0, unused)
        for (int idx = threadIdx.x; idx < BC * D; idx += blockDim.x) {
            const int c = idx / D, e = idx % D, gpos = kt + c;
            float kk = 0.0f, vv = 0.0f;
            if (gpos < N) {
                kk = K[((size_t)g * N + gpos) * D + e];
                vv = V[((size_t)g * N + gpos) * D + e];
            }
            Ksh[idx] = kk;
            Vsh[idx] = vv;
        }
        __syncthreads();

        if (active) {
            int cmax = row - kt + 1;            // attend keys kt..row within this tile (causal)
            if (cmax > BC) cmax = BC;
            for (int c = 0; c < cmax; ++c) {
                float partial = q0 * Ksh[c * D + lane] + q1 * Ksh[c * D + lane + 32];
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) partial += __shfl_down_sync(0xFFFFFFFF, partial, off);
                float score = __shfl_sync(0xFFFFFFFF, partial, 0) * scale;   // dot, broadcast to all lanes

                float new_m = fmaxf(m, score);
                float corr = expf(m - new_m);
                float p = expf(score - new_m);
                l = l * corr + p;
                acc0 = acc0 * corr + p * Vsh[c * D + lane];
                acc1 = acc1 * corr + p * Vsh[c * D + lane + 32];
                m = new_m;
            }
        }
        __syncthreads();                        // all threads, before the tile is overwritten
    }

    if (active) {
        float inv = 1.0f / l;
        float* o = O + ((size_t)h * N + row) * D;
        o[lane] = acc0 * inv;
        o[lane + 32] = acc1 * inv;
    }
}

class FlashAttention {
public:
    FlashAttention(int H, int Hkv, int d, int /*max_N*/) : H(H), Hkv(Hkv), d(d) {}

    // Q:[H,N,d] K,V:[Hkv,N,d] O:[H,N,d], device-accessible (managed) pointers. No scratch:
    // the N x N matrices never exist -- that is the whole point.
    void forward(const float* Q, const float* K, const float* V, float* O, int N, cudaStream_t stream) {
        constexpr int RPB = 16, BC = 32, D = 64;   // tile sizes (16KB shared); tuned in profiling
        dim3 grid(H, (N + RPB - 1) / RPB);
        flash_attention_kernel<RPB, BC, D><<<grid, RPB * 32, 0, stream>>>(
            Q, K, V, O, H, Hkv, N, 1.0f / sqrtf(static_cast<float>(d)));
        checkCuda(cudaGetLastError());
    }

private:
    int H, Hkv, d;
};
