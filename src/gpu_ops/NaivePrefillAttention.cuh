#pragma once
//
// Naive CUDA baseline for PREFILL causal attention (query length N), FP32, GQA.
//
// This is the "GPU naive" cell of the project's comparison matrix: the three-kernel, 
// fully-materialized attention that FlashAttention
// later replaces. It deliberately writes the full N x N score matrix S to HBM, reads
// it back for softmax into a second N x N matrix P in HBM, then reads P back for P*V
// -- i.e. ~2N^2 + 2Nd HBM accesses. That O(N^2) traffic is exactly the cost the fused
// kernel removes, so this baseline is what we measure the speedup against. No tiling,
// no shared-memory reuse: correctness and an honest memory-bound baseline, nothing more.
//
// Layout matches src/cpu_ref/AttentionCPU.h so the test can compare directly:
//   Q : [H, N, d]      K,V : [Hkv, N, d]      O : [H, N, d]
//   scratch S, P : [H, N, N]
// GQA broadcast: kv(h) = h * Hkv / H. Scale: 1/sqrt(d). Causal: query i attends j <= i.
//
#include "../CudaBuffer.cuh"
#include "../ErrorCheck.h"
#include <cmath>

// (1) S[h,i,j] = scale * <Q[h,i], K[kv(h),j]>, with causal mask j > i -> -inf.
//     grid = (ceil(N/BS), N, H), block = BS over the j (key) axis.
static __global__ void naive_qk_scores_kernel(
    const float* __restrict__ Q, const float* __restrict__ K, float* __restrict__ S,
    int H, int Hkv, int d, int N, float scale
) {
    int h = blockIdx.z;
    int i = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    int g = h * Hkv / H;
    size_t s_idx = ((size_t)h * N + i) * N + j;
    if (j > i) { S[s_idx] = -INFINITY; return; }                 // causal: future key forbidden
    const float* q = Q + ((size_t)h * N + i) * d;
    const float* k = K + ((size_t)g * N + j) * d;
    float dot = 0.0f;
    for (int t = 0; t < d; ++t) dot += q[t] * k[t];
    S[s_idx] = dot * scale;
}

// (2) row-wise softmax over j with max-subtraction: P = softmax(S) row by row.
//     grid = (N, H), one block per (h,i) row; block reduces over the N-length row.
template <int BS>
static __global__ void naive_softmax_rows_kernel(
    const float* __restrict__ S, float* __restrict__ P, int N
) {
    int i = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;
    const float* row = S + ((size_t)h * gridDim.x + i) * N;       // gridDim.x == N
    float* prow      = P + ((size_t)h * gridDim.x + i) * N;
    __shared__ float red[BS];

    float m = -INFINITY;                                         // running max over this row
    for (int j = tid; j < N; j += BS) m = fmaxf(m, row[j]);
    red[tid] = m; __syncthreads();
    for (int s = BS / 2; s > 0; s >>= 1) { if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]); __syncthreads(); }
    float row_max = red[0]; __syncthreads();

    float l = 0.0f;                                              // sum of exp(s - max); masked -inf -> 0
    for (int j = tid; j < N; j += BS) l += expf(row[j] - row_max);
    red[tid] = l; __syncthreads();
    for (int s = BS / 2; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
    float inv = 1.0f / red[0];                                   // red[0] >= 1 (diagonal term present)

    for (int j = tid; j < N; j += BS) prow[j] = expf(row[j] - row_max) * inv;
}

// (3) O[h,i,k] = sum_j P[h,i,j] * V[kv(h),j,k].
//     grid = (ceil(d/BS), N, H), block = BS over the head-dim axis (coalesced V/O loads).
static __global__ void naive_pv_kernel(
    const float* __restrict__ P, const float* __restrict__ V, float* __restrict__ O,
    int H, int Hkv, int d, int N
) {
    int h = blockIdx.z;
    int i = blockIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= d) return;
    int g = h * Hkv / H;
    const float* prow = P + ((size_t)h * N + i) * N;
    float acc = 0.0f;
    for (int j = 0; j < N; ++j) acc += prow[j] * V[((size_t)g * N + j) * d + k];
    O[((size_t)h * N + i) * d + k] = acc;
}

class NaivePrefillAttention {
public:
    NaivePrefillAttention(int H, int Hkv, int d, int max_N)
        : H(H), Hkv(Hkv), d(d),
          scores((size_t)H * max_N * max_N * sizeof(float)),
          probs((size_t)H * max_N * max_N * sizeof(float)) {}

    // Q:[H,N,d] K,V:[Hkv,N,d] O:[H,N,d], all device-accessible (managed) pointers.
    void forward(const float* Q, const float* K, const float* V, float* O, int N, cudaStream_t stream) {
        float* S = static_cast<float*>(scores.data);
        float* P = static_cast<float*>(probs.data);
        float scale = 1.0f / sqrtf(static_cast<float>(d));

        constexpr int BS_J = 128;
        dim3 g1((N + BS_J - 1) / BS_J, N, H);
        naive_qk_scores_kernel<<<g1, BS_J, 0, stream>>>(Q, K, S, H, Hkv, d, N, scale);

        constexpr int BS_S = 256;
        dim3 g2(N, H);
        naive_softmax_rows_kernel<BS_S><<<g2, BS_S, 0, stream>>>(S, P, N);

        constexpr int BS_K = 64;
        dim3 g3((d + BS_K - 1) / BS_K, N, H);
        naive_pv_kernel<<<g3, BS_K, 0, stream>>>(P, V, O, H, Hkv, d, N);

        checkCuda(cudaGetLastError());
    }

private:
    int H, Hkv, d;
    CudaBuffer scores, probs;   // [H, max_N, max_N] each, materialized in HBM (the naive cost)
};
