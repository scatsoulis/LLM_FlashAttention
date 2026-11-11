#pragma once
//
// Batched (N-token) bf16 kernels: the single-token decode ops mirrored to process
// a whole prompt of N tokens at once, for the naive prefill baseline. Each kernel
// matches the math of its decode kernel counterpart exactly (so prefill == decode), just over
// N rows instead of 1:
//   batched_matmul       <- MatrixVectorMultiply (matvec)   : out[N,M] = vecs[N,K] . mat[M,K]^T + bias
//   batched_layernorm    <- LayerNorm                        : per-row T5 RMSNorm, eps 1e-6
//   batched_rope         <- RoPE                             : GPT-NeoX, position = start + token
//   scatter_to_cache     <- the per-token cache write in Qwen2Layer
//   batched_silu_mult    <- SiLUMult                         : elementwise over N*inter
//   residual_add         <- bf16_add_kernel                  : elementwise over N*hidden
//
#include <cuda_bf16.h>
#include "../ErrorCheck.h"

// out[n,m] = sum_k mat[m,k] * vecs[n,k] (+ bias[m]).  mat is [M,K] row-major.
// grid = (M, N), block = 256 -- same per-row reduction as matvec_single_row.
template <typename in_t>
static __global__ void batched_matmul_kernel(
    int32_t M, int32_t K,
    const __nv_bfloat16 * __restrict__ mat, const __nv_bfloat16 * __restrict__ bias,
    const in_t * __restrict__ vecs, __nv_bfloat16 * __restrict__ out
) {
    int32_t m = blockIdx.x;
    int32_t n = blockIdx.y;
    int32_t tid = threadIdx.x;
    const __nv_bfloat16 *row = mat + static_cast<size_t>(m) * K;
    const in_t *v = vecs + static_cast<size_t>(n) * K;

    float sum = 0.0f;
    for (int32_t j = tid; j < K; j += blockDim.x) sum += float(row[j]) * float(v[j]);

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xFFFFFFFF, sum, o);
    __shared__ float warp_sums[8];
    int32_t wid = tid >> 5, lane = tid & 31;
    if (lane == 0) warp_sums[wid] = sum;
    __syncthreads();
    if (wid == 0) {
        float val = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) val += __shfl_down_sync(0xFFFFFFFF, val, o);
        if (lane == 0) {
            if (bias) val += float(bias[m]);
            out[static_cast<size_t>(n) * M + m] = __float2bfloat16(val);
        }
    }
}

// Per-row T5 RMSNorm: out[n,i] = w[i] * x[n,i] * rsqrt(mean_i x[n,i]^2 + 1e-6).
// grid = (N), block = 256.
static __global__ void batched_layernorm_kernel(
    int32_t H, const __nv_bfloat16 * __restrict__ in,
    const __nv_bfloat16 * __restrict__ w, __nv_bfloat16 * __restrict__ out
) {
    int32_t n = blockIdx.x, tid = threadIdx.x;
    const __nv_bfloat16 *x = in + static_cast<size_t>(n) * H;
    float ss = 0.0f;
    for (int32_t i = tid; i < H; i += blockDim.x) { float v = float(x[i]); ss += v * v; }
    __shared__ float sm[256];
    sm[tid] = ss; __syncthreads();
    for (int32_t s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) sm[tid] += sm[tid + s]; __syncthreads(); }
    float inv_rms = rsqrtf(sm[0] / static_cast<float>(H) + 1.0e-6f);
    for (int32_t i = tid; i < H; i += blockDim.x)
        out[static_cast<size_t>(n) * H + i] = __float2bfloat16(float(w[i]) * float(x[i]) * inv_rms);
}

// GPT-NeoX RoPE in place, position of token n = start + n. x is [N, row_stride], each
// row holds num_heads*head_dim. grid = (ceil(num_heads*half / 256), N), block = 256.
static __global__ void batched_rope_kernel(
    __nv_bfloat16 * __restrict__ x, int32_t num_heads, int32_t head_dim,
    int32_t start, float theta_base, int32_t row_stride
) {
    int32_t half = head_dim / 2;
    int32_t total_pairs = num_heads * half;
    int32_t pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= total_pairs) return;
    int32_t n = blockIdx.y;
    int32_t head = pair / half, theta_idx = pair % half;

    float frac = static_cast<float>(theta_idx) / static_cast<float>(half);
    float theta = powf(theta_base, -frac);
    float angle = static_cast<float>(start + n) * theta;
    float c, s; sincosf(angle, &s, &c);

    __nv_bfloat16 *row = x + static_cast<size_t>(n) * row_stride;
    int32_t i1 = head * head_dim + theta_idx, i2 = i1 + half;
    float a = float(row[i1]), b = float(row[i2]);
    row[i1] = __float2bfloat16(a * c - b * s);
    row[i2] = __float2bfloat16(b * c + a * s);
}

// Write src[N, row_size] into the KV cache [seq, num_layers, row_size] at positions
// start..start+N-1, layer `layer`. grid = (ceil(row_size/256), N), block = 256.
static __global__ void scatter_to_cache_kernel(
    const __nv_bfloat16 * __restrict__ src, __nv_bfloat16 * __restrict__ cache,
    int32_t start, int32_t row_size, int32_t layer, int32_t num_layers
) {
    int32_t n = blockIdx.y;
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= row_size) return;
    size_t p = static_cast<size_t>(start) + n;
    cache[p * num_layers * row_size + static_cast<size_t>(layer) * row_size + i] =
        src[static_cast<size_t>(n) * row_size + i];
}

// x = silu(x) * y, elementwise over `len` (= N * intermediate_size). Matches SiLUMult.
static __global__ void batched_silu_mult_kernel(
    __nv_bfloat16 * __restrict__ x, const __nv_bfloat16 * __restrict__ y, int32_t len
) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    float xv = float(x[i]);
    x[i] = __float2bfloat16(xv / (1.0f + expf(-xv)) * float(y[i]));
}

// dst += src, elementwise over `len` (= N * hidden_size).
static __global__ void residual_add_kernel(
    __nv_bfloat16 * __restrict__ dst, const __nv_bfloat16 * __restrict__ src, int32_t len
) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) dst[i] = __float2bfloat16(float(dst[i]) + float(src[i]));
}
