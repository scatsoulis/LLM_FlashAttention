#include "RoPE.cuh"
#include <cuda_bf16.h>
#include "../ErrorCheck.h"

__global__ void rope_kernel(
    __nv_bfloat16 * __restrict__ x,
    int32_t num_heads, int32_t head_dim,
    int32_t position_idx, float theta_base
) {
    int32_t half_dim = head_dim / 2;
    int32_t total_pairs = num_heads * half_dim;
    int32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_idx >= total_pairs) return;

    int32_t head = pair_idx / half_dim;
    int32_t theta_idx = pair_idx % half_dim;

    float theta_frac = static_cast<float>(theta_idx) / static_cast<float>(half_dim);
    float theta = powf(theta_base, -theta_frac);
    float angle = static_cast<float>(position_idx) * theta;
    float cos_val, sin_val;
    sincosf(angle, &sin_val, &cos_val);

    int32_t idx_first = head * head_dim + theta_idx;
    int32_t idx_second = idx_first + half_dim;

    float x_first = __bfloat162float(x[idx_first]);
    float x_second = __bfloat162float(x[idx_second]);

    x[idx_first] = __float2bfloat16(x_first * cos_val - x_second * sin_val);
    x[idx_second] = __float2bfloat16(x_second * cos_val + x_first * sin_val);
}

void RoPE::apply_rope_to_qk(__nv_bfloat16 *x, int32_t num_heads, int32_t head_dim,
        int32_t position_idx, float theta_base, cudaStream_t stream) {
    int32_t total_pairs = num_heads * (head_dim / 2);
    constexpr int32_t BLOCK_SIZE = 256;
    int32_t num_blocks = (total_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    rope_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(x, num_heads, head_dim, position_idx, theta_base);
}
