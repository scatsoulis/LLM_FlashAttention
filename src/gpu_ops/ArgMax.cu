#include "ArgMax.cuh"
#include <cuda_bf16.h>
#include "../ErrorCheck.h"

static constexpr int32_t ARGMAX_BLOCK_SIZE = 256;

__global__ void argmax_reduce_kernel(
    const __nv_bfloat16 * __restrict__ data,
    float * __restrict__ block_vals,
    int32_t * __restrict__ block_indices,
    int32_t len
) {
    extern __shared__ char smem[];
    float *s_vals = reinterpret_cast<float *>(smem);
    int32_t *s_indices = reinterpret_cast<int32_t *>(s_vals + blockDim.x);

    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t tid = threadIdx.x;

    if (idx < len) {
        s_vals[tid] = __bfloat162float(data[idx]);
        s_indices[tid] = idx;
    } else {
        s_vals[tid] = -INFINITY;
        s_indices[tid] = INT32_MAX;
    }
    __syncthreads();

    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other_val = s_vals[tid + stride];
            int32_t other_idx = s_indices[tid + stride];
            if (other_val > s_vals[tid] ||
                (other_val == s_vals[tid] && other_idx < s_indices[tid])) {
                s_vals[tid] = other_val;
                s_indices[tid] = other_idx;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_vals[blockIdx.x] = s_vals[0];
        block_indices[blockIdx.x] = s_indices[0];
    }
}

__global__ void argmax_final_kernel(
    const float * __restrict__ block_vals,
    const int32_t * __restrict__ block_indices,
    int32_t * __restrict__ result,
    int32_t num_blocks
) {
    extern __shared__ char smem[];
    float *s_vals = reinterpret_cast<float *>(smem);
    int32_t *s_indices = reinterpret_cast<int32_t *>(s_vals + blockDim.x);

    int32_t tid = threadIdx.x;

    float my_val = -INFINITY;
    int32_t my_idx = INT32_MAX;
    for (int32_t i = tid; i < num_blocks; i += blockDim.x) {
        float v = block_vals[i];
        int32_t idx = block_indices[i];
        if (v > my_val || (v == my_val && idx < my_idx)) {
            my_val = v;
            my_idx = idx;
        }
    }
    s_vals[tid] = my_val;
    s_indices[tid] = my_idx;
    __syncthreads();

    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other_val = s_vals[tid + stride];
            int32_t other_idx = s_indices[tid + stride];
            if (other_val > s_vals[tid] ||
                (other_val == s_vals[tid] && other_idx < s_indices[tid])) {
                s_vals[tid] = other_val;
                s_indices[tid] = other_idx;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *result = s_indices[0];
    }
}

ArgMax::ArgMax(int32_t len) {
    int32_t num_blocks = (len + ARGMAX_BLOCK_SIZE - 1) / ARGMAX_BLOCK_SIZE;
    size_t temp_size = num_blocks * sizeof(float) + num_blocks * sizeof(int32_t) + sizeof(int32_t);
    temp_space = std::make_shared<CudaBuffer>(temp_size);
}

int32_t *ArgMax::bf16_argmax(const std::shared_ptr<CudaBuffer> &bf16_data, cudaStream_t stream) {
    int32_t len = bf16_data->size / sizeof(__nv_bfloat16);
    int32_t num_blocks = (len + ARGMAX_BLOCK_SIZE - 1) / ARGMAX_BLOCK_SIZE;

    float *block_vals = static_cast<float *>(temp_space->data);
    int32_t *block_indices = reinterpret_cast<int32_t *>(block_vals + num_blocks);
    int32_t *result = block_indices + num_blocks;

    size_t smem_size = ARGMAX_BLOCK_SIZE * (sizeof(float) + sizeof(int32_t));
    argmax_reduce_kernel<<<num_blocks, ARGMAX_BLOCK_SIZE, smem_size, stream>>>(
        static_cast<const __nv_bfloat16 *>(bf16_data->data), block_vals, block_indices, len);

    constexpr int32_t FINAL_BLOCK_SIZE = 256;
    size_t final_smem_size = FINAL_BLOCK_SIZE * (sizeof(float) + sizeof(int32_t));
    argmax_final_kernel<<<1, FINAL_BLOCK_SIZE, final_smem_size, stream>>>(
        block_vals, block_indices, result, num_blocks);

    return result;
}
