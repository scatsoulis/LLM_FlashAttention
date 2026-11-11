#include "LayerNorm.cuh"
#include <cuda_bf16.h>
#include "../ErrorCheck.h"

static constexpr int32_t BLOCK_SIZE = 256;

__global__ void layer_norm_partial_sum(
    const __nv_bfloat16 * __restrict__ input,
    float * __restrict__ partial_sums,
    int32_t len
) {
    extern __shared__ float smem[];
    int32_t tid = threadIdx.x;
    int32_t gid = blockIdx.x * blockDim.x + tid;
    int32_t stride = blockDim.x * gridDim.x;

    float my_sum = 0.0f;
    for (int32_t i = gid; i < len; i += stride) {
        float val = __bfloat162float(input[i]);
        my_sum += val * val;
    }
    smem[tid] = my_sum;
    __syncthreads();

    for (int32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = smem[0];
    }
}

__global__ void layer_norm_normalize(
    const __nv_bfloat16 * __restrict__ input,
    const __nv_bfloat16 * __restrict__ weights,
    __nv_bfloat16 * __restrict__ output,
    const float * __restrict__ partial_sums,
    int32_t len, int32_t num_partials
) {
    float total = 0.0f;
    for (int32_t i = 0; i < num_partials; i++) {
        total += partial_sums[i];
    }
    float inv_rms = rsqrtf(total / static_cast<float>(len) + 1.0e-6f);

    int32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride = blockDim.x * gridDim.x;
    for (int32_t i = gid; i < len; i += stride) {
        float val = __bfloat162float(input[i]);
        float w = __bfloat162float(weights[i]);
        output[i] = __float2bfloat16(w * val * inv_rms);
    }
}

LayerNorm::LayerNorm(int32_t len) : len_(len) {
    int32_t num_blocks = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    temp_storage = std::make_shared<CudaBuffer>(num_blocks * sizeof(float));
}

void LayerNorm::normalize_hidden_state(const std::shared_ptr<CudaBuffer> &hidden_state, const std::shared_ptr<CudaBuffer> &output, cudaStream_t stream) {
    int32_t len = hidden_state->size / sizeof(__nv_bfloat16);
    int32_t num_blocks = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t smem_size = BLOCK_SIZE * sizeof(float);

    layer_norm_partial_sum<<<num_blocks, BLOCK_SIZE, smem_size, stream>>>(
        static_cast<const __nv_bfloat16 *>(hidden_state->data),
        static_cast<float *>(temp_storage->data),
        len);

    layer_norm_normalize<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(hidden_state->data),
        static_cast<const __nv_bfloat16 *>(weights->data),
        static_cast<__nv_bfloat16 *>(output->data),
        static_cast<const float *>(temp_storage->data),
        len, num_blocks);
}
