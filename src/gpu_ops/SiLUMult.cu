#include "SiLUMult.cuh"
#include <cuda_bf16.h>
#include "../ErrorCheck.h"

__global__ void silu_mult_kernel(
    __nv_bfloat16 * __restrict__ x,
    const __nv_bfloat16 * __restrict__ y,
    int32_t len
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) return;
    float x_val = __bfloat162float(x[idx]);
    float y_val = __bfloat162float(y[idx]);
    x[idx] = __float2bfloat16(x_val / (1.0f + expf(-x_val)) * y_val);
}

void SiLUMult::silu_mult_in_place(const std::shared_ptr<CudaBuffer> &x, const std::shared_ptr<CudaBuffer> &y, cudaStream_t stream) {
    int32_t len = x->size / sizeof(__nv_bfloat16);
    constexpr int32_t BLOCK_SIZE = 256;
    int32_t num_blocks = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    silu_mult_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        static_cast<__nv_bfloat16 *>(x->data),
        static_cast<const __nv_bfloat16 *>(y->data),
        len);
}
