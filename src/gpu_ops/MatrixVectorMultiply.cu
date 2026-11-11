#include "MatrixVectorMultiply.cuh"
#include "../ErrorCheck.h"

template<typename input_float_t>
__global__ void matvec_single_row(
    int32_t k,
    const __nv_bfloat16 * __restrict__ mat,
    const __nv_bfloat16 * __restrict__ bias,
    const input_float_t * __restrict__ vec,
    __nv_bfloat16 * __restrict__ out
) {
    int32_t row = blockIdx.x;
    int32_t tid = threadIdx.x;
    const __nv_bfloat16 *row_data = mat + static_cast<int64_t>(row) * k;

    float sum = 0.0f;
    for (int32_t j = tid; j < k; j += blockDim.x) {
        sum += float(row_data[j]) * float(vec[j]);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    __shared__ float warp_sums[8];
    int32_t warp_id = tid >> 5;
    int32_t lane = tid & 31;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (lane == 0) {
            if (bias) val += float(bias[row]);
            out[row] = __float2bfloat16(val);
        }
    }
}

template<typename input_float_t, int ROWS_PER_BLOCK>
__global__ void matvec_multi_row(
    int32_t m, int32_t k,
    const __nv_bfloat16 * __restrict__ mat,
    const __nv_bfloat16 * __restrict__ bias,
    const input_float_t * __restrict__ vec,
    __nv_bfloat16 * __restrict__ out
) {
    constexpr int THREADS_PER_ROW = 32;
    constexpr int TILE_SIZE = ROWS_PER_BLOCK * THREADS_PER_ROW;

    extern __shared__ float s_vec[];

    int32_t local_row = threadIdx.x / THREADS_PER_ROW;
    int32_t lane = threadIdx.x & (THREADS_PER_ROW - 1);
    int32_t row = blockIdx.x * ROWS_PER_BLOCK + local_row;

    const __nv_bfloat16 *row_data = (row < m) ? mat + static_cast<int64_t>(row) * k : nullptr;

    float sum = 0.0f;

    for (int32_t tile_start = 0; tile_start < k; tile_start += TILE_SIZE) {
        int32_t load_idx = tile_start + threadIdx.x;
        s_vec[threadIdx.x] = (load_idx < k) ? float(vec[load_idx]) : 0.0f;
        __syncthreads();

        if (row_data) {
            int32_t tile_end = k - tile_start;
            if (tile_end > TILE_SIZE) tile_end = TILE_SIZE;
            for (int32_t j = lane; j < tile_end; j += THREADS_PER_ROW) {
                sum += float(row_data[tile_start + j]) * s_vec[j];
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int offset = THREADS_PER_ROW / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    if (lane == 0 && row < m) {
        if (bias) sum += float(bias[row]);
        out[row] = __float2bfloat16(sum);
    }
}

template<typename input_float_t>
void MatrixVectorMultiply::bf16_matmul(int32_t m, int32_t k, __nv_bfloat16 *mat, __nv_bfloat16 *bias, input_float_t *vec, __nv_bfloat16 *out, cudaStream_t stream) {
    constexpr int32_t BLOCK_SIZE = 256;
    if (m >= 4096) {
        constexpr int RPB = 8;
        int32_t grid = (m + RPB - 1) / RPB;
        size_t smem = BLOCK_SIZE * sizeof(float);
        matvec_multi_row<input_float_t, RPB><<<grid, BLOCK_SIZE, smem, stream>>>(m, k, mat, bias, vec, out);
    } else {
        matvec_single_row<input_float_t><<<m, BLOCK_SIZE, 0, stream>>>(k, mat, bias, vec, out);
    }
}

template void MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(int32_t m, int32_t k, __nv_bfloat16 *mat, __nv_bfloat16 *bias, __nv_bfloat16 *vec, __nv_bfloat16 *out, cudaStream_t stream);
template void MatrixVectorMultiply::bf16_matmul<float>(int32_t m, int32_t k, __nv_bfloat16 *mat, __nv_bfloat16 *bias, float *vec, __nv_bfloat16 *out, cudaStream_t stream);
