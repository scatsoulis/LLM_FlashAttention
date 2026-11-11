#pragma once

#include "../qwen2/Qwen2Config.h"
#include "../CudaBuffer.cuh"
#include <cuda_bf16.h>
#include <memory>
#include "../ErrorCheck.h"

static __global__ void gqa_sdpa_kernel(
    const __nv_bfloat16 * __restrict__ queries,
    const __nv_bfloat16 * __restrict__ k_cache,
    const __nv_bfloat16 * __restrict__ v_cache,
    float * __restrict__ output,
    int32_t seq_len,
    int32_t num_query_heads,
    int32_t num_kv_heads,
    int32_t head_size,
    int32_t value_size,
    int32_t k_seq_stride,
    int32_t v_seq_stride,
    int32_t k_head_offset,
    int32_t v_head_offset
) {
    int32_t head_idx = blockIdx.x;
    int32_t tid = threadIdx.x;
    int32_t half = head_size / 2;

    int32_t kv_head = head_idx * num_kv_heads / num_query_heads;

    float q0 = __bfloat162float(queries[head_idx * head_size + tid]);
    float q1 = __bfloat162float(queries[head_idx * head_size + tid + half]);

    const __nv_bfloat16 *k_base = k_cache + k_head_offset + kv_head * head_size;
    const __nv_bfloat16 *v_base = v_cache + v_head_offset + kv_head * value_size;

    float inv_sqrt_d = rsqrtf(static_cast<float>(head_size));

    float m = -INFINITY;
    float d = 0.0f;
    float o0 = 0.0f, o1 = 0.0f;

    for (int32_t s = 0; s < seq_len; s++) {
        int32_t k_off = s * k_seq_stride;
        float k0 = __bfloat162float(k_base[k_off + tid]);
        float k1 = __bfloat162float(k_base[k_off + tid + half]);
        float partial = q0 * k0 + q1 * k1;

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
        }
        float score = __shfl_sync(0xFFFFFFFF, partial, 0) * inv_sqrt_d;

        float new_m = fmaxf(m, score);
        float correction = expf(m - new_m);
        float exp_s = expf(score - new_m);
        d = d * correction + exp_s;

        int32_t v_off = s * v_seq_stride;
        float v0 = __bfloat162float(v_base[v_off + tid]);
        float v1 = __bfloat162float(v_base[v_off + tid + half]);
        o0 = o0 * correction + exp_s * v0;
        o1 = o1 * correction + exp_s * v1;

        m = new_m;
    }

    float inv_d = 1.0f / d;
    o0 *= inv_d;
    o1 *= inv_d;

    output[head_idx * value_size + tid] = o0;
    output[head_idx * value_size + tid + half] = o1;
}

template<Qwen2Size QWEN2_SIZE>
class GroupQueryAttention {
public:
    using Qwen2Config = Qwen2Config<QWEN2_SIZE>;

    explicit GroupQueryAttention(int32_t max_seq_len) {
    }

    void sdpa(__nv_bfloat16 *queries, __nv_bfloat16 *k_cache, __nv_bfloat16 *v_cache, float *weighted_values, int32_t layer_num, int32_t seq_len, cudaStream_t stream) {
        static_assert(Qwen2Config::head_size() == Qwen2Config::value_size(),
            "GQA kernel requires head_size == value_size");
        constexpr int32_t BLOCK_SIZE = Qwen2Config::head_size() / 2;
        int32_t grid = Qwen2Config::num_query_heads();

        gqa_sdpa_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            queries, k_cache, v_cache, weighted_values,
            seq_len,
            Qwen2Config::num_query_heads(),
            Qwen2Config::num_kv_heads(),
            Qwen2Config::head_size(),
            Qwen2Config::value_size(),
            Qwen2Config::num_layers() * Qwen2Config::keys_size(),
            Qwen2Config::num_layers() * Qwen2Config::values_size(),
            layer_num * Qwen2Config::keys_size(),
            layer_num * Qwen2Config::values_size()
        );
    }
};
