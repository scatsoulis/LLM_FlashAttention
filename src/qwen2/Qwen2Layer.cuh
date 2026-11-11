#pragma once

#include <cuda_bf16.h>

#include "Qwen2Config.h"
#include "../CudaBuffer.cuh"
#include <memory>

#include "../gpu_ops/MatrixVectorMultiply.cuh"
#include "../gpu_ops/LayerNorm.cuh"
#include "../ErrorCheck.h"
#include "../gpu_ops/RoPE.cuh"
#include "../gpu_ops/GroupQueryAttention.cuh"
#include "../gpu_ops/SiLUMult.cuh"

static __global__ void bf16_add_kernel(
    __nv_bfloat16 * __restrict__ dst,
    const __nv_bfloat16 * __restrict__ src,
    int32_t len
) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        dst[i] = __float2bfloat16(__bfloat162float(dst[i]) + __bfloat162float(src[i]));
    }
}

template<Qwen2Size QWEN2_SIZE>
class Qwen2Layer {
public:
    using Qwen2Config = Qwen2Config<QWEN2_SIZE>;

    Qwen2Layer(uint32_t layer_num, uint32_t max_seq_len):
    layer_num(layer_num), input_layernorm(Qwen2Config::hidden_size()), post_attention_layernorm(Qwen2Config::hidden_size()), gqa(max_seq_len) {
        scratch_norm = std::make_shared<CudaBuffer>(Qwen2Config::hidden_size() * sizeof(__nv_bfloat16));
        scratch_queries = std::make_shared<CudaBuffer>(Qwen2Config::queries_size() * sizeof(__nv_bfloat16));
        scratch_kv = std::make_shared<CudaBuffer>(Qwen2Config::keys_size() * sizeof(__nv_bfloat16));
        scratch_weighted = std::make_shared<CudaBuffer>(
            Qwen2Config::num_query_heads() * Qwen2Config::value_size() * sizeof(float));
        scratch_gate = std::make_shared<CudaBuffer>(Qwen2Config::intermediate_size() * sizeof(__nv_bfloat16));
        scratch_up = std::make_shared<CudaBuffer>(Qwen2Config::intermediate_size() * sizeof(__nv_bfloat16));
    }

    uint32_t layer_num;
    LayerNorm input_layernorm;                              // (hidden_size,)
    std::shared_ptr<CudaBuffer> q_proj_weight;              // (queries_size, hidden_size)
    std::shared_ptr<CudaBuffer> q_proj_bias;                // (queries_size,)
    std::shared_ptr<CudaBuffer> k_proj_weight;              // (keys_size, hidden_size)
    std::shared_ptr<CudaBuffer> k_proj_bias;                // (keys_size,)
    std::shared_ptr<CudaBuffer> v_proj_weight;              // (values_size, hidden_size)
    std::shared_ptr<CudaBuffer> v_proj_bias;                // (values_size,)
    std::shared_ptr<CudaBuffer> o_proj_weight;              // (hidden_size, queries_size)
    LayerNorm post_attention_layernorm;                     // (hidden_size,)
    std::shared_ptr<CudaBuffer> up_proj_weight;             // (intermediate_size, intermediate_size)
    std::shared_ptr<CudaBuffer> gate_proj_weight;           // (intermediate_size, hidden_size)
    std::shared_ptr<CudaBuffer> down_proj_weight;           // (hidden_size, intermediate_size)

    GroupQueryAttention<QWEN2_SIZE> gqa;
    std::shared_ptr<CudaBuffer> scratch_norm;
    std::shared_ptr<CudaBuffer> scratch_queries;
    std::shared_ptr<CudaBuffer> scratch_kv;
    std::shared_ptr<CudaBuffer> scratch_weighted;
    std::shared_ptr<CudaBuffer> scratch_gate;
    std::shared_ptr<CudaBuffer> scratch_up;

    void forward(const std::shared_ptr<CudaBuffer>& k_cache, const std::shared_ptr<CudaBuffer> &v_cache, const std::shared_ptr<CudaBuffer> &hidden_state, int32_t seq_len, cudaStream_t stream) {
        auto *hs = static_cast<__nv_bfloat16*>(hidden_state->data);
        auto *norm = static_cast<__nv_bfloat16*>(scratch_norm->data);
        auto *queries = static_cast<__nv_bfloat16*>(scratch_queries->data);
        auto *kv = static_cast<__nv_bfloat16*>(scratch_kv->data);

        // Pre-attention layernorm
        input_layernorm.normalize_hidden_state(hidden_state, scratch_norm, stream);

        // Q projection + RoPE
        MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(
            Qwen2Config::queries_size(), Qwen2Config::hidden_size(),
            static_cast<__nv_bfloat16*>(q_proj_weight->data),
            static_cast<__nv_bfloat16*>(q_proj_bias->data),
            norm, queries, stream);
        RoPE::apply_rope_to_qk(queries, Qwen2Config::num_query_heads(), Qwen2Config::head_size(),
            seq_len - 1, Qwen2Config::rope_theta_base(), stream);

        // K projection + RoPE + write to cache
        MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(
            Qwen2Config::keys_size(), Qwen2Config::hidden_size(),
            static_cast<__nv_bfloat16*>(k_proj_weight->data),
            static_cast<__nv_bfloat16*>(k_proj_bias->data),
            norm, kv, stream);
        RoPE::apply_rope_to_qk(kv, Qwen2Config::num_kv_heads(), Qwen2Config::head_size(),
            seq_len - 1, Qwen2Config::rope_theta_base(), stream);
        {
            size_t offset = (static_cast<size_t>(seq_len - 1) * Qwen2Config::num_layers() * Qwen2Config::keys_size()
                            + layer_num * Qwen2Config::keys_size());
            checkCuda(cudaMemcpyAsync(
                static_cast<__nv_bfloat16*>(k_cache->data) + offset,
                kv, Qwen2Config::keys_size() * sizeof(__nv_bfloat16),
                cudaMemcpyDeviceToDevice, stream));
        }

        // V projection + write to cache
        MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(
            Qwen2Config::values_size(), Qwen2Config::hidden_size(),
            static_cast<__nv_bfloat16*>(v_proj_weight->data),
            static_cast<__nv_bfloat16*>(v_proj_bias->data),
            norm, kv, stream);
        {
            size_t offset = (static_cast<size_t>(seq_len - 1) * Qwen2Config::num_layers() * Qwen2Config::values_size()
                            + layer_num * Qwen2Config::values_size());
            checkCuda(cudaMemcpyAsync(
                static_cast<__nv_bfloat16*>(v_cache->data) + offset,
                kv, Qwen2Config::values_size() * sizeof(__nv_bfloat16),
                cudaMemcpyDeviceToDevice, stream));
        }

        // Group Query Attention
        gqa.sdpa(queries,
            static_cast<__nv_bfloat16*>(k_cache->data),
            static_cast<__nv_bfloat16*>(v_cache->data),
            static_cast<float*>(scratch_weighted->data),
            layer_num, seq_len, stream);

        // Output projection (float32 input from GQA)
        MatrixVectorMultiply::bf16_matmul<float>(
            Qwen2Config::hidden_size(), Qwen2Config::queries_size(),
            static_cast<__nv_bfloat16*>(o_proj_weight->data),
            nullptr,
            static_cast<float*>(scratch_weighted->data),
            norm, stream);

        // Residual connection
        {
            constexpr int BLOCK = 256;
            int grid = (Qwen2Config::hidden_size() + BLOCK - 1) / BLOCK;
            bf16_add_kernel<<<grid, BLOCK, 0, stream>>>(hs, norm, Qwen2Config::hidden_size());
        }

        // Post-attention layernorm
        post_attention_layernorm.normalize_hidden_state(hidden_state, scratch_norm, stream);

        // Gate projection
        MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(
            Qwen2Config::intermediate_size(), Qwen2Config::hidden_size(),
            static_cast<__nv_bfloat16*>(gate_proj_weight->data),
            nullptr, norm,
            static_cast<__nv_bfloat16*>(scratch_gate->data), stream);

        // Up projection
        MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(
            Qwen2Config::intermediate_size(), Qwen2Config::hidden_size(),
            static_cast<__nv_bfloat16*>(up_proj_weight->data),
            nullptr, norm,
            static_cast<__nv_bfloat16*>(scratch_up->data), stream);

        // SiLU + multiply
        SiLUMult::silu_mult_in_place(scratch_gate, scratch_up, stream);

        // Down projection
        MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(
            Qwen2Config::hidden_size(), Qwen2Config::intermediate_size(),
            static_cast<__nv_bfloat16*>(down_proj_weight->data),
            nullptr,
            static_cast<__nv_bfloat16*>(scratch_gate->data),
            norm, stream);

        // Residual connection
        {
            constexpr int BLOCK = 256;
            int grid = (Qwen2Config::hidden_size() + BLOCK - 1) / BLOCK;
            bf16_add_kernel<<<grid, BLOCK, 0, stream>>>(hs, norm, Qwen2Config::hidden_size());
        }
    }
};
