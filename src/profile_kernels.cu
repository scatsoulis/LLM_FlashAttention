#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <nvtx3/nvToolsExt.h>
#include "gpu_ops/ArgMax.cuh"
#include "gpu_ops/LayerNorm.cuh"
#include "gpu_ops/MatrixVectorMultiply.cuh"
#include "gpu_ops/RoPE.cuh"
#include "gpu_ops/SiLUMult.cuh"
#include "CudaBuffer.cuh"
#include "ErrorCheck.h"

static void fill_random(CudaBuffer &buf, std::mt19937 &gen) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    auto *ptr = static_cast<__nv_bfloat16 *>(buf.data);
    for (size_t i = 0; i < buf.size / sizeof(__nv_bfloat16); i++) {
        ptr[i] = __float2bfloat16(dist(gen));
    }
}

int main() {
    // Qwen2 0.5B dimensions
    constexpr int32_t hidden_size = 896;
    constexpr int32_t intermediate_size = 4864;
    constexpr int32_t vocab_size = 151936;
    constexpr int32_t num_query_heads = 14;
    constexpr int32_t num_kv_heads = 2;
    constexpr int32_t head_size = 64;

    cudaStream_t stream;
    checkCuda(cudaStreamCreate(&stream));
    std::mt19937 gen(42);

    // --- ArgMax: over vocab logits (151936) ---
    {
        auto buf = std::make_shared<CudaBuffer>(vocab_size * sizeof(__nv_bfloat16));
        fill_random(*buf, gen);
        ArgMax argmax(vocab_size);

        nvtxRangePush("ArgMax");
        argmax.bf16_argmax(buf, stream);
        checkCuda(cudaStreamSynchronize(stream));
        nvtxRangePop();
    }

    // --- LayerNorm: hidden_size = 896 ---
    {
        auto input = std::make_shared<CudaBuffer>(hidden_size * sizeof(__nv_bfloat16));
        auto weights = std::make_shared<CudaBuffer>(hidden_size * sizeof(__nv_bfloat16));
        auto output = std::make_shared<CudaBuffer>(hidden_size * sizeof(__nv_bfloat16));
        fill_random(*input, gen);
        fill_random(*weights, gen);

        LayerNorm ln(hidden_size);
        ln.weights = weights;

        nvtxRangePush("LayerNorm");
        ln.normalize_hidden_state(input, output, stream);
        checkCuda(cudaStreamSynchronize(stream));
        nvtxRangePop();
    }

    // --- MatVecMultiply: gate_proj (4864, 896) - largest FFN projection ---
    {
        int32_t m = intermediate_size, k = hidden_size;
        auto mat = std::make_shared<CudaBuffer>(m * k * sizeof(__nv_bfloat16));
        auto vec = std::make_shared<CudaBuffer>(k * sizeof(__nv_bfloat16));
        auto bias = std::make_shared<CudaBuffer>(m * sizeof(__nv_bfloat16));
        auto out = std::make_shared<CudaBuffer>(m * sizeof(__nv_bfloat16));
        fill_random(*mat, gen);
        fill_random(*vec, gen);

        nvtxRangePush("MatVec_gate_proj_4864x896");
        MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(m, k,
            static_cast<__nv_bfloat16 *>(mat->data), nullptr,
            static_cast<__nv_bfloat16 *>(vec->data),
            static_cast<__nv_bfloat16 *>(out->data), stream);
        checkCuda(cudaStreamSynchronize(stream));
        nvtxRangePop();
    }

    // --- MatVecMultiply: down_proj (896, 4864) ---
    {
        int32_t m = hidden_size, k = intermediate_size;
        auto mat = std::make_shared<CudaBuffer>(m * k * sizeof(__nv_bfloat16));
        auto vec = std::make_shared<CudaBuffer>(k * sizeof(__nv_bfloat16));
        auto out = std::make_shared<CudaBuffer>(m * sizeof(__nv_bfloat16));
        fill_random(*mat, gen);
        fill_random(*vec, gen);

        nvtxRangePush("MatVec_down_proj_896x4864");
        MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(m, k,
            static_cast<__nv_bfloat16 *>(mat->data), nullptr,
            static_cast<__nv_bfloat16 *>(vec->data),
            static_cast<__nv_bfloat16 *>(out->data), stream);
        checkCuda(cudaStreamSynchronize(stream));
        nvtxRangePop();
    }

    // --- MatVecMultiply: q_proj (896, 896) with bias ---
    {
        int32_t m = hidden_size, k = hidden_size;
        auto mat = std::make_shared<CudaBuffer>(m * k * sizeof(__nv_bfloat16));
        auto vec = std::make_shared<CudaBuffer>(k * sizeof(__nv_bfloat16));
        auto bias = std::make_shared<CudaBuffer>(m * sizeof(__nv_bfloat16));
        auto out = std::make_shared<CudaBuffer>(m * sizeof(__nv_bfloat16));
        fill_random(*mat, gen);
        fill_random(*vec, gen);
        fill_random(*bias, gen);

        nvtxRangePush("MatVec_q_proj_896x896_bias");
        MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(m, k,
            static_cast<__nv_bfloat16 *>(mat->data),
            static_cast<__nv_bfloat16 *>(bias->data),
            static_cast<__nv_bfloat16 *>(vec->data),
            static_cast<__nv_bfloat16 *>(out->data), stream);
        checkCuda(cudaStreamSynchronize(stream));
        nvtxRangePop();
    }

    // --- MatVecMultiply: output logits (151936, 896) - largest overall ---
    {
        int32_t m = vocab_size, k = hidden_size;
        auto mat = std::make_shared<CudaBuffer>((size_t)m * k * sizeof(__nv_bfloat16));
        auto vec = std::make_shared<CudaBuffer>(k * sizeof(__nv_bfloat16));
        auto out = std::make_shared<CudaBuffer>(m * sizeof(__nv_bfloat16));
        fill_random(*mat, gen);
        fill_random(*vec, gen);

        nvtxRangePush("MatVec_logits_151936x896");
        MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(m, k,
            static_cast<__nv_bfloat16 *>(mat->data), nullptr,
            static_cast<__nv_bfloat16 *>(vec->data),
            static_cast<__nv_bfloat16 *>(out->data), stream);
        checkCuda(cudaStreamSynchronize(stream));
        nvtxRangePop();
    }

    // --- RoPE: queries (14 heads, head_dim=64) ---
    {
        auto queries = std::make_shared<CudaBuffer>(num_query_heads * head_size * sizeof(__nv_bfloat16));
        fill_random(*queries, gen);

        nvtxRangePush("RoPE_queries");
        RoPE::apply_rope_to_qk(static_cast<__nv_bfloat16 *>(queries->data),
            num_query_heads, head_size, 100, 1e6f, stream);
        checkCuda(cudaStreamSynchronize(stream));
        nvtxRangePop();
    }

    // --- RoPE: keys (2 heads, head_dim=64) ---
    {
        auto keys = std::make_shared<CudaBuffer>(num_kv_heads * head_size * sizeof(__nv_bfloat16));
        fill_random(*keys, gen);

        nvtxRangePush("RoPE_keys");
        RoPE::apply_rope_to_qk(static_cast<__nv_bfloat16 *>(keys->data),
            num_kv_heads, head_size, 100, 1e6f, stream);
        checkCuda(cudaStreamSynchronize(stream));
        nvtxRangePop();
    }

    // --- SiLUMult: intermediate_size = 4864 ---
    {
        auto x = std::make_shared<CudaBuffer>(intermediate_size * sizeof(__nv_bfloat16));
        auto y = std::make_shared<CudaBuffer>(intermediate_size * sizeof(__nv_bfloat16));
        fill_random(*x, gen);
        fill_random(*y, gen);

        nvtxRangePush("SiLUMult");
        SiLUMult::silu_mult_in_place(x, y, stream);
        checkCuda(cudaStreamSynchronize(stream));
        nvtxRangePop();
    }

    checkCuda(cudaStreamDestroy(stream));
    std::cout << "Profiling complete." << std::endl;
    return 0;
}
