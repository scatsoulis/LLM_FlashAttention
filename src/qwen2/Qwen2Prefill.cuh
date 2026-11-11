#pragma once
//
// Naive N-query PREFILL forward for Qwen2-0.5B on GPU (bf16), reusing the weights of a
// loaded Qwen2Model. This is the "mirror the decode kernel to N tokens" baseline: it processes a
// whole prompt of N tokens in ONE batched pass instead of looping the query-length-1
// decode N times (which is what main.cu::send_text does today). The decode path is left
// untouched; this is an additive path for prefill + benchmarking.
//
// Attention here is the decode kernel (gqa_sdpa_kernel) generalized to N query
// tokens with causal masking: query token i (global position start+i) streams the online
// softmax over key positions 0..start+i from the KV cache. No tiling / no shared-memory
// K/V reuse -- that is what the FlashAttention kernel will add next.
//
#include <cuda_bf16.h>
#include <vector>
#include <memory>
#include "Qwen2Config.h"
#include "Qwen2Model.cuh"
#include "../CudaBuffer.cuh"
#include "../ErrorCheck.h"
#include "../gpu_ops/BatchedPrefillOps.cuh"

// Causal prefill attention. grid = (num_query_heads, N), block = head_size/2.
// queries: [N, num_query_heads*head_size]   output: [N, num_query_heads*value_size] (float)
static __global__ void prefill_attention_kernel(
    const __nv_bfloat16 * __restrict__ queries,
    const __nv_bfloat16 * __restrict__ k_cache,
    const __nv_bfloat16 * __restrict__ v_cache,
    float * __restrict__ output,
    int32_t start, int32_t num_query_heads, int32_t num_kv_heads,
    int32_t head_size, int32_t value_size,
    int32_t k_seq_stride, int32_t v_seq_stride, int32_t k_head_offset, int32_t v_head_offset
) {
    int32_t head_idx = blockIdx.x;
    int32_t token = blockIdx.y;
    int32_t tid = threadIdx.x;
    int32_t half = head_size / 2;
    int32_t kv_head = head_idx * num_kv_heads / num_query_heads;
    int32_t pos = start + token;   // global position of this query (attends keys 0..pos)

    const __nv_bfloat16 *q = queries + static_cast<size_t>(token) * (num_query_heads * head_size)
                             + head_idx * head_size;
    float q0 = __bfloat162float(q[tid]);
    float q1 = __bfloat162float(q[tid + half]);

    const __nv_bfloat16 *k_base = k_cache + k_head_offset + kv_head * head_size;
    const __nv_bfloat16 *v_base = v_cache + v_head_offset + kv_head * value_size;
    float inv_sqrt_d = rsqrtf(static_cast<float>(head_size));

    float m = -INFINITY, d = 0.0f, o0 = 0.0f, o1 = 0.0f;
    for (int32_t s = 0; s <= pos; s++) {          // causal: only keys up to this query's position
        int32_t k_off = s * k_seq_stride;
        float k0 = __bfloat162float(k_base[k_off + tid]);
        float k1 = __bfloat162float(k_base[k_off + tid + half]);
        float partial = q0 * k0 + q1 * k1;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
        float score = __shfl_sync(0xFFFFFFFF, partial, 0) * inv_sqrt_d;

        float new_m = fmaxf(m, score);
        float correction = expf(m - new_m);
        float exp_s = expf(score - new_m);
        d = d * correction + exp_s;
        int32_t v_off = s * v_seq_stride;
        o0 = o0 * correction + exp_s * __bfloat162float(v_base[v_off + tid]);
        o1 = o1 * correction + exp_s * __bfloat162float(v_base[v_off + tid + half]);
        m = new_m;
    }
    float inv_d = 1.0f / d;
    size_t out_base = static_cast<size_t>(token) * (num_query_heads * value_size) + head_idx * value_size;
    output[out_base + tid] = o0 * inv_d;
    output[out_base + tid + half] = o1 * inv_d;
}

// bf16, cache-aware tiled FlashAttention for the in-model prefill path. Same online-softmax
// math as prefill_attention_kernel, but K/V tiles are streamed into shared memory and reused
// across RPB query rows per block (the FlashAttention.cuh kernel, adapted to read the bf16 KV
// cache with a global-position `start` offset and write float output for o_proj).
// grid = (num_query_heads, ceil(N/RPB)), block = RPB*32.  D = head_size = value_size = 64.
template <int RPB, int BC, int D>
static __global__ void flash_attention_cache_kernel(
    const __nv_bfloat16* __restrict__ queries,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    float* __restrict__ output,
    int N, int start, int num_query_heads, int num_kv_heads, int value_size,
    int k_seq_stride, int v_seq_stride, int k_head_offset, int v_head_offset, float scale
) {
    static_assert(D == 64, "warp-per-row mapping assumes head_dim 64");
    const int h = blockIdx.x;
    const int row_base = blockIdx.y * RPB;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = row_base + warp;
    const int kv_head = h * num_kv_heads / num_query_heads;
    const int pos = start + row;                       // global position; attends keys 0..pos

    __shared__ float Ksh[BC * D];
    __shared__ float Vsh[BC * D];

    const bool active = (row < N);
    float q0 = 0.0f, q1 = 0.0f;
    if (active) {
        const __nv_bfloat16* q = queries + (size_t)row * (num_query_heads * D) + h * D;
        q0 = __bfloat162float(q[lane]);
        q1 = __bfloat162float(q[lane + 32]);
    }
    float m = -INFINITY, l = 0.0f, acc0 = 0.0f, acc1 = 0.0f;

    const int block_max_pos = start + min(row_base + RPB - 1, N - 1);
    const __nv_bfloat16* k_base = k_cache + k_head_offset + kv_head * D;
    const __nv_bfloat16* v_base = v_cache + v_head_offset + kv_head * value_size;

    for (int kt = 0; kt <= block_max_pos; kt += BC) {
        for (int idx = threadIdx.x; idx < BC * D; idx += blockDim.x) {
            const int c = idx / D, e = idx % D, gpos = kt + c;
            float kk = 0.0f, vv = 0.0f;
            if (gpos <= block_max_pos) {
                kk = __bfloat162float(k_base[(size_t)gpos * k_seq_stride + e]);
                vv = __bfloat162float(v_base[(size_t)gpos * v_seq_stride + e]);
            }
            Ksh[idx] = kk;
            Vsh[idx] = vv;
        }
        __syncthreads();

        if (active) {
            int cmax = pos - kt + 1;                   // attend keys kt..pos within this tile
            if (cmax > BC) cmax = BC;
            for (int c = 0; c < cmax; ++c) {
                float partial = q0 * Ksh[c * D + lane] + q1 * Ksh[c * D + lane + 32];
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) partial += __shfl_down_sync(0xFFFFFFFF, partial, off);
                float score = __shfl_sync(0xFFFFFFFF, partial, 0) * scale;
                float new_m = fmaxf(m, score);
                float corr = expf(m - new_m);
                float p = expf(score - new_m);
                l = l * corr + p;
                acc0 = acc0 * corr + p * Vsh[c * D + lane];
                acc1 = acc1 * corr + p * Vsh[c * D + lane + 32];
                m = new_m;
            }
        }
        __syncthreads();
    }

    if (active) {
        float inv = 1.0f / l;
        size_t out_base = (size_t)row * (num_query_heads * value_size) + h * value_size;
        output[out_base + lane] = acc0 * inv;
        output[out_base + lane + 32] = acc1 * inv;
    }
}

template <Qwen2Size QWEN2_SIZE>
class Qwen2Prefill {
public:
    using Config = Qwen2Config<QWEN2_SIZE>;

    Qwen2Prefill(std::shared_ptr<Qwen2Model<QWEN2_SIZE>> model, int32_t max_N)
        : model(std::move(model)), max_N(max_N),
          hidden(bytes<__nv_bfloat16>(max_N, Config::hidden_size())),
          norm(bytes<__nv_bfloat16>(max_N, Config::hidden_size())),
          q(bytes<__nv_bfloat16>(max_N, Config::queries_size())),
          k(bytes<__nv_bfloat16>(max_N, Config::keys_size())),
          v(bytes<__nv_bfloat16>(max_N, Config::values_size())),
          weighted(bytes<float>(max_N, Config::queries_size())),
          gate(bytes<__nv_bfloat16>(max_N, Config::intermediate_size())),
          up(bytes<__nv_bfloat16>(max_N, Config::intermediate_size())),
          logits_buf(bytes<__nv_bfloat16>(max_N, Config::vocab_size())) {}

    // Process tokens[0..N-1] at positions start..start+N-1 in one batched pass, filling
    // the KV cache. Returns the [N, vocab_size] bf16 logits buffer (managed, host-readable).
    CudaBuffer& prefill(const std::shared_ptr<CudaBuffer>& k_cache,
                        const std::shared_ptr<CudaBuffer>& v_cache,
                        const std::vector<int32_t>& tokens, int32_t start, cudaStream_t stream) {
        const int32_t N = static_cast<int32_t>(tokens.size());
        auto *emb = ptr<__nv_bfloat16>(model->embedding_weight);
        auto *h = static_cast<__nv_bfloat16*>(hidden.data);

        // embedding gather
        for (int32_t i = 0; i < N; ++i)
            checkCuda(cudaMemcpyAsync(h + static_cast<size_t>(i) * Config::hidden_size(),
                emb + static_cast<size_t>(tokens[i]) * Config::hidden_size(),
                Config::hidden_size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));

        for (int32_t L = 0; L < static_cast<int32_t>(Config::num_layers()); ++L) {
            auto &layer = *model->layers[L];

            layernorm(h, ptr<__nv_bfloat16>(layer.input_layernorm.weights), N, stream);
            matmul<__nv_bfloat16>(Config::queries_size(), Config::hidden_size(),
                ptr<__nv_bfloat16>(layer.q_proj_weight), ptr<__nv_bfloat16>(layer.q_proj_bias),
                static_cast<__nv_bfloat16*>(norm.data), static_cast<__nv_bfloat16*>(q.data), N, stream);
            matmul<__nv_bfloat16>(Config::keys_size(), Config::hidden_size(),
                ptr<__nv_bfloat16>(layer.k_proj_weight), ptr<__nv_bfloat16>(layer.k_proj_bias),
                static_cast<__nv_bfloat16*>(norm.data), static_cast<__nv_bfloat16*>(k.data), N, stream);
            matmul<__nv_bfloat16>(Config::values_size(), Config::hidden_size(),
                ptr<__nv_bfloat16>(layer.v_proj_weight), ptr<__nv_bfloat16>(layer.v_proj_bias),
                static_cast<__nv_bfloat16*>(norm.data), static_cast<__nv_bfloat16*>(v.data), N, stream);

            rope(static_cast<__nv_bfloat16*>(q.data), Config::num_query_heads(), N, start, Config::queries_size(), stream);
            rope(static_cast<__nv_bfloat16*>(k.data), Config::num_kv_heads(), N, start, Config::keys_size(), stream);

            scatter(static_cast<__nv_bfloat16*>(k.data), static_cast<__nv_bfloat16*>(k_cache->data),
                    N, start, Config::keys_size(), L, stream);
            scatter(static_cast<__nv_bfloat16*>(v.data), static_cast<__nv_bfloat16*>(v_cache->data),
                    N, start, Config::values_size(), L, stream);

            attention(static_cast<__nv_bfloat16*>(k_cache->data), static_cast<__nv_bfloat16*>(v_cache->data),
                      N, start, L, stream);

            // o_proj (float weighted -> bf16 norm), residual into hidden
            matmul<float>(Config::hidden_size(), Config::queries_size(),
                ptr<__nv_bfloat16>(layer.o_proj_weight), nullptr,
                static_cast<float*>(weighted.data), static_cast<__nv_bfloat16*>(norm.data), N, stream);
            residual(h, static_cast<__nv_bfloat16*>(norm.data), N * Config::hidden_size(), stream);

            // FFN
            layernorm(h, ptr<__nv_bfloat16>(layer.post_attention_layernorm.weights), N, stream);
            matmul<__nv_bfloat16>(Config::intermediate_size(), Config::hidden_size(),
                ptr<__nv_bfloat16>(layer.gate_proj_weight), nullptr,
                static_cast<__nv_bfloat16*>(norm.data), static_cast<__nv_bfloat16*>(gate.data), N, stream);
            matmul<__nv_bfloat16>(Config::intermediate_size(), Config::hidden_size(),
                ptr<__nv_bfloat16>(layer.up_proj_weight), nullptr,
                static_cast<__nv_bfloat16*>(norm.data), static_cast<__nv_bfloat16*>(up.data), N, stream);
            silu(static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(up.data),
                 N * Config::intermediate_size(), stream);
            matmul<__nv_bfloat16>(Config::hidden_size(), Config::intermediate_size(),
                ptr<__nv_bfloat16>(layer.down_proj_weight), nullptr,
                static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(norm.data), N, stream);
            residual(h, static_cast<__nv_bfloat16*>(norm.data), N * Config::hidden_size(), stream);
        }

        layernorm(h, ptr<__nv_bfloat16>(model->final_layernorm.weights), N, stream);
        matmul<__nv_bfloat16>(Config::vocab_size(), Config::hidden_size(), emb, nullptr,
            static_cast<__nv_bfloat16*>(norm.data), static_cast<__nv_bfloat16*>(logits_buf.data), N, stream);
        return logits_buf;
    }

    // Select the in-model prefill attention kernel: false = online streaming
    // (prefill_attention_kernel), true = tiled FlashAttention (flash_attention_cache_kernel).
    void set_flash(bool f) { use_flash_ = f; }

private:
    std::shared_ptr<Qwen2Model<QWEN2_SIZE>> model;
    int32_t max_N;
    bool use_flash_ = false;
    CudaBuffer hidden, norm, q, k, v, weighted, gate, up, logits_buf;

    template <typename T> static size_t bytes(int32_t a, int32_t b) { return static_cast<size_t>(a) * b * sizeof(T); }
    template <typename T> static T* ptr(const std::shared_ptr<CudaBuffer>& b) { return static_cast<T*>(b->data); }

    template <typename in_t>
    void matmul(int32_t M, int32_t K, const __nv_bfloat16* mat, const __nv_bfloat16* bias,
                const in_t* vecs, __nv_bfloat16* out, int32_t N, cudaStream_t stream) {
        dim3 grid(M, N);
        batched_matmul_kernel<in_t><<<grid, 256, 0, stream>>>(M, K, mat, bias, vecs, out);
    }
    void layernorm(const __nv_bfloat16* in, const __nv_bfloat16* w, int32_t N, cudaStream_t stream) {
        batched_layernorm_kernel<<<N, 256, 0, stream>>>(Config::hidden_size(), in, w, static_cast<__nv_bfloat16*>(norm.data));
    }
    void rope(__nv_bfloat16* x, int32_t num_heads, int32_t N, int32_t start, int32_t row_stride, cudaStream_t stream) {
        int32_t pairs = num_heads * (Config::head_size() / 2);
        dim3 grid((pairs + 255) / 256, N);
        batched_rope_kernel<<<grid, 256, 0, stream>>>(x, num_heads, Config::head_size(), start, Config::rope_theta_base(), row_stride);
    }
    void scatter(const __nv_bfloat16* src, __nv_bfloat16* cache, int32_t N, int32_t start, int32_t row_size, int32_t layer, cudaStream_t stream) {
        dim3 grid((row_size + 255) / 256, N);
        scatter_to_cache_kernel<<<grid, 256, 0, stream>>>(src, cache, start, row_size, layer, Config::num_layers());
    }
    void attention(const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache, int32_t N, int32_t start, int32_t layer, cudaStream_t stream) {
        const int32_t H = Config::num_query_heads();
        const int32_t k_seq = Config::num_layers() * Config::keys_size();
        const int32_t v_seq = Config::num_layers() * Config::values_size();
        const int32_t k_off = layer * Config::keys_size(), v_off = layer * Config::values_size();
        if (use_flash_) {
            constexpr int RPB = 16, BC = 32, D = 64;
            dim3 grid(H, (N + RPB - 1) / RPB);
            flash_attention_cache_kernel<RPB, BC, D><<<grid, RPB * 32, 0, stream>>>(
                static_cast<__nv_bfloat16*>(q.data), k_cache, v_cache, static_cast<float*>(weighted.data),
                N, start, H, Config::num_kv_heads(), Config::value_size(),
                k_seq, v_seq, k_off, v_off, 1.0f / sqrtf(static_cast<float>(Config::head_size())));
        } else {
            dim3 grid(H, N);
            prefill_attention_kernel<<<grid, Config::head_size() / 2, 0, stream>>>(
                static_cast<__nv_bfloat16*>(q.data), k_cache, v_cache, static_cast<float*>(weighted.data),
                start, H, Config::num_kv_heads(), Config::head_size(), Config::value_size(),
                k_seq, v_seq, k_off, v_off);
        }
    }
    void silu(__nv_bfloat16* x, const __nv_bfloat16* y, int32_t len, cudaStream_t stream) {
        batched_silu_mult_kernel<<<(len + 255) / 256, 256, 0, stream>>>(x, y, len);
    }
    void residual(__nv_bfloat16* dst, const __nv_bfloat16* src, int32_t len, cudaStream_t stream) {
        residual_add_kernel<<<(len + 255) / 256, 256, 0, stream>>>(dst, src, len);
    }
};
