#pragma once
//
// CPU reference implementation of causal scaled dot-product attention (SDPA)
// with Grouped-Query Attention (GQA) broadcasting.  FP32 throughout.
//
// The CPU reference must be a known-correct,
// dependency-free baseline that covers EVERY feature planned for the GPU kernels:
//   * FP32 precision
//   * causal masking (query position i attends only to key positions j <= i)
//   * Grouped-Query Attention (num_kv_heads KV heads broadcast across query heads)
//   * the PREFILL regime: N query positions attended at once (not decode, N_q = 1),
//     which is what makes FlashAttention's O(N^2) -> O(N) HBM-traffic win observable.
//
// Two implementations are provided so the test harness can cross-check them, and
// so the algorithm that the GPU kernel will run is validated before any CUDA is
// written:
//
//   (1) attention_materialized  -- the textbook "naive" path. For each query row it
//       materializes the row of the N x N score matrix S = QK^T / sqrt(d), applies
//       the causal mask, takes a row-wise (max-subtracted) softmax, then multiplies
//       by V. This mirrors PyTorch's scaled_dot_product_attention and is the
//       ground-truth reference. It is also, structurally, the "naive CUDA
//       baseline" (separate QK^T / softmax / PV passes).
//
//   (2) attention_online        -- a single streaming pass per query row using the
//       online-softmax recurrence (Milakov & Gimelshein 2018): a running max `m`, a
//       running denominator `l`, and a running (unnormalized) output accumulator that
//       is rescaled whenever the max grows. This is the EXACT arithmetic the fused
//       FlashAttention CUDA kernel will perform (cf. the existing decode kernel
//       gqa_sdpa_kernel). Verifying (2) == (1) on the CPU de-risks the project's
//       hardest piece -- correct online softmax -- before touching the GPU.
//
// Tensor layouts (row-major, contiguous):
//   Q : [num_query_heads, N, head_dim]      query  for head h, position i, dim k
//   K : [num_kv_heads,    N, head_dim]      key    for kv-head g, position j, dim k
//   V : [num_kv_heads,    N, head_dim]      value  for kv-head g, position j, dim k
//   O : [num_query_heads, N, head_dim]      output for head h, position i, dim k
//
// The header is pure standard C++ (no CUDA include) on purpose: the CPU
// proof-of-concept must build and run before any GPU work begins.

#include <cmath>
#include <vector>
#include <cstdint>
#include <limits>

struct AttentionDims {
    int num_query_heads;  // H   -- 14 for Qwen2-0.5B
    int num_kv_heads;     // Hkv -- 2  for Qwen2-0.5B
    int head_dim;         // d   -- 64 for Qwen2-0.5B
    int seq_len;          // N   -- prefill: # of query AND key positions

    // GQA broadcast: which KV head feeds a given query head. Integer division,
    // identical to the existing decode kernel (head_idx * num_kv_heads / num_query_heads)
    // and to PyTorch's enable_gqa repeat_interleave mapping when H is a multiple of Hkv.
    int kv_head_for(int q_head) const {
        return q_head * num_kv_heads / num_query_heads;
    }

    // Softmax temperature 1/sqrt(d_k). 1/8 for head_dim = 64.
    float scale() const {
        return 1.0f / std::sqrt(static_cast<float>(head_dim));
    }

    // Flat index helpers for the [heads, N, head_dim] layouts above.
    long q_index(int h, int i, int k) const {
        return (static_cast<long>(h) * seq_len + i) * head_dim + k;
    }
    long kv_index(int g, int j, int k) const {
        return (static_cast<long>(g) * seq_len + j) * head_dim + k;
    }
};

// -----------------------------------------------------------------------------
// (1) Materialized / naive reference. Ground truth for correctness.
//
// PARALLELIZATION MAP (how this maps onto the naive and FlashAttention GPU kernels):
//   for h          ... independent per query head      [GPU] grid dimension (one block-group per head)
//     for i        ... independent per query row        [GPU] grid dimension (one block / warp per query row or row-tile)
//       for j      ... the QK^T reduction over keys      [GPU] tiled into shared-memory K blocks of width Bc; causal mask
//                                                              splits tiles into skip / partial / full classes
//         for k    ... the head_dim dot-product          [GPU] warp reduction (__shfl_down) -> needs sync within the warp
//       softmax    ... row max + row sum over j          [GPU] block/warp reduction -> needs __syncthreads
//       for k,j    ... weighted sum P*V                   [GPU] tiled into shared-memory V blocks; accumulate per row
// -----------------------------------------------------------------------------
inline void attention_materialized(const float* Q, const float* K, const float* V,
                                   float* O, const AttentionDims& dims) {
    const int H = dims.num_query_heads;
    const int d = dims.head_dim;
    const int N = dims.seq_len;
    const float scale = dims.scale();

    // One materialized row of the N x N score matrix S at a time. The full matrix is
    // never needed simultaneously, but each row IS written out in HBM in the naive
    // GPU baseline -- that O(N^2) traffic is exactly what FlashAttention removes.
    std::vector<float> scores(N);

    for (int h = 0; h < H; ++h) {
        const int g = dims.kv_head_for(h);

        for (int i = 0; i < N; ++i) {
            // --- 1) S[i, j] = (Q_h,i . K_g,j) * scale, with causal mask j > i -> -inf ---
            float row_max = -std::numeric_limits<float>::infinity();
            for (int j = 0; j < N; ++j) {
                if (j > i) {                       // causal mask: future keys are forbidden
                    scores[j] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                float dot = 0.0f;
                for (int k = 0; k < d; ++k) {
                    dot += Q[dims.q_index(h, i, k)] * K[dims.kv_index(g, j, k)];
                }
                float s = dot * scale;
                scores[j] = s;
                row_max = std::fmax(row_max, s);
            }

            // --- 2) numerically-stable row softmax (subtract row max before exp) ---
            float denom = 0.0f;
            for (int j = 0; j <= i; ++j) {         // masked entries (j > i) contribute exp(-inf) = 0
                scores[j] = std::exp(scores[j] - row_max);
                denom += scores[j];
            }
            float inv_denom = 1.0f / denom;        // denom >= 1 (the diagonal term is always present)

            // --- 3) O[i, :] = sum_j softmax[j] * V_g,j ---
            for (int k = 0; k < d; ++k) {
                float acc = 0.0f;
                for (int j = 0; j <= i; ++j) {
                    acc += scores[j] * V[dims.kv_index(g, j, k)];
                }
                O[dims.q_index(h, i, k)] = acc * inv_denom;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// (2) Online-softmax (streaming) reference -- the FlashAttention recurrence.
//
// For each query row we stream over keys j = 0..i in a single pass, never
// materializing the score row. State carried across the stream:
//   m   running max of the scores seen so far
//   l   running sum of exp(score - m)
//   acc running UNNORMALIZED output  sum exp(score - m) * V_j
// When a new score raises the max (m -> m_new), the prior l and acc are rescaled
// by alpha = exp(m - m_new) so every term is expressed relative to the same max.
//
// This is identical to the math in gpu_ops/GroupQueryAttention.cuh, generalized
// from decode (one query) to prefill (a query per position).
//
// PARALLELIZATION MAP:
//   for h, for i ... same grid mapping as (1)
//     for j      ... [GPU] OUTER loop over K/V tiles (block columns of width Bc) loaded into
//                          shared memory. Fully-masked tiles (all j > i) are SKIPPED entirely;
//                          the partially-masked diagonal tile applies a per-element check.
//       online update ... [GPU] the m/l/acc recurrence is the FlashAttention inner step; the
//                          acc rescale-by-alpha is the one subtlety the GPU kernel must get right.
// -----------------------------------------------------------------------------
inline void attention_online(const float* Q, const float* K, const float* V,
                             float* O, const AttentionDims& dims) {
    const int H = dims.num_query_heads;
    const int d = dims.head_dim;
    const int N = dims.seq_len;
    const float scale = dims.scale();

    std::vector<float> acc(d);  // running unnormalized output for the current row (GPU: in registers)

    for (int h = 0; h < H; ++h) {
        const int g = dims.kv_head_for(h);

        for (int i = 0; i < N; ++i) {
            float m = -std::numeric_limits<float>::infinity();  // running max
            float l = 0.0f;                                     // running denominator
            for (int k = 0; k < d; ++k) acc[k] = 0.0f;

            // Causal: stream only over allowed keys j <= i (fully-masked tiles skipped).
            for (int j = 0; j <= i; ++j) {
                float dot = 0.0f;
                for (int k = 0; k < d; ++k) {
                    dot += Q[dims.q_index(h, i, k)] * K[dims.kv_index(g, j, k)];
                }
                float s = dot * scale;

                float m_new = std::fmax(m, s);
                float alpha = std::exp(m - m_new);   // rescale factor for prior state (1.0 when m unchanged)
                float p     = std::exp(s - m_new);   // weight of the new key, relative to the new max

                l = l * alpha + p;
                for (int k = 0; k < d; ++k) {
                    acc[k] = acc[k] * alpha + p * V[dims.kv_index(g, j, k)];
                }
                m = m_new;
            }

            float inv_l = 1.0f / l;
            for (int k = 0; k < d; ++k) {
                O[dims.q_index(h, i, k)] = acc[k] * inv_l;
            }
        }
    }
}
