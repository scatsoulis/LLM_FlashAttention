//
// End-to-end test for the naive N-query prefill path (Qwen2Prefill) on the real
// Qwen2-0.5B GPU model. Verifies prefill (one batched pass over N tokens) reproduces
// the trusted query-length-1 decode loop, and times it.
//
// Correctness is checked at the LOGIT level, not just argmax: Qwen2Model::logits is a
// public buffer holding each decode step's full [vocab] logits, so for every position we
// compare prefill's logit row against decode's elementwise (max abs diff). bf16 rounding
// + the different matmul reduction orders between decode and prefill make a small
// nonzero diff expected; a logic bug (wrong position/causal/GQA) would blow it up.
// We also exercise the start>0 continuation path (chunked prefill on a non-empty cache).
//
#include "../qwen2/Qwen2Config.h"
#include "../qwen2/Qwen2Model.cuh"
#include "../qwen2/Qwen2Loader.h"
#include "../qwen2/Qwen2Prefill.cuh"
#include "../CudaBuffer.cuh"
#include <cstdio>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>

using Config = Qwen2Config<QWEN2_0_5B>;
using Clock = std::chrono::high_resolution_clock;
constexpr int32_t VOCAB = Config::vocab_size();

namespace {
int g_failures = 0;
// bf16 reduction-order differences between decode and prefill make some logit drift
// unavoidable; a real logic divergence is orders of magnitude larger. Calibrated to the
// observed worst (~0.1-0.5 over 24 layers); generous but far below a real bug's effect.
constexpr double LOGIT_TOL = 2.0;

std::shared_ptr<CudaBuffer> make_k(int32_t max_seq) {
    return std::make_shared<CudaBuffer>((size_t)max_seq * Config::num_layers() * Config::keys_size() * sizeof(__nv_bfloat16));
}
std::shared_ptr<CudaBuffer> make_v(int32_t max_seq) {
    return std::make_shared<CudaBuffer>((size_t)max_seq * Config::num_layers() * Config::values_size() * sizeof(__nv_bfloat16));
}

// Trusted reference: autoregressive decode, one token at a time, capturing each step's
// full logit vector (Qwen2Model::logits) and its argmax.
struct DecodeRef { std::vector<int32_t> preds; std::vector<float> logits; };  // logits: [N, VOCAB]
DecodeRef decode_loop(Qwen2Model<QWEN2_0_5B>& model,
                      const std::shared_ptr<CudaBuffer>& kc, const std::shared_ptr<CudaBuffer>& vc,
                      const std::vector<int32_t>& toks) {
    DecodeRef r;
    r.logits.reserve(toks.size() * (size_t)VOCAB);
    for (size_t i = 0; i < toks.size(); ++i) {
        r.preds.push_back(model.forward(kc, vc, (int32_t)(i + 1), toks[i], 0.0f));  // syncs internally
        const auto* dl = static_cast<const __nv_bfloat16*>(model.logits->data);
        for (int32_t j = 0; j < VOCAB; ++j) r.logits.push_back(float(dl[j]));
    }
    return r;
}

int32_t row_argmax(const __nv_bfloat16* row) {
    float best = -INFINITY; int32_t bi = 0;
    for (int32_t j = 0; j < VOCAB; ++j) { float v = float(row[j]); if (v > best) { best = v; bi = j; } }
    return bi;
}

// Compare prefill logit rows [r0..) against decode logit positions [d0..) for `count`
// rows: worst elementwise abs diff + argmax agreement.
void compare_rows(const char* name, const __nv_bfloat16* prefill_rows,
                  const DecodeRef& ref, int d0, int count) {
    double worst = 0.0; int amatch = 0;
    for (int t = 0; t < count; ++t) {
        const __nv_bfloat16* prow = prefill_rows + (size_t)t * VOCAB;
        const float* drow = &ref.logits[(size_t)(d0 + t) * VOCAB];
        double mx = 0.0;
        for (int32_t j = 0; j < VOCAB; ++j) mx = std::max(mx, std::fabs((double)float(prow[j]) - (double)drow[j]));
        worst = std::max(worst, mx);
        amatch += (row_argmax(prow) == ref.preds[d0 + t]);
    }
    bool ok = (worst <= LOGIT_TOL);
    std::printf("  [%s] %-26s argmax %d/%d | worst |logit_prefill - logit_decode| = %.4f (tol %.1f)\n",
                ok ? "PASS" : "FAIL", name, amatch, count, worst, LOGIT_TOL);
    if (!ok) g_failures++;
}
double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
}  // namespace

int main() {
    std::printf("=== Naive N-query prefill vs query-length-1 decode loop (Qwen2-0.5B, GPU bf16) ===\n");
    constexpr int32_t MAX_N = 256;
    auto model = Qwen2Loader::load_qwen2<QWEN2_0_5B>(Qwen2Loader::get_model_dir() + "/model.safetensors", MAX_N);
    Qwen2Prefill<QWEN2_0_5B> pf(model, MAX_N);
    cudaStream_t stream; checkCuda(cudaStreamCreate(&stream));

    // ---- (1) full-prefill (start=0) logits vs decode, per position ----
    auto verify = [&](std::vector<int32_t> toks, const char* name) {
        DecodeRef ref = decode_loop(*model, make_k(MAX_N), make_v(MAX_N), toks);
        CudaBuffer& lg = pf.prefill(make_k(MAX_N), make_v(MAX_N), toks, 0, stream);
        checkCuda(cudaStreamSynchronize(stream));
        compare_rows(name, static_cast<__nv_bfloat16*>(lg.data), ref, 0, (int)toks.size());
    };
    verify({785, 6722, 315, 9625, 374}, "\"The capital...\" N=5");
    std::vector<int32_t> seq64; for (int i = 0; i < 64; ++i) seq64.push_back((i * 131 + 7) % 151000);
    verify(seq64, "synthetic N=64");

    // ---- (2) start>0 continuation: prefill in two chunks on one cache, vs decode ----
    {
        std::vector<int32_t> toks; for (int i = 0; i < 40; ++i) toks.push_back((i * 53 + 17) % 151000);
        const int split = 24;
        DecodeRef ref = decode_loop(*model, make_k(MAX_N), make_v(MAX_N), toks);
        auto kc = make_k(MAX_N), vc = make_v(MAX_N);
        pf.prefill(kc, vc, {toks.begin(), toks.begin() + split}, 0, stream);             // fills 0..split-1
        CudaBuffer& lg = pf.prefill(kc, vc, {toks.begin() + split, toks.end()}, split, stream);  // start=split
        checkCuda(cudaStreamSynchronize(stream));
        compare_rows("chunked start>0 (24+16)", static_cast<__nv_bfloat16*>(lg.data), ref, split, (int)toks.size() - split);
    }

    // ---- (3) speed: prefill one batched pass vs the decode loop (send_text-style), N=256 ----
    // send_text-style baseline: just forward() per token, NO host logit readback (that
    // readback is test-only instrumentation and must stay out of the timed region).
    std::vector<int32_t> seqT; for (int i = 0; i < MAX_N; ++i) seqT.push_back((i * 97 + 11) % 151000);
    auto kcd = make_k(MAX_N), vcd = make_v(MAX_N), kcp = make_k(MAX_N), vcp = make_v(MAX_N);  // alloc outside timing
    auto decode_bare = [&]() { for (size_t i = 0; i < seqT.size(); ++i) model->forward(kcd, vcd, (int32_t)(i + 1), seqT[i], 0.0f); };
    decode_bare();                                                                           // warmup
    pf.prefill(kcp, vcp, seqT, 0, stream); checkCuda(cudaStreamSynchronize(stream));         // warmup
    double decode_ms = 1e30, prefill_ms = 1e30;
    for (int rep = 0; rep < 5; ++rep) {
        auto d0 = Clock::now(); decode_bare(); auto d1 = Clock::now();
        decode_ms = std::min(decode_ms, ms(d0, d1));
        auto p0 = Clock::now(); pf.prefill(kcp, vcp, seqT, 0, stream);
        checkCuda(cudaStreamSynchronize(stream)); auto p1 = Clock::now();
        prefill_ms = std::min(prefill_ms, ms(p0, p1));
    }
    std::printf("\n  ingest N=%d tokens (best of 5; both compute all-N logits):\n", MAX_N);
    std::printf("    decode loop (1 query x %d, sync/token) : %8.2f ms\n", MAX_N, decode_ms);
    std::printf("    naive prefill (N queries, 1 pass)      : %8.2f ms   (%.2fx faster)\n", prefill_ms, decode_ms / prefill_ms);
    std::printf("    (speedup = batching + amortized launches/syncs; FlashAttention is the next, separate win)\n");

    std::printf("================================================================================\n");
    if (g_failures == 0) { std::printf("ALL CHECKS PASSED\n"); return 0; }
    std::printf("%d CHECK(S) FAILED\n", g_failures);
    return 1;
}
