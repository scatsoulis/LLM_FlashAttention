//
// End-to-end: FlashAttention wired into the real Qwen2-0.5B prefill (bf16). Verifies the
// flash-attention prefill produces the same predictions as (a) the trusted decode loop and
// (b) the model's current online-streaming prefill attention, then measures full-prefill
// tok/s with each attention kernel. The full-model delta is small by design (attention is a
// slice of the model, and the current prefill already avoids N^2 materialization) -- this is
// the "tok/s for full inference" diagnostic, not the kernel-level FlashAttention
// result (that is flash_vs_naive.md).
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
std::shared_ptr<CudaBuffer> make_k(int32_t s){ return std::make_shared<CudaBuffer>((size_t)s*Config::num_layers()*Config::keys_size()*sizeof(__nv_bfloat16)); }
std::shared_ptr<CudaBuffer> make_v(int32_t s){ return std::make_shared<CudaBuffer>((size_t)s*Config::num_layers()*Config::values_size()*sizeof(__nv_bfloat16)); }
int32_t row_argmax(const __nv_bfloat16* r){ float best=-INFINITY; int32_t bi=0; for(int32_t j=0;j<VOCAB;++j){float v=float(r[j]); if(v>best){best=v;bi=j;}} return bi; }
std::vector<int32_t> decode_loop(Qwen2Model<QWEN2_0_5B>& m, const std::shared_ptr<CudaBuffer>& kc, const std::shared_ptr<CudaBuffer>& vc, const std::vector<int32_t>& t){
    std::vector<int32_t> p; for(size_t i=0;i<t.size();++i) p.push_back(m.forward(kc,vc,(int32_t)(i+1),t[i],0.0f)); return p;
}
double ms(Clock::time_point a, Clock::time_point b){ return std::chrono::duration<double,std::milli>(b-a).count(); }
}

int main() {
    std::printf("=== FlashAttention in the real Qwen2-0.5B prefill (bf16, end-to-end) ===\n");
    constexpr int32_t MAX_N = 512;
    auto model = Qwen2Loader::load_qwen2<QWEN2_0_5B>(Qwen2Loader::get_model_dir() + "/model.safetensors", MAX_N);
    Qwen2Prefill<QWEN2_0_5B> pf(model, MAX_N);
    cudaStream_t stream; checkCuda(cudaStreamCreate(&stream));

    // ---- correctness: flash prefill vs decode, and flash vs current prefill ----
    std::vector<int32_t> toks = {785, 6722, 315, 9625, 374};   // "The capital of France is"
    auto dec = decode_loop(*model, make_k(MAX_N), make_v(MAX_N), toks);

    std::vector<float> cur_logits;
    pf.set_flash(false);                                  // current online-streaming prefill attention
    {
        CudaBuffer& lc = pf.prefill(make_k(MAX_N), make_v(MAX_N), toks, 0, stream);
        checkCuda(cudaStreamSynchronize(stream));
        const auto* lp = static_cast<__nv_bfloat16*>(lc.data);
        cur_logits.assign(toks.size() * (size_t)VOCAB, 0.0f);
        for (size_t i = 0; i < cur_logits.size(); ++i) cur_logits[i] = float(lp[i]);
    }

    pf.set_flash(true);                                  // tiled FlashAttention prefill attention
    CudaBuffer& lf = pf.prefill(make_k(MAX_N), make_v(MAX_N), toks, 0, stream);
    checkCuda(cudaStreamSynchronize(stream));
    const auto* fp = static_cast<__nv_bfloat16*>(lf.data);

    int flash_vs_decode = 0; double max_logit_diff = 0.0;
    for (size_t i = 0; i < toks.size(); ++i) {
        if (row_argmax(fp + i * (size_t)VOCAB) == dec[i]) flash_vs_decode++;
        for (int32_t j = 0; j < VOCAB; ++j)
            max_logit_diff = std::max(max_logit_diff, std::fabs((double)float(fp[i*(size_t)VOCAB+j]) - (double)cur_logits[i*(size_t)VOCAB+j]));
    }
    bool ok = (flash_vs_decode == (int)toks.size()) && (max_logit_diff < 2.0);
    std::printf("  [%s] flash prefill: %d/%zu next-token preds == decode | max|logit_flash - logit_current| = %.3f\n",
                ok ? "PASS" : "FAIL", flash_vs_decode, toks.size(), max_logit_diff);
    std::printf("       last-position predicted token: flash=%d decode=%d\n", row_argmax(fp + (toks.size()-1)*(size_t)VOCAB), dec.back());
    if (!ok) g_failures++;

    // ---- end-to-end tok/s: full prefill with current vs flash attention, N=512 ----
    std::vector<int32_t> seqT; for (int i=0;i<MAX_N;++i) seqT.push_back((i*97+11)%151000);
    auto kc = make_k(MAX_N), vc = make_v(MAX_N);
    auto bench = [&](bool flash)->double {
        pf.set_flash(flash);
        pf.prefill(kc, vc, seqT, 0, stream); checkCuda(cudaStreamSynchronize(stream));   // warmup
        double best = 1e30;
        for (int r=0;r<5;++r){ auto t0=Clock::now(); pf.prefill(kc,vc,seqT,0,stream); checkCuda(cudaStreamSynchronize(stream)); best=std::min(best, ms(t0, Clock::now())); }
        return best;
    };
    double cur_ms = bench(false), flash_ms = bench(true);
    std::printf("\n  full prefill of N=%d tokens (best of 5):\n", MAX_N);
    std::printf("    current online attention : %7.2f ms  (%6.0f tok/s)\n", cur_ms, MAX_N/(cur_ms/1000.0));
    std::printf("    FlashAttention           : %7.2f ms  (%6.0f tok/s)   (%.2fx end-to-end)\n", flash_ms, MAX_N/(flash_ms/1000.0), cur_ms/flash_ms);
    std::printf("    (attention is a slice of the model -> small end-to-end delta, as expected)\n");

    std::printf("========================================================================\n");
    if (g_failures==0){ std::printf("ALL CHECKS PASSED\n"); return 0; }
    std::printf("%d CHECK(S) FAILED\n", g_failures); return 1;
}
