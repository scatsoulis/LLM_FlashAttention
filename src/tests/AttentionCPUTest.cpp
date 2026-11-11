//
// Correctness harness for the CPU attention reference (src/cpu_ref/AttentionCPU.h).
//
// Pure standard C++ (no CUDA), so the proof-of-concept builds and runs before any GPU
// work begins. Build/run standalone with:
//     g++ -std=c++20 -O2 src/tests/AttentionCPUTest.cpp -o build/attentioncputest && ./build/attentioncputest
// or via CMake/ctest as the `attentioncputest` target.
//
// The reference is verified against THREE independent sources of truth:
//
//   [1] An independent, double-precision reference (independent_reference below) that
//       shares NO code with AttentionCPU.h -- its scale (1/sqrt(d)), GQA grouping, and
//       index arithmetic are all written out locally. Both the materialized and the
//       online FP32 paths are checked against it on random Q,K,V. This pins down
//       scale / GQA-broadcast / memory-layout correctness WITHOUT needing PyTorch, and
//       (because the ref is FP64) is a genuinely higher-precision oracle. We also report
//       the direct online-vs-materialized delta -- the project's core claim that the
//       online-softmax recurrence equals an ordinary row softmax.
//
//   [2] Analytic known-answers derived by hand (expected computed with local, helper-free
//       indexing so a layout/mapping bug cannot hide):
//         [2a] N=1   -> output equals V[0] exactly.
//         [2b] Q=0   -> output is the causal prefix-mean of V (uniform softmax).
//
//   [3] A PyTorch oracle: torch.nn.functional.scaled_dot_product_attention(is_causal=True).
//       Optional (needs the fixture from pyref/gen_attention_fixture.py); if absent we
//       say so LOUDLY and the final banner reflects that the external check did not run.
//
// Tolerance: the target is 1e-5. compare() uses |a-b| <= atol + rtol*|b| with
// rtol=atol=1e-5. For output components near zero the atol term is load-bearing (FP32
// cancellation makes the relative error large on a tiny value); max_abs is therefore the
// meaningful headline metric and is always reported.
//
#include "../cpu_ref/AttentionCPU.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
bool g_oracle_ran = false;

constexpr float RTOL = 1e-5f;
constexpr float ATOL = 1e-5f;

struct Error {
    double max_abs = 0.0;
    double max_rel = 0.0;
    int    n_bad   = 0;
};

// allclose in the same spirit as TestUtils.cuh::check_fp32_allclose: |a-b| <= atol + rtol*|b|
Error compare(const std::vector<float>& a, const std::vector<float>& b, float rtol, float atol) {
    Error e;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        double rel  = diff / (std::fabs(static_cast<double>(b[i])) + 1e-30);
        e.max_abs = std::max(e.max_abs, diff);
        e.max_rel = std::max(e.max_rel, rel);
        if (diff > atol + static_cast<double>(rtol) * std::fabs(static_cast<double>(b[i]))) e.n_bad++;
    }
    return e;
}

void report(const std::string& name, const Error& e, int total, float rtol, float atol) {
    bool ok = (e.n_bad == 0);
    std::printf("  [%s] %-34s max_abs=%.3e  max_rel=%.3e  (rtol=%.0e atol=%.0e, %d/%d bad)\n",
                ok ? "PASS" : "FAIL", name.c_str(), e.max_abs, e.max_rel, rtol, atol, e.n_bad, total);
    if (!ok) g_failures++;
}

void fill_random(std::vector<float>& v, std::mt19937& gen, std::normal_distribution<float>& dist) {
    for (auto& x : v) x = dist(gen);
}

// -----------------------------------------------------------------------------
// Independent, FP64, helper-free reference. Deliberately shares NO code with
// AttentionCPU.h so that a bug in AttentionDims::{scale,kv_head_for,q_index,kv_index}
// cannot be masked by a matching bug in the expected values. Plain materialized
// causal softmax with the GQA mapping written in its repeat_interleave form (h/group).
// Requires H % Hkv == 0 (true for every Qwen2 config and every case here).
// -----------------------------------------------------------------------------
void independent_reference(const float* Q, const float* K, const float* V, float* O,
                           int H, int Hkv, int d, int N) {
    const double scale = 1.0 / std::sqrt(static_cast<double>(d));  // independent literal
    const int group = H / Hkv;                                     // contiguous GQA group size
    std::vector<double> s(N);
    for (int h = 0; h < H; ++h) {
        const int g = h / group;                                   // independent mapping form
        for (int i = 0; i < N; ++i) {
            double mx = -1e300;
            for (int j = 0; j <= i; ++j) {
                double dot = 0.0;
                for (int k = 0; k < d; ++k)
                    dot += static_cast<double>(Q[(static_cast<long>(h) * N + i) * d + k]) *
                           static_cast<double>(K[(static_cast<long>(g) * N + j) * d + k]);
                s[j] = dot * scale;
                if (s[j] > mx) mx = s[j];
            }
            double denom = 0.0;
            for (int j = 0; j <= i; ++j) { s[j] = std::exp(s[j] - mx); denom += s[j]; }
            for (int k = 0; k < d; ++k) {
                double acc = 0.0;
                for (int j = 0; j <= i; ++j)
                    acc += s[j] * static_cast<double>(V[(static_cast<long>(g) * N + j) * d + k]);
                O[(static_cast<long>(h) * N + i) * d + k] = static_cast<float>(acc / denom);
            }
        }
    }
}

// ---- [1] random check vs the independent FP64 reference ----------------------
void test_vs_independent() {
    std::printf("[1] materialized & online (FP32) vs independent FP64 reference (random Q,K,V ~ N(0,1))\n");
    const std::vector<AttentionDims> configs = {
        {14, 2, 64, 1}, {14, 2, 64, 2}, {14, 2, 64, 8}, {14, 2, 64, 64},
        {14, 2, 64, 128}, {14, 2, 64, 256}, {14, 2, 64, 512}, {14, 2, 64, 1024},
    };
    std::mt19937 gen{123};
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (const auto& d : configs) {
        const int H = d.num_query_heads, Hkv = d.num_kv_heads, N = d.seq_len, hd = d.head_dim;
        std::vector<float> Q(static_cast<size_t>(H) * N * hd);
        std::vector<float> K(static_cast<size_t>(Hkv) * N * hd);
        std::vector<float> V(static_cast<size_t>(Hkv) * N * hd);
        std::vector<float> O_mat(Q.size()), O_onl(Q.size()), O_ref(Q.size());
        fill_random(Q, gen, dist);
        fill_random(K, gen, dist);
        fill_random(V, gen, dist);

        attention_materialized(Q.data(), K.data(), V.data(), O_mat.data(), d);
        attention_online(Q.data(), K.data(), V.data(), O_onl.data(), d);
        independent_reference(Q.data(), K.data(), V.data(), O_ref.data(), H, Hkv, hd, N);

        const int total = static_cast<int>(Q.size());
        char name[48];
        std::snprintf(name, sizeof(name), "N=%-4d materialized vs indep", N);
        report(name, compare(O_mat, O_ref, RTOL, ATOL), total, RTOL, ATOL);
        std::snprintf(name, sizeof(name), "N=%-4d online       vs indep", N);
        report(name, compare(O_onl, O_ref, RTOL, ATOL), total, RTOL, ATOL);
        std::snprintf(name, sizeof(name), "N=%-4d online       vs materialized", N);
        report(name, compare(O_onl, O_mat, RTOL, ATOL), total, RTOL, ATOL);
    }
}

// ---- [2a] analytic: N=1 -> output == V[0] exactly (helper-free expected) -----
void test_single_token() {
    std::printf("[2a] analytic: N=1 -> O[h] == V[group-mapped kv][0] exactly\n");
    const int H = 14, Hkv = 2, d = 64, N = 1, group = H / Hkv;
    std::mt19937 gen{7};
    std::normal_distribution<float> dist(0.0f, 3.0f);
    std::vector<float> Q(static_cast<size_t>(H) * d), K(static_cast<size_t>(Hkv) * d), V(static_cast<size_t>(Hkv) * d);
    std::vector<float> O(Q.size()), expected(Q.size());
    fill_random(Q, gen, dist);
    fill_random(K, gen, dist);
    fill_random(V, gen, dist);

    for (int h = 0; h < H; ++h) {
        int g = h / group;  // helper-free mapping
        for (int k = 0; k < d; ++k) expected[h * d + k] = V[g * d + k];
    }
    AttentionDims dims{H, Hkv, d, N};
    attention_materialized(Q.data(), K.data(), V.data(), O.data(), dims);
    report("materialized", compare(O, expected, 0.0f, 0.0f), static_cast<int>(O.size()), 0.0f, 0.0f);
    attention_online(Q.data(), K.data(), V.data(), O.data(), dims);
    report("online", compare(O, expected, 0.0f, 0.0f), static_cast<int>(O.size()), 0.0f, 0.0f);
}

// ---- [2b] analytic: Q=0 -> output == causal prefix-mean of V (helper-free) ---
void test_uniform_softmax() {
    std::printf("[2b] analytic: Q=0 -> O[h,i] == mean_{j<=i} V[group-mapped kv][j]\n");
    const int H = 14, Hkv = 2, d = 64, N = 96, group = H / Hkv;
    std::mt19937 gen{99};
    std::normal_distribution<float> dist(0.0f, 2.0f);
    std::vector<float> Q(static_cast<size_t>(H) * N * d, 0.0f);  // all zeros
    std::vector<float> K(static_cast<size_t>(Hkv) * N * d), V(static_cast<size_t>(Hkv) * N * d);
    std::vector<float> O(Q.size()), expected(Q.size());
    fill_random(K, gen, dist);
    fill_random(V, gen, dist);

    // Ground truth: equal scores -> causal softmax is uniform over j in [0, i].
    for (int h = 0; h < H; ++h) {
        int g = h / group;  // helper-free mapping
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < d; ++k) {
                double acc = 0.0;
                for (int j = 0; j <= i; ++j) acc += V[(static_cast<long>(g) * N + j) * d + k];
                expected[(static_cast<long>(h) * N + i) * d + k] = static_cast<float>(acc / (i + 1));
            }
        }
    }
    AttentionDims dims{H, Hkv, d, N};
    attention_materialized(Q.data(), K.data(), V.data(), O.data(), dims);
    report("materialized", compare(O, expected, RTOL, ATOL), static_cast<int>(O.size()), RTOL, ATOL);
    attention_online(Q.data(), K.data(), V.data(), O.data(), dims);
    report("online", compare(O, expected, RTOL, ATOL), static_cast<int>(O.size()), RTOL, ATOL);
}

// ---- [3] PyTorch oracle (optional; LOUD skip if fixture absent) --------------
template <typename T>
bool read_pod(std::FILE* f, T* out, size_t n = 1) { return std::fread(out, sizeof(T), n, f) == n; }

void test_pytorch_oracle() {
    const char* candidates[] = {
        std::getenv("ATTENTION_FIXTURE"),
        "attention_fixture.bin", "../pyref/attention_fixture.bin", "pyref/attention_fixture.bin",
        "/home/scatsoulis/project/pyref/attention_fixture.bin",
    };
    std::FILE* f = nullptr;
    std::string used;
    for (const char* path : candidates) {
        if (!path) continue;
        f = std::fopen(path, "rb");
        if (f) { used = path; break; }
    }
    if (!f) {
        std::printf("[3] PyTorch oracle: *** SKIPPED *** (no attention_fixture.bin; run "
                    "pyref/gen_attention_fixture.py to enable the external check)\n");
        return;
    }
    std::printf("[3] PyTorch oracle (scaled_dot_product_attention, is_causal): %s\n", used.c_str());

    int32_t num_cases = 0;
    if (!read_pod(f, &num_cases)) { std::printf("  [FAIL] bad fixture header\n"); g_failures++; std::fclose(f); return; }
    bool ok_all = true;
    for (int c = 0; c < num_cases; ++c) {
        int32_t H, Hkv, hd, N;
        if (!read_pod(f, &H) || !read_pod(f, &Hkv) || !read_pod(f, &hd) || !read_pod(f, &N)) {
            std::printf("  [FAIL] truncated fixture\n"); g_failures++; ok_all = false; break;
        }
        AttentionDims d{H, Hkv, hd, N};
        std::vector<float> Q(static_cast<size_t>(H) * N * hd), K(static_cast<size_t>(Hkv) * N * hd);
        std::vector<float> V(static_cast<size_t>(Hkv) * N * hd), O_ref(Q.size());
        std::vector<float> O_mat(Q.size()), O_onl(Q.size());
        if (!read_pod(f, Q.data(), Q.size()) || !read_pod(f, K.data(), K.size()) ||
            !read_pod(f, V.data(), V.size()) || !read_pod(f, O_ref.data(), O_ref.size())) {
            std::printf("  [FAIL] truncated fixture arrays\n"); g_failures++; ok_all = false; break;
        }
        attention_materialized(Q.data(), K.data(), V.data(), O_mat.data(), d);
        attention_online(Q.data(), K.data(), V.data(), O_onl.data(), d);
        char name[48];
        std::snprintf(name, sizeof(name), "materialized vs torch N=%-4d", N);
        report(name, compare(O_mat, O_ref, RTOL, ATOL), static_cast<int>(O_ref.size()), RTOL, ATOL);
        std::snprintf(name, sizeof(name), "online       vs torch N=%-4d", N);
        report(name, compare(O_onl, O_ref, RTOL, ATOL), static_cast<int>(O_ref.size()), RTOL, ATOL);
    }
    std::fclose(f);
    if (ok_all) g_oracle_ran = true;
}

}  // namespace

int main() {
    std::printf("=== CPU causal SDPA reference (FP32, GQA, prefill) ===\n");
    std::printf("(tolerance rtol=atol=1e-5; max_abs is the headline metric -- near-zero\n");
    std::printf(" outputs pass on the atol term, which is expected FP32 behavior)\n");
    test_vs_independent();
    test_single_token();
    test_uniform_softmax();
    test_pytorch_oracle();
    std::printf("=============================================================\n");
    if (g_failures != 0) {
        std::printf("%d CHECK(S) FAILED\n", g_failures);
        return 1;
    }
    if (g_oracle_ran) {
        std::printf("ALL CHECKS PASSED (independent FP64 reference + PyTorch oracle)\n");
    } else {
        std::printf("SELF-CONTAINED CHECKS PASSED (independent FP64 reference covers "
                    "scale/GQA/layout).\n*** PyTorch oracle was SKIPPED -- external "
                    "cross-check not performed. ***\n");
    }
    return 0;
}
