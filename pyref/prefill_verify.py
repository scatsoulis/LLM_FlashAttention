"""
Verification gate: the CPU prefill (query length N) is a correct upgrade of the
query-length-1 decode, on the real Qwen2-0.5B model (fp32).

Per-prompt checks:
  (1) prefill == decode : the all-N-at-once prefill predicts the SAME next token at
      every position as the trusted one-token-at-a-time decode path (the established
      pyref code that already generates correct Qwen text). Independent oracle.
  (2) online == materialized : running the prefill attention with the FlashAttention
      online-softmax recurrence vs a plain materialized softmax gives identical logits.
  (info) logit drift prefill(prompt[:k])[-1] vs prefill(prompt)[k-1]: ~1e-4 fp noise
      from BLAS computing row k-1 differently at M=k vs M=N; argmax is unaffected.

Plus a standalone, exact CAUSALITY check (test_causal_isolation): perturbing future
K/V leaves every earlier output row bit-identical -- a zero-tolerance proof that
position i attends only to 0..i (no future leakage), with no BLAS/depth confound.

"""
import torch
from transformers import AutoTokenizer
from qwen2 import Qwen2Model
from prefill_demo import prefill_forward, decode_predictions, causal_attention, MODEL_DIR

PROMPTS = [
    "The capital of France is",
    "Once upon a time, there was a",
    "The opposite of hot is",
    "Water is made of hydrogen and",
    "1, 2, 3, 4,",
]


@torch.no_grad()
def test_causal_isolation():
    """Exact causality: perturbing future K/V must leave earlier output rows
       bit-identical. Zero-tolerance, no BLAS/depth confound."""
    torch.manual_seed(0)
    N, H, Hkv, d = 16, 14, 2, 64
    Q = torch.randn(N, H, d); K = torch.randn(N, Hkv, d); V = torch.randn(N, Hkv, d)
    base = causal_attention(Q, K, V, H, Hkv, d, "materialized")
    K2, V2 = K.clone(), V.clone()
    K2[N - 1] += 10.0; V2[N - 1] += 10.0          # corrupt the last (future) position
    pert = causal_attention(Q, K2, V2, H, Hkv, d, "materialized")
    leak = (base[:N - 1] - pert[:N - 1]).abs().max().item()   # rows 0..N-2: must be 0
    changed = (base[N - 1] - pert[N - 1]).abs().max().item()  # row N-1: must move
    ok = (leak == 0.0) and (changed > 0.0)
    print(f"[{'PASS' if ok else 'FAIL'}] causal isolation: future-K/V leak into earlier rows "
          f"= {leak:.1e} (must be 0); perturbed row moved by {changed:.2f} (must be >0)")
    return ok


@torch.no_grad()
def main():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = Qwen2Model(MODEL_DIR)
    print("model: Qwen2-0.5B fp32 (CPU) | prefill (query length N) vs decode (query length 1)\n")

    fails = 0
    fails += not test_causal_isolation()
    print()
    for prompt in PROMPTS:
        ids = tok(prompt)["input_ids"]
        N = len(ids)

        # (1) prefill vs trusted decode, per position
        decode_next = decode_predictions(model, ids)
        logits = prefill_forward(model, ids, "materialized")
        prefill_next = logits.argmax(-1).tolist()
        matches = sum(d == p for d, p in zip(decode_next, prefill_next))

        # (2) online-softmax (FlashAttention core) vs materialized
        online_err = (prefill_forward(model, ids, "online") - logits).abs().max().item()

        # (info) logit self-consistency: ~1e-4 fp noise, argmax unaffected
        drift = 0.0
        for k in range(1, N + 1):
            row = prefill_forward(model, ids[:k], "materialized")[-1]
            drift = max(drift, (row - logits[k - 1]).abs().max().item())

        ok = (matches == N) and (online_err < 1e-3)
        fails += not ok
        cont = tok.decode([prefill_next[-1]])
        print(f"[{'PASS' if ok else 'FAIL'}] N={N:<2d} prefill==decode {matches}/{N} | "
              f"online_err={online_err:.1e} | logit_drift={drift:.1e} (fp) | "
              f'next("{prompt}")="{cont}"')

    print("\nRESULT:", "ALL PREFILL CHECKS PASSED" if fails == 0 else f"{fails} CHECK(S) FAILED")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
