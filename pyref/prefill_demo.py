"""
Demo: prefill (query length N) causal attention inside the REAL Qwen2-0.5B model.

This shows two things, end to end on the actual LLM weights (fp32):

  (A) PREFILL == DECODE.  The existing pyref path is autoregressive *decode* (query
      length 1: one token at a time against a growing KV cache) -- the trusted code
      that already generates correct Qwen text. Here we add a *prefill* forward that
      processes all N prompt tokens at once with causal attention (query length N),
      and check it predicts the SAME next token at every position as decode. That is
      exactly the "extend decode to N tokens" generalization, verified in-model.

  (B) ONLINE-SOFTMAX == MATERIALIZED.  We run the prefill attention two ways -- the
      plain materialized softmax and the streaming online-softmax recurrence (the
      core of FlashAttention) -- and check the model's logits are identical. Same
      math as src/cpu_ref/AttentionCPU.h, now exercised on real activations.

It also dumps a fixture of REAL layer-0 activations so the C++ CPU reference
(src/tests/AttentionCPUTest.cpp) can be checked on real model data, not just N(0,1).

"""
import os, math, struct
import torch
from pathlib import Path
from transformers import AutoTokenizer
from qwen2 import Qwen2Model, t5_layer_norm, silu
from rope import RoPE

MODEL_DIR = Path(os.getenv("TRANSFORMER_MODEL_DIR") or Path(__file__).resolve().parent.parent / "models" / "Qwen2.5-0.5B-Instruct")
PROMPT = "The capital of France is"
THETA = 1e6


def causal_attention(Q, K, V, H, Hkv, d, impl):
    """Q:[N,H,d]  K,V:[N,Hkv,d] -> ctx:[N,H,d].  Mirrors src/cpu_ref/AttentionCPU.h.
       GQA broadcast kv = h*Hkv//H; scale 1/sqrt(d); causal j<=i."""
    N = Q.shape[0]
    scale = 1.0 / math.sqrt(d)
    ctx = torch.zeros(N, H, d, dtype=Q.dtype)
    for h in range(H):
        g = h * Hkv // H
        Qh, Kg, Vg = Q[:, h, :], K[:, g, :], V[:, g, :]
        if impl == "materialized":
            scores = (Qh @ Kg.T) * scale                      # [N,N]
            mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
            ctx[:, h, :] = torch.softmax(scores, dim=-1) @ Vg
        elif impl == "online":                                # streaming, like attention_online
            for i in range(N):
                m, l, acc = float("-inf"), 0.0, torch.zeros(d, dtype=Q.dtype)
                for j in range(i + 1):
                    s = float(Qh[i] @ Kg[j]) * scale
                    m_new = max(m, s)
                    alpha = math.exp(m - m_new) if m != float("-inf") else 0.0
                    p = math.exp(s - m_new)
                    l = l * alpha + p
                    acc = acc * alpha + p * Vg[j]
                    m = m_new
                ctx[i, h, :] = acc / l
    return ctx


@torch.no_grad()
def prefill_forward(model, token_ids, impl="materialized", capture_layer0=False):
    cfg = model.config
    N = len(token_ids)
    H, Hkv, d = cfg.num_query_heads, cfg.num_kv_heads, cfg.head_size
    hidden = model.embedding_weight[token_ids].clone()        # [N, hidden]
    cap = None
    for layer in model.layers:
        attn_in = torch.stack([t5_layer_norm(layer.input_layernorm, hidden[i]) for i in range(N)])
        Q = (attn_in @ layer.q_proj_weight.T + layer.q_proj_bias).view(N, H, d)
        K = (attn_in @ layer.k_proj_weight.T + layer.k_proj_bias).view(N, Hkv, d)
        V = (attn_in @ layer.v_proj_weight.T + layer.v_proj_bias).view(N, Hkv, d)
        for i in range(N):                                    # per-position RoPE (position = i)
            Q[i] = RoPE.apply_rope_to_qk(Q[i], i, THETA)
            K[i] = RoPE.apply_rope_to_qk(K[i], i, THETA)
        if capture_layer0 and cap is None:
            cap = (Q.clone(), K.clone(), V.clone())
        ctx = causal_attention(Q, K, V, H, Hkv, d, impl).reshape(N, H * d)
        hidden = hidden + ctx @ layer.o_proj_weight.T
        ffn_in = torch.stack([t5_layer_norm(layer.post_attention_layernorm, hidden[i]) for i in range(N)])
        g = silu(ffn_in @ layer.gate_proj_weight.T)
        u = ffn_in @ layer.up_proj_weight.T
        hidden = hidden + (g * u) @ layer.down_proj_weight.T
    hidden = torch.stack([t5_layer_norm(model.final_layernorm, hidden[i]) for i in range(N)])
    logits = hidden @ model.embedding_weight.T                # [N, vocab]
    return (logits, cap) if capture_layer0 else logits


@torch.no_grad()
def decode_predictions(model, token_ids):
    """Trusted reference: existing autoregressive decode, token by token.
       Returns predicted next token after each prefix [0..i]."""
    cfg = model.config
    N = len(token_ids)
    k_cache = torch.zeros((N, cfg.num_layers, cfg.keys_size()))
    v_cache = torch.zeros((N, cfg.num_layers, cfg.values_size()))
    return [model.forward(k_cache[:i + 1], v_cache[:i + 1], token_ids[i], 0.0) for i in range(N)]


def dump_fixture(path, Q, K, V, H, Hkv, d, N):
    """Write real layer-0 activations in the format of gen_attention_fixture.py so
       the C++ reference can be checked on them. O = torch SDPA (is_causal) oracle."""
    Qh = Q.permute(1, 0, 2).contiguous()                      # [H,N,d]
    Kh = K.permute(1, 0, 2).contiguous()                      # [Hkv,N,d]
    Vh = V.permute(1, 0, 2).contiguous()
    kv_for = [h * Hkv // H for h in range(H)]
    Kx = torch.stack([Kh[g] for g in kv_for], 0)              # [H,N,d]
    Vx = torch.stack([Vh[g] for g in kv_for], 0)
    O = torch.nn.functional.scaled_dot_product_attention(Qh, Kx, Vx, is_causal=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<i", 1))
        f.write(struct.pack("<iiii", H, Hkv, d, N))
        for t in (Qh, Kh, Vh, O):
            f.write(t.numpy().astype("<f4").tobytes())


def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    ids = tok(PROMPT)["input_ids"]
    model = Qwen2Model(MODEL_DIR)                             # fp32
    cfg = model.config
    H, Hkv, d, N = cfg.num_query_heads, cfg.num_kv_heads, cfg.head_size, len(ids)

    print(f"model: Qwen2-0.5B fp32 | H={H} Hkv={Hkv} d={d}")
    print(f'prompt: "{PROMPT}"  ->  {N} tokens {ids}')
    print("        ", [tok.decode([i]) for i in ids])

    decode_preds = decode_predictions(model, ids)
    logits_mat, cap = prefill_forward(model, ids, "materialized", capture_layer0=True)
    prefill_preds = logits_mat.argmax(-1).tolist()
    logits_onl = prefill_forward(model, ids, "online")

    print("\n(A) prefill (query length N) vs trusted decode (query length 1), per position:")
    print("    pos | context so far                  | decode next | prefill next | match")
    all_match = True
    for i in range(N):
        ctx_txt = tok.decode(ids[: i + 1]).strip()[:30].ljust(30)
        dnext, pnext = decode_preds[i], prefill_preds[i]
        ok = (dnext == pnext)
        all_match &= ok
        print(f"    {i:3d} | {ctx_txt} | {repr(tok.decode([dnext]))[:11].ljust(11)} | "
              f"{repr(tok.decode([pnext]))[:12].ljust(12)} | {'OK' if ok else 'DIFF'}")
    print(f"    => prefill==decode at all {N} positions: {all_match}")

    onl_diff = (logits_onl - logits_mat).abs().max().item()
    print(f"\n(B) online-softmax vs materialized prefill logits: max_abs_diff = {onl_diff:.2e} "
          f"(== FlashAttention core gives identical model output)")

    print(f'\n    greedy continuation from prefill: "{PROMPT}"', end="")
    cur = list(ids)
    for _ in range(6):
        nxt = int(prefill_forward(model, cur, "materialized").argmax(-1)[-1])
        cur.append(nxt)
        print(tok.decode([nxt]), end="")
    print()

    Q0, K0, V0 = cap
    out_path = "real_activation_fixture.bin"
    dump_fixture(out_path, Q0, K0, V0, H, Hkv, d, N)
    print(f"\n    wrote {out_path} (real layer-0 activations) for the C++ reference check")

    print("\nRESULT:", "PASS" if (all_match and onl_diff < 1e-3) else "FAIL")


if __name__ == "__main__":
    main()
