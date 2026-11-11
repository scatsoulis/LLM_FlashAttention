"""
Generate a PyTorch oracle fixture for the CPU attention reference.

We use torch.nn.functional.scaled_dot_product_attention with is_causal=True as the
known-correct reference. GQA broadcasting is done
EXPLICITLY here (expand the Hkv KV heads to H query heads with the mapping
g = h * Hkv // H) so the oracle uses identical semantics to AttentionCPU.h, with no
dependence on torch's enable_gqa flag / version.

Everything is float32 on CPU (the math backend) for the most deterministic reference.

Output binary layout (little-endian), read by src/tests/AttentionCPUTest.cpp:
    int32  num_cases
    repeated per case:
        int32 H, int32 Hkv, int32 d, int32 N
        float32 Q[H * N * d]
        float32 K[Hkv * N * d]
        float32 V[Hkv * N * d]
        float32 O[H * N * d]
"""
import struct
import torch
import torch.nn.functional as F

torch.manual_seed(179)

# (H, Hkv, d, N). Qwen2-0.5B uses H=14, Hkv=2, d=64; several N to exercise scaling.
CASES = [
    (14, 2, 64, 1),
    (14, 2, 64, 2),
    (14, 2, 64, 8),
    (14, 2, 64, 64),
    (14, 2, 64, 128),
    (14, 2, 64, 256),
    (14, 2, 64, 512),
]

OUT_PATH = "attention_fixture.bin"


def make_case(H, Hkv, d, N):
    # N(0,1) inputs -> O(1)-O(10) scores, well-conditioned for a tight 1e-5 check.
    Q = torch.randn(H, N, d, dtype=torch.float32)
    K = torch.randn(Hkv, N, d, dtype=torch.float32)
    V = torch.randn(Hkv, N, d, dtype=torch.float32)

    # Broadcast KV heads to query heads using the same mapping as AttentionDims::kv_head_for.
    kv_for = [h * Hkv // H for h in range(H)]
    Kx = torch.stack([K[g] for g in kv_for], dim=0)  # [H, N, d]
    Vx = torch.stack([V[g] for g in kv_for], dim=0)  # [H, N, d]

    # Default scale = 1/sqrt(d); is_causal masks j > i for the square NxN prefill case.
    O = F.scaled_dot_product_attention(Q, Kx, Vx, is_causal=True)  # [H, N, d]
    return Q, K, V, O


def main():
    with open(OUT_PATH, "wb") as f:
        f.write(struct.pack("<i", len(CASES)))
        for (H, Hkv, d, N) in CASES:
            Q, K, V, O = make_case(H, Hkv, d, N)
            f.write(struct.pack("<iiii", H, Hkv, d, N))
            for t in (Q, K, V, O):
                f.write(t.contiguous().numpy().astype("<f4").tobytes())
            print(f"case H={H} Hkv={Hkv} d={d} N={N}: "
                  f"O mean={O.mean().item():+.6f} std={O.std().item():.6f}")
    print(f"wrote {OUT_PATH} ({len(CASES)} cases)")


if __name__ == "__main__":
    main()
