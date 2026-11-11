# FlashAttention vs naive prefill attention

Both are standalone FP32 attention kernels with the **same inputs/layout/precision/causal/GQA** (Q:[H,N,d], K,V:[Hkv,N,d], O:[H,N,d], H=14, Hkv=2, d=64), so this is an apples-to-apples kernel-vs-kernel comparison. RTX A5000 (CC 8.6). Naive = 3-kernel materialized (src/gpu_ops/NaivePrefillAttention.cuh); Flash = fused tiled `flash_attention_kernel<16,32,64>` (src/gpu_ops/FlashAttention.cuh), online softmax with K/V tiles in shared memory, **no N×N matrices in HBM and no scratch buffers**.

## Correctness (flashattentiontest)
FlashAttention matches the CPU reference (AttentionCPU.h `attention_materialized`) AND the naive GPU kernel to **max_abs ≤ 1.2e-6** for N ∈ {1,2,8,64,128,256,512,1024} — exact attention. The test gates on BOTH the per-element allclose AND a tight max_abs bound (≤ 5e-6), so the headline accuracy number is the actual pass criterion (not just a printed diagnostic).

## HBM traffic (ncu, same protocol as the naive baseline)
| N | naive HBM (rd+wr) | flash HBM | **traffic reduction** |
|---|---|---|---|
| 512 | 64.0 MB | **3.8–3.9 MB** | **16–17×** |
| 1024 | 245 MB | **8.4 MB** | **29×** |

(N=512 measures 16.3× via targeted `--metrics` and 17.1× via `--set full` — small per-mode variance.)

FlashAttention removes the S/P materialization (write S → read S → write P → read P ≈ 4·H·N²·4 B),
which was ~90% of the naive traffic. The reduction grows with N (16× → 29×) because the eliminated
term is O(N²) while flash's traffic (Q,K,V,O) is O(N·d).

## Roofline position — memory-bound → compute-bound
| | naive (N=512) | flash (N=512) |
|---|---|---|
| arithmetic intensity | ~7 FLOP/byte | **~120 FLOP/byte** |
| A5000 ridge | ~36 FLOP/byte | ~36 |
| SM (compute) throughput | 12.8% (QK) | **80.5%** |
| DRAM throughput | 1–35% | <1% |
| achieved occupancy | 63–95% | 78.5% |
| bound | **memory-subsystem** | **compute** |

FlashAttention crosses the roofline ridge: AI rises ~7 → ~120, and the kernel becomes compute-bound
(SM ~80–90%) instead of memory/uncoalesced-bound. This is the "compute-bound is the destination"
point, now measured.

## Wall-clock speedup (cudaEvent kernel time, best of 20)
| N | naive | flash | speedup |
|---|---|---|---|
| 512 | 2.83 ms | 0.65 ms | 4.4× |
| 1024 | 10.08 ms | 2.62 ms | 3.9× |
| 2048 | 40.22 ms | 10.47 ms | 3.8× |
| 4096 | 162.46 ms | 42.03 ms | 3.9× |

**Honest note on why wall-clock (~4×) < HBM-traffic cut (16–29×):** the naive kernel was *not* purely DRAM-bandwidth-bound — its dominant QK kernel was L1TEX/uncoalesced-bound (DRAM only 1.3%) and softmax/PV were compute/L2-bound. So cutting HBM *bytes* doesn't translate 1:1 to wall-clock. FlashAttention is now compute-bound, so further wall-clock gains would come from compute/tile tuning (tensor cores, better tile sizes), not more IO reduction. The IO win (16–29× HBM) is the headline metric; the ~4× wall-clock is the realized end speedup at these sizes.

## End-to-end (FlashAttention in the full bf16 prefill) — the diagnostic `flashprefilltest` wires a bf16, cache-aware tiled FlashAttention(`flash_attention_cache_kernel`) into `Qwen2Prefill` behind a flag and runs the real model. Findings:
- **Correct**: flash prefill predicts 5/5 next-tokens == the decode loop, and is **bit-identical** to the model's current online-streaming prefill attention (max |logit diff| = 0.000) — flash is a faithful re-tiling of the same arithmetic.
- **End-to-end speedup ≈ 1.00×** (full prefill N=512: 2079 ms current vs 2082 ms flash). Two reasons, both expected: (1) attention is a small slice of the model (the projection/FFN/logit matmuls dominate — Amdahl), and (2) the model's prefill attention was **already** online-streaming, so it never paid the N² materialization cost that FlashAttention removes. The kernel-level win above (16–29× HBM, ~4× wall-clock) is vs the *materialized* naive baseline; the in-model baseline was already "flash-like", so there's no further end-to-end gain to capture here.

This is the "tok/s for full inference" diagnostic; the kernel-level comparison is where the FlashAttention result actually lives. (The non-attention matmuls here are themselves naive and dominate the 2079 ms; optimizing them is a separate axis, unrelated to attention.)

## Reproduce (same protocol both kernels)
```
ncu --set full --launch-skip 1 --launch-count 1 -f -o profiling/flash_attention_N512 \
    ./build/profile_flash_attention 512     # flash (1 kernel)
ncu --set full --launch-skip 3 --launch-count 3 -f -o profiling/naive_attention_N512 \
    ./build/profile_naive_attention 512     # naive (3 kernels)
```
Tile sizes RPB=16, BC=32 (16 KB shared). A sweep over RPB∈{4,8,16,32} × BC∈{16,32,64}(`flashattentionsweep`) shows all configs within ~5% (kernel is compute-bound, so tiling mostly affects occupancy, not the dominant cost); RPB=16/BC=32 is already near-optimal I guess
