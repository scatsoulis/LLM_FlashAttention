# Naive prefill-attention baseline — ncu profile (the FlashAttention comparison point)

Kernels profiled: `naive_qk_scores_kernel`, `naive_softmax_rows_kernel<256>`, `naive_pv_kernel` (src/gpu_ops/NaivePrefillAttention.cuh), FP32, Qwen2-0.5B head config H=14, Hkv=2, d=64, causal, GQA. GPU: RTX A5000 (CC 8.6, 24 GB, ~768 GB/s HBM, ~27.8 TFLOP/s FP32). Tool: Nsight Compute 2024.2.1, default clock control (locked base clocks — so durations here are larger than the boosted-clock `cudaEvent` numbers, but consistent run-to-run).

Reproduce:
```
ncu --set full --launch-skip 3 --launch-count 3 -f -o profiling/naive_attention_N512 \
    ./build/profile_naive_attention 512
```
(`--launch-skip 3` skips the warmup forward's 3 kernels; the runner is src/profile_naive_attention.cu.) The command writes the full report (roofline chart, openable in the Nsight Compute GUI) to `profiling/naive_attention_N512.ncu-rep`.

## Per-kernel metrics

### N = 512
| kernel | DRAM rd+wr (MB) | dur (ms) | DRAM BW % | mem-pipe % | compute % | L2 hit % | occ % |
|---|---|---|---|---|---|---|---|
| QK (write S) | 2.2 + 14.5 = **16.7** | 1.67 | 1.3 | 97.3 | 12.8 | 97.8 | 76.8 |
| softmax (S→P) | 14.7 + 13.9 = **28.6** | 0.108 | 34.7 | 86.9 | 86.9 | 50.9 | 94.6 |
| PV (read P,V) | 15.0 + 3.8 = **18.7** | 0.416 | 5.9 | 96.2 | 96.2 | 95.4 | 63.2 |
| **total** | **≈ 64 MB** | **2.19** | | | | | |

### N = 1024
| kernel | DRAM rd+wr (MB) | dur (ms) | DRAM BW % | compute % | L2 hit % | occ % |
|---|---|---|---|---|---|---|
| QK (write S) | 4.2 + 58.5 = **62.7** | 6.64 | 1.2 | 12.5 | 99.3 | 82.3 |
| softmax (S→P) | 58.7 + 58.2 = **117.0** | 0.238 | 64.1 | 89.1 | 56.1 | 95.5 |
| PV (read P,V) | 59.3 + 6.0 = **65.2** | 1.61 | 5.3 | 98.0 | 96.3 | 65.0 |
| **total** | **≈ 245 MB** | **8.49** | | | | |

## What this baseline shows
- **HBM traffic is O(N²) and dominated by materializing S and P**: write S → read S → write P → read P ≈ 4·N²·H·4 bytes ≈ 58 MB at N=512, 235 MB at N=1024 — i.e. ~90% of the ~64 MB / ~245 MB total. This is exactly the traffic FlashAttention removes by keeping S/P tiles in SRAM and never writing the N×N matrices to HBM. **Expected FlashAttention HBM traffic ≈ O(N·d) (Q,K,V,O only) ≈ a few MB → ~15–25× less HBM traffic at N=512–1024.**
- **All three kernels are memory-subsystem-bound, not compute-bound.** Memory-pipe utilization is 87–97% while compute is as low as 12.8% (QK). Arithmetic intensity ≈ 4.7e8 FLOP / 64 MB ≈ **7.3 FLOP/byte at N=512, well below the A5000 FP32 roofline ridge (~36 FLOP/byte)** → memory-bound; achieved ≈ 0.2 TFLOP/s ≈ <1% of FP32 peak.
- **QK is the worst kernel** (1.67 ms / 6.64 ms — the dominant cost): ncu flags *"uncoalesced global accesses, 102,760,448 excessive sectors (84%)"*. Each thread reloads
Q[h,i,:]/K[g,j,:] from global with poor coalescing and no reuse — precisely what shared-memory tiling fixes.
- **Occupancy is already decent (63–95%)** — so the bottleneck is *traffic/coalescing*, not occupancy. FlashAttention will likely trade some occupancy (large SRAM tiles limit blocks/SM) for a large drop in HBM traffic and far better coalescing.

## Consistency protocol for the FlashAttention comparison (apples-to-apples)
Profile the FlashAttention kernel(s) **identically**:
- same standalone FP32 runner pattern (a `profile_flash_attention.cu` mirroring src/profile_naive_attention.cu), **same Q/K/V/O layout, precision (FP32), seed, H/Hkv/d, causal+GQA**;
- same N values (512, 1024) and the same `ncu --set full --launch-skip <warmup> --launch-count <K>` command;
- compare the same headline numbers: **total HBM bytes, duration, DRAM-BW%, mem-pipe%, compute%, arithmetic intensity / roofline position, achieved occupancy.**

Headline baseline to beat — **total HBM traffic: 64 MB @ N=512, 245 MB @ N=1024.**

## Scaling with N — does the naive prefill become compute-bound? No.
| N | total HBM | total dur (ncu) | QK compute % | QK DRAM % | softmax DRAM % | AI (FLOP/byte) |
|---|---|---|---|---|---|---|
| 512 | 64 MB | 2.19 ms | 12.8 | 1.3 | 35 | 7.3 |
| 1024 | 245 MB | 8.49 ms | 12.5 | 1.2 | 64 | 7.7 |
| 2048 | 1.02 GB | 33.7 ms | 12.4 | 1.2 | 83 | 7.4 |
| 4096 | 4.93 GB | 135.6 ms | 12.3 | 1.2 | 94 | 6.1 |

Both FLOPs (≈ 2·H·N²·d) and the dominant HBM traffic (S/P materialization ≈ 4·H·N²·4 B) scale as N²,
so **AI is constant in N (~7)** — the naive prefill stays memory-bound at every N, and softmax even
approaches DRAM-BW saturation (94%) as N grows. Increasing N does NOT move it toward the compute roofline.

This is the squandered-potential point: prefill *could* have AI ≈ O(N) (each K/V reused across N queries, vs decode's AI ≈ 2), but materializing S to HBM throws that away. FlashAttention recovers it — no S/P in HBM → memory ≈ O(N·d) → AI ≈ O(N) → compute-bound at large N. Compute-bound is the *destination* FlashAttention delivers, not the starting point.
