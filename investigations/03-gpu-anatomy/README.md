# Investigation 03: the anatomy of a graphics card

**Status: in progress. Milestone 1 (seven controlled experiments) and milestone 2 (follow-up experiments on the gaps) complete.**

Investigation 02 found its speed-ups by reading hardware counters. This investigation takes the effects behind those counters one at a time. Each experiment is a small kernel that isolates one mechanism, and each comes with a simple model written down before the run. The measurement then either confirms the model or shows where it breaks, and where it breaks is reported, not tuned away.

Hardware: NVIDIA GeForce RTX 3050 4 GB Laptop GPU (Ampere GA107, compute capability 8.6), 20 SMs, 1 MB L2, 64-bit GDDR6 at 7.0 GHz (112.0 GB/s by specification). All code is in `src/anat.cu`, one mode per experiment.

## A note on the clock
The SM clock is not fixed. Every kernel measures its own clock by reading `clock64()` against `%globaltimer` in block 0, which does not depend on how many blocks are resident. Two states appear. Under heavy DRAM or FP32 load the GPU sits at **1.207 GHz**, its base clock, and otherwise it boosts to about **1.80 GHz** (for example FP64 at high arithmetic intensity, where DRAM is idle). All rates below are compared with peaks at the clock measured in the same run.

## Milestone 1

| # | experiment | model | result |
|---|---|---|---|
| 1 | pointer chase, one thread, random cycle over 128-byte lines | separate latency plateaus per memory level | L1 **38.9** cycles up to 92 KB, L2 **212.9** up to 741 KB, DRAM **525.1** cycles (290.7 ns), and 535.2 beyond 64 MB |
| 2 | streaming read at controlled warps per SM (W) and independent loads per thread (K) | Little's law, BW = min(B_peak, 20 · W · K · 512 B / L_DRAM) | peak 109.0 GB/s (97% of spec). Where latency limits, measured is **0.83 to 0.88** of the model |
| 3 | 4-byte reads at stride S | 32-byte sectors, useful BW = B_peak / min(8, S) | within **2%** for S = 3 to 48; counters show exactly 4S sectors per request up to 32, and 8.00 DRAM bytes per useful byte from S = 8 |
| 4 | F FMAs per loaded element | roofline, P = min(P_peak(f), AI · B_peak) | FP32 reaches **98%** of 6.17 TFLOP/s at 1.207 GHz; FP64 plateaus at **84%** of its model |
| 5 | FP64 work under divergence, `active` lanes and `paths` distinct code paths | time ∝ min(active, paths) | within **0.5%** on 16 combinations; one active lane costs as much as 32 |
| 6 | shared-memory reads, word stride S | bank conflict degree gcd(S, 32) | exactly **max(2, gcd(S, 32))** cycles for all 34 strides; counters show gcd(S, 32) wavefronts |
| 7 | shared-memory reads of 4, 8 and 16 bytes per lane | cycles = max(2, wavefronts), wavefronts counted per 128-byte group of lanes | 55 of 67 points within 1%, worst 13% |

### What the experiments show

**1. Three latencies.** A dependent chain of loads costs 39 cycles from L1, 213 from L2 and 525 from DRAM. The L1 plateau ends between 92 KB and 131 KB and the L2 plateau between 741 KB and 1.48 MB, consistent with the reported 1 MB L2. Beyond 64 MB every load costs about 10 cycles more. Milestone 2 identifies this as a 16-entry translation cache with 2 MB pages, which covers 32 MB.

**2. Bandwidth needs bytes in flight.** One warp per SM issuing one 16-byte load per thread keeps 10 KB in flight across the GPU, and Little's law predicts 35.2 GB/s against 31.0 measured. Four warps per SM reach 105 GB/s. The model is consistently 12 to 17% optimistic in the latency-bound region, which would follow if latency under load is longer than the unloaded chase latency. More independent loads per thread help only up to a point. With one warp per SM, raising K from 4 to 8 does not pass about 87 GB/s. Milestone 2 shows that this cap came from the loop structure, not from the hardware: a software-pipelined loop reaches the full peak with one warp.

**3. Coalescing is a sector count.** A warp of 4-byte loads at stride S touches min(32, 4S) sectors of 32 bytes. Nsight Compute reports exactly that, and DRAM bytes per useful byte are 1.00, 2.00, 4.00 and then 8.00 from S = 8 on. Useful bandwidth follows B_peak / min(8, S) to within 2%. At S = 64 it falls a further 11%. Milestone 2 traces that to the 512 MB span of the test, not to the stride. Stride 1 reaches 91% of the model because the 8 MB transfer is short enough for launch overhead to show.

**4. The roofline, traced by one kernel.** FP32 throughput rises linearly with arithmetic intensity until about 57 FLOP per byte and then reaches 6.03 TFLOP/s, 98% of 20 SMs × 128 lanes × 2 × 1.207 GHz. Two points near the ridge fall short (82% at F = 16 and 89% at F = 128). FP64 plateaus at 84% of the two-lanes-per-SM model at both clocks, and the divergence experiment gives the same 84% independently (19.1 cycles per FP64 warp instruction per SM against 16). Investigation 02 measured the FP64 pipe as active for exactly 16 cycles per instruction, so the pipe model is right about occupancy but sustained issue falls short of it. Milestone 2 measures the gap as 3 idle cycles per instruction and finds warps stalled on the short scoreboard.

**5. Divergence costs paths, not lanes.** A warp pays for every distinct path any of its lanes takes. One active lane costs as much as 32, and 32 lanes on 32 different paths cost 31.8× one path. min(active, paths) predicts all 16 combinations to within 0.5%. Investigation 02's smoother lost half its FP64 throughput to exactly this.

**6 and 7. Banks, and a limit that is not banks.** Shared-memory time follows gcd(S, 32) exactly, and stride 33 (the usual padding trick) is back to full speed. But a conflict-free 32-bit load still takes 2 cycles per warp instruction although the counters show a single wavefront. 32-bit shared loads therefore issue at most once every 2 cycles per SM, which is 64 bytes per cycle, half of what the banks can deliver. 64-bit loads move 128 bytes in the same 2 cycles, the full bank bandwidth, and 128-bit loads stay at about 124 bytes per cycle. Counting wavefronts per 128-byte group of lanes (32 lanes for 4 bytes, 16 for 8, 8 for 16) predicts every width and stride tested. One mistake along the way is worth recording. The first version of the wide-load kernel used ordinary C++ loads, the compiler hoisted the few distinct addresses out of the loop, and the kernel reported an impossible 178 bytes per cycle. The loads are now explicit `ld.volatile.shared`.

## Milestone 2: following up the gaps

Each gap from milestone 1 had several candidate explanations. Milestone 2 runs one experiment per gap, chosen so that the candidates predict different outcomes (`run_milestone2.py`, results in `results/milestone2.json`).

### FP64 runs at 19 cycles per instruction, and the scoreboard is why
`anat fp64` puts W warps on each SM, each thread running either one dependent FMA chain or eight independent ones.

| warps per SM | 1 | 2 | 3 | 4 | 8 | 32 |
|---|---|---|---|---|---|---|
| one dependent chain, cycles per FP64 warp instruction per SM | 49.0 | 24.5 | 19.0 | 19.0 | 19.0 | 19.0 |
| eight independent chains | 22.8 | 19.0 | 19.0 | 19.0 | 19.0 | 19.0 |

- A dependent DFMA takes **49 cycles**, and the SM cannot issue FP64 warp instructions faster than one per **19.0 cycles**, however much parallelism is available. Little's law applied to the arithmetic unit says 49 / 19 = 2.6 warps of dependent work are needed to reach the floor, and the floor appears at 3 warps.
- One warp with eight independent chains reaches 22.8 cycles, so a single warp cannot quite saturate the unit; two can.
- Nsight Compute reports the FP64 pipe active for exactly **16.0** cycles per instruction in every configuration, so 3 of every 19 cycles the pipe sits idle. The stall reasons are the informative part. Warps wait on the **short scoreboard** (54 to 97% of stalls) and almost never on math-pipe throttling, even with eight independent chains, where no instruction waits on a result. On this chip FP64 instructions therefore behave like a variable-latency unit, tracked by the scoreboard used for shared memory and special functions, and the 3 idle cycles per instruction are consistent with a fixed hand-off cost on that path. The exact hardware mechanism is not identifiable from outside, but the 84% in milestone 1 is now measured to be 16/19 in the cycle domain, independent of clock, occupancy and dependency structure.

### The single-warp cap was the loop, not the hardware
With one warp per SM, a software-pipelined loop that issues the next round of loads before summing the current one reaches **108 GB/s at K = 8**, the full peak, while the plain loop stops near 88 to 101 GB/s depending on the run. The cap seen in milestone 1 was the loop structure (wait for all K loads, then compute, then issue), not a limit on outstanding requests. With L1 bypassed (`--cg 1`), a 512 KB L2-resident array read by one warp per SM reaches 303 GB/s with pipelining, so there is no fixed per-warp rate limit either.

The 12 to 17% shortfall at K = 1 is **not** explained. Pipelining does not change it (30.1 against 31.0 GB/s), which rules out serial work in the loop as its cause. It is also present for L2-resident data (56 against 87 GB/s predicted). In addition, latency-bound runs vary by up to 25% between sessions (31.0 and 24.8 GB/s at K = 1 on two runs of the same command), while bandwidth-bound runs repeat to 1%. One plausible reason is that the memory clock drops under light load, which would change latency between runs. The next step is to measure latency in the same process as the stream.

### The 64 MB step is a 16-entry translation cache with 2 MB pages
`anat chase --stride g --jitter 1 --cg 1` touches one 128-byte line in each g-byte block, at a random offset so blocks do not alias in L2, and bypasses L1, so the baseline is the L2 latency (212 cycles) and only address translation can add to it.

| block size g | last flat (blocks, span) | first step (blocks, span) | step |
|---|---|---|---|
| 4 KB | 4096, 16 MB | none | |
| 64 KB | 512, 32 MB | 1024, 64 MB | +8 cycles |
| 2 MB | 16, 32 MB | 17, 34 MB | +12 |
| 4 MB | 16, 64 MB | 24, 96 MB | +9 |
| 8 MB | 16, 128 MB | 17, 136 MB | +11 |

For blocks of 2 MB and larger the step comes at **17 blocks** whatever the block size, so the limit is a count of entries: **16**. For 64 KB blocks it comes when the span passes 16 × 2 MB = 32 MB, so each entry covers **2 MB**. Pages cannot be larger than 2 MB, because then 17 blocks of 2 MB would fit in 9 entries, and they cannot be smaller, because then 64 KB blocks would step at 17 blocks. A miss costs about 10 cycles, which means it is served by a larger second level, not by a page-table walk. The step is abrupt, as expected when a cyclic access pattern meets a small cache with LRU-like replacement. This explains the +10 cycles beyond 64 MB in the DRAM chase of milestone 1 (flat at exactly 32 MB, partly raised at 47 MB). One of 26 points (13 blocks of 8 MB) read 224 cycles once and 212 in three re-runs; each run allocates memory afresh, so a different physical layout is a plausible cause, and the point is kept in the data.

### Stride 64 was the span, but other strides misbehave
At a fixed 128 MB span, stride 64 delivers 13.0 GB/s, the same as strides 56 to 65, so its 11% loss in milestone 1 came from the 512 MB span that test used, not from the stride. The fixed-span sweep found something else instead: strides 96, 192 and 256 lose 35 to 55% (8.9, 6.0 and 8.0 GB/s), while 128 does not (12.6). Shifting every other warp by 128 bytes does not recover stride 256 (7.2 GB/s). These are consistent with an interaction with how physical addresses map to memory channels and banks, which is not documented for this chip. That is open.

## Open questions
1. The 12 to 17% Little's-law shortfall at low K, and the run-to-run variation of latency-bound runs. Measure loaded latency and the memory clock in the same process.
2. The dips at strides 96, 192 and 256.
3. The hardware reason for the 3 idle cycles per FP64 instruction. The cycle count is established; the mechanism behind it is not visible from software.

## Reproduce
```
./build.sh                 # bin/anat (CUDA 13.5, sm_86)
python run_milestone1.py   # all sweeps, about 5 minutes; close anything else using the GPU first
python profile_counters.py # Nsight Compute counters for the stride and bank experiments
python run_milestone2.py   # follow-up experiments on the gaps, about 6 minutes
python analyze.py          # model against measurement, results/analysis.json
```
Results are in `results/`: `milestone1.json` and `milestone2.json` (raw), `counters.json`, `analysis.json`.

## References
- H. Wong, M.-M. Papadopoulou, M. Sadooghi-Alvandi, A. Moshovos. Demystifying GPU microarchitecture through microbenchmarking. ISPASS 2010.
- Z. Jia, M. Maggioni, B. Staiger, D. P. Scarpazza. Dissecting the NVIDIA Volta GPU architecture via microbenchmarking. arXiv:1804.06826, 2018.
- J. D. C. Little. A proof for the queuing formula L = λW. Operations Research 9(3), 383–387, 1961.
- S. Williams, A. Waterman, D. Patterson. Roofline: an insightful visual performance model for multicore architectures. Communications of the ACM 52(4), 65–76, 2009.
