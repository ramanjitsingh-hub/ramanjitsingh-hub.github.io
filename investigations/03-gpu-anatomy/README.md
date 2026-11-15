# Investigation 03: the anatomy of a graphics card

**Status: in progress. Milestone 1 (seven controlled experiments) complete.**

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

**1. Three latencies.** A dependent chain of loads costs 39 cycles from L1, 213 from L2 and 525 from DRAM. The L1 plateau ends between 92 KB and 131 KB and the L2 plateau between 741 KB and 1.48 MB, consistent with the reported 1 MB L2. Beyond 64 MB every load costs about 10 cycles more, which I attribute to TLB misses but have not confirmed with counters.

**2. Bandwidth needs bytes in flight.** One warp per SM issuing one 16-byte load per thread keeps 10 KB in flight across the GPU, and Little's law predicts 35.2 GB/s against 31.0 measured. Four warps per SM reach 105 GB/s. The model is consistently 12 to 17% optimistic in the latency-bound region, which would follow if latency under load is longer than the unloaded chase latency. More independent loads per thread help only up to a point. With one warp per SM, raising K from 4 to 8 does not pass about 87 GB/s, so a single warp has a limit on outstanding requests that the simple model lacks.

**3. Coalescing is a sector count.** A warp of 4-byte loads at stride S touches min(32, 4S) sectors of 32 bytes. Nsight Compute reports exactly that, and DRAM bytes per useful byte are 1.00, 2.00, 4.00 and then 8.00 from S = 8 on. Useful bandwidth follows B_peak / min(8, S) to within 2%. At S = 64 it falls a further 11%, which the sector model does not explain. Stride 1 reaches 91% of the model because the 8 MB transfer is short enough for launch overhead to show.

**4. The roofline, traced by one kernel.** FP32 throughput rises linearly with arithmetic intensity until about 57 FLOP per byte and then reaches 6.03 TFLOP/s, 98% of 20 SMs × 128 lanes × 2 × 1.207 GHz. Two points near the ridge fall short (82% at F = 16 and 89% at F = 128). FP64 plateaus at 84% of the two-lanes-per-SM model at both clocks, and the divergence experiment gives the same 84% independently (19.1 cycles per FP64 warp instruction per SM against 16). Investigation 02 measured the FP64 pipe as active for exactly 16 cycles per instruction, so the pipe model is right about occupancy but sustained issue falls short of it. The cause is open.

**5. Divergence costs paths, not lanes.** A warp pays for every distinct path any of its lanes takes. One active lane costs as much as 32, and 32 lanes on 32 different paths cost 31.8× one path. min(active, paths) predicts all 16 combinations to within 0.5%. Investigation 02's smoother lost half its FP64 throughput to exactly this.

**6 and 7. Banks, and a limit that is not banks.** Shared-memory time follows gcd(S, 32) exactly, and stride 33 (the usual padding trick) is back to full speed. But a conflict-free 32-bit load still takes 2 cycles per warp instruction although the counters show a single wavefront. 32-bit shared loads therefore issue at most once every 2 cycles per SM, which is 64 bytes per cycle, half of what the banks can deliver. 64-bit loads move 128 bytes in the same 2 cycles, the full bank bandwidth, and 128-bit loads stay at about 124 bytes per cycle. Counting wavefronts per 128-byte group of lanes (32 lanes for 4 bytes, 16 for 8, 8 for 16) predicts every width and stride tested. One mistake along the way is worth recording. The first version of the wide-load kernel used ordinary C++ loads, the compiler hoisted the few distinct addresses out of the loop, and the kernel reported an impossible 178 bytes per cycle. The loads are now explicit `ld.volatile.shared`.

## Open questions
1. Why sustained FP64 issue is 84% of the pipe model.
2. Why Little's law over-predicts by 12 to 17% where latency limits, and what caps one warp near 87 GB/s. Measuring loaded latency directly would test the first part.
3. The 10-cycle step beyond 64 MB (TLB, to be confirmed with counters) and the extra loss at stride 64.

## Reproduce
```
./build.sh                 # bin/anat (CUDA 13.5, sm_86)
python run_milestone1.py   # all sweeps, about 5 minutes; close anything else using the GPU first
python profile_counters.py # Nsight Compute counters for the stride and bank experiments
python analyze.py          # model against measurement, results/analysis.json
```
Results are in `results/`: `milestone1.json` (raw), `counters.json`, `analysis.json`.

## References
- H. Wong, M.-M. Papadopoulou, M. Sadooghi-Alvandi, A. Moshovos. Demystifying GPU microarchitecture through microbenchmarking. ISPASS 2010.
- Z. Jia, M. Maggioni, B. Staiger, D. P. Scarpazza. Dissecting the NVIDIA Volta GPU architecture via microbenchmarking. arXiv:1804.06826, 2018.
- J. D. C. Little. A proof for the queuing formula L = λW. Operations Research 9(3), 383–387, 1961.
- S. Williams, A. Waterman, D. Patterson. Roofline: an insightful visual performance model for multicore architectures. Communications of the ACM 52(4), 65–76, 2009.
