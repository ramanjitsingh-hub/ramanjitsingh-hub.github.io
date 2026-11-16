# 02. A GPU Navier–Stokes solver

**Status. In progress.** Milestones 1 to 4 complete: verified periodic solver, latency removal and mixed-precision multigrid, wall-bounded lid-driven cavity, a shared-memory smoother, a WebGPU port running on the website, and a hardware-counter roofline that found and fixed warp divergence in the smoother.

## Question
On a bandwidth-limited consumer GPU, how accurately and how fast can a from-scratch incompressible flow solver run, and where does its time go?

## Method (milestone 1)
| Component | Choice | File |
|---|---|---|
| Equations | 2D incompressible Navier–Stokes, doubly periodic box [0, 2π)² | `src/ns2d.cu` |
| Grid | MAC staggered, N × N cells | |
| Advection | conservative form ∂(uu)/∂x + ∂(uv)/∂y, central interpolation, 2nd order | `k_rhs` |
| Diffusion | 5-point Laplacian | `k_rhs` |
| Time | SSP-RK3 (Shu–Osher), exact projection after every stage | `Solver::step` |
| Pressure | geometric multigrid V(2,2), red-black Gauss–Seidel, full-weighting restriction, bilinear prolongation, down to 4 × 4 | `MG` |
| Precision | templated, FP64 and FP32 from the same source | |
| Reference | independent NumPy implementation with an exact FFT pressure solve | `reference.py` |

Build with `./build.sh` (CUDA 13.5, MSVC 2022, sm_86), run with `python run_experiments.py`.

## Results (milestone 1, `results/summary.json`, run time 1,060 s)

Hardware. RTX 3050 Laptop, 4 GB, sm_86, 20 SMs, 64-bit memory bus. Measured device copy bandwidth **104.4 GB/s** (median of 20, 256 MB float4 copy).

### 1. Verification
The FP64 GPU solution at N = 64, T = 1 agrees with the independent NumPy/FFT implementation to **5.6 × 10⁻¹⁶** (max norm), which is machine precision. FP32 agrees to 5.7 × 10⁻⁷. This checks the advection stencils, the RK3 stages and the multigrid solve against code that shares none of them.

### 2. Grid convergence, Taylor–Green vortex (ν = 0.01, T = 1, CFL 0.5)
| N | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|
| FP64 max error | 6.26e-5 | 1.57e-5 | 3.93e-6 | 9.84e-7 | 2.46e-7 | 6.15e-8 |
| observed order | | 1.993 | 1.998 | 2.000 | 2.000 | 2.000 |
| FP32 max error | 6.29e-5 | 1.63e-5 | 4.89e-6 | 2.89e-6 | 7.45e-6 | 2.86e-5 |

FP64 is second order to three decimal places across the range, with max |∇·u| ≤ 7 × 10⁻¹². Since dt ∝ h, the RK3 time error is O(h³) and does not show. FP32 follows second order to N = 128 and then **gets worse as the grid is refined**. The cause is section 4.

### 3. Multigrid
V(2,2) cycles reduce the FP64 residual by a factor of **0.053 to 0.059 per cycle at every N from 64 to 2048**, so convergence is grid-independent as theory predicts, and the solver reaches 10⁻¹⁴ relative residual. A projection needs 7 to 8 cycles for a 10⁻¹⁰ tolerance.

### 4. FP32 multigrid stagnates
In FP32 the relative residual cannot fall below a floor that grows with N, from 1.3 × 10⁻⁶ at N = 64 to 5.9 × 10⁻⁵ at N = 2048. The residual r = f − Lp subtracts quantities of size |p|/h², so its rounding error grows like ε/h². With a 10⁻⁵ tolerance the solve never converges at large N and runs to the 40-cycle cap, which is why the FP32 runs are **slower** than FP64 (157 vs 33 ms per step at N = 128) and why the FP32 Taylor–Green error rises with N. The standard remedy is mixed-precision iterative refinement, with the residual computed in higher precision. That is now the first experiment of the precision study.

### 5. Where the time goes (FP64, ms per time step = 3 RK stages)
| N | ms/step | projection share | rhs kernel GB/s |
|---|---|---|---|
| 128 | 33.5 | 99% | 21.3 |
| 512 | 58.9 | 98% | 43.9 |
| 2048 | 474.5 | 96% | 41.8 |

- **The pressure projection takes 96 to 99% of every step.** A V-cycle costs 1.3 to 2.2 ms at N from 64 to 512, almost independent of N, so it is bound by launch latency and host synchronisation, not bandwidth. Each cycle launches hundreds of small kernels (120 for the 4 × 4 coarse solve alone) and every convergence check copies a reduction back to the host.
- **The FP32 advection kernel runs at 98.9 GB/s at N = 2048, 95% of measured copy bandwidth.** It is bandwidth bound, as it should be.
- **The FP64 advection kernel stalls near 42 GB/s.** At about 55 to 60 operations per cell (hand count) that is roughly 75 GFLOP/s, the same order as this card's FP64 peak, which consumer Ampere runs at 1/64 of FP32. Confirming it needs Nsight Compute counters, the next step.

## Limitations
- Periodic box only. No walls, so no lid-driven cavity validation yet.
- Wall-clock numbers are from a laptop GPU whose clocks vary with temperature. Each figure is one run of 30 steps after 5 warm-up steps, no variance reported yet.
- The FLOP-bound claim for FP64 is an estimate until profiled.

## Milestone 2. Removing the latency floor (`results/milestone2.json`, run time 547 s)

Every change is a switch (`--graph --coarse1 --warm --adaptive`, `--prec mixed`), so each one is measured on its own, added in order. FP64, ms per time step, median of 5 repeats (the min–max spread is under 4% everywhere).

| change, cumulative | N = 128 | N = 512 | N = 2048 |
|---|---|---|---|
| baseline (milestone 1) | 30.63 | 57.12 | 472.7 |
| + single-launch coarse solve | 11.39 | 36.40 | 451.2 |
| + CUDA Graph capture of the V-cycle | 4.06 | 26.96 | 441.1 |
| + run the expected cycles blind, check once | 3.39 | 24.44 | 406.6 |
| + warm start from the previous solve | 3.20 | 22.82 | 394.3 |
| + mixed-precision multigrid | **2.28** | **15.99** | **276.4** |
| total speed-up | **13.4×** | **3.6×** | **1.7×** |

- **At small N the cost was launches, not arithmetic.** Replacing 120 coarse-grid launches with one shared-memory kernel, then capturing each V-cycle as a CUDA Graph, cut N = 128 from 30.6 to 4.1 ms. A V-cycle at N = 64 fell from 0.345 to 0.087 ms.
- **Host synchronisations per step fell from 30 to 6** by running the expected number of cycles (from the measured contraction factor of about 0.06) before the first convergence check. Warm start then saves about 0.7 cycles per solve but adds 3 checks back. Its net gain is small, 5%.
- **At large N the solver is now bandwidth bound.** A V-cycle at N = 2048 takes 15.5 ms with or without graphs. By my traffic estimate (about 35 grid-sized arrays read or written per cycle, with coarse levels adding a third) that is roughly 75 GB/s, about 70% of the measured 104.4 GB/s. This is an estimate until the hardware counters are available.

### Mixed-precision multigrid
Each correction is computed by one FP32 V-cycle on the FP64 residual, then added to the FP64 iterate (iterative refinement, Higham and Mary 2022).

| relative residual after cycle k, N = 1024 | 3 | 6 | 9 | 12 | 16 |
|---|---|---|---|---|---|
| FP64 | 8.0e-5 | 4.0e-8 | 2.4e-11 | 4.7e-14 | 4.6e-14 |
| FP32 | 8.0e-5 | 2.5e-5 | 2.5e-5 | 2.4e-5 | 2.5e-5 |
| mixed | 8.0e-5 | 4.0e-8 | 2.4e-11 | 4.6e-14 | 4.4e-14 |

The mixed solver follows the FP64 contraction cycle for cycle down to 4.6 × 10⁻¹⁴, while pure FP32 stalls at 2.5 × 10⁻⁵. On Taylor–Green the mixed solver reproduces **every FP64 error digit at every N from 32 to 1024** (6.150632 × 10⁻⁸ at N = 1024) and runs 1.38 to 1.44× faster (85 s against 123 s for the full run at N = 1024). Halving the bytes moved by the smoother is where the gain comes from, which is consistent with the bandwidth-bound reading above.

### Throughput after milestone 2
| N | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|
| FP64, ms/step | 3.20 | 5.42 | 22.98 | 92.37 | 394.4 |
| mixed, ms/step | 2.22 | 3.80 | 16.03 | 64.40 | 276.7 |
| mixed, cell-steps per second | 7.4e6 | 1.73e7 | 1.64e7 | 1.63e7 | 1.52e7 |

## Milestone 3

### 3a. Lid-driven cavity against Ghia, Ghia and Shin (1982)
`src/cavity.cu` adds walls to the same scheme: no-slip through ghost values (u_ghost = 2U_wall − u_inside), a lid moving at U = 1, and a multigrid whose stencil drops neighbours outside the domain (homogeneous Neumann for the pressure correction). Each case runs until max |∂u/∂t| < 10⁻⁷. The reference tables (`data/ghia1982.json`) come from three independent reproductions cross-checked value by value, since the publisher's copy could not be fetched. The Re = 100 and Re = 1000 columns agree in every source. Run with `python run_cavity.py` (32 min).

| Re | N | steps to steady | max dev. u | rms dev. u | max dev. v | rms dev. v | ψ_min |
|---|---|---|---|---|---|---|---|
| 100 | 64 | 5,508 | 0.0038 | 0.0019 | 0.0086 | 0.0046 | −0.10318 |
| 100 | 128 | 22,086 | 0.0049 | 0.0024 | 0.0091 | 0.0050 | −0.10343 |
| 100 | 256 | 88,452 | 0.0050 | 0.0025 | 0.0092 | 0.0051 | −0.10350 |
| 1000 | 64 | 18,900 | 0.0178 | 0.0113 | 0.0200 | 0.0115 | −0.11360 |
| 1000 | 128 | 37,376 | 0.0030 | 0.0019 | 0.0125 | 0.0062 | −0.11751 |
| 1000 | 256 | 74,496 | 0.0053 | 0.0030 | 0.0168 | 0.0089 | −0.11858 |

Deviations are measured at Ghia's 17 centreline points, excluding the imposed wall values. Every case converged, with max |∇·u| ≤ 1.7 × 10⁻⁹.

**How to read it.** The solver converges on itself at second order. Between successive grids the centreline profiles change by 1.1 × 10⁻³ then 2.6 × 10⁻⁴ at Re = 100 (observed order 2.11), and by 2.3 × 10⁻² then 6.2 × 10⁻³ at Re = 1000 (order 1.88). At Re = 100 the remaining difference from Ghia, about 0.005 in u and 0.009 in v, does not shrink as the grid is refined. It is therefore larger than this solver's own discretisation error. Ghia's values come from a 129 × 129 grid and carry discretisation error of their own, so the comparison cannot say which side the residual difference belongs to. At Re = 1000 the deviation grows slightly from N = 128 to N = 256 while ψ_min keeps moving, consistent with the same explanation. Settling it needs a higher-accuracy reference, for example a spectral benchmark solution. That is listed as next work, not claimed here.

### 3b. Temporally blocked smoother
`k_smooth_tb` loads a 32 × 32 tile with a 4-cell halo into shared memory and performs both red-black sweeps of a smoothing call there, so the fine grid is read and written once per smoothing call instead of four times.

- **A race found and fixed.** The first version wrote back in place. Neighbouring blocks that had not yet loaded their halo then read already-updated values. On small grids every block finishes loading before any block writes, so the results matched and the bug was invisible. At N = 2048 blocks run in waves and the residual history diverged in the second digit. The fixed version writes to a second buffer and swaps. Each level is smoothed exactly twice per V-cycle, so the pointers return to their original buffers after every cycle and the captured CUDA Graph stays valid.
- **Bitwise identical.** After the fix, the Taylor–Green solution at N = 512 (FP64) and N = 1024 (mixed) is identical to the unblocked solver to the last bit (max difference 0.0).

| ms per step, median of 5 | N = 512 | N = 1024 | N = 2048 |
|---|---|---|---|
| FP64 | 22.76 | 88.65 | 368.6 |
| FP64 + blocked smoother | 15.17 | 52.66 | 221.3 |
| mixed | 15.85 | 62.29 | 267.0 |
| mixed + blocked smoother | **12.09** | **44.18** | **190.4** |

Re-measured on 2026-11-15 with the colour-compacted smoother from section 4 (`results/milestone3_perf.json`). The first session's numbers, with the branchy smoother, are kept in `results/milestone3_perf_branch.json`. Between the two sessions the same configuration can differ by up to 20% on this laptop GPU. For example mixed precision at N = 128 does not use the blocked smoother at all and measured 2.28 ms in one session and 2.81 ms in the other. Comparisons in this README are therefore only made within one session, or as interleaved A/B runs.

The blocked smoother is applied only to levels with N ≥ 512. Below that, whole time steps got slower (5.39 → 7.14 ms at N = 256). Hardware counters (section 4) show why. At N = 256 the blocked kernel launches 64 blocks, about one wave on 20 SMs, and the next level down launches 16 blocks, a quarter of a wave, while every block also recomputes its halo. Against the milestone 1 baseline, a step was **13.5× faster at N = 128, 4.8× at N = 512 and 2.7× at N = 2048** in the first session, and 11.9×, 4.9× and 2.4× when re-measured on 2026-11-15, with the same answer. The full mixed-precision Taylor–Green run at N = 1024 takes 57 s, down from 85 s in milestone 2 and 123 s in FP64.

### 3c. WebGPU port on the website
`site/assets/flow-gpu.js` ports the solver to WGSL compute shaders: MAC grid, conservative advection, multigrid V(2,2) with a single-workgroup coarse solve, 190 dispatches per frame on a 256 × 128 periodic grid. It differs from the CUDA solver in ways stated on the page: FP32, Heun (RK2) instead of SSP-RK3, a fixed two V-cycles per projection warm-started from the previous pressure, linear drag, and a passive dye advected semi-Lagrangian for display. On this laptop in Chrome it runs at 60 fps (display-capped) with max |∇·u| ≈ 3 × 10⁻⁴ read back asynchronously. Browsers without WebGPU fall back to the earlier CPU solver. That fallback path has not been tested in a browser without WebGPU yet.

## 4. Hardware counters (milestone 4)

`profile_roofline.py` profiles every kernel on its finest level at N = 1024 with Nsight Compute and writes `results/roofline.json`.

**Ceilings are measured, not taken from a datasheet.** The profiler locks the SM clock at 1.207 GHz. On `k_rhs<double>` the FP64 pipe is active for exactly 16.0 cycles per FP64 warp instruction, so each of the 20 SMs retires 2 FP64 lanes per cycle. That is 48.3 G FP64 instructions per second, or 96.6 GFLOP/s if every instruction were an FMA. DRAM bandwidth from memory clock and bus width is 112.0 GB/s, and Nsight's own percentages imply 111.8 GB/s.

| finding | evidence |
|---|---|
| Every streaming kernel, both precisions, is memory bound and near the limit | 91 to 95% of peak DRAM throughput |
| `k_rhs<double>` is compute bound | 1.6 FLOP/byte, 50.6 GFLOP/s, FP64 pipe busy 84%; only 24% of its FP64 instructions are FMAs, so its practical ceiling is about 60 GFLOP/s. The hand estimate of 75 GFLOP/s in milestone 1 was too high |
| The blocked smoother wasted half its FP64 lanes | 13.8 active lanes per FP64 warp instruction, FP64 pipe 82% busy, 20.2 GFLOP/s |

**Divergence fix.** The red-black update skipped cells of the other colour with a branch, so every FP64 instruction ran with fewer than half its lanes active. `k_smooth_tb<T, true>` maps threads onto same-colour cells only. Since N is even, wrapping preserves parity and the colour of local cell (lx, ly) is (bx + by + lx + ly) & 1. Arithmetic and update order are unchanged. The Taylor–Green solutions at N = 512 (FP64) and N = 1024 (mixed) are byte-identical to the previous kernel and to the unblocked solver (checked with `--dump` and `cmp`).

| `run_tb_compact.py`, median of interleaved repeats | N = 512 | N = 1024 | N = 2048 |
|---|---|---|---|
| FP64 V-cycle, ms, branch → compact | 0.772 → 0.575 (1.34×) | 2.587 → 1.819 (1.42×) | 9.602 → 6.909 (1.39×) |
| mixed V-cycle, ms | 1.005× | 1.014× | 1.006× |
| full time step, ms | | | FP64 278.0 → 221.3 (1.26×), mixed 214.9 → 190.7 (1.13×) |

The kernel alone goes from 750 µs to 354 µs (2.1×), at 31.1 active lanes per FP64 instruction and 42.7 GFLOP/s. The old kernel stays available as `--tbcompact 0`. An isolated mixed V-cycle barely changes because its smoother runs in FP32 and was already near the memory limit. A full mixed time step nevertheless gains 1.13×, and the reason is not yet established.

The plain red-black smoother `k_smooth` has the same divergence (16.0 lanes) but is limited by DRAM at 93% of peak, so fixing it would not help.

## Next
1. A higher-accuracy reference for the cavity to settle the residual difference from Ghia.
2. Find why a full mixed time step gains more from the divergence fix than its V-cycle does.
3. Systems study S2, the anatomy of GPU computation, built on these kernels and on the divergence finding.
