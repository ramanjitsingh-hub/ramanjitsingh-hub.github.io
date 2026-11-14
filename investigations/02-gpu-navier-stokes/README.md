# 02. A GPU Navier–Stokes solver

**Status. In progress.** Milestone 1 (verified solver, convergence, multigrid, baseline) and milestone 2 (latency removal, mixed-precision multigrid) complete. Wall-bounded flows, shared-memory smoothers and hardware-counter profiling are next.

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

The mixed solver follows the FP64 contraction cycle for cycle down to 4.6 × 10⁻¹⁴, while pure FP32 stalls at 2.5 × 10⁻⁵. On Taylor–Green the mixed solver reproduces **every FP64 error digit at every N from 32 to 1024** (6.150632 × 10⁻⁸ at N = 1024) and runs 1.4 to 1.45× faster (5.3 s against 7.6 s at N = 1024). Halving the bytes moved by the smoother is where the gain comes from, which is consistent with the bandwidth-bound reading above.

### Throughput after milestone 2
| N | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|
| FP64, ms/step | 3.20 | 5.42 | 22.98 | 92.37 | 394.4 |
| mixed, ms/step | 2.22 | 3.80 | 16.03 | 64.40 | 276.7 |
| mixed, cell-steps per second | 7.4e6 | 1.73e7 | 1.64e7 | 1.63e7 | 1.52e7 |

## Not done yet, and why
- **Hardware-counter profiling.** Nsight Compute 2026.4 is installed but Windows blocks counter access (`ERR_NVGPUCTRPERM`) until it is enabled in the NVIDIA Control Panel with administrator rights.
- **Lid-driven cavity.** It needs walls (no-slip velocity, Neumann pressure in the multigrid) and the published benchmark table, which I will take from the source rather than from memory. Moved to milestone 3.

## Next (milestone 3)
1. Shared-memory, temporally blocked red-black smoother to cut fine-level traffic, measured against the roofline once counters are on.
2. Walls and the lid-driven cavity against the Ghia, Ghia and Shin (1982) tables.
3. A live WebGPU port of this solver for the website, replacing the CPU-only hero flow.
