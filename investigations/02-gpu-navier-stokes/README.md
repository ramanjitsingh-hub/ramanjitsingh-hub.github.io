# 02. A GPU Navier–Stokes solver

**Status. In progress.** Milestone 1 (verified 2D periodic solver, convergence, multigrid, first performance baseline) complete. Kernel optimisation, wall-bounded flows and the CUDA anatomy study are next.

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

## Next (milestone 2)
1. Remove the latency floor: single-block coarse solve, device-side convergence check, CUDA Graph capture of the V-cycle.
2. Nsight Compute roofline of every kernel (this is where investigation "The anatomy of GPU computation" starts).
3. Mixed-precision multigrid: FP32 smoothing with FP64 residuals and refinement.
4. Walls and the lid-driven cavity against published benchmark data.
