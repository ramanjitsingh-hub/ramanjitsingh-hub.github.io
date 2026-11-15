# Ramanjit Singh

Source for [ramanjitsingh-hub.github.io](https://ramanjitsingh-hub.github.io): my personal site and the research investigations it presents.

Each investigation is a question, a derivation, an implementation written from scratch, and measured results, including the ones that did not go my way. Every figure on the site is rendered from data exported by the code in this repository.

| | Investigation | Status | Code |
|---|---|---|---|
| 01 | Re-deriving a production cycle predictor. The Kalman-filter predictor I shipped at FemoraAI, ported line by line, its failure modes, and an exact Bayesian replacement. | in progress, synthetic-data study complete | [`investigations/01-femora-predictor`](investigations/01-femora-predictor) |
| 02 | A GPU Navier–Stokes solver. A from-scratch CUDA solver on a 4 GB laptop GPU: verification, multigrid, mixed precision, latency removal, a shared-memory smoother, the lid-driven cavity, a hardware-counter roofline, and a WebGPU port. | in progress, milestones 1 to 4 complete | [`investigations/02-gpu-navier-stokes`](investigations/02-gpu-navier-stokes) |
| 03 | The anatomy of a graphics card. Microbenchmarks that each isolate one hardware mechanism, tested against a model written down first: memory latency, Little's law, coalescing, the roofline, divergence, shared-memory banks and address translation. | in progress, milestones 1 and 2 complete | [`investigations/03-gpu-anatomy`](investigations/03-gpu-anatomy) |

## Layout
```
site/                     the website (static HTML, no build step), deployed by .github/workflows/pages.yml
  index.html              homepage
  01/, 02/, 03/           investigation pages
  data/                   numbers exported from the experiment runs, never edited by hand
  assets/                 shared CSS, plotting helpers, the WebGPU solver and its CPU fallback
investigations/
  01-femora-predictor/    Python. port, generator, grid filter, experiments, tests
  02-gpu-navier-stokes/   CUDA C++ and Python. solvers, reference implementation, experiment runners
  03-gpu-anatomy/         CUDA C++ and Python. microbenchmarks, counter profiling, model-against-measurement analysis
```

## Reproduce
Each investigation folder has a README with its exact commands. In short:
```
cd investigations/01-femora-predictor && python -m pytest -q && python run_experiments.py && python export_site_data.py
cd investigations/02-gpu-navier-stokes && ./build.sh && python run_experiments.py && python export_site_data.py
cd investigations/03-gpu-anatomy && ./build.sh && python run_milestone1.py && python run_milestone2.py && python analyze.py && python export_site_data.py
```
Investigations 02 and 03 need an NVIDIA GPU (developed on an RTX 3050 Laptop, sm_86) with CUDA 13.x.

## Data
Investigation 01 studies a model that ran in production on real FemoraAI users' data. Every result here is computed on synthetic data. No user data is in this repository.

## Contact
hackit830@gmail.com · [LinkedIn](https://www.linkedin.com/in/ramanjitsingh1)
