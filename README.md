# Ramanjit Singh

Source for [ramanjitsingh-hub.github.io](https://ramanjitsingh-hub.github.io): my personal site and the research investigations it presents.

Each investigation is a question, a derivation, an implementation written from scratch, and measured results, including the ones that did not go my way. Every figure on the site is rendered from data exported by the code in this repository.

| | Investigation | Status | Code |
|---|---|---|---|
| 01 | Re-deriving a production cycle predictor. The Kalman-filter predictor I shipped at FemoraAI, ported line by line, its failure modes, and an exact Bayesian replacement. | in progress, milestones 1 and 2 complete | [`investigations/01-femora-predictor`](investigations/01-femora-predictor) |
| 02 | When is a cheap model good enough? Routing prompts between 13 language models as a decision under a budget: the optimal rule and its frontier, what a router learns from text, where it fails, how much of the hindsight oracle is luck, and a certified quality guarantee, then learning under partial feedback and off-policy evaluation. | in progress, milestones 1 and 2 complete | [`investigations/02-model-routing`](investigations/02-model-routing) |

### Systems (earlier work on GPU computing)
| | Investigation | Status | Code |
|---|---|---|---|
| S1 | A GPU Navier–Stokes solver. From-scratch CUDA on a 4 GB laptop GPU: verification, multigrid, mixed precision, a shared-memory smoother, the lid-driven cavity, a hardware-counter roofline, and a WebGPU port. | in progress, milestones 1 to 4 complete | [`investigations/s1-gpu-navier-stokes`](investigations/s1-gpu-navier-stokes) |
| S2 | The anatomy of a graphics card. Microbenchmarks that each isolate one hardware mechanism, tested against a model written down first. | in progress, milestones 1 and 2 complete | [`investigations/s2-gpu-anatomy`](investigations/s2-gpu-anatomy) |

## Layout
```
site/                     the website (static HTML, no build step), deployed by .github/workflows/pages.yml
  index.html              homepage
  01/, 02/                research investigation pages
  s1/, s2/                systems studies (03/ and 04/ redirect to their new addresses)
  data/                   numbers exported from the experiment runs, never edited by hand
  assets/                 shared CSS, plotting helpers, the WebGPU solver and its CPU fallback
investigations/
  01-femora-predictor/    Python. port, generator, grid filter, experiments, conformal ranges, tests
  02-model-routing/       Python. decision theory, text routers, certification, bandits, off-policy evaluation, on the public SPROUT data
  s1-gpu-navier-stokes/   CUDA C++ and Python. solvers, reference implementation, experiment runners
  s2-gpu-anatomy/         CUDA C++ and Python. microbenchmarks, counter profiling, model-against-measurement analysis
```

## Reproduce
Each investigation folder has a README with its exact commands. In short:
```
cd investigations/01-femora-predictor && python -m pytest -q && python run_experiments.py && python run_conformal.py && python export_site_data.py
cd investigations/02-model-routing && python prepare.py && python fit_routers.py && python run_milestone1.py && python run_milestone2.py && python export_site_data.py   # SPROUT parquet files in data/raw/ first
cd investigations/s1-gpu-navier-stokes && ./build.sh && python run_experiments.py && python export_site_data.py
cd investigations/s2-gpu-anatomy && ./build.sh && python run_milestone1.py && python run_milestone2.py && python analyze.py && python export_site_data.py
```
Systems studies S1 and S2 need an NVIDIA GPU (developed on an RTX 3050 Laptop, sm_86) with CUDA 13.x.

## Data
Investigation 01 studies a model that ran in production on real FemoraAI users' data. Every result there is computed on synthetic data. Investigation 02 uses the public SPROUT benchmark, which is not redistributed here. No user data from FemoraAI or Luna is in this repository.

## Contact
hackit830@gmail.com · [LinkedIn](https://www.linkedin.com/in/ramanjitsingh1)
