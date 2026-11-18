# Ramanjit Singh

Source for [ramanjitsingh-hub.github.io](https://ramanjitsingh-hub.github.io): my personal site and the research investigations it presents.

Each investigation starts inside FemoraAI or Luna, from a decision the product makes or one we faced as founders. It is a question, a derivation, an implementation written from scratch, and measured results, including the ones that did not go my way. Every figure on the site is rendered from data exported by the code in this repository.

| | Investigation | Status | Code |
|---|---|---|---|
| 01 | Re-deriving a production cycle predictor. The Kalman-filter predictor I shipped at FemoraAI, ported line by line, its failure modes, and an exact Bayesian replacement. | in progress, milestones 1 and 2 complete | [`investigations/01-femora-predictor`](investigations/01-femora-predictor) |
| 02 | When is a cheap model good enough? Routing prompts between 13 language models as a decision under a budget: the optimal rule and its frontier, what a router learns from text, where it fails, how much of the hindsight oracle is luck, and a certified quality guarantee, then learning under partial feedback and off-policy evaluation. | in progress, milestones 1 and 2 complete | [`investigations/02-model-routing`](investigations/02-model-routing) |
| 03 | What can a startup learn from fifty users? Interval coverage, cohort noise, empirical Bayes shrinkage, the peeking problem and always-valid tests, checked against a real customer base used as ground truth. | in progress, milestone 1 complete | [`investigations/03-startup-statistics`](investigations/03-startup-statistics) |
| 04 | When does mixing keyword and meaning search help? BM25 and a GPU sentence encoder on five BEIR datasets, six fusion methods with paired tests, and a two-signal detection model that predicts the result. | in progress, milestone 1 complete | [`investigations/04-retrieval-fusion`](investigations/04-retrieval-fusion) |

## Layout
```
site/                     the website (static HTML, no build step), deployed by .github/workflows/pages.yml
  index.html              homepage
  01/ .. 04/              research investigation pages
  data/                   numbers exported from the experiment runs, never edited by hand
  assets/                 shared CSS and plotting helpers
investigations/
  01-femora-predictor/    Python. port, generator, grid filter, experiments, conformal ranges, tests
  02-model-routing/       Python. decision theory, text routers, certification, bandits, off-policy evaluation, on the public SPROUT data
  03-startup-statistics/  Python. interval coverage, shrinkage, sequential tests, on the public Online Retail II data
  04-retrieval-fusion/    Python and PyTorch. BM25, dense retrieval, fusion, detection-theory model, on public BEIR data
```

## Reproduce
Each investigation folder has a README with its exact commands. In short:
```
cd investigations/01-femora-predictor && python -m pytest -q && python run_experiments.py && python run_conformal.py && python export_site_data.py
cd investigations/02-model-routing && python prepare.py && python fit_routers.py && python run_milestone1.py && python run_milestone2.py && python export_site_data.py   # SPROUT parquet files in data/raw/ first
cd investigations/03-startup-statistics && python prepare.py && python run_milestone1.py && python export_site_data.py   # Online Retail II in data/raw/ first
cd investigations/04-retrieval-fusion && python run_fusion.py && python export_site_data.py   # BEIR datasets in data/raw/ first
```

## Data
Investigation 01 studies a model that ran in production on real FemoraAI users' data. Every result there is computed on synthetic data. Investigations 02, 03 and 04 use public data (SPROUT, Online Retail II, BEIR), which is not redistributed here. No user data from FemoraAI or Luna is in this repository.

## Contact
hackit830@gmail.com · [LinkedIn](https://www.linkedin.com/in/ramanjitsingh1)
