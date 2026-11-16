# Investigation 04 · When is a cheap model good enough?

Luna sends each request to the cheapest language model that can handle it. This investigation studies that rule as a decision under uncertainty, on public data where every prompt was answered by every model.

Status: **in progress, milestones 1 and 2 complete** (2026-11-16). Page: [ramanjitsingh-hub.github.io/04](https://ramanjitsingh-hub.github.io/04/).

## Data
SPROUT (Somerstep et al., *CARROT: a cost aware rate optimal router*, arXiv:2502.03261), from the Hugging Face hub (`CARROT-LLM-Routing/SPROUT`, Parquet). It has 44,241 prompts from six benchmark families (MATH, MMLU-Pro, GPQA, MuSR, RAGBench, OpenHermes), each answered by 13 models and scored from 0 to 1 by a Llama 3.1 70B judge against a reference answer. Cost per answer = input tokens × input price + output tokens × output price, with the prices from Table 2 of the paper (verified against the paper's HTML). The official split is used: 30,968 train, 6,636 validation, 6,637 test.

The dataset card gives no licence, so `data/raw/` is not committed and only statistics computed from it are published. No Luna user data is used anywhere.

## Theory
See [THEORY.md](THEORY.md). It covers:
- the Lagrangian routing rule;
- the proof that the achievable frontier is the upper concave hull, so the hull of single models is the right baseline;
- a plug-in regret bound;
- why the hindsight oracle is biased upward by luck (Jensen);
- the fixed-sequence certification with Hoeffding and betting-martingale p-values.

## Milestone 1 results (test set)

| | cost to match GPT-4o | score at GPT-4o mini's cost | best score |
|---|---|---|---|
| single models, mixed | 100% | 0.807 | 0.848 |
| benchmark-label router | 17.0% | 0.810 | 0.866 |
| text ridge (TF-IDF) | 12.5% | 0.826 | 0.867 |
| text boosted trees | 11.9% | 0.821 | 0.862 |
| ensemble | **10.3%** (95% CI 8.2 to 13.5%) | 0.829 | 0.871 |
| hindsight oracle | 0.9% | · | 0.979 |

Findings:
1. **Router gains are real but modest.** Against the hull of fixed models, the ensemble gains 4.0 points of score averaged over log cost.
2. **Most of the gain is task recognition.** Knowing only which benchmark a prompt came from already brings the cost of matching GPT-4o to 17%. Within a single benchmark, the text router adds 0.4 to 3.1 points on five benchmarks and loses 0.8 on GPQA.
3. **Routers do not transfer to unseen task types.** Retrained without one benchmark and tested on it, the router falls below the best fixed mix for that benchmark on all six, by 2.4 to 8.8 points. It beats an equally uninformed fixed ranking on only 2 of 6.
4. **The oracle is mostly unreachable.** The binary oracle scores 0.986. Calibrated beliefs with independent errors would give 0.9998, so model errors are shared: the residual pairwise correlation is 0.25, against 0.39 raw.
5. **Optimizer's curse.** The router expects 0.907 for its chosen models and realises 0.881.
6. **Validation-picked thresholds miss.** The λ picked on validation gives 0.835 at 7.4% of the cost on test, 1.3 points below GPT-4o although it matched GPT-4o on validation.
7. **Certification holds.** Using the test set as the population with 1,000 calibration draws per size:
   - The naive cheapest-passing rule misses the target in 42 to 44% of runs.
   - Fixed-sequence testing misses in at most 4.2% of runs, against δ = 10%.
   - The cost of that safety: with 4,000 calibration prompts the betting version averages 52% of GPT-4o's cost, against 10.8% truly achievable.

Limitation: the judge is a Llama model and may favour Llama-style answers. This was not tested.

## Reproduce
```
# put the five SPROUT parquet files in data/raw/
python prepare.py           # arrays + costs
python fit_routers.py       # features, ridge and boosted trees (~10 min)
python run_milestone1.py    # results/milestone1.json (leave-one-family-out ~20 min the first time, cached after)
python export_site_data.py  # site/data/inv04.json
```

## Milestone 2 · partial feedback (`run_milestone2.py`)
**Contextual bandits.**
- **Setup:** 37,604 train+validation prompts streamed in 5 random orders, reward = score − λ·cost with λ = 30, 100 and 300. Per-model linear models run over 64 SVD dimensions.
- **Greedy wins.** It earns the highest stream reward of the six partial-feedback learners (greedy, ε-greedy 5% and 20%, LinUCB α 0.3 and 1, Thompson) at all three λ. Deliberate exploration only costs here, which matches Bastani, Bayati and Khosravi (2021).
- **Partial feedback still costs.** No partial-feedback learner beats always using GPT-4o mini over the stream, while the full-information learner does: at λ = 100, greedy finishes 469 behind and full information 380 ahead.
- **The learned policy catches up.** Greedy's late slope is negative at λ = 30 and 100. Its final model routes the test set almost as well as the full-information one: 0.820 vs 0.826 at 8.6% of GPT-4o's cost.

**Off-policy evaluation.**
- **Setup:** the logging policy is ε = 0.2 greedy over a linear router, run on test and replayed 300 times. The reward model is cross-fitted.
- **DM:** low variance, but biased by −0.024 for the milestone 1 router.
- **IPS:** unbiased, with sd 0.058.
- **SNIPS and DR:** close to unbiased, with sd about 0.023 to 0.025.
- **The logs can't settle the decision.** The true gap between the milestone 1 router and the logging router (0.015) is smaller than DR's sd.
- **Intervals undercover for rarely logged policies.** For always-GPT-4o, which matches the logged choice 3.5% of the time, nominal 95% intervals cover 86% (DR) and 91% (IPS).

## Next
- Sentence-embedding features computed on the GPU.
- Certification under distribution shift.
- Cascades.
