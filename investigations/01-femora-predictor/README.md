# 01. Re-deriving a production cycle predictor

**Status. In progress.** Synthetic-data study complete.

**Data note.** The production model studied here ran in FemoraAI on real users' data. Every result in this investigation is computed on synthetic data with known ground truth. No FemoraAI user data is used or published.

## Question
What does the cycle predictor shipped in FemoraAI's CycleService (Java, Aug 2025) actually estimate, where does it fail, and what does a principled replacement buy?

## What was built
| File | Purpose |
|---|---|
| `femora_port.py` | Line-by-line port of the production estimator. Kalman filter (q = 0.5, r = 4), EWMA (α = 0.3, tuned), 3σ outlier gate, 50/50 blend, σ² = P_KF + EWMA variance, 80% interval, hit-rate feedback. Two feedback modes, "as written" and "as intended" (see finding 4). |
| `generator.py` | Synthetic cohort with known truth. Drifting level, level shifts (15% of users), missed logs that merge two cycles (5%), occasional long cycles (4%), a long-cycle subgroup (10%). All values are assumptions, not literature estimates. |
| `grid_filter.py` | Robust replacement. Exact Bayesian grid filter over level x, noise sd s and drift q jointly, with a contamination mixture (missed log = 2x, long cycle = ex-Gaussian). Calibrated integer-day predictive pmf, proper hazard, conditioning on elapsed days. |
| `run_experiments.py` | Three cohorts × 2,000 users × 24 cycles. Writes `results/summary.json` and `results/figdata.json`. |
| `test_investigation.py` | 7 tests. Hand-computed Kalman step, steady-state fixed point, absorbing gate, lost feedback update, robustness to a missed log, hazard algebra. |

Reproduce with `python run_experiments.py` (≈ 270 s on one laptop CPU core) and `python -m pytest -q`.

## Findings

**1. The Kalman filter is an EWMA after about five cycles.** For a random walk plus noise the prior variance satisfies M² − qM − qr = 0. At q = 0.5, r = 4 this gives K∞ = 0.2965 and P∞ = 1.186. In simulation the mean absolute gap between the Kalman mean and the α = 0.3 EWMA falls from 0.99 days after cycle 1 to 0.06 days by cycle 9 and below 0.01 days after cycle 17. The blend is two copies of almost the same smoother.

**2. The variance formula is wrong in principle but works on clean data.** The predictive variance for the next cycle under the filter's own model is P + q + r ≈ 5.69. The service uses P_KF + EWMA variance, and P_KF alone (≈ 1.19) omits r. *Correction to my earlier note.* The EWMA variance term largely stands in for the missing observation noise. On clean data production coverage of its nominal 80% interval is **0.822**, close to nominal. The problem appears under contamination (finding 3), not by construction.

**3. Under realistic contamination production is overconfident in the tails.**

| Cohort (2,000 users) | Model | MAE (d) | 80% coverage | Width (d) | CRPS | Log score |
|---|---|---|---|---|---|---|
| main | production (as written) | 3.95 | 0.749 | 7.2 | 3.297 | 4.170 |
| main | minimal fix (P + q + r, no gate) | 4.55 | 0.659 | 7.3 | 3.792 | 4.737 |
| main | robust grid Bayes | **3.84** | 0.848 | 10.2 | **3.085** | **2.729** |
| clean | production | 2.07 | 0.822 | 7.1 | 1.495 | 2.447 |
| clean | minimal fix | **2.03** | 0.846 | 7.3 | **1.475** | 2.430 |
| clean | robust grid Bayes | 2.05 | 0.918 | 9.8 | 1.565 | **2.425** |
| misspecified (t₃ noise, 2× contamination) | production | 5.54 | 0.713 | 6.8 | 4.931 | 5.742 |
| misspecified | robust grid Bayes | **5.41** | 0.792 | 9.8 | **4.509** | **2.908** |

Lower is better for MAE, CRPS and log score. The log-score gap (4.17 vs 2.73) comes from the Gaussian forecast putting almost no mass on merged or long cycles. Two honest caveats. The robust model **over-covers on clean data** (0.918). Part of that is its contamination prior, part is that equal-tailed intervals on integer days are conservative. Removing the gate without a robust likelihood (the "minimal fix") is **worse** than production under contamination, so the gate was doing real work.

**4. The 3σ gate is an absorbing state.** If the level shifts by more than three EWMA standard deviations, every later cycle is rejected, the state never moves, so the gate never reopens. In the main cohort **9.1%** of users with a level shift (1.65% of all users) had all of their final 8 cycles rejected. The robust filter has no gate and cannot lock out.

**5. The feedback loop does not persist (by code reading).** `AutoEvaluator` loads, tunes and saves its own copy of the model state. `updateOnNewPeriod` then saves the copy it loaded earlier, overwriting the tuned α and σ-scale. Simulating the loop as intended changes little (coverage 0.749 → 0.763, lockout among shifted users 9.1% → 6.8%). This is inferred from Spring Data's whole-document `save()`, not reproduced against a running service.

**6. The "hazard" endpoint returns a pmf, not a hazard, and ignores elapsed days.** On daily "does it start today?" forecasts over each user's final cycle, the proper hazard from the grid model has Brier score **0.0251** against **0.0273** for the production pmf (main cohort).

## Limitations
- Synthetic data only, so the size of every effect depends on the generator's assumptions. The direction of findings 1, 2, 4 and 5 does not, since they follow from the algebra or the code.
- The robust model's contamination rates are fixed, not learned. Learning them hierarchically across users (empirical Bayes) is the obvious next step and should fix the clean-data over-coverage.
- The symptom personalization layer was not studied.

## Next
1. Learn contamination rates and the noise prior by empirical Bayes on a training cohort.
2. Real-data validation on a licensed public dataset.
3. Write the investigation page with live figures from `results/figdata.json`.
