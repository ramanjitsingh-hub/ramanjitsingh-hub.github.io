# Investigation 03 · What can a startup learn from fifty users?

Founders decide on small numbers. This investigation measures what those numbers can support, using a real customer base as the truth and drawing startup-sized samples from it.

Status: **in progress, milestone 1 complete** (2026-11-16). Page: [ramanjitsingh-hub.github.io/03](https://ramanjitsingh-hub.github.io/03/).

## Data
Online Retail II (D. Chen, UCI Machine Learning Repository, 2019, doi:10.24432/C5CG6D, CC BY 4.0). It holds the transactions of a UK online retailer from 1 Dec 2009 to 9 Dec 2011, 1,067,371 lines in all.

Cleaning:
- Drop cancellations, rows without a customer id, and non-positive quantity or price.
- Assign each customer to the monthly cohort of their first purchase.

That leaves 5,878 customers. Cohorts used:
- Dec 2009 is dropped, because it is left-censored and so includes all existing customers.
- The last two months are dropped, because their following month is incomplete.
- The result is 4,704 new customers in 22 cohorts.

Metric: next-month retention, meaning bought again in the following calendar month. Its population value is **21.0%**.

`data/raw/` is not committed; download `online_retail_ii.zip` from UCI into it.

## Milestone 1 results
1. **Intervals.** 20,000 random samples at each n.
   - Wilson half-width is ±21.9 points at n = 10, ±10.9 at 50, ±7.9 at 100 and ±2.5 at 1,000.
   - Wald coverage is 89.5% at n = 10 and 91.6% at n = 50, against 95%.
   - Exact binomial coverage at n = 50 drops to 64% for Wald near the edges, and stays at 92% or above for Wilson.
   - At n ≥ 500 every method over-covers, because sampling is without replacement from 4,704.
2. **Cohorts differ, but less than they appear to.**
   - The χ² test of equal rates gives 55.4 on 21 df, p = 6 × 10⁻⁵.
   - The spread of cohort rates (sd 5.1 points) combines a true between-cohort sd of 3.9 with a sampling sd of 3.2, so about 59% of the variance between cohorts is real.
3. **Empirical Bayes shrinkage.** This uses a beta-binomial model with a moment-matched prior, on 2,000 subsamples of the 14 cohorts with at least 150 customers.

   | users per cohort | raw RMSE | pooled RMSE | EB RMSE | EB better |
   |---|---|---|---|---|
   | 20 | 8.6 pts | 5.1 pts | 5.0 pts | 99% of runs |
   | 50 | 5.2 pts | 4.7 pts | 4.0 pts | 80% |
   | 100 | 3.2 pts | 4.6 pts | 3.1 pts | 57% |

4. **Peeking.** These are A/A tests with customers in their real arrival order, up to 2,000 per arm.
   - A z-test checked once has a false-positive rate of 5.5%. Checked after every 10 users per arm (200 looks) it is 42.1%.
   - The rate rises with the number of looks: 8% at 2, 19% at 10, 32% at 50 and 41% at 200.
   - The mixture SPRT (τ = 2 points, threshold 1/α = 20) has a false-positive rate of 0.3%.
5. **Power.**
   - At a base rate of 21%, the minimum detectable effect (5% significance, 80% power) is 22.8 points at 50 per arm, 7.2 at 500 and 2.4 at 4,600.
   - Detecting +2, +5 or +10 points needs 6,587, 1,068 or 272 users per arm.
   - The always-valid test on planted lifts stops at a median of 1,280 users per arm for +5 points (76% power within 2,000) and 610 for +10 points (100%).

## Reproduce
```
python prepare.py && python run_milestone1.py && python export_site_data.py
```

## Next
Revenue per customer: heavy tails and the fragility of means at small n.
