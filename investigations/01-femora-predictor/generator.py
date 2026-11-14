"""Synthetic cohort with known ground truth.

Every distribution below is an ASSUMPTION chosen to exercise the failure modes of interest
(drift, level shifts, contamination by missed logs and long cycles). None of the parameter
values is claimed to be a literature estimate. Real-data validation is a separate step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CohortSpec:
    n_users: int = 2000
    n_cycles: int = 24
    mu_mean: float = 29.0
    mu_sd: float = 3.0
    long_frac: float = 0.10        # subgroup with long mean cycles
    long_mu_mean: float = 38.0
    long_mu_sd: float = 4.0
    q_median: float = 0.3          # latent drift variance per cycle (days^2)
    q_logsd: float = 0.7
    sigma_median: float = 2.0      # within-user observation sd (days)
    sigma_logsd: float = 0.4
    shift_prob: float = 0.15       # user experiences one level shift
    shift_sd: float = 6.0
    p_missed: float = 0.05         # a start is not logged -> two cycles merge
    p_long: float = 0.04           # an occasional long cycle
    long_extra_mean: float = 10.0
    self_report_sd: float = 2.0
    noise: str = "normal"          # or "t3" for a misspecification check


def simulate(spec: CohortSpec, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n, k = spec.n_users, spec.n_cycles
    is_long = rng.random(n) < spec.long_frac
    mu0 = np.where(is_long,
                   rng.normal(spec.long_mu_mean, spec.long_mu_sd, n),
                   rng.normal(spec.mu_mean, spec.mu_sd, n))
    mu0 = np.clip(mu0, 21, 50)
    q = spec.q_median * np.exp(spec.q_logsd * rng.standard_normal(n))
    sig = spec.sigma_median * np.exp(spec.sigma_logsd * rng.standard_normal(n))
    # +k extra cycles so missed logs can merge and still leave k observed cycles
    kk = 2 * k
    latent = mu0[:, None] + np.cumsum(np.sqrt(q)[:, None] * rng.standard_normal((n, kk)), axis=1)
    has_shift = rng.random(n) < spec.shift_prob
    shift_at = rng.integers(3, k, n)
    shift = rng.normal(0, spec.shift_sd, n) * has_shift
    latent += shift[:, None] * (np.arange(kk)[None, :] >= shift_at[:, None])
    latent = np.clip(latent, 18, 55)
    if spec.noise == "t3":
        eps = rng.standard_t(3, (n, kk)) / np.sqrt(3.0)
    else:
        eps = rng.standard_normal((n, kk))
    true_len = latent + sig[:, None] * eps
    long_ev = rng.random((n, kk)) < spec.p_long
    true_len += long_ev * rng.exponential(spec.long_extra_mean, (n, kk))
    true_len = np.maximum(np.rint(true_len), 15).astype(int)

    observed = np.zeros((n, k), dtype=int)
    latent_obs = np.zeros((n, k))
    missed = np.zeros((n, k), dtype=bool)
    for i in range(n):
        j = 0
        for c in range(k):
            v, lat = true_len[i, j], latent[i, j]
            j += 1
            if rng.random() < spec.p_missed:
                v += true_len[i, j]
                j += 1
                missed[i, c] = True
            observed[i, c] = v
            latent_obs[i, c] = lat
    self_report = np.rint(mu0 + rng.normal(0, spec.self_report_sd, n)).astype(int)
    return {"observed": observed, "latent": latent_obs, "missed": missed, "mu0": mu0,
            "q": q, "sigma": sig, "is_long": is_long, "has_shift": has_shift,
            "shift_at": shift_at, "shift": shift, "self_report": self_report}
