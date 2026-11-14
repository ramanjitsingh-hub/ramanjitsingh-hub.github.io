"""Robust Bayesian replacement: an exact grid filter over (level x, noise sd s, drift q).

Model, per user
  x_k = x_{k-1} + w_k,  w_k ~ N(0, q)                         random-walk level
  z_k | x_k, s ~ (1 - pm - pl) N(x, s^2)                       ordinary cycle
               +  pm          N(2x, 2 s^2)                      missed log (two cycles merged)
               +  pl          ExGauss(x, s, tau)                occasional long cycle
Unknown x, s and q are carried jointly on a grid, so the noise level and drift are learned
per user by exact Bayesian updating instead of being global constants. Nothing is ever
discarded as an outlier, so there is no absorbing state.

The predictive distribution is integrated onto integer days, which gives calibrated
intervals, a proper hazard h(d) = p(d) / P(T >= d), and conditioning on elapsed days.
"""
from __future__ import annotations

import numpy as np
from scipy.special import erfc
from scipy.stats import norm

X = np.arange(14.0, 70.0 + 1e-9, 0.25)                 # level grid (days)
S = np.exp(np.linspace(np.log(0.7), np.log(7.0), 14))  # noise sd grid
Q = np.array([0.05, 0.3, 1.5, 8.0])                    # drift variance grid
DAYS = np.arange(1, 181)


def _exgauss_pdf(z, mu, s, tau):
    lam = 1.0 / tau
    arg = (mu + lam * s * s - z) / (np.sqrt(2) * s)
    return 0.5 * lam * np.exp(0.5 * lam * (2 * mu + lam * s * s - 2 * z)) * erfc(arg)


def _build_transitions():
    dx = X[1] - X[0]
    diff = X[:, None] - X[None, :]
    mats = []
    for q in Q:
        t = norm.pdf(diff, 0, np.sqrt(q)) * dx      # t[i, j] = p(x_i | x_j)
        mats.append(t / t.sum(axis=0, keepdims=True))  # reflect mass that would leave the grid
    return mats


def _build_obs_pmf(pm, pl, tau):
    """M[i, j, d] = p(observed length = DAYS[d] | x = X[i], s = S[j]). Independent of state."""
    lo, hi = DAYS - 0.5, DAYS + 0.5
    x = X[:, None, None]
    s = S[None, :, None]
    base = norm.cdf(hi, x, s) - norm.cdf(lo, x, s)
    miss = norm.cdf(hi, 2 * x, np.sqrt(2) * s) - norm.cdf(lo, 2 * x, np.sqrt(2) * s)
    lng = _exgauss_pdf(DAYS[None, None, :].astype(float), x, s, tau)
    return (1 - pm - pl) * base + pm * miss + pl * lng


_T = _build_transitions()
_M_CACHE: dict = {}


class GridFilter:
    def __init__(self, prior_mean: float, prior_sd: float = 5.0, pm: float = 0.05,
                 pl: float = 0.04, tau: float = 10.0, s_prior=(np.log(2.0), 0.6),
                 q_prior=(0.25, 0.35, 0.25, 0.15)):
        self.pm, self.pl, self.tau = pm, pl, tau
        key = (pm, pl, tau)
        if key not in _M_CACHE:
            _M_CACHE[key] = _build_obs_pmf(pm, pl, tau)
        self.M = _M_CACHE[key]
        px = norm.pdf(X, prior_mean, prior_sd)
        ps = norm.pdf(np.log(S), *s_prior)
        pq = np.asarray(q_prior, float)
        post = px[:, None, None] * ps[None, :, None] * pq[None, None, :]
        self.post = post / post.sum()                 # shape (|X|, |S|, |Q|)
        self.pred = self._propagate(self.post)

    @staticmethod
    def _propagate(post):
        out = np.empty_like(post)
        for iq, t in enumerate(_T):
            out[:, :, iq] = t @ post[:, :, iq]
        return out / out.sum()

    def _lik(self, z):
        d = int(round(z))
        if 1 <= d <= DAYS[-1]:
            return self.M[:, :, d - 1][:, :, None]
        x = X[:, None]; s = S[None, :]
        return ((1 - self.pm - self.pl) * norm.pdf(z, x, s) + self.pm * norm.pdf(z, 2 * x, np.sqrt(2) * s)
                + self.pl * _exgauss_pdf(z, x, s, self.tau))[:, :, None]

    def predictive_pmf(self) -> np.ndarray:
        """p(next observed length = d) for d in DAYS (integer-day likelihood)."""
        w = self.pred.sum(axis=2)
        pmf = np.tensordot(w, self.M, axes=([0, 1], [0, 1]))
        return pmf / pmf.sum()

    def update(self, z: float):
        post = self.pred * self._lik(z)
        tot = post.sum()
        self.post = post / tot if tot > 0 else self.pred
        self.pred = self._propagate(self.post)

    def level_marginal(self):
        return self.post.sum(axis=(1, 2))

    def noise_marginal(self):
        return self.post.sum(axis=(0, 2))


def interval_from_pmf(pmf: np.ndarray, level: float = 0.80):
    c = np.cumsum(pmf)
    a = (1 - level) / 2
    lo = DAYS[np.searchsorted(c, a)]
    hi = DAYS[np.searchsorted(c, 1 - a)]
    med = DAYS[np.searchsorted(c, 0.5)]
    return int(lo), int(hi), int(med)


def conditional_on_elapsed(pmf: np.ndarray, elapsed: int) -> np.ndarray:
    """p(T = d | T > elapsed)."""
    out = pmf.copy()
    out[DAYS <= elapsed] = 0.0
    tot = out.sum()
    return out / tot if tot > 0 else out


def hazard(pmf: np.ndarray) -> np.ndarray:
    surv = 1.0 - np.concatenate([[0.0], np.cumsum(pmf)[:-1]])   # P(T >= d)
    return np.where(surv > 1e-12, pmf / np.maximum(surv, 1e-12), 1.0)
