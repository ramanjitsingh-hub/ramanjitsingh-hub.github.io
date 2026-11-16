"""Core routines for Investigation 02: routing prompts between language models.

Notation. n prompts, K models. S[i,k] in [0,1] is the judged score of model k on
prompt i, C[i,k] its cost in USD. A router sees an estimate mu_hat[i,k] of the
expected score and an estimate c_hat[i,k] of the cost, and for a price of
quality lambda picks argmax_k mu_hat - lambda * c_hat (the Lagrangian rule).
"""
import numpy as np


# ---------------------------------------------------------------- frontiers
def lagrangian_points(mu_hat, c_hat, S, C, lambdas):
    """Mean realised (cost, score) of the Lagrangian router for each lambda."""
    out = []
    idx = np.arange(len(S))
    for lam in lambdas:
        k = np.argmax(mu_hat - lam * c_hat, axis=1)
        out.append((C[idx, k].mean(), S[idx, k].mean()))
    return np.array(out)


def upper_hull(points):
    """Upper concave hull of (cost, score) points, increasing in cost.

    Randomising between two policies realises every point on the segment
    joining them, so the hull is the achievable frontier (Section 2).
    Points dominated in score by a cheaper point are dropped first.
    """
    p = np.unique(np.asarray(points), axis=0)
    p = p[np.argsort(p[:, 0])]
    keep = []
    best = -np.inf
    for q in p:
        if q[1] > best + 1e-15:
            keep.append(q)
            best = q[1]
    hull = []
    for q in keep:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (q[1] - y1) - (y2 - y1) * (q[0] - x1) >= 0:
                hull.pop()
            else:
                break
        hull.append(q)
    return np.array(hull)


def hull_score_at(hull, cost):
    """Best score reachable at a given mean cost on a hull (flat after its end)."""
    x, y = hull[:, 0], hull[:, 1]
    if cost <= x[0]:
        return y[0] if cost >= x[0] - 1e-18 else np.nan
    return float(np.interp(cost, x, y))


def hull_cost_for(hull, score):
    """Cheapest mean cost on a hull reaching a target score (nan if never)."""
    x, y = hull[:, 0], hull[:, 1]
    if score > y.max() + 1e-12:
        return np.nan
    if score <= y[0]:
        return x[0]
    j = min(int(np.searchsorted(y, score)), len(y) - 1)
    x0, x1, y0, y1 = x[j - 1], x[j], y[j - 1], y[j]
    return float(x0 + (score - y0) * (x1 - x0) / (y1 - y0))


def lambda_grid(c_scale, n=160):
    """Log-spaced prices of quality, in score units per USD, plus zero."""
    return np.concatenate([[0.0], np.logspace(-2, 7, n) / max(c_scale, 1e-12)])


def router_hull(mu_hat, c_hat, S, C, lambdas):
    pts = lagrangian_points(mu_hat, c_hat, S, C, lambdas)
    return upper_hull(pts), pts


def single_model_hull(S, C):
    return upper_hull(np.c_[C.mean(0), S.mean(0)])


# ---------------------------------------------------------------- oracle
def oracle_points(S, C, lambdas):
    """Hindsight router that sees the realised scores: argmax S - lambda C."""
    return lagrangian_points(S, C, S, C, lambdas)


# ---------------------------------------------------------------- guarantees
def hoeffding_p(x, target):
    """p-value for H0: mean < target, for x in [0,1] (one-sided Hoeffding)."""
    gap = x.mean() - target
    if gap <= 0:
        return 1.0
    return float(np.exp(-2 * len(x) * gap * gap))


def betting_p(x, target, delta):
    """p-value for H0: mean <= target via a nonnegative betting martingale.

    Capital K_t = prod (1 + eta_i (x_i - target)), eta_i predictable and in
    [0, 0.5/target]. Under H0 K_t is a nonnegative supermartingale, so by
    Ville's inequality P(sup K_t >= 1/delta) <= delta and 1/sup K_t is a valid
    p-value (Waudby-Smith and Ramdas, 2024, predictable plug-in bets).
    """
    n = len(x)
    cap, best = 1.0, 1.0
    m, v = 0.5, 0.25  # running mean and variance with a weak prior
    cmax = 0.5 / max(target, 1e-9)
    for t in range(n):
        eta = np.sqrt(2 * np.log(1 / delta) / ((t + 1) * np.log(t + 2) * max(v, 1e-4)))
        eta = min(max(eta, 0.0), cmax) if m > target else 0.0
        cap *= 1 + eta * (x[t] - target)
        best = max(best, cap)
        # update running estimates
        m = (m * (t + 1) + x[t]) / (t + 2)
        v = (v * (t + 1) + (x[t] - m) ** 2) / (t + 2)
    return float(min(1.0, 1.0 / best))


def learn_then_test(score_by_lambda, target, delta, method="betting"):
    """Fixed-sequence testing over lambdas ordered from expensive to cheap.

    score_by_lambda[j] holds the calibration scores of the router run at the
    j-th lambda (increasing lambda, decreasing cost). Walk forward while H0
    (mean score < target) is rejected at level delta; return the last rejected
    index or -1. Family-wise error <= delta (Angelopoulos et al., 2021).
    """
    chosen = -1
    for j, x in enumerate(score_by_lambda):
        p = betting_p(x, target, delta) if method == "betting" else hoeffding_p(x, target)
        if p <= delta:
            chosen = j
        else:
            break
    return chosen


# ---------------------------------------------------------------- statistics
def bootstrap(fn, n, B=1000, seed=0):
    """Percentile bootstrap over prompts: fn(index array) -> scalar or array."""
    rng = np.random.default_rng(seed)
    vals = [fn(rng.integers(0, n, n)) for _ in range(B)]
    return np.array(vals)
