"""Investigation 01, milestone 2: prediction ranges with a guarantee.

The production service promised an 80% range and delivered 0.749 under
contamination. Split conformal prediction turns any forecaster into one whose
range covers a new user's next cycle with probability at least 80%, by a
finite-sample argument that needs only exchangeability across users.

Calibration is done separately for each cycle index k, using one forecast per
user, so the calibration scores and a new user's score at cycle k are i.i.d.
across users. Within-user dependence over time never enters the argument.

Writes results/conformal.json.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

import grid_filter as gf
from femora_port import FemoraPredictor
from generator import CohortSpec, simulate

OUT = Path(__file__).parent / "results"
ALPHA = 0.20
R4 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 4)


def record(data):
    """Run production and the grid filter over every user; keep what conformal needs."""
    obs = data["observed"]
    n, k = obs.shape
    rec = {key: np.zeros((n, k)) for key in ["y", "mu", "sd", "lo", "hi", "glo", "ghi", "gmed", "rej_prev"]}
    for i in range(n):
        init = int(data["self_report"][i])
        prod = FemoraPredictor(init, feedback_mode="as_written")
        gb = gf.GridFilter(prior_mean=init, prior_sd=5.0)
        prev_rej = 0.0
        for c in range(k):
            y = int(obs[i, c])
            pr = prod.predict()
            g_lo, g_hi, g_med = gf.interval_from_pmf(gb.predictive_pmf(), 1 - ALPHA)
            for key, v in [("y", y), ("mu", prod.s.mu), ("sd", pr["sigma"]), ("lo", pr["lower"]), ("hi", pr["upper"]),
                           ("glo", g_lo), ("ghi", g_hi), ("gmed", g_med), ("rej_prev", prev_rej)]:
                rec[key][i, c] = v
            prev_rej = 0.0 if prod.observe(y) else 1.0
            gb.update(y)
    rec["has_shift"] = data["has_shift"].astype(bool)
    rec["is_long"] = data["is_long"].astype(bool)
    rec["missed"] = data["missed"].astype(bool)
    rec["shift_at"] = data["shift_at"]
    return rec


def conf_quantile(scores, alpha):
    """The ceil((n+1)(1-alpha))-th smallest score: the split-conformal threshold."""
    n = len(scores)
    j = math.ceil((n + 1) * (1 - alpha))
    if j > n:
        return np.inf
    return np.partition(scores, j - 1)[j - 1]


def scores(rec, method):
    y = rec["y"]
    if method == "prod":
        return np.abs(y - rec["mu"]) / rec["sd"]
    if method == "grid":
        return np.maximum(rec["glo"] - y, y - rec["ghi"])
    raise ValueError(method)


def width(rec, method, q):
    """Number of whole days in the conformal range for each forecast."""
    if method == "prod":
        lo = np.ceil(rec["mu"] - q * rec["sd"])
        hi = np.floor(rec["mu"] + q * rec["sd"])
    else:
        lo, hi = rec["glo"] - q, rec["ghi"] + q
    return np.maximum(hi - lo + 1, 0)


def evaluate(cal, test, method, groups=None):
    """Per-cycle split conformal. Returns hit and width matrices for the test users.
    With groups (a boolean matrix of an observable state), calibrate within each group
    separately (Mondrian conformal), which gives coverage within each group."""
    Sc, St = scores(cal, method), scores(test, method)
    n, k = St.shape
    hit = np.zeros((n, k), bool)
    wid = np.zeros((n, k))
    for c in range(k):
        if groups is None:
            q = conf_quantile(Sc[:, c], ALPHA)
            qv = np.full(n, q)
        else:
            gc, gt = groups[0][:, c], groups[1][:, c]
            qv = np.zeros(n)
            for g in (False, True):
                sc = Sc[gc == g, c]
                qv[gt == g] = conf_quantile(sc, ALPHA) if len(sc) else np.inf
        hit[:, c] = St[:, c] <= qv
        wid[:, c] = width({key: v[:, c] for key, v in test.items() if np.ndim(v) == 2}, method, qv)
    return hit, wid


def subset(rec, idx):
    return {key: v[idx] for key, v in rec.items()}


def main():
    t0 = time.time()
    main_data = simulate(CohortSpec(n_users=4000), seed=5001)
    R = record(main_data)
    print(f"main cohort recorded {time.time()-t0:.0f}s")
    mis_data = simulate(CohortSpec(n_users=2000, noise="t3", p_missed=0.10, p_long=0.08), seed=5002)
    M = record(mis_data)
    print(f"misspecified cohort recorded {time.time()-t0:.0f}s")
    n = len(R["y"])
    k = R["y"].shape[1]
    rng = np.random.default_rng(0)

    # subgroup masks on the test side (cycle level)
    def masks(rec):
        after_shift = rec["has_shift"][:, None] & (np.arange(k)[None, :] >= rec["shift_at"][:, None])
        return {
            "all": np.ones_like(rec["y"], bool),
            "after a level shift": after_shift,
            "long-cycle users": np.repeat(rec["is_long"][:, None], k, 1),
            "missed log this cycle": rec["missed"],
            "gate rejected last cycle": rec["rej_prev"] == 1,
            "gate accepted last cycle": rec["rej_prev"] == 0,
        }

    reps = 200
    methods = ["production", "conformal production", "conformal production, by gate state", "grid Bayes", "conformal grid Bayes"]
    cov = {m: [] for m in methods}
    wid = {m: [] for m in methods}
    sub = {m: {g: [] for g in masks(R)} for m in methods}
    bycycle = {m: [] for m in methods}
    for r in range(reps):
        perm = rng.permutation(n)
        cal, test = subset(R, perm[: n // 2]), subset(R, perm[n // 2:])
        mk = masks(test)
        out = {}
        out["production"] = ((test["y"] >= test["lo"]) & (test["y"] <= test["hi"]), test["hi"] - test["lo"] + 1)
        out["grid Bayes"] = ((test["y"] >= test["glo"]) & (test["y"] <= test["ghi"]), test["ghi"] - test["glo"] + 1)
        out["conformal production"] = evaluate(cal, test, "prod")
        out["conformal production, by gate state"] = evaluate(cal, test, "prod", groups=(cal["rej_prev"] == 1, test["rej_prev"] == 1))
        out["conformal grid Bayes"] = evaluate(cal, test, "grid")
        for m, (h, w) in out.items():
            cov[m].append(h.mean())
            wid[m].append(w.mean())
            bycycle[m].append(h.mean(0))
            for g, msk in mk.items():
                sub[m][g].append(h[msk].mean() if msk.any() else np.nan)
    res = {"alpha": ALPHA, "reps": reps, "n_users": n, "n_cal": n // 2, "cycles": k, "methods": {}}
    for m in methods:
        c = np.array(cov[m])
        res["methods"][m] = {
            "coverage_mean": R4(c.mean()), "coverage_q05": R4(np.quantile(c, 0.05)), "coverage_q95": R4(np.quantile(c, 0.95)),
            "coverage_min": R4(c.min()), "width_mean": R4(np.mean(wid[m])),
            "coverage_by_cycle": [R4(v) for v in np.mean(bycycle[m], 0)],
            "coverage_by_cycle_min": [R4(v) for v in np.min(bycycle[m], 0)],
            "subgroups": {g: R4(np.nanmean(v)) for g, v in sub[m].items()},
            "hist": np.histogram(c, bins=24, range=(0.70, 0.94))[0].tolist(),
        }
        print(f"{m:40s} cov {c.mean():.4f} [{np.quantile(c,.05):.4f}, {np.quantile(c,.95):.4f}] width {np.mean(wid[m]):.2f}  ",
              {g: round(float(np.nanmean(v)), 3) for g, v in sub[m].items()})
    mk0 = masks(R)
    res["subgroup_share"] = {g: R4(v.mean()) for g, v in mk0.items()}

    # theory check: for continuous scores coverage lies in [1-a, 1-a+1/(n+1)]
    res["theory_upper"] = R4(1 - ALPHA + 1 / (n // 2 + 1))

    # distribution shift: calibrate on the main cohort, deploy on the misspecified one
    shift = {}
    for m, meth in [("conformal production", "prod"), ("conformal grid Bayes", "grid")]:
        h, w = evaluate(R, M, meth)
        shift[m] = {"coverage": R4(h.mean()), "width": R4(w.mean())}
    shift["production"] = {"coverage": R4(((M["y"] >= M["lo"]) & (M["y"] <= M["hi"])).mean()), "width": R4((M["hi"] - M["lo"] + 1).mean())}
    shift["grid Bayes"] = {"coverage": R4(((M["y"] >= M["glo"]) & (M["y"] <= M["ghi"])).mean()), "width": R4((M["ghi"] - M["glo"] + 1).mean())}
    # recalibrating on half of the new cohort restores the guarantee
    perm = rng.permutation(len(M["y"]))
    h, w = evaluate(subset(M, perm[:1000]), subset(M, perm[1000:]), "prod")
    shift["conformal production, recalibrated on new cohort"] = {"coverage": R4(h.mean()), "width": R4(w.mean())}
    res["shift"] = shift
    print(json.dumps(shift))
    OUT.mkdir(exist_ok=True)
    (OUT / "conformal.json").write_text(json.dumps(res, indent=1))
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
