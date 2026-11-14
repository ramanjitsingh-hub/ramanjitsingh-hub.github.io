"""Investigation 01. Runs every experiment and writes results/summary.json and results/figdata.json.

    python run_experiments.py            # full run (2000 users per cohort)
    python run_experiments.py --quick    # 300 users, for iteration
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

import grid_filter as gf
from femora_port import FemoraPredictor, Params, steady_state_gain
from generator import CohortSpec, simulate

OUT = Path(__file__).parent / "results"
LEVEL = 0.80
Z80 = 1.2815515655446004


# ---------------------------------------------------------------- scoring helpers
def crps_pmf(pmf: np.ndarray, y: int) -> float:
    c = np.cumsum(pmf)
    step = (gf.DAYS >= y).astype(float)
    return float(np.sum((c - step) ** 2))


def logscore(pmf: np.ndarray, y: int) -> float:
    p = pmf[y - 1] if 1 <= y <= len(pmf) else 0.0
    return float(-math.log(max(p, 1e-12)))


def gauss_pmf(mu: float, sigma: float) -> np.ndarray:
    w = np.exp(-0.5 * ((gf.DAYS - mu) / sigma) ** 2)
    return w / w.sum()


def pit(pmf: np.ndarray, y: int, rng) -> float:
    """Randomised PIT for discrete forecasts."""
    c = np.concatenate([[0.0], np.cumsum(pmf)])
    y = min(max(y, 1), len(pmf))
    lo, hi = c[y - 1], c[y]
    return float(lo + rng.random() * (hi - lo))


# ---------------------------------------------------------------- forecasters
class MinimalFix:
    """Same scalar Kalman filter, but correct predictive variance P + q + r, no gate, no blend."""

    def __init__(self, init, q=0.5, r=4.0):
        self.x, self.p, self.q, self.r = float(init), 9.0, q, r

    def forecast(self):
        mu, var = self.x, self.p + self.q + self.r
        return gauss_pmf(mu, math.sqrt(var)), mu, math.sqrt(var)

    def observe(self, z):
        z = min(60, max(15, z))
        pp = self.p + self.q
        k = pp / (pp + self.r)
        self.x += k * (z - self.x)
        self.p = (1 - k) * pp


def run_cohort(data: dict, mode: str, seed: int, keep_examples=()):
    rng = np.random.default_rng(seed + 99)
    obs = data["observed"]
    n, k = obs.shape
    names = ["production", "minimal_fix", "grid_bayes"]
    rec = {m: {"abs_err": np.zeros((n, k)), "hit": np.zeros((n, k), bool), "crps": np.zeros((n, k)),
               "log": np.zeros((n, k)), "pit": np.zeros((n, k)), "width": np.zeros((n, k))} for m in names}
    locked = np.zeros(n, bool)
    examples = {}
    brier = {"production": [], "grid_bayes": []}
    for i in range(n):
        init = int(data["self_report"][i])
        prod = FemoraPredictor(init, feedback_mode=mode)
        mfix = MinimalFix(init)
        gb = gf.GridFilter(prior_mean=init, prior_sd=5.0)
        accepted = []
        ex = {"obs": [], "latent": [], "prod_mu": [], "prod_lo": [], "prod_hi": [], "gb_med": [],
              "gb_lo": [], "gb_hi": [], "accepted": [], "gb_level": [], "missed": []} if i in keep_examples else None
        for c in range(k):
            y = int(obs[i, c])
            # --- forecasts issued before cycle c is observed
            pr = prod.predict()
            p_pmf = gauss_pmf(prod.s.mu, pr["sigma"])
            m_pmf, m_mu, m_sd = mfix.forecast()
            g_pmf = gb.predictive_pmf()
            g_lo, g_hi, g_med = gf.interval_from_pmf(g_pmf, LEVEL)
            m_lo, m_hi = round(m_mu - Z80 * m_sd), round(m_mu + Z80 * m_sd)
            for name, pmf, pt, lo, hi in (
                ("production", p_pmf, pr["expected"], pr["lower"], pr["upper"]),
                ("minimal_fix", m_pmf, round(m_mu), m_lo, m_hi),
                ("grid_bayes", g_pmf, g_med, g_lo, g_hi),
            ):
                r = rec[name]
                r["abs_err"][i, c] = abs(y - pt)
                r["hit"][i, c] = lo <= y <= hi
                r["crps"][i, c] = crps_pmf(pmf, y)
                r["log"][i, c] = logscore(pmf, y)
                r["pit"][i, c] = pit(pmf, y, rng)
                r["width"][i, c] = hi - lo + 1
            # daily "does it start today?" forecasts over the final cycle
            if c == k - 1 and y <= 90:
                ph = gf.hazard(g_pmf)
                for t in range(1, y + 1):
                    yt = 1.0 if t == y else 0.0
                    brier["grid_bayes"].append((ph[t - 1] - yt) ** 2)
                    brier["production"].append((p_pmf[t - 1] - yt) ** 2)
            if ex is not None:
                ex["obs"].append(y); ex["latent"].append(round(float(data["latent"][i, c]), 2))
                ex["prod_mu"].append(round(prod.s.mu, 2)); ex["prod_lo"].append(pr["lower"]); ex["prod_hi"].append(pr["upper"])
                ex["gb_med"].append(g_med); ex["gb_lo"].append(g_lo); ex["gb_hi"].append(g_hi)
                ex["missed"].append(bool(data["missed"][i, c]))
            # --- observe
            acc = prod.observe(y)
            accepted.append(acc)
            mfix.observe(y)
            gb.update(y)
            if ex is not None:
                ex["accepted"].append(acc)
                lm = gb.level_marginal()
                ex["gb_level"].append([round(float(v), 5) for v in lm[::2]])
        locked[i] = not any(accepted[-8:])
        if ex is not None:
            ex["meta"] = {"self_report": init, "mu0": round(float(data["mu0"][i]), 2),
                          "has_shift": bool(data["has_shift"][i]), "shift": round(float(data["shift"][i]), 2),
                          "shift_at": int(data["shift_at"][i]), "sigma": round(float(data["sigma"][i]), 2),
                          "q": round(float(data["q"][i]), 3), "level_grid": [float(v) for v in gf.X[::2]]}
            examples[int(i)] = ex
    return rec, locked, examples, {k2: float(np.mean(v)) for k2, v in brier.items()}


def summarise(rec, locked, data, brier):
    out = {}
    for name, r in rec.items():
        late = slice(6, None)
        out[name] = {
            "mae_days": round(float(r["abs_err"].mean()), 3),
            "mae_days_cycles_7plus": round(float(r["abs_err"][:, late].mean()), 3),
            "coverage_80": round(float(r["hit"].mean()), 4),
            "coverage_80_cycles_7plus": round(float(r["hit"][:, late].mean()), 4),
            "mean_interval_width_days": round(float(r["width"].mean()), 2),
            "crps": round(float(r["crps"].mean()), 4),
            "log_score": round(float(r["log"].mean()), 4),
            "coverage_by_cycle": [round(float(v), 4) for v in r["hit"].mean(axis=0)],
            "mae_by_cycle": [round(float(v), 3) for v in r["abs_err"].mean(axis=0)],
            "pit_hist": np.histogram(r["pit"].ravel(), bins=20, range=(0, 1))[0].tolist(),
        }
    out["lockout"] = {
        "fraction_all_users": round(float(locked.mean()), 4),
        "fraction_with_level_shift": round(float(locked[data["has_shift"]].mean()), 4) if data["has_shift"].any() else None,
        "fraction_long_cycle_group": round(float(locked[data["is_long"]].mean()), 4) if data["is_long"].any() else None,
        "definition": "production gate rejected all of the final 8 observed cycles",
    }
    out["daily_brier_final_cycle"] = brier
    return out


def kf_vs_ewma(seed: int, n=500, k=24):
    """With feedback off and no gating events, how far apart are the two estimators?"""
    d = simulate(CohortSpec(n_users=n, n_cycles=k, shift_prob=0, p_missed=0, p_long=0), seed)
    diffs = np.zeros((n, k))
    for i in range(n):
        p = FemoraPredictor(int(d["self_report"][i]), feedback_mode="off", params=Params(z_cutoff=1e9))
        for c in range(k):
            p.observe(int(d["observed"][i, c]))
            diffs[i, c] = abs(p.s.mu_kf - p.s.ewma_mean)
    gains, P = [], 9.0
    for _ in range(k):
        pp = P + 0.5
        g = pp / (pp + 4.0)
        P = (1 - g) * pp
        gains.append(round(g, 5))
    kss, pss = steady_state_gain(0.5, 4.0)
    return {"gain_by_cycle": gains, "steady_state_gain": round(kss, 5), "steady_state_P": round(pss, 5),
            "predictive_var_correct": round(pss + 0.5 + 4.0, 4),
            "mean_abs_diff_kf_vs_ewma_by_cycle": [round(float(v), 4) for v in diffs.mean(axis=0)]}


def gain_surface():
    qs = np.exp(np.linspace(np.log(0.05), np.log(10), 40))
    rs = np.exp(np.linspace(np.log(0.5), np.log(25), 40))
    return {"q": qs.round(4).tolist(), "r": rs.round(4).tolist(),
            "K": [[round(steady_state_gain(q, r)[0], 4) for q in qs] for r in rs]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    n = 300 if a.quick else 2000
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    cohorts = {
        "main": CohortSpec(n_users=n),
        "clean": CohortSpec(n_users=n, shift_prob=0, p_missed=0, p_long=0, long_frac=0),
        "misspecified": CohortSpec(n_users=n, noise="t3", p_missed=0.10, p_long=0.08),
    }
    summary = {"meta": {"python": platform.python_version(), "numpy": np.__version__,
                        "users_per_cohort": n, "cycles_per_user": 24, "interval_level": LEVEL}}
    try:
        summary["meta"]["git_commit"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                                                cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        summary["meta"]["git_commit"] = None
    figdata = {}
    for ci, (cname, spec) in enumerate(cohorts.items()):
        data = simulate(spec, seed=1000 + ci)
        keep = ()
        if cname == "main":
            # examples: first user with a level shift >= 8 days, a long-cycle user, a regular user
            sh = np.where(data["has_shift"] & (np.abs(data["shift"]) >= 8) & (data["shift_at"] < 12))[0]
            lg = np.where(data["is_long"] & ~data["has_shift"])[0]
            rg = np.where(~data["is_long"] & ~data["has_shift"])[0]
            keep = tuple(int(x[0]) for x in (sh, lg, rg) if len(x))
        for mode in (["as_written", "as_intended"] if cname == "main" else ["as_written"]):
            rec, locked, ex, brier = run_cohort(data, mode, seed=ci, keep_examples=keep if mode == "as_written" else ())
            summary[f"{cname}/{mode}"] = summarise(rec, locked, data, brier)
            if ex:
                figdata["examples"] = ex
            print(f"{cname}/{mode} done  t={time.time() - t0:.0f}s")
    figdata["kf_vs_ewma"] = kf_vs_ewma(seed=7)
    figdata["gain_surface"] = gain_surface()
    summary["kf_vs_ewma"] = {k: v for k, v in figdata["kf_vs_ewma"].items() if k != "gain_by_cycle"}
    summary["meta"]["runtime_s"] = round(time.time() - t0, 1)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    (OUT / "figdata.json").write_text(json.dumps(figdata))
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items()
                     if not isinstance(vv, list)}) for k, v in summary.items()}, indent=1))


if __name__ == "__main__":
    main()
