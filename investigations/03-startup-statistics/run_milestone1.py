"""Investigation 03, milestone 1: what can a startup learn from fifty users?

The full customer base of a real retailer is treated as the truth. Startup-
sized samples are drawn from it, so every interval, test and estimate can be
checked against a known answer.

Metric: next-month retention, the share of a month's new customers who buy
again in the following calendar month (the retail analogue of "still active
after four weeks").

A. Intervals at small n: coverage and width of Wald, Wilson, Agresti-Coull,
   Clopper-Pearson and Jeffreys intervals, n = 10 to 1,000.
B. Cohorts: how much of the month-to-month movement in retention is real,
   and empirical-Bayes shrinkage of small cohorts toward the pooled rate.
C. Peeking: A/A tests in true arrival order, checked after every 10 users with
   a fixed-n z-test, against an always-valid mixture sequential test.
D. Power: users needed per arm to detect a given lift, and the sequential
   test's stopping time on planted lifts.

Writes results/milestone1.json.
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import beta, norm

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
OUT = HERE / "results"
R4 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 5)
Z = norm.ppf(0.975)


def intervals(x, n):
    """Five 95% intervals for a binomial proportion, vectorised over x."""
    p = x / n
    out = {}
    h = Z * np.sqrt(p * (1 - p) / n)
    out["Wald"] = (p - h, p + h)
    c = p + Z ** 2 / (2 * n)
    d = Z * np.sqrt(p * (1 - p) / n + Z ** 2 / (4 * n * n))
    out["Wilson"] = ((c - d) / (1 + Z ** 2 / n), (c + d) / (1 + Z ** 2 / n))
    nt = n + Z ** 2
    pt = (x + Z ** 2 / 2) / nt
    ht = Z * np.sqrt(pt * (1 - pt) / nt)
    out["Agresti-Coull"] = (pt - ht, pt + ht)
    lo = np.where(x > 0, beta.ppf(0.025, x, n - x + 1), 0.0)
    hi = np.where(x < n, beta.ppf(0.975, x + 1, n - x), 1.0)
    out["Clopper-Pearson"] = (lo, hi)
    lo = np.where(x > 0, beta.ppf(0.025, x + 0.5, n - x + 0.5), 0.0)
    hi = np.where(x < n, beta.ppf(0.975, x + 0.5, n - x + 0.5), 1.0)
    out["Jeffreys"] = (lo, hi)
    return out


def msprt_stat(n, dbar, var, tau2):
    """Mixture SPRT likelihood ratio for a mean difference (Johari et al., 2017).
    n pairs, mean pairwise difference dbar, per-pair variance var, N(0, tau2) mixture."""
    return np.sqrt(var / (var + n * tau2)) * np.exp(n * n * tau2 * dbar * dbar / (2 * var * (var + n * tau2)))


def main():
    t0 = time.time()
    d = np.load(RAW / "activity.npz")
    first, active = d["first"], d["active"]
    M = active.shape[1]
    # the last month (Dec 2011) is only 9 days long; cohorts need a full next month
    # Dec 2009 is the first month of data, so its 'new' customers include every existing
    # customer; it is excluded. Cohorts need a full following month, so the last two go too.
    usable = (first >= 1) & (first <= M - 3)
    ret = np.zeros(len(first), bool)
    ret[usable] = active[np.flatnonzero(usable), first[usable] + 1]
    pop = np.flatnonzero(usable)
    y = ret[pop].astype(float)
    p_true = y.mean()
    res = {"n_customers": int(len(first)), "n_population": int(len(pop)), "months": int(M), "p_true": R4(p_true)}
    print(f"population {len(pop)} customers, next-month retention {p_true:.4f}")
    rng = np.random.default_rng(0)

    # ------------------------------------------------ A. intervals
    A = {}
    reps = 20000
    for n in [10, 20, 50, 100, 200, 500, 1000]:
        idx = np.array([rng.choice(len(pop), n, replace=False) for _ in range(reps)])
        x = y[idx].sum(1)
        iv = intervals(x, n)
        A[n] = {}
        for k, (lo, hi) in iv.items():
            A[n][k] = {"coverage": R4(np.mean((lo <= p_true) & (p_true <= hi))), "width": R4(np.mean(hi - lo)),
                       "width_pts_half": R4(50 * np.mean(hi - lo))}
        A[n]["sd_estimate"] = R4(np.std(x / n))
        A[n]["example"] = {"q10": R4(np.quantile(x / n, 0.1)), "q90": R4(np.quantile(x / n, 0.9))}
    res["intervals"] = A
    # exact coverage curve of Wald vs Wilson at n = 50 over p (binomial, not resampled)
    n = 50
    ps = np.linspace(0.02, 0.98, 97)
    xs = np.arange(n + 1)
    iv = intervals(xs.astype(float), n)
    from scipy.stats import binom
    curve = {k: [] for k in iv}
    for p in ps:
        w = binom.pmf(xs, n, p)
        for k, (lo, hi) in iv.items():
            curve[k].append(R4(np.sum(w * ((lo <= p) & (p <= hi)))))
    res["coverage_curve_n50"] = {"p": [R4(v) for v in ps], **curve}
    print(f"A done {time.time()-t0:.0f}s")

    # ------------------------------------------------ B. cohorts and shrinkage
    cohorts = [m for m in range(1, M - 2) if (first == m).sum() >= 30]
    sizes = np.array([(first == m).sum() for m in cohorts])
    rates = np.array([ret[first == m].mean() for m in cohorts])
    pooled = ret[np.isin(first, cohorts)].mean()
    # method of moments for the between-cohort variance
    sampling_var = pooled * (1 - pooled) / sizes
    tau2 = max(0.0, np.var(rates, ddof=1) - sampling_var.mean())
    chi2 = np.sum((rates - pooled) ** 2 / sampling_var)
    from scipy.stats import chi2 as chi2d
    res["cohorts"] = {"month": [int(m) for m in cohorts], "size": sizes.tolist(), "rate": [R4(v) for v in rates],
                      "pooled": R4(pooled), "tau": R4(np.sqrt(tau2)), "sd_rates": R4(np.std(rates, ddof=1)),
                      "mean_sampling_sd": R4(np.sqrt(sampling_var.mean())),
                      "chi2": R4(chi2), "df": int(len(cohorts) - 1), "chi2_p": R4(chi2d.sf(chi2, len(cohorts) - 1))}
    # shrinkage: subsample every cohort to n users, estimate, compare with the full cohort
    shr = {}
    big = [m for m in cohorts if (first == m).sum() >= 150]
    rates_big = np.array([ret[first == m].mean() for m in big])
    res["shrinkage_cohorts"] = len(big)
    for n in [20, 50, 100]:
        err_raw, err_eb, err_pool = [], [], []
        for r in range(2000):
            xs_ = np.array([ret[rng.choice(np.flatnonzero(first == m), n, replace=False)].sum() for m in big])
            raw = xs_ / n
            pb = raw.mean()
            sv = pb * (1 - pb) / n
            t2 = max(1e-6, np.var(raw, ddof=1) - sv)
            # beta-binomial posterior mean with a moment-matched beta prior
            k_ = pb * (1 - pb) / t2 - 1
            a0, b0 = max(pb * k_, 0.01), max((1 - pb) * k_, 0.01)
            eb = (xs_ + a0) / (n + a0 + b0)
            err_raw.append(np.mean((raw - rates_big) ** 2))
            err_eb.append(np.mean((eb - rates_big) ** 2))
            err_pool.append(np.mean((pb - rates_big) ** 2))
        shr[n] = {"rmse_raw": R4(np.sqrt(np.mean(err_raw))), "rmse_eb": R4(np.sqrt(np.mean(err_eb))),
                  "rmse_pooled": R4(np.sqrt(np.mean(err_pool))),
                  "eb_better_share": R4(np.mean(np.array(err_eb) < np.array(err_raw)))}
    res["shrinkage"] = shr
    print(f"B done {time.time()-t0:.0f}s", res["cohorts"]["tau"], shr)

    # ------------------------------------------------ C. peeking, A/A in real arrival order
    order = pop[np.argsort(first[pop], kind="stable")]
    # customers arrive by first-purchase month; shuffle within month for the replicate
    C = {}
    reps = 4000
    looks = np.arange(10, 2001, 10)
    naive_fp, msprt_fp, fixed_fp, done = 0, 0, 0, 0
    tau2_m = 0.01 ** 2 * 4  # mixture sd of 2 points of retention
    for r in range(reps):
        perm = order.copy()
        for m in np.unique(first[order]):
            sl = np.flatnonzero(first[perm] == m)
            perm[sl] = rng.permutation(perm[sl])
        stream = ret[perm[: 4600]].astype(float)
        arm = rng.random(len(stream)) < 0.5
        a, b = stream[arm][:2000], stream[~arm][:2000]
        if min(len(a), len(b)) < 2000:
            continue
        done += 1
        ca, cb = np.cumsum(a), np.cumsum(b)
        hit_naive, hit_ms = False, False
        for L in looks:
            pa, pb = ca[L - 1] / L, cb[L - 1] / L
            pp = (pa + pb) / 2
            se = np.sqrt(max(pp * (1 - pp), 1e-9) * 2 / L)
            if abs(pa - pb) / se > Z:
                hit_naive = True
            var = max(pa * (1 - pa) + pb * (1 - pb), 1e-6)
            if msprt_stat(L, pb - pa, var, tau2_m) >= 20:
                hit_ms = True
        naive_fp += hit_naive
        msprt_fp += hit_ms
        pa, pb = a.mean(), b.mean()
        pp = (pa + pb) / 2
        fixed_fp += abs(pa - pb) / np.sqrt(pp * (1 - pp) * 2 / 2000) > Z
    C = {"reps": done, "looks": len(looks), "max_per_arm": 2000,
         "fpr_fixed_once": R4(fixed_fp / done), "fpr_naive_peeking": R4(naive_fp / done), "fpr_msprt": R4(msprt_fp / done)}
    # false-positive rate as a function of how many looks have been taken
    curve_looks = [1, 2, 5, 10, 20, 50, 100, 200]
    fp_by_looks = {k: 0 for k in curve_looks}
    for r in range(2000):
        stream = ret[rng.permutation(pop)[:4000]].astype(float)
        a, b = stream[:2000], stream[2000:]
        ca, cb = np.cumsum(a), np.cumsum(b)
        for k in curve_looks:
            Ls = np.unique(np.linspace(2000 / k, 2000, k).astype(int))
            pa, pb = ca[Ls - 1] / Ls, cb[Ls - 1] / Ls
            pp = (pa + pb) / 2
            zz = np.abs(pa - pb) / np.sqrt(np.maximum(pp * (1 - pp), 1e-9) * 2 / Ls)
            fp_by_looks[k] += np.any(zz > Z)
    C["fpr_by_looks"] = {str(k): R4(v / 2000) for k, v in fp_by_looks.items()}
    res["peeking"] = C
    print(f"C done {time.time()-t0:.0f}s", C)

    # ------------------------------------------------ D. power and stopping
    D = {}
    p0 = p_true
    for n in [50, 100, 500, 1000, 4600]:
        mde = (Z + norm.ppf(0.8)) * np.sqrt(2 * p0 * (1 - p0) / n)
        D[str(n)] = {"mde_points": R4(100 * mde), "mde_relative": R4(mde / p0)}
    need = {}
    for lift in [0.02, 0.05, 0.10]:
        p1 = p0 + lift
        need[str(lift)] = int(np.ceil((Z * np.sqrt(2 * p0 * (1 - p0)) + norm.ppf(0.8) * np.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2 / lift ** 2))
    stop = {}
    for lift in [0.05, 0.10]:
        q = lift / (1 - p0)  # flip this share of zeros in arm B to ones
        times, power_fixed, power_ms = [], 0, 0
        n_fixed = need[str(lift)]
        for r in range(1500):
            s = ret[rng.permutation(pop)[:2 * max(n_fixed, 2000)]].astype(float)
            a, b = s[0::2].copy(), s[1::2].copy()
            b[(b == 0) & (rng.random(len(b)) < q)] = 1
            ca, cb = np.cumsum(a), np.cumsum(b)
            st = None
            for L in range(10, len(a) + 1, 10):
                pa, pb = ca[L - 1] / L, cb[L - 1] / L
                var = max(pa * (1 - pa) + pb * (1 - pb), 1e-6)
                if msprt_stat(L, pb - pa, var, tau2_m) >= 20:
                    st = L
                    break
            if st is not None:
                times.append(st)
                power_ms += 1
            L = n_fixed
            pa, pb = ca[L - 1] / L, cb[L - 1] / L
            pp = (pa + pb) / 2
            power_fixed += (pb - pa) / np.sqrt(pp * (1 - pp) * 2 / L) > Z
        stop[str(lift)] = {"n_fixed_80pct": n_fixed, "power_fixed": R4(power_fixed / 1500),
                           "power_msprt_within_cap": R4(power_ms / 1500), "cap": int(len(a)),
                           "median_stop": R4(np.median(times)) if times else None,
                           "q25_stop": R4(np.percentile(times, 25)) if times else None,
                           "q75_stop": R4(np.percentile(times, 75)) if times else None}
    res["power"] = {"p0": R4(p0), "mde": D, "n_needed_per_arm": need, "sequential": stop}
    print(f"D done {time.time()-t0:.0f}s", res["power"])
    OUT.mkdir(exist_ok=True)
    (OUT / "milestone1.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
