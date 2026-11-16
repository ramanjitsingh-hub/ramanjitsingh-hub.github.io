"""Investigation 02, milestone 1. Every number on the page comes from here.

Reads data/raw/sprout.npz and data/raw/preds.npz (from prepare.py and
fit_routers.py) and writes results/milestone1.json.
"""
import json
import time
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

from routing import (lagrangian_points, upper_hull, hull_cost_for, hull_score_at, lambda_grid,
                     router_hull, single_model_hull, oracle_points, hoeffding_p)
from features import load, TextFeatures, ridge_predict

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
OUT = HERE / "results"
GPT4O, MINI = 2, 3
R4 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 5)


def curve(h):
    """Hull as [cost per 1,000 prompts in USD, mean score]."""
    return [[R4(1000 * x), R4(y)] for x, y in h]


def main():
    t0 = time.time()
    d = np.load(RAW / "sprout.npz")
    S, C, sp, fam = d["S"], d["C"], d["split"], d["family"]
    meta = json.loads((RAW / "models.json").read_text())
    names = [meta["short"][m] for m in meta["models"]]
    K = len(names)
    p = np.load(RAW / "preds.npz")
    tr, va, te = sp == "train", sp == "validation", sp == "test"
    St, Ct, Sv, Cv = S[te], C[te], S[va], C[va]
    mu_te = (p["ridge_mte"] + p["gbm_mte"]) / 2
    mu_va = (p["ridge_mva"] + p["gbm_mva"]) / 2
    ch_te, ch_va = p["ridge_cte"], p["ridge_cva"]
    lam = lambda_grid(Ct.mean())
    res = {"models": names, "prices": [meta["prices"][m] for m in meta["models"]]}

    # ------------------------------------------------ 1. the data
    res["data"] = {
        "n": int(len(S)), "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
        "families": {f: int((fam == f).sum()) for f in np.unique(fam)},
        "frac_zero": R4(np.mean(S == 0)), "frac_one": R4(np.mean(S == 1)),
        "frac_partial": R4(np.mean((S > 0) & (S < 1))),
    }
    res["single"] = [{"model": names[k], "score": R4(St[:, k].mean()), "cost_per_1k": R4(1000 * Ct[:, k].mean()),
                      "score_all": R4(S[:, k].mean()), "cost_per_1k_all": R4(1000 * C[:, k].mean()),
                      "by_family": {f: R4(S[fam == f, k].mean()) for f in np.unique(fam)}}
                     for k in range(K)]
    sm = single_model_hull(St, Ct)
    res["single_hull"] = curve(sm)
    g4 = (Ct[:, GPT4O].mean(), St[:, GPT4O].mean())
    mini = (Ct[:, MINI].mean(), St[:, MINI].mean())

    # ------------------------------------------------ 2. oracle
    oh = upper_hull(oracle_points(St, Ct, lam))
    best = St.max(1)
    cand = np.where(St == best[:, None], Ct, np.inf)
    pick = cand.argmin(1)
    res["oracle"] = {
        "hull": curve(oh), "max_score": R4(best.mean()),
        "cheapest_best_cost_frac": R4(Ct[np.arange(len(St)), pick].mean() / g4[0]),
        "cost_frac_match_gpt4o": R4(hull_cost_for(oh, g4[1]) / g4[0]),
        "pick_share": {names[k]: R4(np.mean(pick == k)) for k in range(K)},
        "none_correct": R4(np.mean(best == 0)),
    }

    # ------------------------------------------------ 3. structure: who beats whom
    B = (S >= 0.5).astype(float)
    order = np.argsort(-B.mean(0))
    inv = np.array([[np.mean((B[:, a] == 1) & (B[:, b] == 0)) for b in order] for a in order])
    # Guttman errors: weaker model right while stronger wrong, against independence
    acc = B.mean(0)
    obs, exp_ind = [], []
    for a in range(K):
        for b in range(K):
            if acc[a] < acc[b]:
                obs.append(np.mean((B[:, a] == 1) & (B[:, b] == 0)))
                exp_ind.append(acc[a] * (1 - acc[b]))
    # one-dimensional benchmark: if errors followed a single difficulty axis
    # (Rasch), the weaker model would almost never win.
    Bt = (St >= 0.5)
    rng = np.random.default_rng(1)
    samp = rng.choice(np.flatnonzero(np.ones(len(Bt))), 360, replace=False)
    Hs = Bt[samp][:, order]
    Hs = Hs[np.argsort(-Hs.sum(1), kind="stable")]
    res["structure"] = {
        "order": [names[k] for k in order], "acc_binary": [R4(acc[k]) for k in order],
        "inversion": [[R4(v) for v in row] for row in inv],
        "guttman_obs": R4(np.mean(obs)), "guttman_indep": R4(np.mean(exp_ind)),
        "gpt4o_wrong_mini_right": R4(np.mean((B[:, GPT4O] == 0) & (B[:, MINI] == 1))),
        "gpt4o_right_mini_wrong": R4(np.mean((B[:, GPT4O] == 1) & (B[:, MINI] == 0))),
        "heatmap": ["".join("1" if v else "0" for v in row) for row in Hs],
    }
    Xc = B - B.mean(0)
    sv = np.linalg.svd(Xc - Xc.mean(1, keepdims=True), compute_uv=False)
    res["structure"]["svd_var"] = [R4(v) for v in (sv ** 2 / np.sum(sv ** 2))[:6]]

    # ------------------------------------------------ 4. learned routers on test
    routers = {}
    for key, mu, ch in [("family", p["family_mte"], p["family_cte"]), ("ridge", p["ridge_mte"], ch_te),
                        ("gbm", p["gbm_mte"], ch_te), ("ensemble", mu_te, ch_te)]:
        h, pts = router_hull(mu, ch, St, Ct, lam)
        xs = np.exp(np.linspace(np.log(Ct.mean(0).min()), np.log(Ct.mean(0).max()), 60))
        routers[key] = {
            "hull": curve(h),
            "area_gain": R4(np.nanmean([hull_score_at(h, x) - hull_score_at(sm, x) for x in xs])),
            "cost_frac_match_gpt4o": R4(hull_cost_for(h, g4[1]) / g4[0]),
            "score_at_mini_cost": R4(hull_score_at(h, mini[0])),
            "max_score": R4(h[:, 1].max()),
            "max_score_cost_frac": R4(h[np.argmax(h[:, 1]), 0] / g4[0]),
        }
    res["routers"] = routers
    res["refs"] = {"gpt4o_score": R4(g4[1]), "gpt4o_cost_per_1k": R4(1000 * g4[0]),
                   "mini_score": R4(mini[1]), "mini_cost_per_1k": R4(1000 * mini[0])}

    # routing shares as the price of quality changes (for the interactive figure)
    lam_show = np.concatenate([[0.0], np.logspace(0.5, 6.5, 70)])
    shares = []
    for L in lam_show:
        k = np.argmax(mu_te - L * ch_te, axis=1)
        shares.append({"lam": R4(L), "cost_per_1k": R4(1000 * Ct[np.arange(len(k)), k].mean()),
                       "score": R4(St[np.arange(len(k)), k].mean()),
                       "share": [R4(np.mean(k == j)) for j in range(K)]})
    res["shares"] = shares

    # bootstrap over test prompts, hull recomputed each time
    rng = np.random.default_rng(0)
    bs = {"cost_frac": [], "score_at_mini": [], "gain_vs_gpt4o": []}
    for _ in range(400):
        ix = rng.integers(0, len(St), len(St))
        h, _ = router_hull(mu_te[ix], ch_te[ix], St[ix], Ct[ix], lam)
        gs, gc = St[ix, GPT4O].mean(), Ct[ix, GPT4O].mean()
        bs["cost_frac"].append(hull_cost_for(h, gs) / gc)
        bs["score_at_mini"].append(hull_score_at(h, Ct[ix, MINI].mean()) - St[ix, MINI].mean())
        bs["gain_vs_gpt4o"].append(h[:, 1].max() - gs)
    res["bootstrap"] = {k: [R4(np.nanpercentile(v, 2.5)), R4(np.nanpercentile(v, 97.5))] for k, v in bs.items()}

    # honest operating point: lambda chosen on validation, read on test
    pv = lagrangian_points(mu_va, ch_va, Sv, Cv, lam)
    okv = np.flatnonzero(pv[:, 1] >= Sv[:, GPT4O].mean())
    jv = okv[np.argmin(pv[okv, 0])]
    pt = lagrangian_points(mu_te, ch_te, St, Ct, lam[[jv]])[0]
    res["val_chosen"] = {"lam": R4(lam[jv]), "test_score": R4(pt[1]), "test_cost_frac": R4(pt[0] / g4[0]),
                         "val_score": R4(pv[jv, 1]), "val_gpt4o": R4(Sv[:, GPT4O].mean())}

    # per family, router trained on everything
    res["by_family"] = {}
    for f in np.unique(fam[te]):
        m = fam[te] == f
        h, _ = router_hull(mu_te[m], ch_te[m], St[m], Ct[m], lam)
        hr, _ = router_hull(p["ridge_mte"][m], ch_te[m], St[m], Ct[m], lam)
        smf = single_model_hull(St[m], Ct[m])
        xs = np.exp(np.linspace(np.log(Ct[m].mean(0).min()), np.log(Ct[m].mean(0).max()), 60))
        res["by_family"][f] = {"n": int(m.sum()), "gpt4o": R4(St[m, GPT4O].mean()),
                               "ridge_area_gain": R4(np.nanmean([hull_score_at(hr, x) - hull_score_at(smf, x) for x in xs])),
                               "ridge_cost_frac_match_gpt4o": R4(hull_cost_for(hr, St[m, GPT4O].mean()) / Ct[m, GPT4O].mean()),
                               "router_max": R4(h[:, 1].max()),
                               "cost_frac_match_gpt4o": R4(hull_cost_for(h, St[m, GPT4O].mean()) / Ct[m, GPT4O].mean()),
                               "single_best": R4(St[m].mean(0).max()),
                               "single_best_model": names[int(np.argmax(St[m].mean(0)))]}
    print(f"routers done {time.time()-t0:.0f}s")

    # ------------------------------------------------ 5. luck: posterior predictive check
    Bv, Bte = (Sv >= 0.5).astype(float), (St >= 0.5).astype(float)
    cal = np.zeros_like(mu_te)
    for k in range(K):
        iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip").fit(mu_va[:, k], Bv[:, k])
        cal[:, k] = iso.predict(mu_te[:, k])
    sims = []
    for r in range(200):
        Y = (rng.random(cal.shape) < cal).astype(float)
        sims.append(Y.max(1).mean())
    rel = [[R4(np.mean(cal[:, GPT4O][(cal[:, GPT4O] >= a) & (cal[:, GPT4O] < a + 0.1)])),
            R4(np.mean(Bte[:, GPT4O][(cal[:, GPT4O] >= a) & (cal[:, GPT4O] < a + 0.1)])),
            int(np.sum((cal[:, GPT4O] >= a) & (cal[:, GPT4O] < a + 0.1)))] for a in np.arange(0, 1, 0.1)]
    # predicted-best model under calibrated probabilities: what a perfect router with
    # these beliefs would score, versus the realised oracle
    res["luck"] = {
        "oracle_binary": R4(Bte.max(1).mean()),
        "sim_oracle_mean": R4(np.mean(sims)), "sim_oracle_sd": R4(np.std(sims)),
        "belief_router": R4(Bte[np.arange(len(Bte)), cal.argmax(1)].mean()),
        "belief_router_expected": R4(cal.max(1).mean()),
        "best_single_binary": R4(Bte.mean(0).max()),
        "reliability_gpt4o": rel,
        "mean_pairwise_corr_resid": None,
    }
    resid = Bte - cal
    cr = np.corrcoef(resid.T)
    res["luck"]["mean_pairwise_corr_resid"] = R4(cr[np.triu_indices(K, 1)].mean())
    raw_c = np.corrcoef(Bte.T)
    res["luck"]["mean_pairwise_corr_raw"] = R4(raw_c[np.triu_indices(K, 1)].mean())
    print(f"luck done {time.time()-t0:.0f}s")

    # ------------------------------------------------ 6. certified operating point
    res["guarantee"] = certify(mu_te, ch_te, St, Ct, g4[1], rng)
    print(f"guarantee done {time.time()-t0:.0f}s")

    # ------------------------------------------------ 7. unseen benchmarks
    res["lofo"] = leave_one_family_out(S, C, sp, fam, lam)
    print(f"lofo done {time.time()-t0:.0f}s")

    OUT.mkdir(exist_ok=True)
    (OUT / "milestone1.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ["oracle", "routers", "val_chosen", "luck", "bootstrap"]}, indent=1)[:6000])


def betting_p_matrix(X, target, delta):
    """Vectorised betting p-values, one per row of X (see routing.betting_p)."""
    R, n = X.shape
    cap = np.ones(R)
    best = np.ones(R)
    m = np.full(R, 0.5)
    v = np.full(R, 0.25)
    cmax = 0.5 / target
    for t in range(n):
        eta = np.sqrt(2 * np.log(1 / delta) / ((t + 1) * np.log(t + 2) * np.maximum(v, 1e-4)))
        eta = np.where(m > target, np.minimum(eta, cmax), 0.0)
        x = X[:, t]
        cap = cap * (1 + eta * (x - target))
        best = np.maximum(best, cap)
        m = (m * (t + 1) + x) / (t + 2)
        v = (v * (t + 1) + (x - m) ** 2) / (t + 2)
    return np.minimum(1.0, 1.0 / best)


def certify(mu, ch, S, C, target, rng, delta=0.1, reps=1000):
    """Choose lambda from a calibration sample so that the population score is at
    least `target` with probability >= 1 - delta (fixed-sequence testing).

    The 6,637 test prompts are treated as the population: calibration samples are
    drawn from it with replacement, so the true score of every lambda is known and
    the violation rate can be counted exactly.
    """
    lams = np.concatenate([[0.0], np.logspace(1, 5.5, 46)])
    idx = np.arange(len(S))
    K_choice = [np.argmax(mu - L * ch, axis=1) for L in lams]
    pop_score = np.array([S[idx, k].mean() for k in K_choice])
    pop_cost = np.array([C[idx, k].mean() for k in K_choice])
    per = np.stack([S[idx, k] for k in K_choice])  # (L, n)
    feasible = np.flatnonzero(pop_score >= target)
    oracle_j = feasible[np.argmin(pop_cost[feasible])] if len(feasible) else None
    out = {"target": R4(target), "delta": delta, "reps": reps,
           "curve": [[R4(1000 * c), R4(s)] for c, s in zip(pop_cost, pop_score)],
           "best_feasible_cost_frac": R4(pop_cost[oracle_j] / C[:, GPT4O].mean()) if oracle_j is not None else None,
           "runs": []}
    for n in [250, 500, 1000, 2000, 4000]:
        draw = rng.integers(0, len(S), (reps, n))
        for method in ["hoeffding", "betting"]:
            P = np.zeros((len(lams), reps))
            for j in range(len(lams)):
                X = per[j][draw]
                if method == "betting":
                    P[j] = betting_p_matrix(X, target, delta)
                else:
                    gap = X.mean(1) - target
                    P[j] = np.where(gap > 0, np.exp(-2 * n * gap ** 2), 1.0)
            rej = P <= delta
            # fixed sequence: last j such that all of 0..j rejected
            run = np.cumprod(rej, axis=0)
            nrej = run.sum(0)
            chosen = nrej - 1  # -1 means abstain: fall back to always using GPT-4o
            sc = np.where(chosen >= 0, pop_score[np.maximum(chosen, 0)], np.nan)
            cc = np.where(chosen >= 0, pop_cost[np.maximum(chosen, 0)], C[:, GPT4O].mean())
            viol = np.mean(np.where(chosen >= 0, pop_score[np.maximum(chosen, 0)] < target, False))
            out["runs"].append({"n": n, "method": method, "violation": R4(viol),
                                "abstain": R4(np.mean(chosen < 0)),
                                "cost_frac_mean": R4(np.mean(cc) / C[:, GPT4O].mean()),
                                "cost_frac_q": [R4(np.percentile(cc, q) / C[:, GPT4O].mean()) for q in (10, 50, 90)],
                                "score_mean": R4(np.nanmean(sc))})
            print(n, method, out["runs"][-1])
    # sanity check of validity under a deliberately tight target: the naive rule
    # "pick the cheapest lambda whose sample mean clears the target" for contrast
    naive = []
    for n in [250, 500, 1000, 2000, 4000]:
        draw = rng.integers(0, len(S), (reps, n))
        means = np.stack([per[j][draw].mean(1) for j in range(len(lams))])
        ok = means >= target
        run = np.cumprod(ok, axis=0)
        ch_j = run.sum(0) - 1
        viol = np.mean(np.where(ch_j >= 0, pop_score[np.maximum(ch_j, 0)] < target, False))
        naive.append({"n": n, "violation": R4(viol)})
    out["naive"] = naive
    return out


def leave_one_family_out(S, C, sp, fam, lam):
    """Train the text router without one benchmark family; test on that family."""
    d = load()
    P = d["prompt"]
    tr, te = sp == "train", sp == "test"
    out = {}
    logC = np.log(C + 1e-7)
    for f in ["MATH", "MMLU-Pro", "OpenHermes", "RAGBench", "MuSR", "GPQA"]:
        trf = tr & (fam != f)
        tef = te & (fam == f)
        cache = RAW / f"lofo_{f}.npz"
        if cache.exists():
            z = np.load(cache)
            mu, lc = z["mu"], z["lc"]
        else:
            tf = TextFeatures()
            Xtr = tf.fit_transform(list(P[trf]))
            Xte = tf.transform(list(P[tef]))
            mu, = ridge_predict(Xtr, S[trf], [Xte], alpha=30)
            lc, = ridge_predict(Xtr, logC[trf], [Xte], alpha=3)
            np.savez_compressed(cache, mu=mu, lc=lc)
        St, Ct = S[tef], C[tef]
        h, _ = router_hull(mu, np.exp(lc), St, Ct, lam)
        sm = single_model_hull(St, Ct)
        # fair transfer baseline: a fixed ranking learned on the other families
        const_mu = np.tile(S[trf].mean(0), (len(St), 1))
        const_c = np.tile(C[trf].mean(0), (len(St), 1))
        hc, _ = router_hull(const_mu, const_c, St, Ct, lam)
        xs = np.exp(np.linspace(np.log(Ct.mean(0).min()), np.log(Ct.mean(0).max()), 60))
        gain = np.nanmean([hull_score_at(h, x) - hull_score_at(sm, x) for x in xs])
        gain_c = np.nanmean([hull_score_at(hc, x) - hull_score_at(sm, x) for x in xs])
        out[f] = {"n_test": int(tef.sum()), "area_gain": R4(gain), "area_gain_fixed_ranking": R4(gain_c),
                  "fixed_ranking_hull": curve(hc),
                  "router_max": R4(h[:, 1].max()), "gpt4o": R4(St[:, GPT4O].mean()),
                  "cost_frac_match_gpt4o": R4(hull_cost_for(h, St[:, GPT4O].mean()) / Ct[:, GPT4O].mean()),
                  "hull": curve(h), "single_hull": curve(sm)}
        print(f, out[f]["area_gain"], out[f]["cost_frac_match_gpt4o"])
    return out


if __name__ == "__main__":
    main()
