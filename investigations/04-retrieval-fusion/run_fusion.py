"""Investigation 04, milestone 1: when does mixing keyword and meaning search help?

For each BEIR dataset: BM25, dense (bge-small), and four fusions
(reciprocal rank fusion, equal-weight z-score sum, and z-score and min-max sums
with the weight chosen by 5-fold cross-validation over queries). Per-query
nDCG@10 and recall@100, paired bootstrap intervals and sign-flip tests with a
Holm correction, and a test of the two-signal Gaussian model (THEORY.md) that
predicts the fused separation from d1, d2 and the score correlation rho.

Writes results/milestone1.json.
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

from retrieval import DATASETS, load, BM25, dense_scores, ndcg_at, recall_at, rrf, minmax, zscore

HERE = Path(__file__).parent
OUT = HERE / "results"
R4 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 5)
ALPHAS = np.round(np.linspace(0, 1, 21), 2)


def auc(pos, neg):
    """Probability a relevant document outscores a random non-relevant one."""
    allv = np.concatenate([pos, neg])
    r = allv.argsort().argsort() + 1.0
    rp = r[: len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def paired(delta, rng, B=10000):
    n = len(delta)
    boots = delta[rng.integers(0, n, (B, n))].mean(1)
    flips = (rng.choice([-1, 1], (B, n)) * delta).mean(1)
    p = (np.sum(np.abs(flips) >= abs(delta.mean())) + 1) / (B + 1)
    return {"mean": R4(delta.mean()), "lo": R4(np.percentile(boots, 2.5)), "hi": R4(np.percentile(boots, 97.5)), "p": R4(p)}


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.zeros(m)
    run = 0
    for i, j in enumerate(order):
        run = max(run, (m - i) * ps[j])
        adj[j] = min(1, run)
    return adj


def run_dataset(name, rng):
    t0 = time.time()
    ids, texts, qids, queries, qrels = load(name)
    S1 = BM25(texts).scores(queries)
    S2 = dense_scores(name, texts, queries)
    nq, nd = S1.shape
    id_index = {d: i for i, d in enumerate(ids)}
    print(f"{name}: {nd} docs, {nq} test queries, scored in {time.time()-t0:.0f}s")
    sys_names = ["BM25", "dense", "RRF", "min-max, equal", "min-max, tuned", "z over corpus, equal", "z over top 200, equal", "z over top 200, tuned"]
    nd10 = {s: np.zeros(nq) for s in sys_names}
    r100 = {s: np.zeros(nq) for s in sys_names}
    grid_z = np.zeros((nq, len(ALPHAS)))
    grid_m = np.zeros((nq, len(ALPHAS)))
    theory = {"d1": [], "d2": [], "rho": [], "auc1": [], "auc2": [], "auc_eq": [], "pred_auc_eq": [], "pred_auc1": [], "pred_auc2": []}
    cands = []
    for qi, q in enumerate(qids):
        rel = qrels[q]
        relidx = np.array([id_index[d] for d in rel if d in id_index])
        cand = np.setdiff1d(rng.choice(nd, min(2000, nd), replace=False), relidx)
        s1, s2 = S1[qi], S2[qi]
        z1, z2 = zscore(s1, cand), zscore(s2, cand)
        m1, m2 = minmax(s1), minmax(s2)
        # the pool where ranking happens: top 100 of each system
        pool = np.union1d(np.argpartition(-s1, 100)[:100], np.argpartition(-s2, 100)[:100])
        p1, p2 = zscore(s1, pool), zscore(s2, pool)
        for sname, sc in [("BM25", s1), ("dense", s2), ("RRF", rrf(s1, s2)), ("min-max, equal", m1 + m2),
                          ("z over corpus, equal", z1 + z2), ("z over top 200, equal", p1 + p2)]:
            nd10[sname][qi] = ndcg_at(sc, ids, rel)
            r100[sname][qi] = recall_at(sc, ids, rel)
        for ai, a in enumerate(ALPHAS):
            grid_z[qi, ai] = ndcg_at(a * p1 + (1 - a) * p2, ids, rel)
            grid_m[qi, ai] = ndcg_at(a * m1 + (1 - a) * m2, ids, rel)
        cands.append(pool)
        # two-signal Gaussian model, per query, measured among the pool's non-relevant documents
        neg = np.setdiff1d(pool, relidx)
        if len(relidx) and len(neg) > 20 and s1[neg].std() > 0 and s2[neg].std() > 0:
            z1, z2 = zscore(s1, neg), zscore(s2, neg)
            cand = neg
            d1, d2 = z1[relidx].mean(), z2[relidx].mean()
            rho = np.corrcoef(z1[cand], z2[cand])[0, 1]
            deq = (d1 + d2) / np.sqrt(2 + 2 * rho)
            for k, v in [("d1", d1), ("d2", d2), ("rho", rho), ("auc1", auc(z1[relidx], z1[cand])), ("auc2", auc(z2[relidx], z2[cand])),
                         ("auc_eq", auc((z1 + z2)[relidx], (z1 + z2)[cand])), ("pred_auc_eq", norm.cdf(deq / np.sqrt(2))),
                         ("pred_auc1", norm.cdf(d1 / np.sqrt(2))), ("pred_auc2", norm.cdf(d2 / np.sqrt(2)))]:
                theory[k].append(v)
    # cross-validated weights: choose alpha on 4 folds, apply to the 5th
    folds = np.array_split(rng.permutation(nq), 5)
    chosen_z, chosen_m = [], []
    for f in range(5):
        test = folds[f]
        train = np.concatenate([folds[g] for g in range(5) if g != f])
        az = ALPHAS[np.argmax(grid_z[train].mean(0))]
        am = ALPHAS[np.argmax(grid_m[train].mean(0))]
        chosen_z.append(float(az)); chosen_m.append(float(am))
        for qi in test:
            nd10["z over top 200, tuned"][qi] = grid_z[qi, list(ALPHAS).index(az)]
            nd10["min-max, tuned"][qi] = grid_m[qi, list(ALPHAS).index(am)]
            z1, z2 = zscore(S1[qi], cands[qi]), zscore(S2[qi], cands[qi])
            r100["z over top 200, tuned"][qi] = recall_at(az * z1 + (1 - az) * z2, ids, qrels[qids[qi]])
            r100["min-max, tuned"][qi] = recall_at(am * minmax(S1[qi]) + (1 - am) * minmax(S2[qi]), ids, qrels[qids[qi]])
    best_single = "BM25" if nd10["BM25"].mean() >= nd10["dense"].mean() else "dense"
    out = {"n_docs": nd, "n_queries": nq, "best_single": best_single,
           "ndcg10": {s: R4(v.mean()) for s, v in nd10.items()},
           "recall100": {s: R4(v.mean()) for s, v in r100.items()},
           "alpha_curve_z": [R4(v) for v in grid_z.mean(0)], "alpha_curve_minmax": [R4(v) for v in grid_m.mean(0)],
           "alpha_chosen_z": chosen_z, "alpha_chosen_minmax": chosen_m,
           "vs_best_single": {s: paired(nd10[s] - nd10[best_single], rng) for s in ["RRF", "min-max, equal", "min-max, tuned", "z over corpus, equal", "z over top 200, equal", "z over top 200, tuned"]},
           "dense_vs_bm25": paired(nd10["dense"] - nd10["BM25"], rng),
           "per_query_win_rate": {s: R4(np.mean(nd10[s] > nd10[best_single])) for s in ["RRF", "min-max, equal", "z over top 200, tuned"]},
           "per_query_loss_rate": {s: R4(np.mean(nd10[s] < nd10[best_single])) for s in ["RRF", "min-max, equal", "z over top 200, tuned"]}}
    T = {k: np.array(v) for k, v in theory.items()}
    d1, d2, rho = np.median(T["d1"]), np.median(T["d2"]), np.median(T["rho"])
    deq = (d1 + d2) / np.sqrt(2 + 2 * rho)
    dopt = np.sqrt((d1 ** 2 + d2 ** 2 - 2 * rho * d1 * d2) / (1 - rho ** 2))
    out["theory"] = {
        "d1": R4(d1), "d2": R4(d2), "rho_median": R4(rho), "rho_iqr": [R4(np.percentile(T["rho"], 25)), R4(np.percentile(T["rho"], 75))],
        "d_equal": R4(deq), "d_optimal": R4(dopt), "w_opt_bm25": R4((d1 - rho * d2) / ((d1 - rho * d2) + (d2 - rho * d1))),
        "helps_threshold_rho": R4((d1 + d2) ** 2 / (2 * max(d1, d2) ** 2) - 1),
        "auc": {"BM25": R4(T["auc1"].mean()), "dense": R4(T["auc2"].mean()), "equal": R4(T["auc_eq"].mean())},
        "pred_auc": {"BM25": R4(T["pred_auc1"].mean()), "dense": R4(T["pred_auc2"].mean()), "equal": R4(T["pred_auc_eq"].mean())},
        "per_query_pred_vs_meas": {"corr": R4(np.corrcoef(T["pred_auc_eq"], T["auc_eq"])[0, 1]),
                                   "mae": R4(np.mean(np.abs(T["pred_auc_eq"] - T["auc_eq"])))},
        "scatter": [[R4(a), R4(b)] for a, b in zip(T["pred_auc_eq"][:400], T["auc_eq"][:400])],
        "pred_gain_sign_auc": bool(norm.cdf(deq / np.sqrt(2)) > norm.cdf(max(d1, d2) / np.sqrt(2))),
        "n_queries_used": int(len(T["d1"])),
        "auc_gain_equal_vs_best_meas": R4(T["auc_eq"].mean() - max(T["auc1"].mean(), T["auc2"].mean())),
        "auc_gain_equal_vs_best_pred": R4(T["pred_auc_eq"].mean() - max(T["pred_auc1"].mean(), T["pred_auc2"].mean())),
        "rho_hist": np.histogram(T["rho"], bins=20, range=(-0.6, 0.6))[0].tolist(),
    }
    print(f"  ndcg@10 {out['ndcg10']}  theory d1 {d1:.2f} d2 {d2:.2f} rho {rho:.2f} deq {deq:.2f}  [{time.time()-t0:.0f}s]")
    return out


def main():
    rng = np.random.default_rng(0)
    res = {"datasets": {}}
    for name in DATASETS:
        res["datasets"][name] = run_dataset(name, rng)
    for s in ["RRF", "min-max, equal", "min-max, tuned", "z over corpus, equal", "z over top 200, equal", "z over top 200, tuned"]:
        ps = np.array([res["datasets"][n]["vs_best_single"][s]["p"] for n in DATASETS])
        for n, a in zip(DATASETS, holm(ps)):
            res["datasets"][n]["vs_best_single"][s]["p_holm"] = R4(a)
    OUT.mkdir(exist_ok=True)
    (OUT / "milestone1.json").write_text(json.dumps(res, indent=1))
    for n in DATASETS:
        d = res["datasets"][n]
        print(n, d["best_single"], {k: (v["mean"], v["p_holm"]) for k, v in d["vs_best_single"].items()})


if __name__ == "__main__":
    main()
