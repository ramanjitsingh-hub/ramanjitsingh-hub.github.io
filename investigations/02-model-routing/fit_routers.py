"""Fit the learned routers on SPROUT train, tune on validation, predict test.

Saves data/raw/preds.npz with predicted scores and costs for validation and test
for each router. Run once; run_milestone1.py reads it.
"""
import time
import numpy as np
from features import load, TextFeatures, ridge_predict, family_means, gbm_predict, svd_embed
from routing import router_hull, lambda_grid, single_model_hull, hull_cost_for, hull_score_at
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"


def area_gain(mu, ch, S, C):
    """Mean score advantage of a router hull over the single-model hull,
    averaged over log-cost between the cheapest and the most expensive model."""
    lam = lambda_grid(C.mean())
    h, _ = router_hull(mu, ch, S, C, lam)
    s = single_model_hull(S, C)
    xs = np.exp(np.linspace(np.log(C.mean(0).min()), np.log(C.mean(0).max()), 60))
    return np.nanmean([hull_score_at(h, x) - hull_score_at(s, x) for x in xs])


def main():
    t0 = time.time()
    d = load()
    sp, S, C, fam = d["split"], d["S"], d["C"], d["family"]
    tr, va, te = sp == "train", sp == "validation", sp == "test"
    P = d["prompt"]
    tf = TextFeatures()
    Xtr = tf.fit_transform(list(P[tr]))
    Xva, Xte = tf.transform(list(P[va])), tf.transform(list(P[te]))
    print(f"features {Xtr.shape} in {time.time()-t0:.0f}s")

    # costs are predicted on a log scale, then mapped back
    logC = np.log(C + 1e-7)
    out = {}

    # per-model constants: routing collapses to choosing one model
    for name, M in [("constant", lambda F: np.tile(S[tr].mean(0), (F.sum(), 1)))]:
        pass

    # benchmark-label router (uses metadata a deployed router would not have)
    out["family"] = (family_means(fam[tr], S[tr], fam[va]), family_means(fam[tr], S[tr], fam[te]),
                     np.exp(family_means(fam[tr], logC[tr], fam[va])), np.exp(family_means(fam[tr], logC[tr], fam[te])))

    # text ridge: tune alpha on validation by area gain over single models
    cva, cte = ridge_predict(Xtr, logC[tr], [Xva, Xte], alpha=3.0)
    cva, cte = np.exp(cva), np.exp(cte)
    best = None
    for a in [0.3, 1, 3, 10, 30]:
        mva, = ridge_predict(Xtr, S[tr], [Xva], alpha=a)
        g = area_gain(mva, cva, S[va], C[va])
        print(f"ridge alpha {a}: val area gain {g:.4f}")
        if best is None or g > best[0]:
            best = (g, a)
    mva, mte = ridge_predict(Xtr, S[tr], [Xva, Xte], alpha=best[1])
    out["ridge"] = (mva, mte, cva, cte)
    print("ridge alpha", best[1], f"{time.time()-t0:.0f}s")

    # gradient boosting on a 256-d SVD of the TF-IDF plus length features
    Ztr, (Zva, Zte) = svd_embed(Xtr, [Xva, Xte], dim=256)
    gva, gte = gbm_predict(Ztr, S[tr], [Zva, Zte])
    out["gbm"] = (gva, gte, cva, cte)
    print("gbm", f"{time.time()-t0:.0f}s")

    for k, (mva, mte, cv, ct) in out.items():
        print(f"{k:8s} val gain {area_gain(mva, cv, S[va], C[va]):.4f}  test gain {area_gain(mte, ct, S[te], C[te]):.4f}")
    np.savez_compressed(RAW / "preds.npz", **{f"{k}_{p}": v for k, vals in out.items()
                                             for p, v in zip(["mva", "mte", "cva", "cte"], vals)})
    np.savez_compressed(RAW / "svd.npz", Ztr=Ztr, Zva=Zva, Zte=Zte)


if __name__ == "__main__":
    main()
