"""Investigation 04, milestone 2: routing under partial feedback.

A deployed router sees only the score and cost of the model it chose. SPROUT
records every model's answer, so both questions below can be checked against
the truth, which is never possible with real logs.

A. Online learning (contextual bandit). Prompts arrive one at a time; the router
   picks a model, observes reward r = score - lambda * cost for that model only,
   and updates. Compared: greedy, epsilon-greedy, LinUCB, linear Thompson
   sampling, and a full-information learner that sees all 13 rewards.

B. Off-policy evaluation. From logs of one router (with known propensities),
   estimate the value of another router without running it: direct method,
   inverse propensity scoring, self-normalised IPS and doubly robust. True
   values are known, so bias, spread and interval coverage are measured.

Writes results/milestone2.json.
"""
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
OUT = HERE / "results"
R4 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 5)
GPT4O, MINI = 2, 3


def load():
    d = np.load(RAW / "sprout.npz")
    z = np.load(RAW / "svd.npz")
    sp = d["split"]
    S, C = d["S"], d["C"]
    Z = np.zeros((len(S), z["Ztr"].shape[1]))
    Z[sp == "train"], Z[sp == "validation"], Z[sp == "test"] = z["Ztr"], z["Zva"], z["Zte"]
    return S, C, sp, d["family"], Z


def features(Z, dim, ref):
    X = Z[:, :dim]
    m, s = X[ref].mean(0), X[ref].std(0) + 1e-9
    X = (X - m) / s
    return np.c_[np.ones(len(X)), X]


# ------------------------------------------------------------ A. bandits
class LinearArms:
    """Per-arm ridge regression with Sherman-Morrison updates of A^-1."""

    def __init__(self, K, d, reg=1.0):
        self.Ainv = np.stack([np.eye(d) / reg for _ in range(K)])
        self.b = np.zeros((K, d))
        self.theta = np.zeros((K, d))

    def update(self, k, x, r):
        Ax = self.Ainv[k] @ x
        self.Ainv[k] -= np.outer(Ax, Ax) / (1 + x @ Ax)
        self.b[k] += r * x
        self.theta[k] = self.Ainv[k] @ self.b[k]

    def mean(self, x):
        return self.theta @ x

    def width(self, x):
        return np.sqrt(np.einsum("kij,i,j->k", self.Ainv, x, x))


def run_bandit(policy, X, R, rng, param, warm=0):
    """Play one stream. R[t, k] is the reward model k would have earned."""
    T, K = R.shape
    arms = LinearArms(K, X.shape[1])
    got = np.zeros(T)
    pulls = np.zeros(K)
    for t in range(T):
        x = X[t]
        if t < warm:  # round-robin start so every arm has data
            k = t % K
        elif policy == "greedy":
            k = int(np.argmax(arms.mean(x)))
        elif policy == "egreedy":
            k = int(rng.integers(K)) if rng.random() < param else int(np.argmax(arms.mean(x)))
        elif policy == "linucb":
            k = int(np.argmax(arms.mean(x) + param * arms.width(x)))
        elif policy == "thompson":
            # sample each arm's mean from N(theta x, v^2 x' A^-1 x)
            k = int(np.argmax(arms.mean(x) + param * arms.width(x) * rng.standard_normal(K)))
        elif policy == "full":
            k = int(np.argmax(arms.mean(x)))
        got[t] = R[t, k]
        pulls[k] += 1
        if policy == "full":
            for j in range(K):
                arms.update(j, x, R[t, j])
        else:
            arms.update(k, x, R[t, k])
    return got, pulls, arms


def part_a(S, C, sp, Z, lams, seeds=5):
    stream = np.flatnonzero(sp != "test")
    test = np.flatnonzero(sp == "test")
    X = features(Z, 64, stream)
    out = {}
    configs = [("greedy", 0.0), ("egreedy", 0.05), ("egreedy", 0.2), ("linucb", 0.3), ("linucb", 1.0),
               ("thompson", 0.3), ("full", 0.0)]
    for lam in lams:
        Rall = S - lam * C
        best_fixed = int(np.argmax(Rall[stream].mean(0)))
        res = {"lam": lam, "best_fixed_model": best_fixed, "algos": {}}
        for pol, par in configs:
            key = f"{pol}" + (f"_{par}" if pol in ("egreedy", "linucb", "thompson") else "")
            curves, finals, tests, pulls_all = [], [], [], []
            for s in range(seeds):
                rng = np.random.default_rng(100 + s)
                order = rng.permutation(stream)
                t0 = time.time()
                got, pulls, arms = run_bandit(pol, X[order], Rall[order], rng, par, warm=13 * 5)
                # regret against the best fixed model in hindsight on this stream
                reg = np.cumsum(Rall[order, best_fixed] - got)
                pts = np.unique(np.geomspace(50, len(got), 40).astype(int)) - 1
                curves.append(reg[pts])
                finals.append(got.mean())
                # freeze the learned model and route the test set greedily
                kt = np.argmax(X[test] @ arms.theta.T, axis=1)
                tests.append((S[test, kt].mean(), C[test, kt].mean() / C[test, GPT4O].mean(),
                              (S[test, kt] - lam * C[test, kt]).mean()))
                pulls_all.append(pulls / pulls.sum())
            res["algos"][key] = {
                "t": [int(p + 1) for p in pts],
                "regret_mean": [R4(v) for v in np.mean(curves, 0)],
                "regret_lo": [R4(v) for v in np.min(curves, 0)],
                "regret_hi": [R4(v) for v in np.max(curves, 0)],
                "avg_reward": R4(np.mean(finals)), "avg_reward_sd": R4(np.std(finals)),
                "test_score": R4(np.mean([t[0] for t in tests])),
                "test_cost_frac": R4(np.mean([t[1] for t in tests])),
                "test_reward": R4(np.mean([t[2] for t in tests])),
                "pull_share": [R4(v) for v in np.mean(pulls_all, 0)],
            }
            print(f"lam {lam} {key:14s} avg reward {np.mean(finals):.4f}  final regret {np.mean(curves,0)[-1]:.1f}"
                  f"  test score {res['algos'][key]['test_score']}  cost {res['algos'][key]['test_cost_frac']}")
        res["best_fixed_reward"] = R4(Rall[stream, best_fixed].mean())
        res["oracle_reward"] = R4(Rall[stream].max(1).mean())
        out[str(lam)] = res
    return out


# ------------------------------------------------------------ B. off-policy evaluation
def part_b(S, C, sp, Z, lam, reps=300, eps=0.2):
    """Logging policy: epsilon-greedy over a reward model fitted on train.
    Targets: several deterministic routers. Truth: their mean reward on test."""
    tr, te = np.flatnonzero(sp == "train"), np.flatnonzero(sp == "test")
    X = features(Z, 64, tr)
    K = S.shape[1]
    R = S - lam * C
    # a deliberately simple, imperfect logging router (ridge on 64 SVD features)
    W = np.linalg.solve(X[tr].T @ X[tr] + 10 * np.eye(X.shape[1]), X[tr].T @ R[tr])
    pred = X[te] @ W
    log_greedy = pred.argmax(1)
    n = len(te)
    prop = np.full((n, K), eps / K)
    prop[np.arange(n), log_greedy] += 1 - eps
    # target policies, each a deterministic choice per prompt
    W2 = np.linalg.solve(X[tr].T @ X[tr] + 1000 * np.eye(X.shape[1]), X[tr].T @ R[tr])
    targets = {
        "always GPT-4o mini": np.full(n, MINI),
        "always GPT-4o": np.full(n, GPT4O),
        "always Llama 3.2 3B": np.full(n, 9),
        "logging router, no exploration": log_greedy,
        "milestone 1 router": None,
    }
    p1 = np.load(RAW / "preds.npz")
    mu1 = (p1["ridge_mte"] + p1["gbm_mte"]) / 2
    targets["milestone 1 router"] = np.argmax(mu1 - lam * p1["ridge_cte"], axis=1)
    truth = {k: float(R[te][np.arange(n), a].mean()) for k, a in targets.items()}
    Rt = R[te]
    Xt = X[te]
    rng = np.random.default_rng(7)
    est = {k: {m: [] for m in ["DM", "IPS", "SNIPS", "DR"]} for k in targets}
    cover = {k: {m: [] for m in ["IPS", "DR"]} for k in targets}
    half = n // 2
    for r in range(reps):
        # draw the logged actions anew: this is the randomness OPE must handle
        u = rng.random(n)
        cum = np.cumsum(prop, axis=1)
        a = (u[:, None] > cum).sum(1)
        a = np.minimum(a, K - 1)
        rew = Rt[np.arange(n), a]
        p = prop[np.arange(n), a]
        # direct-method reward model, cross-fitted on two halves using logged data only
        perm = rng.permutation(n)
        folds = [perm[:half], perm[half:]]
        qhat = np.zeros((n, K))
        for f in range(2):
            fit, ev = folds[1 - f], folds[f]
            for k in range(K):
                m = fit[a[fit] == k]
                # unpenalised arm mean, ridge only on the centred features
                mu_k = rew[m].mean() if len(m) else rew[fit].mean()
                Xm = Xt[m][:, 1:]
                xm = Xm.mean(0) if len(m) else np.zeros(Xt.shape[1] - 1)
                wk = np.linalg.solve((Xm - xm).T @ (Xm - xm) + 30 * np.eye(Xt.shape[1] - 1), (Xm - xm).T @ (rew[m] - mu_k)) if len(m) > 1 else np.zeros(Xt.shape[1] - 1)
                qhat[ev, k] = mu_k + (Xt[ev][:, 1:] - xm) @ wk
        for name, tgt in targets.items():
            match = (a == tgt).astype(float)
            w = match / p
            dm = qhat[np.arange(n), tgt]
            ips_terms = w * rew
            dr_terms = dm + w * (rew - qhat[np.arange(n), a])
            est[name]["DM"].append(dm.mean())
            est[name]["IPS"].append(ips_terms.mean())
            est[name]["SNIPS"].append((w * rew).sum() / max(w.sum(), 1e-12))
            est[name]["DR"].append(dr_terms.mean())
            for m, terms in [("IPS", ips_terms), ("DR", dr_terms)]:
                se = terms.std(ddof=1) / np.sqrt(n)
                cover[name][m].append(abs(terms.mean() - truth[name]) <= 1.96 * se)
    out = {"lam": lam, "eps": eps, "reps": reps, "n": n, "targets": {}}
    for name in targets:
        row = {"truth": R4(truth[name]), "match_rate": R4(np.mean(prop[np.arange(n), targets[name]]))}
        for m, v in est[name].items():
            v = np.array(v)
            row[m] = {"bias": R4(v.mean() - truth[name]), "sd": R4(v.std()), "rmse": R4(np.sqrt(np.mean((v - truth[name]) ** 2)))}
        for m, c in cover[name].items():
            row[m]["coverage95"] = R4(np.mean(c))
        out["targets"][name] = row
        print(name, json.dumps(row))
    return out


def main():
    t0 = time.time()
    S, C, sp, fam, Z = load()
    import sys
    res = json.loads((OUT / "milestone2.json").read_text()) if (OUT / "milestone2.json").exists() else {}
    if "ope" in sys.argv or not res:
        res["ope"] = part_b(S, C, sp, Z, lam=100.0)
        print(f"OPE done {time.time()-t0:.0f}s")
    if "bandit" in sys.argv or "bandit" not in res:
        res["bandit"] = part_a(S, C, sp, Z, lams=[30.0, 100.0, 300.0])
    print(f"bandits done {time.time()-t0:.0f}s")
    OUT.mkdir(exist_ok=True)
    (OUT / "milestone2.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
