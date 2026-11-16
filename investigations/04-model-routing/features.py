"""Prompt features and score predictors for the learned routers."""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"


def load():
    d = dict(np.load(RAW / "sprout.npz"))
    prompts = pd.read_parquet(RAW / "prompts.parquet")
    assert (prompts["key"].to_numpy() == d["key"]).all()
    d["prompt"] = prompts["prompt"].to_numpy()
    return d


def length_features(texts):
    L = np.array([len(t) for t in texts], float)
    digits = np.array([sum(ch.isdigit() for ch in t[:4000]) for t in texts], float)
    lines = np.array([t.count("\n") for t in texts], float)
    dollars = np.array([t.count("$") + t.count("\\") for t in texts], float)
    return np.c_[np.log1p(L), np.log1p(digits), np.log1p(lines), np.log1p(dollars)]


class TextFeatures:
    """TF-IDF over words (1-2 grams) and characters (3-5 grams), plus lengths."""

    def __init__(self, max_word=60000, max_char=60000):
        self.w = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=max_word,
                                 sublinear_tf=True, strip_accents="unicode")
        self.c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                                 max_features=max_char, sublinear_tf=True)

    @staticmethod
    def _clip(texts, n=6000):
        # long RAG contexts dominate otherwise; the question sits at the start and end
        return [t if len(t) <= n else t[: n // 2] + " " + t[-n // 2:] for t in texts]

    def fit_transform(self, texts):
        t = self._clip(texts)
        Xw = self.w.fit_transform(t)
        Xc = self.c.fit_transform(t)
        self.mu_len = None
        Lf = length_features(texts)
        self.m, self.s = Lf.mean(0), Lf.std(0) + 1e-9
        return sparse.hstack([Xw, Xc, sparse.csr_matrix((Lf - self.m) / self.s)]).tocsr()

    def transform(self, texts):
        t = self._clip(texts)
        Lf = (length_features(texts) - self.m) / self.s
        return sparse.hstack([self.w.transform(t), self.c.transform(t), sparse.csr_matrix(Lf)]).tocsr()


def ridge_predict(Xtr, Ytr, Xev_list, alpha):
    r = Ridge(alpha=alpha, solver="auto")
    r.fit(Xtr, Ytr)
    return [r.predict(X) for X in Xev_list]


def family_means(fam_tr, Y_tr, fam_ev):
    df = pd.DataFrame(Y_tr)
    df["f"] = fam_tr
    g = df.groupby("f").mean()
    glob = Y_tr.mean(0)
    return np.array([g.loc[f].to_numpy() if f in g.index else glob for f in fam_ev])


def gbm_predict(Ztr, Ytr, Zev_list, seed=0):
    outs = [np.zeros((Z.shape[0], Ytr.shape[1])) for Z in Zev_list]
    for k in range(Ytr.shape[1]):
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=seed)
        m.fit(Ztr, Ytr[:, k])
        for o, Z in zip(outs, Zev_list):
            o[:, k] = m.predict(Z)
    return outs


def svd_embed(Xtr, X_list, dim=256, seed=0):
    svd = TruncatedSVD(n_components=dim, random_state=seed)
    Ztr = svd.fit_transform(Xtr)
    return Ztr, [svd.transform(X) for X in X_list]
