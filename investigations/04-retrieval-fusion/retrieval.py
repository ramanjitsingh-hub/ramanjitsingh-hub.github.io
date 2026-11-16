"""Keyword search, meaning search and their fusion on BEIR datasets.

BM25 is implemented here on sparse matrices. Dense retrieval uses the
bge-small-en-v1.5 sentence encoder on the GPU. Per-query score vectors are
kept for the whole corpus so that any fusion can be computed exactly.
"""
import json
import re
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

RAW = Path(__file__).parent / "data" / "raw"
CACHE = RAW / "cache"
DATASETS = ["scifact", "nfcorpus", "arguana", "scidocs", "fiqa"]
MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load(name):
    d = RAW / name
    corpus = [json.loads(l) for l in open(d / "corpus.jsonl", encoding="utf8")]
    queries = {q["_id"]: q["text"] for q in (json.loads(l) for l in open(d / "queries.jsonl", encoding="utf8"))}
    qrels = {}
    with open(d / "qrels" / "test.tsv", encoding="utf8") as f:
        next(f)
        for line in f:
            qid, did, rel = line.rstrip("\n").split("\t")
            if int(rel) > 0:
                qrels.setdefault(qid, {})[did] = int(rel)
    ids = [c["_id"] for c in corpus]
    texts = [((c.get("title") or "") + " " + (c.get("text") or "")).strip() for c in corpus]
    qids = [q for q in qrels if q in queries]
    return ids, texts, qids, [queries[q] for q in qids], qrels


_tok = re.compile(r"[a-z0-9]+")


def _stem(w):
    # a light suffix stripper; enough to merge plurals and common inflections
    for suf in ("ational", "ization", "fulness", "iveness", "ations", "ation", "ingly", "edly", "ies", "ing", "ers", "ed", "es", "ly", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def analyzer(text):
    return [_stem(w) for w in _tok.findall(text.lower()) if w not in ENGLISH_STOP_WORDS]


class BM25:
    """Okapi BM25 with k1 = 0.9, b = 0.4 (the Anserini defaults)."""

    def __init__(self, texts, k1=0.9, b=0.4):
        self.cv = CountVectorizer(analyzer=analyzer)
        tf = self.cv.fit_transform(texts).tocsc().astype(np.float32)
        n = tf.shape[0]
        dl = np.asarray(tf.sum(1)).ravel()
        df = np.diff(tf.indptr)
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        tf = tf.tocoo()
        denom = tf.data + k1 * (1 - b + b * dl[tf.row] / dl.mean())
        w = tf.data * (k1 + 1) / denom * self.idf[tf.col]
        self.W = sparse.csr_matrix((w, (tf.row, tf.col)), shape=tf.shape).T.tocsr()  # terms x docs

    def scores(self, queries):
        Q = self.cv.transform(queries)
        Q.data[:] = 1.0  # each query term counted once
        return np.asarray((Q @ self.W).todense(), dtype=np.float32)


def dense_scores(name, texts, queries, batch=128):
    """Cosine similarities, cached per dataset."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{name}_dense.npy"
    if f.exists():
        return np.load(f)
    import torch
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL, device="cuda")
    m.half()
    D = m.encode(texts, batch_size=batch, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    Q = m.encode([QUERY_PREFIX + q for q in queries], batch_size=batch, convert_to_tensor=True, normalize_embeddings=True)
    S = (Q.float() @ D.float().T).cpu().numpy().astype(np.float32)
    np.save(f, S)
    del m
    torch.cuda.empty_cache()
    return S


# ---------------------------------------------------------------- metrics
def ndcg_at(scores, ids, qrels_q, k=10):
    top = np.argpartition(-scores, k)[:k]
    top = top[np.argsort(-scores[top])]
    gains = np.array([qrels_q.get(ids[j], 0) for j in top], float)
    dcg = np.sum((2 ** gains - 1) / np.log2(np.arange(2, k + 2)))
    ideal = np.sort(np.array(list(qrels_q.values()), float))[::-1][:k]
    idcg = np.sum((2 ** ideal - 1) / np.log2(np.arange(2, len(ideal) + 2)))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at(scores, ids, qrels_q, k=100):
    top = np.argpartition(-scores, k)[:k]
    hit = sum(1 for j in top if ids[j] in qrels_q)
    return hit / len(qrels_q)


# ---------------------------------------------------------------- fusion
def rrf(s1, s2, k=60):
    """Reciprocal rank fusion over the full corpus."""
    r1 = np.empty_like(s1); r1[np.argsort(-s1)] = np.arange(1, len(s1) + 1)
    r2 = np.empty_like(s2); r2[np.argsort(-s2)] = np.arange(1, len(s2) + 1)
    return 1 / (k + r1) + 1 / (k + r2)


def minmax(s, top=1000):
    """Min-max over the query's top-1000, as fusion systems usually do."""
    t = np.partition(s, -top)[-top:]
    lo, hi = t.min(), t.max()
    return (s - lo) / (hi - lo + 1e-12)


def zscore(s, ref):
    return (s - s[ref].mean()) / (s[ref].std() + 1e-12)
