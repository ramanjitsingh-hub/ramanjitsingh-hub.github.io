# Investigation 04 · When does mixing keyword and meaning search help?

Luna's memory mixes keyword search and meaning search. This investigation asks when that mix beats its better half, derives a model that predicts it, and tests both on public benchmarks.

Status: **in progress, milestone 1 complete** (2026-11-16). Page: [ramanjitsingh-hub.github.io/04](https://ramanjitsingh-hub.github.io/04/).

## Data and methods
- **Data:** five BEIR datasets with their test qrels (SciFact, NFCorpus, ArguAna, SCIDOCS, FiQA). That is 100,785 documents and 3,677 test queries. The datasets are not redistributed; download them from the BEIR site into `data/raw/`.
- **Keyword search:** BM25 (k1 = 0.9, b = 0.4), implemented on sparse matrices with a light suffix stemmer.
- **Meaning search:** `BAAI/bge-small-en-v1.5` in fp16 on an RTX 3050 Laptop GPU, scored by cosine similarity, with the bge query instruction.
- **Mixes:**
  - RRF (k = 60);
  - min-max over the top 1,000, with an equal weight and with a weight chosen by cross-validation;
  - z-scores over a random 2,000-document sample of the corpus, with an equal weight;
  - z-scores over the top-200 pool, with an equal weight and with a weight chosen by cross-validation.
- **Statistics:** nDCG@10 and recall@100 per query, compared in pairs with 10,000 bootstrap resamples and sign-flip tests. Holm's correction is applied across the five datasets for each method. Weights are chosen by 5-fold cross-validation over queries.

## Theory
Two-signal Gaussian model (signal detection theory). Scores are standardised on non-relevant documents, relevant documents are shifted by (d1, d2), and the two scores have correlation ρ.
- **Separation of a mix:** a mix with weights w has separation wᵀd/√(wᵀΣw), and AUC = Φ(d/√2).
- **Optimal weighting never hurts:** the best weights are w ∝ Σ⁻¹d, and they gain (d2 − ρd1)²/(1 − ρ²) ≥ 0 in squared separation.
- **When an equal mix helps:** it beats the stronger score only when ρ < (d1 + d2)²/(2d1²) − 1.

The proof is on the page.

## Milestone 1 results
- **Meaning search beats keyword search on all five datasets**, by 2.0 to 16.2 points of nDCG@10.
- **Reciprocal rank fusion** is significantly worse than meaning search alone on ArguAna (−5.7), SCIDOCS (−1.2) and FiQA (−4.5). It is never significantly better.
- **Normalisation matters as much as weighting.** z-scores over the corpus lose up to 9.1 points (FiQA). BM25 scores are zero for most documents, so standardising inflates the few non-zero ones.
- **Tuned top-200 z-score fusion** is never significantly worse. It is significantly better on NFCorpus (+2.3), SCIDOCS (+0.5) and FiQA (+1.7), Holm-adjusted p < 0.05. Cross-validated keyword weights are 0.05 to 0.4.
- **Model test, measured on the top-200 pool:**
  - Per-query predicted against measured AUC of the equal mix correlates at r = 0.79 to 0.97.
  - The predicted direction of the equal mix's AUC change is right on 5 of 5 datasets, and its direction on nDCG@10 on 4 of 5 (NFCorpus is the miss).
  - Predicted optimal weights are higher than the cross-validated ones, and are undefined where a median separation is negative.
  - Measured over random corpus documents, the model fails, because BM25 scores are mostly zero there.

## Reproduce
```
python run_fusion.py && python export_site_data.py
```

## Next
A labelled set of realistic founder queries over a synthetic company archive, to test whether a handful of labelled queries suffices to set the weight.
