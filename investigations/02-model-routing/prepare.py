"""Convert the SPROUT parquet files into compact arrays.

Writes data/raw/sprout.npz (scores, token counts, costs, split, benchmark) and
data/raw/prompts.parquet (prompt text). Both stay out of git: the dataset card
states no licence, so this repository publishes only statistics computed from it.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RAW = Path(__file__).parent / "data" / "raw"

# USD per 1M tokens (input, output). Source: Somerstep et al., "CARROT: A Cost
# Aware Rate Optimal Router", arXiv:2502.03261v2, Table 2.
PRICES = {
    "aws-claude-3-5-sonnet-v1": (3.0, 15.0),
    "aws-titan-text-premier-v1": (0.5, 1.5),
    "openai-gpt-4o": (2.5, 10.0),
    "openai-gpt-4o-mini": (0.15, 0.6),
    "wxai-granite-3-2b-instruct-8k-max-tokens": (0.1, 0.1),
    "wxai-granite-3-8b-instruct-8k-max-tokens": (0.2, 0.2),
    "wxai-llama-3-1-70b-instruct": (0.9, 0.9),
    "wxai-llama-3-1-8b-instruct": (0.2, 0.2),
    "wxai-llama-3-2-1b-instruct": (0.06, 0.06),
    "wxai-llama-3-2-3b-instruct": (0.06, 0.06),
    "wxai-llama-3-3-70b-instruct": (0.9, 0.9),
    "wxai-llama-3-405b-instruct": (3.5, 3.5),
    "wxai-mixtral-8x7b-instruct-v01": (0.6, 0.6),
}
SHORT = {
    "aws-claude-3-5-sonnet-v1": "Claude 3.5 Sonnet",
    "aws-titan-text-premier-v1": "Titan Premier",
    "openai-gpt-4o": "GPT-4o",
    "openai-gpt-4o-mini": "GPT-4o mini",
    "wxai-granite-3-2b-instruct-8k-max-tokens": "Granite 3 2B",
    "wxai-granite-3-8b-instruct-8k-max-tokens": "Granite 3 8B",
    "wxai-llama-3-1-70b-instruct": "Llama 3.1 70B",
    "wxai-llama-3-1-8b-instruct": "Llama 3.1 8B",
    "wxai-llama-3-2-1b-instruct": "Llama 3.2 1B",
    "wxai-llama-3-2-3b-instruct": "Llama 3.2 3B",
    "wxai-llama-3-3-70b-instruct": "Llama 3.3 70B",
    "wxai-llama-3-405b-instruct": "Llama 3.1 405B",
    "wxai-mixtral-8x7b-instruct-v01": "Mixtral 8x7B",
}
MODELS = list(PRICES)


def family(ds: str) -> str:
    """Collapse SPROUT's sub-task names into six benchmark families."""
    for key, name in [("MMLU-Pro", "MMLU-Pro"), ("MATH", "MATH"), ("gpqa", "GPQA"),
                      ("MuSR", "MuSR"), ("ragbench", "RAGBench"), ("openhermes", "OpenHermes")]:
        if key in ds:
            return name
    return "other"


def main():
    frames = []
    for split in ["train", "validation", "test"]:
        for f in sorted(RAW.glob(f"{split}-*.parquet")):
            t = pq.read_table(f).to_pandas()
            t["split"] = split
            frames.append(t)
    t = pd.concat(frames, ignore_index=True)
    n, k = len(t), len(MODELS)
    S = np.full((n, k), np.nan)
    TI = np.zeros((n, k))
    TO = np.zeros((n, k))
    for j, m in enumerate(MODELS):
        col = t[m].to_numpy()
        for i, d in enumerate(col):
            if d is None:
                continue
            S[i, j] = d["score"] if d["score"] is not None else np.nan
            TI[i, j] = d["num_input_tokens"] or 0
            TO[i, j] = d["num_output_tokens"] or 0
    pin = np.array([PRICES[m][0] for m in MODELS])
    pout = np.array([PRICES[m][1] for m in MODELS])
    C = (TI * pin + TO * pout) / 1e6
    ok = ~np.isnan(S).any(axis=1)
    print(f"{n} prompts, {ok.sum()} with a score from all {k} models")
    t = t.loc[ok].reset_index(drop=True)
    S, TI, TO, C = S[ok], TI[ok], TO[ok], C[ok]
    fam = t["dataset"].map(family).to_numpy()
    np.savez_compressed(RAW / "sprout.npz", S=S, TI=TI, TO=TO, C=C,
                        split=t["split"].to_numpy().astype(str), family=fam.astype(str),
                        dataset=t["dataset"].to_numpy().astype(str), key=t["key"].to_numpy().astype(str))
    t[["key", "prompt"]].to_parquet(RAW / "prompts.parquet")
    (RAW / "models.json").write_text(json.dumps({"models": MODELS, "short": SHORT, "prices": PRICES}, indent=1))
    print(pd.Series(fam).value_counts().to_string())
    print(pd.Series(t["split"]).value_counts().to_string())


if __name__ == "__main__":
    main()
