"""Text classification tasks the quantum classifier can be trained on.

Each task provides labelled train/test text and names for its two classes.
Embeddings are computed once and cached in data/<task>_<split>_emb.npz.

  sentiment : Rotten Tomatoes movie-review sentences (Pang & Lee, 2005),
              positive vs negative. 8,530 train / 1,066 test.
  cve       : NVD CVE descriptions published Jan-Apr 2024, HIGH/CRITICAL
              (CVSS v3.1 >= 7.0) vs LOW/MEDIUM. Time-based 80/20 split.
"""
import os
import urllib.request

import numpy as np

from core.text_encoder import embed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

ROTTEN_URL = ("https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes"
              "/resolve/main/{split}.parquet")


def _rotten_tomatoes(split):
    import pandas as pd
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"rotten_tomatoes_{split}.parquet")
    if not os.path.exists(path):
        print(f"Downloading Rotten Tomatoes {split} split -> {path}")
        urllib.request.urlretrieve(ROTTEN_URL.format(split=split), path)
    df = pd.read_parquet(path)
    return df["text"].tolist(), df["label"].to_numpy().astype(int)


def _cve(split):
    from core.cve_data import load_cve_split
    return load_cve_split(split)


TASKS = {
    "sentiment": {
        "load": _rotten_tomatoes,
        "labels": ("Negative", "Positive"),
        "description": "Movie-review sentiment (positive vs negative)",
        "examples": ["What a wonderful, moving film.", "A tedious, lifeless mess."],
    },
    "cve": {
        "load": _cve,
        "labels": ("Low/medium severity", "High/critical severity"),
        "description": "CVE severity from description (HIGH/CRITICAL vs LOW/MEDIUM)",
        "examples": [
            "A remote unauthenticated attacker can execute arbitrary code via a crafted HTTP request to the admin API.",
            "A reflected cross-site scripting issue in the search page allows an attacker to inject script if a user clicks a crafted link.",
        ],
    },
}


def load_texts(task, split):
    return TASKS[task]["load"](split)


def load_embeddings(task, split, limit=None, seed=0):
    """(embeddings, labels) for a task's split; `limit` takes a balanced random subset."""
    os.makedirs(DATA_DIR, exist_ok=True)
    cache = os.path.join(DATA_DIR, f"{task}_{split}_emb.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        X, y = d["X"], d["y"]
    else:
        texts, y = load_texts(task, split)
        print(f"Embedding {len(texts)} {task} {split} texts (one-off, cached to {cache})...")
        X = embed(texts)
        np.savez_compressed(cache, X=X, y=y)

    if limit is not None and limit < len(y):
        rng = np.random.default_rng(seed)
        idx = np.concatenate([rng.permutation(np.flatnonzero(y == c))[: limit // 2] for c in (0, 1)])
        idx = rng.permutation(idx)  # shuffle once; X and y share the same order
        X, y = X[idx], y[idx]
    return X, y
