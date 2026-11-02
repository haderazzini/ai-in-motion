"""AI in Motion · E001 — How do embeddings capture meaning?

Reproduces every number shown in the video, and a few more for the next episodes:
  0. The toy table of neighbouring words (illustrative counts) and its dot products: 12 and 0.
  1. One-hot vectors: every pair of distinct words has cosine similarity 0.
  2. GloVe (static word vectors, 50-d, Wikipedia + Gigaword): 400,000 words; cat in 50 numbers (some negative);
     cosine similarity cat/dog 0.92, cat/car 0.36 — and the raw dot products, which are not the same thing.
  3. all-MiniLM-L6-v2 (modern sentence encoder, 384-d): the same geometry, used for semantic search.

Writes results.json (consumed by the video) and prints a readable report.
usage: uv run main.py
"""
import gzip
import json
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"
GLOVE_URL = "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz"
GLOVE_FILE = CACHE / "glove-wiki-gigaword-50.gz"
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"

WORDS = ["cat", "dog", "car"]
PAIRS = [("cat", "dog"), ("cat", "car"), ("dog", "car"), ("hot", "cold"), ("good", "bad")]
DOCS = [
    "To change your password, open Settings and choose Security.",
    "Our office is closed on public holidays.",
    "Cats and dogs are the most common household pets.",
    "Shipping usually takes three to five business days.",
]
QUERY = "I forgot my login credentials"


def cos(u: np.ndarray, v: np.ndarray) -> float:
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


def load_glove() -> dict[str, np.ndarray]:
    CACHE.mkdir(exist_ok=True)
    if not GLOVE_FILE.exists():
        print(f"downloading {GLOVE_URL} (~66 MB, once)")
        urllib.request.urlretrieve(GLOVE_URL, GLOVE_FILE)
    vecs = {}
    with gzip.open(GLOVE_FILE, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) == 51:  # skip the word2vec-format header line
                vecs[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
    return vecs


def nearest(vecs: dict[str, np.ndarray], target: np.ndarray, exclude: set[str], k: int = 3) -> list[tuple[str, float]]:
    words = list(vecs)
    M = np.stack([vecs[w] for w in words])
    sims = M @ target / (np.linalg.norm(M, axis=1) * np.linalg.norm(target))
    out = []
    for i in np.argsort(-sims):
        if words[i] not in exclude:
            out.append((words[i], round(float(sims[i]), 3)))
        if len(out) == k:
            return out
    return out


def pca2(X: np.ndarray) -> tuple[np.ndarray, list[float]]:
    Xc = X - X.mean(axis=0)
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    explained = (s**2 / (s**2).sum()).tolist()
    return Xc @ vt[:2].T, [round(e, 3) for e in explained[:2]]


def main() -> None:
    results: dict = {}

    # 0. Toy table (illustrative, not counted from a corpus): how often each neighbour shows up
    cols = ["chased", "fed", "fuel"]
    toy = {"cat": np.array([3, 2, 0]), "dog": np.array([2, 3, 0]), "car": np.array([0, 0, 4])}
    results["toy"] = {
        "columns": cols,
        "rows": {w: v.tolist() for w, v in toy.items()},
        "dot": {"cat-dog": int(toy["cat"] @ toy["dog"]), "cat-car": int(toy["cat"] @ toy["car"])},
    }

    # 1. One-hot: orthogonal by construction
    V = sorted({w for p in PAIRS for w in p} | set(WORDS))
    onehot = {w: np.eye(len(V))[i] for i, w in enumerate(V)}
    results["onehot"] = {f"{a}-{b}": cos(onehot[a], onehot[b]) for a, b in PAIRS}

    # 2. GloVe static vectors
    glove = load_glove()
    results["glove"] = {
        "dim": 50,
        "vocab": len(glove),
        "cosine": {f"{a}-{b}": round(cos(glove[a], glove[b]), 3) for a, b in PAIRS},
        "nearest_cat": nearest(glove, glove["cat"], {"cat"}),
        "cat": [round(float(x), 2) for x in glove["cat"]],
        "cat_negative_values": int((glove["cat"] < 0).sum()),
        # the raw dot product depends on the vectors' lengths; cosine divides them out
        "dot": {f"{a}-{b}": round(float(glove[a] @ glove[b]), 2) for a, b in [("cat", "dog"), ("cat", "car")]},
        "norm": {w: round(float(np.linalg.norm(glove[w])), 2) for w in WORDS},
    }
    target = glove["king"] - glove["man"] + glove["woman"]
    results["glove"]["analogy_king-man+woman"] = {
        "excluding_inputs": nearest(glove, target, {"king", "man", "woman"}),
        "including_inputs": nearest(glove, target, set()),
    }
    pts, var = pca2(np.stack([glove[w] for w in WORDS + ["hot", "cold"]]))
    results["glove"]["pca2"] = {
        "words": WORDS + ["hot", "cold"],
        "points": [[round(float(x), 3) for x in p] for p in pts],
        "explained_variance": var,
    }

    # 3. Modern sentence encoder + semantic search
    from fastembed import TextEmbedding

    enc = TextEmbedding(ENCODER, cache_dir=str(CACHE))
    emb = lambda texts: np.stack(list(enc.embed(texts)))
    words = sorted({w for p in PAIRS for w in p})
    W = dict(zip(words, emb(words)))
    D, q = emb(DOCS), emb([QUERY])[0]
    ranked = sorted(((round(cos(q, d), 3) + 0.0, doc) for d, doc in zip(D, DOCS)), reverse=True)
    results["minilm"] = {
        "model": ENCODER,
        "dim": int(D.shape[1]),
        "cosine": {f"{a}-{b}": round(cos(W[a], W[b]), 3) for a, b in PAIRS},
        "search": {"query": QUERY, "ranked": ranked},
    }

    (HERE / "results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
