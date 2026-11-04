"""AI in Motion · E002 — Why measure the angle?

Reproduces every number the episode shows (`on_screen`, script C) and the research candidates:
  0. The E001 toy table (illustrative counts) plus a row for "it", a word found in every context: by raw dot
     product "it" beats "dog" as the closest word to "cat" (50 vs 12); by cosine "dog" wins (0.92 vs 0.80).
     Also: counted in twice as much text, cat·dog goes 12 → 48 while the cosine stays 0.92.
  1. GloVe 50-d: about half of all coordinates are negative (learned numbers, not counts).
  2. GloVe 50-d: the raw dot product rewards long arrows — dog·you beats dog·cat, pizza·plumaged beats
     pizza·sandwich; dividing by the lengths (cosine) puts the right word back on top.
  3. GloVe 50-d: a negative cosine is not "opposite meaning" (cat's most negative words are unrelated names), while
     hot/cold, real opposites, are close.
  4. all-MiniLM-L6-v2: modern encoders output length-1 vectors, so dot product = cosine and distance gives the same
     ranking (|u - v|^2 = 2 - 2 cos).

Writes results.json and prints it.
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
TOP = 50_000  # GloVe lists words by frequency; the top 50k keeps the candidate words recognisable


def cos(u: np.ndarray, v: np.ndarray) -> float:
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


def r(x: float, d: int = 2) -> float:
    return round(float(x), d) + 0.0


def load_glove() -> tuple[list[str], np.ndarray]:
    CACHE.mkdir(exist_ok=True)
    if not GLOVE_FILE.exists():
        print(f"downloading {GLOVE_URL} (~66 MB, once)")
        urllib.request.urlretrieve(GLOVE_URL, GLOVE_FILE)
    words, vecs = [], []
    with gzip.open(GLOVE_FILE, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) == 51:  # skip the word2vec-format header line
                words.append(parts[0])
                vecs.append(np.asarray(parts[1:], dtype=np.float32))
    return words, np.stack(vecs)


def main() -> None:
    results: dict = {}

    # 0. Toy table from E001 (illustrative): columns chased, fed, fuel; "it" shows up in every context
    toy = {"cat": np.array([3, 2, 0]), "dog": np.array([2, 3, 0]), "car": np.array([0, 0, 4]),
           "it": np.array([10, 10, 10])}
    cat2 = 2 * toy["cat"]  # the same word counted in twice as much text
    results["toy"] = {
        "rows": {w: v.tolist() for w, v in toy.items()},
        "dot": {f"cat-{w}": int(toy["cat"] @ toy[w]) for w in ["dog", "car", "it"]},
        "length": {w: r(np.linalg.norm(v), 3) for w, v in toy.items()},
        "cosine": {f"cat-{w}": r(cos(toy["cat"], toy[w]), 3) for w in ["dog", "car", "it"]},
        # both words counted in twice as much text: the sum grows 4x, the angle does not move
        "both_twice_the_text": {
            "rows": {"cat": cat2.tolist(), "dog": (2 * toy["dog"]).tolist()},
            "dot": int(cat2 @ (2 * toy["dog"])),
            "cosine": r(cos(cat2, 2 * toy["dog"]), 3),
        },
        # script C (shots 01-08): every pair scored, each row divided by its length (unit rows)
        "dot_pairs": {f"{a}-{b}": int(toy[a] @ toy[b]) for a, b in
                      [("cat", "dog"), ("cat", "car"), ("cat", "it"), ("dog", "it"), ("car", "it")]},
        "cosine_pairs": {f"{a}-{b}": r(cos(toy[a], toy[b]), 3) for a, b in
                         [("cat", "dog"), ("cat", "car"), ("cat", "it"), ("dog", "it"), ("car", "it")]},
        "unit_rows": {w: [r(x) for x in v / np.linalg.norm(v)] for w, v in toy.items()},
        "cat_twice_the_text": {
            "row": cat2.tolist(),
            "cosine_with_cat": r(cos(toy["cat"], cat2), 3),
            "distance_to_cat": r(np.linalg.norm(toy["cat"] - cat2), 3),
            "distance_cat_dog": r(np.linalg.norm(toy["cat"] - toy["dog"]), 3),
        },
    }

    words, M = load_glove()
    idx = {w: i for i, w in enumerate(words)}
    norm = np.linalg.norm(M, axis=1)
    g = lambda w: M[idx[w]]

    # 1. Negative coordinates
    results["glove_negative"] = {
        "share_of_all_coordinates": r((M < 0).mean(), 3),
        "cat_negative_of_50": int((g("cat") < 0).sum()),
        "cat_first_8": [r(x) for x in g("cat")[:8]],
    }

    # 1b. Cosine of real cat and dog, step by step (shot 09): each row ÷ its length, multiply number by number, add
    uc, ud = g("cat") / norm[idx["cat"]], g("dog") / norm[idx["dog"]]
    results["glove_cat_dog_steps"] = {
        "length": {"cat": r(norm[idx["cat"]]), "dog": r(norm[idx["dog"]])},
        "cat": [r(x) for x in g("cat")], "dog": [r(x) for x in g("dog")],
        "cat_unit": [r(x, 3) for x in uc], "dog_unit": [r(x, 3) for x in ud],
        "products": [r(x, 4) for x in uc * ud],
        "sum": r(uc @ ud, 3),
    }

    # 2. Raw dot product vs cosine
    def ranking(w: str, score: np.ndarray, k: int = 5) -> list[tuple[str, float, float]]:
        s = score.copy()
        s[idx[w]] = -np.inf
        s[TOP:] = -np.inf
        return [(words[i], r(s[i], 3), r(norm[i])) for i in np.argsort(-s)[:k]]  # (word, score, length)

    pairs = [("dog", "cat"), ("dog", "you"), ("pizza", "sandwich"), ("pizza", "plumaged"), ("cat", "car")]
    results["glove_dot_vs_cosine"] = {
        "pairs": {
            f"{a}-{b}": {"dot": r(g(a) @ g(b)), "cosine": r(cos(g(a), g(b)), 3), "distance": r(np.linalg.norm(g(a) - g(b)))}
            for a, b in pairs
        },
        "length": {w: r(norm[idx[w]]) for w in ["cat", "dog", "you", "pizza", "sandwich", "plumaged"]},
        "top_by_dot": {w: ranking(w, M @ g(w)) for w in ["dog", "pizza"]},
        "top_by_cosine": {w: ranking(w, M @ g(w) / (norm * norm[idx[w]])) for w in ["dog", "pizza"]},
        "top_by_distance": {w: ranking(w, -np.linalg.norm(M - g(w), axis=1)) for w in ["dog", "pizza"]},
    }

    # 3. Negative cosine is not "opposite meaning"
    c = M[:TOP] @ g("cat") / (norm[:TOP] * norm[idx["cat"]])
    results["glove_negative_cosine"] = {
        "most_negative_to_cat": [(words[i], r(c[i], 3)) for i in np.argsort(c)[:5]],
        "hot-cold": r(cos(g("hot"), g("cold")), 3),
    }

    # 4. Modern encoder: unit-length vectors
    from fastembed import TextEmbedding

    enc = TextEmbedding(ENCODER, cache_dir=str(CACHE))
    E = dict(zip(["cat", "dog", "car"], np.stack(list(enc.embed(["cat", "dog", "car"])))))
    results["minilm"] = {
        "model": ENCODER,
        "length": {w: r(np.linalg.norm(v), 3) for w, v in E.items()},
        "cat-dog": {"dot": r(E["cat"] @ E["dog"], 3), "cosine": r(cos(E["cat"], E["dog"]), 3)},
    }

    # The numbers the episode shows, in screen order (validation.md); everything else above supports research.md
    t, gp = results["toy"], results["glove_dot_vs_cosine"]["pairs"]
    # script C (approved 2026-11-04), shot by shot
    st = results["glove_cat_dog_steps"]
    results["on_screen"] = {
        "01_dot_cat_dog_car": {k: t["dot_pairs"][k] for k in ["cat-dog", "cat-car"]},
        "03_dot_with_it": {k: t["dot_pairs"][k] for k in ["cat-it", "dog-it", "car-it"]},
        "05_length": {w: r(t["length"][w]) for w in ["cat", "dog", "car", "it"]},
        "05_unit_rows": t["unit_rows"],
        "08_unit_cat_dot_unit_dog": r(t["cosine_pairs"]["cat-dog"]),
        "09_glove": {"length": st["length"], "cat-dog": r(st["sum"]), "cat-car": r(gp["cat-car"]["cosine"])},
        "10_cat_negative_of_50": results["glove_negative"]["cat_negative_of_50"],
    }

    (HERE / "results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
