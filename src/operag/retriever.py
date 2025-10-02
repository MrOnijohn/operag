# src/operag/retriever.py
from pathlib import Path
import argparse
import pickle
import textwrap
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

DATA_DIR = Path("data")
INDEX_PATH = DATA_DIR / "libretti.index"
CHUNKS_PATH = DATA_DIR / "libretti.pkl"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # must match prepare_data.py


# ----------------------
# Load artifacts
# ---------------------
def load_artifacts():
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "Missing artifacts. Expected data/libretti.index and data/libretti.pkl. "
            "Run prepare_data.py first."
        )
    index = faiss.read_index(str(INDEX_PATH))
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def build_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)


# -----------------
# Dense retrieval
# -----------------
def embed_query(embedder, query: str) -> np.ndarray:
    if not query.strip():
        raise ValueError("Empty query.")
    vec = embedder.encode([query], convert_to_numpy=True)  # shape (1, d)
    return vec


def dense_search(index, chunks, embedder, query, k=3):
    q_vec = embed_query(embedder, query)
    D, I = index.search(q_vec, k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        doc = chunks[idx]
        results.append(
            {
                "id": doc["id"],
                "source": doc["source"],
                "text": doc["text"],
                "distance": float(dist),
            }
        )
    return results


# ----------------
# Hybrid retrieval
# ----------------
def hybrid_search(index, chunks, embedder, query, k=3):
    dense_results = dense_search(index, chunks, embedder, query, k)
    bm25 = build_bm25(chunks)
    bm25_results = bm25_search(bm25, chunks, query)

    # Merge results (naive: just interleave, remove duplicates)
    merged = []
    seen = set()
    for r in bm25_results + dense_results:
        if r["id"] not in seen:
            merged.append(r)
            seen.add(r["id"])
    return merged[:k]


# -------------
# BM25 retrieval
# --------------
def build_bm25(chunks):
    tokenized = [c["text"].split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    return bm25


def bm25_search(bm25, chunks, query, k=3):
    query_tokens = query.split()
    scores = bm25.get_scores(query_tokens)
    top_idx = np.argsort(scores)[::-1][:k]
    results = []
    for i in top_idx:
        results.append(
            {
                "id": chunks[i]["id"],
                "source": chunks[i]["source"],
                "text": chunks[i]["text"],
                "score": float(scores[i]),
            }
        )
    return results


# --------------------
# Pretty print results
#  -------------------
def pretty_print(results, max_chars: int, show_scores: bool):
    wrapper = textwrap.TextWrapper(width=100, replace_whitespace=False)
    for rank, r in enumerate(results, start=1):
        header = f"[{rank}] {r['source']}  (id={r['id']})"
        print(header)
        if show_scores:
            if "distance" in r:
                d = r["distance"]
                sim = 1.0 / (1.0 + d)
                print(f"    dense: distance={d:.4f}  similarity≈{sim:.4f}")
            if "score" in r:
                print(f"    bm25: score={r['score']:.4f}")
        excerpt = r["text"]
        if max_chars and len(excerpt) > max_chars:
            excerpt = excerpt[:max_chars].rstrip() + " …"
        for line in wrapper.fill(excerpt).splitlines():
            print("    " + line)
        print()


# ------------------
# Main CLI
# ------------------
def main():
    parser = argparse.ArgumentParser(
        description="Retrieve libretti chunks using dense, bm25, or hybrid."
    )
    parser.add_argument("query", type=str, help="Your question or search query.")
    parser.add_argument(
        "--mode",
        choices=["dense", "bm25", "hybrid"],
        default="dense",
        help="Retreival method.",
    )
    parser.add_argument("-k", type=int, default=3, help="Number of chunks to return.")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=500,
        help="Max characters of each chunk to display (0 for full).",
    )
    parser.add_argument(
        "--no-scores", action="store_true", help="Hide distance/similarity scores."
    )
    args = parser.parse_args()

    index, chunks = load_artifacts()
    embedder = build_embedder()

    if args.mode == "dense":
        results = dense_search(index, chunks, embedder, args.query, args.k)
    elif args.mode == "bm25":
        bm25 = build_bm25(chunks)
        results = bm25_search(bm25, chunks, args.query, args.k)
    else:  # Hybrid
        results = hybrid_search(index, chunks, embedder, args.query, args.k)

    if not results:
        print("No results.")
        return

    pretty_print(
        results, max_chars=max(0, args.max_chars), show_scores=not args.no_scores
    )


if __name__ == "__main__":
    main()
