# src/operag/retriever.py
from pathlib import Path
import argparse
import pickle
import textwrap
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
INDEX_PATH = DATA_DIR / "libretti.index"
CHUNKS_PATH = DATA_DIR / "libretti.pkl"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # must match prepare_data.py


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


def embed_query(embedder, query: str) -> np.ndarray:
    if not query.strip():
        raise ValueError("Empty query.")
    vec = embedder.encode([query], convert_to_numpy=True)  # shape (1, d)
    return vec


def search(index, embed_vec: np.ndarray, k: int):
    # FAISS expects (n_queries, dim)
    D, I = index.search(embed_vec, k)  # D: distances (smaller is better), I: indices
    return D[0], I[0]


def pretty_print(results, max_chars: int, show_scores: bool):
    wrapper = textwrap.TextWrapper(width=100, replace_whitespace=False)
    for rank, r in enumerate(results, start=1):
        header = f"[{rank}] {r['source']}  (id={r['id']})"
        print(header)
        if show_scores:
            # L2 distance (d) -> monotone "similarity" for readability
            d = r["distance"]
            sim = 1.0 / (1.0 + d)
            print(f"    distance(L2)={d:.4f}  similarity≈{sim:.4f}")
        excerpt = r["text"]
        if max_chars and len(excerpt) > max_chars:
            excerpt = excerpt[:max_chars].rstrip() + " …"
        for line in wrapper.fill(excerpt).splitlines():
            print("    " + line)
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve top-k libretti chunks using FAISS (L2)."
    )
    parser.add_argument("query", type=str, help="Your question or search query.")
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

    # Sanity check on dimensions (helps catch model mismatches)
    dim_index = index.d  # FAISS index dimension
    dim_model = embedder.get_sentence_embedding_dimension()
    if dim_index != dim_model:
        print(
            f"Warning: index dim={dim_index} but embedder dim={dim_model}. "
            "This usually means you built the index with a different embedding model."
        )

    q_vec = embed_query(embedder, args.query)
    D, I = search(index, q_vec, args.k)

    results = []
    for dist, idx in zip(D, I):
        if idx == -1:
            continue  # no result for this slot
        doc = chunks[idx]
        results.append(
            {
                "id": doc["id"],
                "source": doc["source"],
                "text": doc["text"],
                "distance": float(dist),
            }
        )

    if not results:
        print("No results.")
        return

    pretty_print(
        results, max_chars=max(0, args.max_chars), show_scores=not args.no_scores
    )


if __name__ == "__main__":
    main()
