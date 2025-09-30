from pathlib import Path
import pickle
import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
LIBRETTI_DIR = Path("libretti")

INDEX_PATH = DATA_DIR / "libretti.index"
CHUNKS_PATH = DATA_DIR / "libretti.pkl"


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """
    Split text into overlapping chunks of approx. `chunk_size` characters.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_chunks():
    """Load libretti and split them into overlapping chunks."""
    chunks = []
    for file in LIBRETTI_DIR.glob("*.txt"):
        text = file.read_text(encoding="utf-8")
        pieces = chunk_text(text, chunk_size=500, overlap=50)
        for i, piece in enumerate(pieces):
            chunks.append(
                {"id": f"{file.stem}-{i}", "source": file.name, "text": piece}
            )
    return chunks


def build_index(chunks):
    """Embed chunks and build a FAISS index."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in chunks]

    if not texts:
        raise ValueError("No texts found to embed.")

    embeddings = model.encode(texts, convert_to_numpy=True)

    print(f"Embedding shape: {embeddings.shape}")  # Debug

    # Ensure embeddings is 2D
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def main():
    DATA_DIR.mkdir(exist_ok=True)

    print("🔹 Building chunks...")
    chunks = build_chunks()
    print(
        f"  -> {len(chunks)} chunks created from {len(list(LIBRETTI_DIR.glob('*.txt')))} libretti."
    )

    print("🔹 Building FAISS index...")
    index = build_index(chunks)
    print("  -> Index built.")

    print("🔹 Saving index and chunks to disk...")
    faiss.write_index(index, str(INDEX_PATH))
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"✅ Done. Saved index to {INDEX_PATH} and chunks to {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
