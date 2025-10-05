import textwrap
import subprocess
from retriever import load_artifacts, build_embedder, dense_search


def ask_ollama(prompt: str, model: str = "mistral") -> str:
    """
    Sends a prompt to the local Ollama model and returns the generated output.
    """
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        capture_output=True
    )
    return result.stdout.decode("utf-8", errors="ignore").strip()


def build_prompt(query: str, retrieved_chunks: list[str]) -> str:
    """
    Builds a context-aware prompt for the LLM, embedding the retrieved text.
    """
    context = "\n\n---\n\n".join(retrieved_chunks)
    prompt = textwrap.dedent(f"""
    You are an assistant that answers questions about opera libretti.
    Use only the information in the provided context. Be concise and factual.

    Context:
    {context}

    Question: {query}

    Answer:
    """)
    return prompt.strip()


def rag_query(query: str, model: str = "mistral", k: int = 3) -> str:
    """
    Runs the full retrieval-augmented generation (RAG) pipeline.
    """
    # Load index + embeddings
    index, chunks = load_artifacts()
    embedder = build_embedder()

    # Retrieve top-k relevant chunks
    results = dense_search(index, chunks, embedder, query, k)
    if not results:
        return "No relevant text found in the corpus."

    # Prepare context for the LLM
    context_chunks = [r["text"] for r in results]
    prompt = build_prompt(query, context_chunks)

    # Generate the answer
    answer = ask_ollama(prompt, model=model)
    return answer


if __name__ == "__main__":
    print("Opera Libretti RAG – Ask your question (empty line to quit)\n")
    while True:
        query = input("🎭 > ").strip()
        if not query:
            break
        print("\n🧠 Thinking...\n")
        try:
            answer = rag_query(query)
            print(answer, "\n")
        except Exception as e:
            print(f"Error: {e}\n")
