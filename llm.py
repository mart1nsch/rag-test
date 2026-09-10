"""LLM calls via a local Ollama server running the gemma4:cloud model.

The model itself runs on Ollama's cloud; the local `ollama serve` process
just forwards requests to it, so no weights are loaded locally.
"""

from functools import lru_cache

from ollama import Client

MODEL = "gemma4:cloud"


@lru_cache(maxsize=1)
def _client() -> Client:
    return Client(host="http://localhost:11434")


def ask(question: str, context_chunks: list[str]) -> str:
    """Answer a question grounded in the given retrieved context chunks."""
    context = "\n\n".join(context_chunks)
    prompt = (
        "Answer the question using only the context below. "
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    response = _client().chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]
