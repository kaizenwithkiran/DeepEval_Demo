"""
src/rag_pipeline.py — the eval-facing wrapper around the chatbot.

The Streamlit app only needs (answer, sources). An evaluation harness needs a third
thing the app throws away: the retrieved context. Faithfulness and contextual relevancy
are scored *against* that context, so without it those metrics cannot be computed at all.

RagPipeline returns all three legs of the RAG triad for one query:

    {"query": ..., "context": [chunk, ...], "answer": ...}

It deliberately reuses the same retriever, the same prompt and the same model as
app.py, including the greeting and "couldn't find that" short-circuits. Evaluating a
reimplementation of the app would measure the reimplementation, not the app.

    python -m src.rag_pipeline
"""

from dataclasses import dataclass, field
from pathlib import Path
import time

from langchain_community.callbacks import get_openai_callback

from . import config
from .rag_chain import (
    GREETING_PATTERN,
    GREETING_REPLY,
    NOT_FOUND_MESSAGE,
    _format_docs,
    build_generator_chain,
    generate_stream,
    get_retriever,
)


@dataclass
class RagPipeline:
    """The whole chatbot as one callable, with the retrieved context exposed."""

    k: int = field(default_factory=lambda: config.RETRIEVER_K)

    def __post_init__(self):
        # build once — each construction re-opens Chroma and re-reads the embeddings model
        self.retriever = get_retriever(self.k)
        self.chain = build_generator_chain()

    def retrieve(self, query: str):
        """Documents only — what the retriever alone produced."""
        return self.retriever.invoke(query)

    def invoke(self, query: str) -> dict:
        """Run the same RAG path as the chatbot and return data needed for evaluation."""
        start_time = time.perf_counter()

        # Preserve the app's greeting shortcut: no retrieval or model call is needed.
        if GREETING_PATTERN.match(query):
            return {
                "query": query,
                "context": [],
                "answer": GREETING_REPLY,
                "sources": [],
                "token_cost": 0.0,
                "latency": round(time.perf_counter() - start_time, 3),
            }

        # 1. RETRIEVE the same document chunks used by the chatbot.
        docs = self.retrieve(query)
        context = [doc.page_content for doc in docs]

        # 2. GENERATE with the same prompt and model, while measuring API usage.
        with get_openai_callback() as cost_counter:
            answer = self.chain.invoke({
                "context": _format_docs(docs),
                "question": query
            })

        token_cost = round(float(cost_counter.total_cost), 6)
        latency = round(time.perf_counter() - start_time, 3)

        # 3. Preserve the app's not-found/abstention behavior.
        if NOT_FOUND_MESSAGE in answer:
            return {
                "query": query,
                "context": context,
                "answer": NOT_FOUND_MESSAGE,
                "sources": [],
                "token_cost": token_cost,
                "latency": latency,
            }

        sources = sorted({Path(d.metadata.get("source", "unknown")).name for d in docs})
        return {
            "query": query,
            "context": context,
            "answer": answer,
            "sources": sources,
            "token_cost": token_cost,
            "latency": latency,
        }
    def invoke_stream(self, query: str):
        """Retrieve, then stream the answer. Returns (context, token_iterator)."""
        docs = self.retrieve(query)
        context = [doc.page_content for doc in docs]
        return context, generate_stream(query, [_format_docs(docs)])

    def abstained(self, answer: str) -> bool:
        return NOT_FOUND_MESSAGE in answer


if __name__ == "__main__":
    rag = RagPipeline()
    result = rag.invoke("How much does the Playwright course cost and how long is it?")
    print("QUERY :", result["query"])
    print("ANSWER:", result["answer"])
    print("SOURCES:", result["sources"])
    print(f"\n{len(result['context'])} CONTEXT CHUNKS:")
    for i, chunk in enumerate(result["context"]):
        print(f"  [{i}] {chunk[:110].replace(chr(10), ' ')}...")
