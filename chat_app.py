# chat_app.py
# Run from your RAG project root: streamlit run chat_app.py

import csv
import json
from pathlib import Path

import streamlit as st

from src import config
from src.ingest import build_index
from src.rag_pipeline import RagPipeline

LOG_FILE = config.BASE_DIR / "chat_log.csv"
CSV_COLUMNS = ["input", "output", "context", "token_cost", "latency"]


def save_chat(question, answer, context, token_cost, latency):
    """Append the raw answer and retrieved text chunks to the CSV."""
    file_needs_header = not LOG_FILE.exists() or LOG_FILE.stat().st_size == 0
    with LOG_FILE.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        if file_needs_header:
            writer.writeheader()
        writer.writerow({
            "input": question,
            "output": answer,
            # Keep all retrieved chunks together in one CSV cell.
            "context": json.dumps(context, ensure_ascii=False),
            "token_cost": token_cost,
            "latency": latency,
        })


st.set_page_config(page_title="RAG Notes Chatbot - Evaluation Mode", page_icon="💬")
st.title("RAG Notes Chatbot")
st.caption("Evaluation mode: each successful chatbot response is automatically saved to chat_log.csv.")

if not config.OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set. Add it to your .env file and restart the app.")
    st.stop()


@st.cache_resource
def ensure_index():
    if not config.CHROMA_DIR.exists():
        build_index()
    return True


@st.cache_resource
def get_rag_pipeline():
    return RagPipeline()


with st.sidebar:
    st.header("Knowledge base")
    st.caption(f"Notes folder: `{config.DATA_DIR}`")
    st.caption(f"CSV log: `{LOG_FILE}`")

    if st.button("Re-index notes"):
        with st.spinner("Indexing notes..."):
            build_index()
        st.cache_resource.clear()
        st.success("Knowledge base re-indexed.")
        st.rerun()

    if st.button("Clear chat displayed on screen"):
        st.session_state.messages = []
        st.rerun()

    if LOG_FILE.exists():
        with LOG_FILE.open("r", newline="", encoding="utf-8") as f:
            saved_rows = sum(1 for _ in csv.DictReader(f))
        st.metric("Saved conversations", saved_rows)
    else:
        st.info("The CSV file will be created after the first question.")


with st.spinner("Loading knowledge base..."):
    ensure_index()

if not config.CHROMA_DIR.exists():
    st.warning("No index found. Add .md/.txt files to the data folder and re-index.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask a question about your notes..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating an answer..."):
            try:
                pipeline = get_rag_pipeline()
                result = pipeline.invoke(question)

                answer = result["answer"]
                context = result["context"]
                sources = result["sources"]
                token_cost = result["token_cost"]
                latency = result["latency"]

                # Display answer and source names, but log the raw answer separately.
                st.markdown(answer)
                if sources:
                    st.caption("Sources: " + ", ".join(sources))
                st.caption(
                    f"Latency: {latency:.3f} seconds | "
                    f"Estimated API cost: ${token_cost:.6f}"
                )

                save_chat(question, answer, context, token_cost, latency)
                st.success(f"Saved automatically to {LOG_FILE.name}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })
            except Exception as exc:
                st.error(f"Chatbot request failed: {exc}")
                st.exception(exc)
