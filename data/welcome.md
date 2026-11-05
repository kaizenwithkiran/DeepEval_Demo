# Welcome to your RAG Chatbot

This is a sample note to help you test the chatbot before adding your own files.

## How it works

1. Drop `.md` or `.txt` files into the `data/` folder.
2. Run `python -m src.ingest` (or click "Re-index notes" in the app sidebar).
3. Ask questions in the chat — answers are grounded in your notes, with sources cited.

## Tech stack

- **LLM**: OpenAI (`gpt-4o-mini` by default)
- **Embeddings**: OpenAI (`text-embedding-3-small` by default)
- **Vector store**: Chroma, persisted locally in `chroma_db/`
- **Orchestration**: LangChain
- **UI**: Streamlit

## Try asking

- "What is this project built with?"
- "How do I add new notes?"

You can delete this file once you've added your own notes.
