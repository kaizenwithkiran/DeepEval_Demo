import streamlit as st

from src import config
from src.ingest import build_index
from src.rag_chain import build_chain

st.set_page_config(page_title="RAG Notes Chatbot", page_icon="💬")
st.title("💬 RAG Notes Chatbot")
st.caption("Ask questions about the notes in your data folder.")

if not config.OPENAI_API_KEY:
    st.error(
        "OPENAI_API_KEY is not set. Copy `.env.example` to `.env`, add your key "
        "(or add it in Streamlit Cloud's app secrets), and restart the app."
    )
    st.stop()


@st.cache_resource
def ensure_index():
    if not config.CHROMA_DIR.exists():
        build_index()
    return True


with st.sidebar:
    st.header("Knowledge base")
    st.caption(f"Notes folder: `{config.DATA_DIR}`")
    st.caption("Drop `.md` or `.txt` files there, then re-index.")
    if st.button("🔁 Re-index notes"):
        with st.spinner("Indexing notes..."):
            build_index()
        st.cache_resource.clear()
        st.success("Re-indexed. Ask away!")

    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()


@st.cache_resource
def get_chain():
    return build_chain()


with st.spinner("Loading knowledge base..."):
    ensure_index()

if not config.CHROMA_DIR.exists():
    st.warning(
        "No index found yet. Add `.md`/`.txt` files to the `data/` folder and "
        "click **Re-index notes** in the sidebar to get started."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask a question about your notes..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chain = get_chain()
            answer, sources = chain(question)
            if sources:
                answer += "\n\n**Sources:** " + ", ".join(sources)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
