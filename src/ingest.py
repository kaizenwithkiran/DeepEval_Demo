from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config


def load_documents():
    documents = []
    for pattern in ("**/*.md", "**/*.txt"):
        loader = DirectoryLoader(
            str(config.DATA_DIR),
            glob=pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
        )
        documents.extend(loader.load())
    return documents


def build_index():
    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")

    documents = load_documents()
    if not documents:
        print(f"No .md or .txt files found in {config.DATA_DIR}. Add some notes and re-run.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL, api_key=config.OPENAI_API_KEY)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.CHROMA_DIR),
    )

    print(f"Indexed {len(chunks)} chunks from {len(documents)} documents into {config.CHROMA_DIR}")


if __name__ == "__main__":
    build_index()
