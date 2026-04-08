import os
import pickle
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi


LANGCHAIN_DOCS = [
    Path("data/langchain-docs/src/oss/langchain"),
    Path("data/langchain-docs/src/oss/langgraph"),
    Path("data/langchain-docs/src/oss/concepts"),
    Path("data/langchain-docs/src/oss/python"),
]
LLAMA_DOCS = [
    Path("data/llama_index/docs/src/content/docs/framework"),
    Path("data/llama_index/docs/examples"),
    Path("data/llama_index/docs/api_reference/api_reference"),
]

def load_docs(path):
    loader = DirectoryLoader(
        str(path),
        glob="**/*.md*",
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True},
        recursive=True,
        show_progress=True,
        exclude=["**/javascript/**", "**/changelog*", "**/CHANGELOG*"]
    )
    return loader.load()


def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    return [c for c in chunks if len(c.page_content.strip()) > 100]


def build_faiss_index(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(chunks, embeddings)


def build_bm25_index(chunks):
    tokenized = [chunk.page_content.lower().split() for chunk in chunks]
    return BM25Okapi(tokenized)

def save_indexes(faiss_index, bm25_index, chunks):
    faiss_index.save_local("indexes/faiss")
    
    with open("indexes/bm25.pkl", "wb") as f:
        pickle.dump(bm25_index, f)
    
    with open("indexes/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)


if __name__ == "__main__":
    print("Loading LangChain docs...")
    lc_docs = []
    for path in LANGCHAIN_DOCS:
        lc_docs += load_docs(path)
    print(f"Loaded {len(lc_docs)} LangChain documents")

    print("Loading LlamaIndex docs...")
    li_docs = []
    for path in LLAMA_DOCS:
        li_docs += load_docs(path)
    print(f"Loaded {len(li_docs)} LlamaIndex documents")

    all_docs = lc_docs + li_docs
    print(f"Total: {len(all_docs)} documents")

    print("Chunking...")
    chunks = chunk_docs(all_docs)
    print(f"Total chunks: {len(chunks)}")

    print("Building FAISS index...")
    faiss_index = build_faiss_index(chunks)

    print("Building BM25 index...")
    bm25_index = build_bm25_index(chunks)

    print("Saving indexes...")
    save_indexes(faiss_index, bm25_index, chunks)
    print("Saved to indexes/")