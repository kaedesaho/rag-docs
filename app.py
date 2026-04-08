import sys
sys.path.append("src")

import streamlit as st
from retriever import load_indexes, hybrid_search
from generator import generate_answer
from huggingface_hub import hf_hub_download, snapshot_download
import os

st.title("RAG Docs QA")
st.caption("Ask questions about LangChain and LlamaIndex documentation")

def download_indexes():
    if not os.path.exists("indexes/faiss"):
        print("Downloading indexes from HF Hub...")
        snapshot_download(
            repo_id="kaedesaho/rag-docs-indexes",
            repo_type="dataset",
            local_dir="indexes/"
        )

@st.cache_resource
def load():
    download_indexes()
    return load_indexes()

faiss_index, bm25_index, chunks = load()

query = st.text_input("Ask a question:")

if query:
    with st.spinner("Searching..."):
        results = hybrid_search(query, faiss_index, bm25_index, chunks)
        answer = generate_answer(query, results)
    
    st.markdown("### Answer")
    st.write(answer)
    
    with st.expander("View sources"):
        for i, chunk in enumerate(results):
            st.markdown(f"**Source {i+1}:** `{chunk.metadata.get('source', 'unknown')}`")
            st.text(chunk.page_content[:300])
            st.divider()
