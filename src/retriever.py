import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi


def load_indexes():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    faiss_index = FAISS.load_local(
        "indexes/faiss",
        embeddings,
        allow_dangerous_deserialization=True
    )
    with open("indexes/bm25.pkl", "rb") as f:
        bm25_index = pickle.load(f)
    with open("indexes/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    
    return faiss_index, bm25_index, chunks


def reciprocal_rank_fusion(faiss_results, bm25_results, k=60):
    scores = {}
    
    for rank, doc in enumerate(faiss_results):
        doc_id = doc.page_content
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    
    for rank, doc in enumerate(bm25_results):
        doc_id = doc.page_content
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    
    sorted_ids = sorted(scores, key=scores.get, reverse=True)
    return sorted_ids


def hybrid_search(query: str, faiss_index, bm25_index, chunks, top_k=5):
    # FAISS search
    faiss_results = faiss_index.similarity_search(query, k=20)
    
    # BM25 search
    tokenized_query = query.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    top_bm25_indices = sorted(range(len(bm25_scores)), 
                               key=lambda i: bm25_scores[i], 
                               reverse=True)[:20]
    bm25_results = [chunks[i] for i in top_bm25_indices]
    
    # Merge with RRF
    merged = reciprocal_rank_fusion(faiss_results, bm25_results)
    
    # Return top_k chunks
    chunk_map = {chunk.page_content: chunk for chunk in chunks}
    return [chunk_map[doc_id] for doc_id in merged[:top_k] if doc_id in chunk_map]


if __name__ == "__main__":
    print("Loading indexes...")
    faiss_index, bm25_index, chunks = load_indexes()
    print("Indexes loaded.")
    
    query = "How do I use FAISS with LangChain?"
    print(f"\nQuery: {query}")
    results = hybrid_search(query, faiss_index, bm25_index, chunks)
    
    for i, chunk in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(chunk.page_content[:200])