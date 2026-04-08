import sys
sys.path.append("src")

from retriever import load_indexes, hybrid_search, faiss_only_search
from generator import generate_answer
from evaluator import run_eval

if __name__ == "__main__":
    print("Loading indexes...")
    faiss_index, bm25_index, chunks = load_indexes()
    
    print("\n==== BASELINE: FAISS ONLY ====")
    baseline_retriever = lambda query: faiss_only_search(query, faiss_index, chunks)
    baseline_llm = lambda query, chunks: generate_answer(query, chunks)
    run_eval(baseline_retriever, baseline_llm)

    print("\n==== HYBRID: BM25 + FAISS + RRF ====")
    hybrid_retriever = lambda query: hybrid_search(query, faiss_index, bm25_index, chunks)
    hybrid_llm = lambda query, chunks: generate_answer(query, chunks)
    run_eval(hybrid_retriever, hybrid_llm)