import sys
sys.path.append("src")

from retriever import load_indexes, hybrid_search
from generator import generate_answer
from evaluator import run_eval

if __name__ == "__main__":
    print("Loading indexes...")
    faiss_index, bm25_index, chunks = load_indexes()
    
    retriever_fn = lambda query: hybrid_search(query, faiss_index, bm25_index, chunks)
    llm_fn = lambda query, chunks: generate_answer(query, chunks)
    
    run_eval(retriever_fn, llm_fn)