import json
import os
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv(override=False)
client = OpenAI()

def judge_response(query, chunks, answer):
    chunk_text = "\n\n".join([c.page_content for c in chunks])
    
    prompt = f"""
    You are evaluating a RAG system. Given a query, retrieved chunks, and a generated answer, score the following:

    1. Relevance (0-1): Are the retrieved chunks relevant to the query?
    2. Faithfulness (0-1): Is the answer grounded in the retrieved chunks only?
    3. Answer quality (0-1): Does the answer actually address the query?

    Query: {query}

    Retrieved chunks:
    {chunk_text}

    Answer: {answer}

    Respond in JSON only with this format:
    {{"relevance": 0.0, "faithfulness": 0.0, "answer_quality": 0.0, "reasoning": "brief explanation"}}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    raw = response.choices[0].message.content
    clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(clean)

TEST_QUERIES = [
    "How do I use FAISS with LangChain?",
    "What is a vector store?",
    "How do I split documents in LangChain?",
    "What is LlamaIndex used for?",
    "How do I create a RAG pipeline?",
    "What embedding models does LangChain support?",
    "How do I persist a vector store to disk?",
    "What is the difference between a retriever and a vector store?",
    "How do I use metadata filtering in LlamaIndex?",
    "What is a node parser in LlamaIndex?"
]


def run_eval(retriever_fn, llm_fn):
    results = []
    
    for query in TEST_QUERIES:
        chunks = retriever_fn(query)
        answer = llm_fn(query, chunks)
        scores = judge_response(query, chunks, answer)
        
        results.append({
            "query": query,
            "answer": answer,
            "scores": scores
        })
        print(f"Query: {query}")
        print(f"Scores: {scores}\n")
    
    avg_relevance = sum(r["scores"]["relevance"] for r in results) / len(results)
    avg_faithfulness = sum(r["scores"]["faithfulness"] for r in results) / len(results)
    avg_quality = sum(r["scores"]["answer_quality"] for r in results) / len(results)
    
    print(f"\n==== EVAL SUMMARY ====")
    print(f"Avg Relevance:    {avg_relevance:.3f}")
    print(f"Avg Faithfulness: {avg_faithfulness:.3f}")
    print(f"Avg Quality:      {avg_quality:.3f}")
    
    return results