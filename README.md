# RAG Docs QA

A RAG system for querying LangChain and LlamaIndex documentation, built with hybrid search and LLM-as-a-judge evaluation.

**Live demo:** https://kaedesaho-rag-docs.hf.space

## Architecture

- **Hybrid Search**: BM25 + FAISS dense retrieval merged with Reciprocal Rank Fusion
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM**: OpenAI `gpt-4o-mini`
- **Eval Framework**: LLM-as-a-judge scoring relevance, faithfulness, and answer quality
- **Frontend**: Streamlit

## Eval Results

| Metric | FAISS Only | Hybrid (BM25 + FAISS + RRF) |
|---|---|---|
| Relevance | 0.920 | 0.900 |
| Faithfulness | 0.900 | 0.900 |
| Answer Quality | 0.910 | 0.800 |

*Evaluated on 10 queries using GPT-4o-mini as judge.*

## Corpus

- LangChain documentation (`langchain-ai/docs`, main branch)
- LlamaIndex documentation (`run-llama/llama_index`, v0.14.19)
- ~2,460 documents, ~33,000 chunks after filtering

## Setup
```bash
git clone https://github.com/kaedesaho/rag-docs
cd rag-docs
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add your OpenAI API key to `.env`:
```
OPENAI_API_KEY=your_key_here
```

## Usage

Build indexes (run once):
```bash
python src/ingest.py
```

Run the app:
```bash
streamlit run app.py
```

Run evaluation:
```bash
python eval/run_eval.py
```