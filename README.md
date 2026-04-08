---
title: RAG Docs QA
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---
# RAG Docs QA

A production-quality RAG system for querying LangChain and LlamaIndex documentation.

## Architecture

- **Hybrid Search**: BM25 + FAISS dense retrieval merged with Reciprocal Rank Fusion
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM**: OpenAI `gpt-4o-mini`
- **Eval Framework**: LLM-as-a-judge scoring relevance, faithfulness, and answer quality
- **Frontend**: Streamlit

## Setup
```bash
git clone https://github.com/kaedesaho/rag-docs
cd rag-docs
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add your OpenAI API key:
OPENAI_API_KEY=your_key_here

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

## Eval Results

| Metric | Score |
|---|---|
| Relevance | 1.000 |
| Faithfulness | 1.000 |
| Answer Quality | 0.960 |

*Evaluated on 5 queries. Baseline vs hybrid comparison in progress.*

## Corpus

- LangChain documentation (langchain-ai/docs, pinned to main)
- LlamaIndex documentation (run-llama/llama_index, pinned to v0.14.19)
- ~3,300 documents, ~42,000 chunks