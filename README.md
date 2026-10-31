# 📄 RAG PDF Chatbot

Ask questions about your own PDFs and get answers with page-level sources.
Built with **Python, LangChain, ChromaDB, Sentence-Transformers and Streamlit**.

## How it works
```
PDFs -> split into chunks -> embeddings -> ChromaDB
Question -> embed -> top-k similar chunks -> LLM answers using only those chunks
```

## Features
- Local, free embeddings (`all-MiniLM-L6-v2`), no embedding API needed
- Choose your LLM: Anthropic, OpenAI, or local Ollama
- Answers only from your documents ("I don't know" otherwise)
- Chat memory for follow-up questions
- Source file and page number under every answer
- Evaluation script to measure answer quality

## Setup
```bash
git clone https://github.com/harini23-dot/rag-pdf-chatbot
cd rag-pdf-chatbot
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then add your API key
```

## Run
**Web app**
```bash
streamlit run app.py
```
Upload PDFs in the sidebar, click *Process documents*, then chat.

**Command line**
```bash
# put PDFs in data/
python ingest.py
python chat_cli.py
```

## Evaluate
Add 10 questions with expected keywords to `eval/questions.json`, then:
```bash
python evaluate.py
```

| chunk_size | top_k | score |
|-----------|-------|-------|
| 500       | 4     | _fill in_ |
| 800       | 4     | _fill in_ |
| 1000      | 6     | _fill in_ |

## Project structure
```
app.py        Streamlit interface
rag.py        Core RAG pipeline
ingest.py     Build the vector DB from data/
chat_cli.py   Terminal chatbot
evaluate.py   Accuracy test
```

## Tech stack
Python · LangChain · ChromaDB · Sentence-Transformers · Streamlit · Claude / OpenAI / Ollama

## Limitations
- Scanned PDFs need OCR first
- Answer quality depends on chunk size and retrieval settings
