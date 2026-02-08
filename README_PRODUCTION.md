# Document RAG App (Production Baseline)

## What this app does
- Ingests PDF/TXT/textClipping documents.
- Splits content with LangChain `RecursiveCharacterTextSplitter`.
- Stores vectors in Chroma (`vector_db_docs/`).
- Supports RAG chat with two prompt modes:
  - Single Prompt
  - Reasoning Prompt (no chain-of-thought exposure)
- Runs QAEvalChain scoring and shows dashboard metrics.

## Production hardening included
- Secret-first API key loading (`st.secrets` then env fallback).
- API key validation guard.
- Upload size limits (`MAX_UPLOAD_MB`, default 25 MB per file).
- Parse-failure and oversized-file tracking.
- Vector DB readiness checks before chat/eval.
- Cached LLM/embedding resources for lower latency.
- Cleaner end-user errors with optional debug details.
- Downloadable QAEval results CSV.

## Run locally
```bash
venv/bin/pip install -r requirements.txt
venv/bin/streamlit run doc_rag_app.py
```

## Deploy on Streamlit Cloud
1. Push repo to GitHub.
2. In Streamlit Cloud, set main file: `doc_rag_app.py`.
3. Add secret in app settings:
   ```toml
   OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
   ```
4. Deploy and share URL.

## Operational notes
- Local path ingestion is for local environments only.
- Hosted runtime storage may be ephemeral; re-ingest docs after restarts.
- For durable production, move vectors/docs to external managed storage.

## Environment vars
- `OPENAI_API_KEY`: OpenAI key if not using Streamlit secrets.
- `MAX_UPLOAD_MB`: Max upload size per file (default 25).
