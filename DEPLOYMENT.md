# Deployment Guide (Streamlit Cloud)

## 1) Push to GitHub
- Ensure these files are committed:
  - `doc_rag_app.py`
  - `requirements.txt`
  - `run.sh` (optional)
- Do **not** commit `.env`.

## 2) Deploy on Streamlit Community Cloud
- Go to https://share.streamlit.io
- Connect your GitHub repo.
- Set:
  - Branch: your deployment branch
  - Main file path: `doc_rag_app.py`

## 3) Configure secrets
In Streamlit Cloud app settings, add:

```toml
OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```

## 4) Share URL
- Click Deploy.
- After build completes, share the generated public URL.

## Notes
- `vector_db_docs/` is local app storage.
- On cloud, runtime storage can be ephemeral; re-ingest docs after restart.
- For durable storage, use external object store + external vector DB.
- In hosted mode, use file upload instead of local file paths.
