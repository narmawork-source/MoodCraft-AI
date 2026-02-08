#!/usr/bin/env sh
set -e

if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

. venv/bin/activate
pip install -r requirements.txt
APP_FILE="${APP_FILE:-doc_rag_app.py}"
streamlit run "$APP_FILE"
