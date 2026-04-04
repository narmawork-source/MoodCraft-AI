#!/usr/bin/env sh
set -e

if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

. venv/bin/activate
pip install -r requirements.txt
APP_FILE="${APP_FILE:-design_agents_app.py}"

if [ ! -f "$APP_FILE" ]; then
  echo "Error: app file not found: $APP_FILE" >&2
  echo "Set APP_FILE to a valid Streamlit script, e.g. APP_FILE=Case2Code/app.py" >&2
  exit 1
fi

streamlit run "$APP_FILE"
