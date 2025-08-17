
.PHONY: setup api ui train lint
setup: ; python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
api: ; uvicorn furniture_ai.api.server:app --host 0.0.0.0 --port 8000 --reload
ui: ; streamlit run apps/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
train: ; python scripts/train_segmenter.py
lint: ; python -m compileall -q furniture_ai || true
