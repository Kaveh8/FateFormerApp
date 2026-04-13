# Hugging Face Spaces (Streamlit + Docker). Port 7860.
# Build context: repository root. Upload `streamlit_hf/cache/*` (pickles + parquet) via Git LFS or CI.

FROM python:3.11-slim-bookworm

WORKDIR /app

COPY streamlit_hf/requirements-docker.txt /app/requirements-docker.txt
RUN pip install --no-cache-dir -r /app/requirements-docker.txt

COPY . /app

ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_HEADLESS=true
EXPOSE 7860

CMD ["streamlit", "run", "streamlit_hf/app.py", "--server.port", "7860", "--server.address", "0.0.0.0", "--browser.gatherUsageStats", "false"]
