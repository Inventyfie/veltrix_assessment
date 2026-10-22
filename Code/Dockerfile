FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CORPUS_DIR=/app/data/corpus \
    MANIFEST_PATH=/app/data/manifest.json \
    DB_PATH=/app/data/veltrix.db \
    INDEX_PATH=/app/data/.peopleops-index.sqlite3

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY peopleops ./peopleops

EXPOSE 8000
CMD ["uvicorn", "peopleops.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
