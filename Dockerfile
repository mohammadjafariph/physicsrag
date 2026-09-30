FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/data/hf-cache

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY config.py main.py ./
COPY src ./src
COPY web ./web

EXPOSE 8000
VOLUME ["/app/data"]

CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]