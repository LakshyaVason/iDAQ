# Dockerfile — iDAQ FastAPI backend for Google Cloud Run
FROM python:3.11-slim

WORKDIR /app

# System deps for scikit-learn / joblib native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create folders expected at runtime
RUN mkdir -p training_data artifacts vector_store

# Cloud Run sets $PORT; default 8000
ENV PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1"]