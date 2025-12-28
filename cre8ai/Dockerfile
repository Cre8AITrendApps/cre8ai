# --- Build Stage ---
FROM python:3.12-slim as builder

WORKDIR /app

# Install build dependencies if needed (e.g., for some python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Final Stage ---
FROM python:3.12-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gosu \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/cache/huggingface \
    DB_DIR=/app/db/chroma_db \
    DATA_DIR=/app/data

WORKDIR /app

# Create a non-root user
RUN addgroup --system appuser && adduser --system --group appuser

# Copy installed python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create necessary directories and set permissions
RUN mkdir -p /app/db /app/data /app/cache/huggingface && \
    chown -R appuser:appuser /app

# Pre-download the embedding model (optional but recommended for faster first start)
RUN python -c "from langchain_community.embeddings import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')"

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
EXPOSE 7860

# Use uvicorn for the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

