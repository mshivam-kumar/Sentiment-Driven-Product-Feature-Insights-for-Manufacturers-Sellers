# Multi-stage build for production deployment
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install additional ML dependencies
RUN pip3 install --no-cache-dir \
    torch \
    transformers \
    sentence-transformers \
    peft \
    scikit-learn \
    numpy \
    pandas

# Copy application code
COPY backend/ .

# Copy fine-tuned model
COPY fine_tuned_tinyllama/ ./fine_tuned_tinyllama/

# Copy real data files
COPY data_ingest/ ./data_ingest/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Start the application
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
