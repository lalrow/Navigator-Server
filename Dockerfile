# Use Python 3.13 slim image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
RUN pip install --no-cache-dir uv

# Copy project files
COPY . /app

# Install Python dependencies using uv
RUN uv sync --extra dev

# Create necessary directories
RUN mkdir -p /app/qdrant_local /tmp

# Expose port 7860 (HuggingFace Spaces standard port)
EXPOSE 7860

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:7860/api/health || exit 1

# Start the FastAPI server
CMD ["uv", "run", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860"]

