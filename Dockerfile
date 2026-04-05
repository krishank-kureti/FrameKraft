# FrameKraft Dockerfile
# ---------------------
# Packages the FastAPI backend + frontend into a single deployable image.
# Note: The final image will be ~4-5GB due to PyTorch & AI model dependencies.
# Models are downloaded on first request. Mount a volume to /app/backend/models/ for caching.

FROM python:3.10-slim

# Install system dependencies required by OpenCV and native Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory to backend (where Python modules live)
WORKDIR /app/backend

# Copy requirements first to leverage Docker layer caching
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY backend/ ./

# Expose the FastAPI port
EXPOSE 10000

# Start the FastAPI server
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
