# ── MSA Arabic TTS — Production Dockerfile ──────────────────────────────────
FROM python:3.11-slim AS base

LABEL maintainer="msa-tts"
LABEL description="Modern Standard Arabic TTS — VITS + HiFi-GAN"
LABEL version="1.0.0"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Application
COPY msa_tts_backend.py .

# Optional: pre-download CAMeL Tools data
# Uncomment for production images (adds ~2 GB to image size)
# RUN python -m camel_tools.cli.camel_data -i defaults

# Non-root user for security
RUN useradd -m -u 1000 ttsuser && chown -R ttsuser:ttsuser /app
USER ttsuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENV TTS_DEVICE=auto \
    TTS_DIACRITIZER=auto \
    TTS_MODEL_PATH=""

CMD ["uvicorn", "msa_tts_backend:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
