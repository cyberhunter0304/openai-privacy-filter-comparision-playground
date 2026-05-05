# ── Base ──────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# HF Spaces runs as a non-root user; create one and own /app before switching
RUN useradd -m -u 1000 user && mkdir -p /app && chown user:user /app
WORKDIR /app

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# ── Switch to non-root user early so caches land in /home/user ───────────────
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# ── Python deps ───────────────────────────────────────────────────────────────
# Install CPU-only torch first (much smaller than the default CUDA build)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── spaCy model (baked into image at build time) ──────────────────────────────
RUN python -m spacy download en_core_web_lg

# ── App files ─────────────────────────────────────────────────────────────────
COPY --chown=user:user main.py .
COPY --chown=user:user index.html .

# HF Spaces requires port 7860
EXPOSE 7860

# ── Start ─────────────────────────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
