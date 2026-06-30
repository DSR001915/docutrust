# DocuTrust backend image.
#
# Builds in "mock" provider mode by default (DOCUTRUST_PROVIDER_MODE=mock
# in docker-compose.yml) so `docker-compose up` works immediately with zero
# external API keys and zero model downloads -- see backend/config.py and
# the various get_*() provider factories for why that's possible.
#
# This image installs requirements-mock.txt (fast, no torch/sentence-
# transformers/ragas) by default. To bake in real local models instead,
# build with: docker build --build-arg REQUIREMENTS_FILE=requirements.txt
# and set DOCUTRUST_PROVIDER_MODE=local + LLM_PROVIDER=anthropic|ollama
# (with the relevant API key/server) in your .env.

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

ARG REQUIREMENTS_FILE=requirements-mock.txt
COPY requirements.txt requirements-mock.txt ./
RUN pip install --no-cache-dir -r ${REQUIREMENTS_FILE}

COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY sample_data/ ./sample_data/
COPY scripts/ ./scripts/

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
