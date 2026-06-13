FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/
COPY migrations/ migrations/

RUN pip install --no-cache-dir .

RUN useradd -m cognita && chown -R cognita /app
USER cognita

EXPOSE 8000 8001
