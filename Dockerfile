FROM python:3.12-slim

# lxml needs the XML toolchain to build; the build tools are removed again below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libxml2-dev libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --no-cache-dir . \
    && apt-get purge -y build-essential libxml2-dev libxslt1-dev \
    && apt-get autoremove -y

RUN useradd --create-home cognita
USER cognita

ENV MCP_HOST=0.0.0.0 \
    MCP_PORT=8080 \
    PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["cognita", "--http"]
