# VeriRAG image: the program, the embedding model and an index of the sample corpus,
# so a question can be asked right after `docker run`. Only the API key is missing.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/opt/models

WORKDIR /app

# Install the program first (its files change often, the model does not).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Download the embedding model once, at build time, into /opt/models.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

# Index the sample documents at build time. Mount your own folder and run
# `verirag index /your/folder` to search other documents.
COPY docs/sample ./docs/sample
COPY eval ./eval
RUN verirag index docs/sample

# Do not run as root.
RUN useradd --create-home app && chown -R app /app /opt/models
USER app

# The API key is NEVER stored in the image: pass it with `docker run -e GEMINI_API_KEY`.
ENTRYPOINT ["verirag"]
CMD ["--help"]
