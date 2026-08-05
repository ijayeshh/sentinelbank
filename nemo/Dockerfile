# NeMo Guardrails server for the §4.4 comparison. Built only under
# docker-compose.nemo.yml, never as part of the main stack.
#
# `nemoguardrails` is deliberately ABSENT from the repository's requirements.txt. It
# pulls sentence-transformers, its own torch, annoy and a FastAPI server — a couple of
# gigabytes of wheels for a comparison that runs a handful of times. Installing it into
# the main venv would also make it possible to import it from the app by accident, which
# §1.1 rule 7 exists to prevent. It lives here, in an image, or nowhere.

FROM python:3.11-slim

# `annoy` — NeMo's default vector store for canonical-form matching — ships no manylinux
# wheel and builds from C++ source, which python:3.11-slim has no compiler for. This is
# the first concrete instance of §4.4's "setup cost" axis: the LangChain implementation
# needs no toolchain because it needs no vector index.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch. The 4 GB GTX 1650 belongs to Ollama (§1.1 rule 5), and the CUDA wheels
# are ~2 GB larger for no benefit to an 80 MB embedding model.
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1+cpu \
    && pip install --no-cache-dir \
        "nemoguardrails==0.11.0" \
        "sentence-transformers==3.3.1" \
        "langchain-community==0.3.14"

# Bake the embedding model into the image rather than downloading it on every container
# start. First boot is slow enough already — it has to index every canonical form — and a
# comparison whose startup cost depends on network weather is not a measurement.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2')"

WORKDIR /config

EXPOSE 8000

# `--config .` reads config.yml plus everything in rails/. Single worker: this serves a
# comparison script, and a second worker would mean a second copy of the embedding model.
CMD ["nemoguardrails", "server", "--config", "/config", "--port", "8000"]
