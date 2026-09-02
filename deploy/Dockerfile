# NOTE: this image has been written against the real package (pyproject.toml, the
# buffdata console-script entry point, and PIIScrubber's actual spacy model) but has not
# been built in this environment -- there was no running Docker daemon reachable from
# here to `docker build` and smoke-test it against. Treat it as a well-grounded starting
# point, not a verified artifact: build and run `buffdata --help` / a real pipeline
# command against it before relying on it.

# --- builder: compile deps into a venv, keep the runtime stage free of build tooling ---
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY pyproject.toml README.md ./
COPY buffdata ./buffdata

# torch is a hard dependency (buffdata/evaluation/accuracy_gate.py's proxy classifier,
# buffdata/optimizers/dedup.py's local embeddings) but the accuracy gate and dedup never
# need a GPU inside this image -- pulling the default PyPI wheel drags in CUDA runtime
# libraries that roughly triple image size for a binary that never gets used here.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir .

# PIIScrubber (buffdata/optimizers/scrubber.py) resolves presidio-analyzer's NLP engine
# to en_core_web_lg. scrub_pii defaults to True on PipelineConfig, so this is not
# optional for a production image -- without it, the pii stage fails on first real use.
RUN python -m spacy download en_core_web_lg

# --- runtime: no compilers, no pip cache, just the venv + spacy model it produced ---
FROM python:3.12-slim AS runtime

RUN groupadd --system buffdata && useradd --system --gid buffdata --create-home buffdata

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# Optional: pre-warm the local dedup embedding model (buffdata/optimizers/dedup.py,
# PipelineConfig.embedding_model default "all-MiniLM-L6-v2") into the HF cache so
# `dedup_method: semantic-local` doesn't pay a network fetch on first run inside a
# network_policy=strict pod. Uncomment if your deployment uses semantic-local dedup.
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

WORKDIR /work
RUN chown buffdata:buffdata /work
USER buffdata

# No secrets baked in -- provider API keys and BUFFDATA_SECRET_BACKEND are supplied at
# `docker run` / Helm deploy time (see deploy/helm/buffdata/templates/secret.yaml and
# buffdata/engine/secrets.py for the full set of supported backends).
ENTRYPOINT ["buffdata"]
CMD ["--help"]
