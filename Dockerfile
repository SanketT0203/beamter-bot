# syntax=docker/dockerfile:1
#
# Beamter-Bot service image: the API, the chat page, and both models baked in.
# Build:  gcloud builds submit --tag <region>-docker.pkg.dev/<project>/beamter/beamter-bot
# See DEPLOY.md for the full sequence.

# ── Stage 1: Python dependencies, exactly as pinned in uv.lock ───────────────
FROM python:3.12-slim AS deps
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_NO_PROGRESS=1 \
    VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
RUN uv venv /opt/venv
WORKDIR /build
COPY pyproject.toml uv.lock ./
# sentence-transformers needs PyTorch, and the default Linux build of PyTorch drags in ~17 NVIDIA GPU
# packages (several GB). Cloud Run has no GPU, so: install the CPU-only build of the *same version the
# lock file pins*, then install everything else from the lock file except torch and the GPU packages.
RUN uv export --frozen --no-dev --no-emit-project --no-hashes --no-annotate -o locked.txt \
 && TORCH="$(sed -nE 's/^torch==([^ ;]+).*/\1/p' locked.txt | head -n1)" \
 && test -n "$TORCH" \
 && uv pip install "torch==${TORCH}" --index-url https://download.pytorch.org/whl/cpu \
 && grep -v -i -E '^(torch|triton|nvidia-|cuda-)' locked.txt > locked.cpu.txt \
 && uv pip install --no-deps -r locked.cpu.txt

# ── Stage 2: the application itself ──────────────────────────────────────────
FROM deps AS app
COPY src ./src
RUN uv pip install --no-deps .

# ── Stage 3: download the models, then prove they load with the network off ──
FROM app AS models
ENV HF_HOME=/opt/hf
RUN python -m beamter.fetch_models
RUN HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m beamter.fetch_models

# ── Stage 4: what actually runs. No uv, no compilers, no build caches ────────
FROM python:3.12-slim AS runtime
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin app
COPY --from=app /opt/venv /opt/venv
COPY --from=models /opt/hf /opt/hf
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
# OMP/MKL threads: keep equal to the vCPUs given to the service (--cpu 2). PyTorch otherwise sees
# all cores of the host machine and the threads fight over the few it is actually allowed to use.
USER app
EXPOSE 8080
# Cloud Run sets $PORT. One worker: each worker would load its own copy of the models.
# Keys (ANTHROPIC_API_KEY, QDRANT_API_KEY, SERVICE_API_KEY) arrive as environment variables at run time.
CMD ["sh", "-c", "exec uvicorn beamter.api:create_app --factory --host 0.0.0.0 --port ${PORT:-8080} --timeout-graceful-shutdown 20"]
