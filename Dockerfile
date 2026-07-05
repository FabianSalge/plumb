# syntax=docker/dockerfile:1

# Model weights are downloaded at first start, not baked into the image:
# LettuceDetect v2 is ~1.2 GB from one Hub repo (no remote code), its revision
# is pinned in config/verifier.yaml, and /readyz already reports "loading
# model" until the load completes. Baking would couple image rebuilds to
# weights that change independently of the code and inflate push/pull times.
# Revisit when vLLM lands.

FROM python:3.13.14-slim-trixie@sha256:eb43ff125d8d58d7449dcba7d336c23bcac412f526d861db493b9994d8010280 AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-install-project --extra model

COPY pyproject.toml uv.lock README.md ./
COPY api/ api/
COPY engine/ engine/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --extra model

FROM python:3.13.14-slim-trixie@sha256:eb43ff125d8d58d7449dcba7d336c23bcac412f526d861db493b9994d8010280

RUN groupadd --system --gid 10001 plumb \
    && useradd --system --uid 10001 --gid plumb --create-home plumb

WORKDIR /app
COPY --from=builder --chown=plumb:plumb /app/.venv .venv/
COPY --chown=plumb:plumb config/ config/

USER plumb
ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/home/plumb/.cache/huggingface

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
