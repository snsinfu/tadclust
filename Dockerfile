FROM ghcr.io/astral-sh/uv:0.13.6 AS uv

FROM python:3.12-slim-bookworm AS build

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

COPY --from=uv /uv /uvx /bin/

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src

RUN uv sync --frozen --no-dev --no-editable


FROM python:3.12-slim-bookworm AS runtime

RUN apt-get update \
 && apt-get install -y --no-install-recommends bash ca-certificates procps \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=build /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8

WORKDIR /data

CMD ["tadclust", "--help"]
