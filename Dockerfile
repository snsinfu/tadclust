FROM ghcr.io/prefix-dev/pixi:0.81.0-bookworm-slim AS build

WORKDIR /app

COPY pixi.toml pixi.lock ./
RUN pixi install --locked -e prod


FROM debian:bookworm-slim AS runtime

RUN apt-get update \
 && apt-get install -y --no-install-recommends bash ca-certificates procps \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=build /app/.pixi/envs/prod /app/.pixi/envs/prod
COPY src /app/src
COPY bin/tadclust /usr/local/bin/tadclust
COPY LICENSE /app/LICENSE

RUN chmod 0755 /usr/local/bin/tadclust

ENV PATH="/app/.pixi/envs/prod/bin:${PATH}" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8

WORKDIR /data

CMD ["tadclust", "--help"]
