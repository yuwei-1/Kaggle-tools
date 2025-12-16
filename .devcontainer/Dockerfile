FROM ghcr.io/astral-sh/uv:python3.12-trixie

WORKDIR /workspaces/

COPY pyproject.toml .

RUN uv sync
