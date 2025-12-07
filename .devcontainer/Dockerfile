FROM ghcr.io/astral-sh/uv:python3.12-trixie

WORKDIR /app

COPY pyproject.toml .

RUN uv pip install --system .