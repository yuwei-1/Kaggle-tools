FROM ghcr.io/astral-sh/uv:python3.12-trixie

ARG WORKSPACE_DIR=/workspaces/Kaggle-tools
ARG VENV_DIR=/opt/venv

WORKDIR ${WORKSPACE_DIR}

ENV UV_PROJECT_ENVIRONMENT=${VENV_DIR} \
	VIRTUAL_ENV=${VENV_DIR} \
	PATH=${VENV_DIR}/bin:${PATH} \
	PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PIP_DISABLE_PIP_VERSION_CHECK=1 \
	PIP_NO_CACHE_DIR=1

# Prime dependency layers from lockfile for faster rebuilds.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project \
	&& rm -rf /root/.cache/uv /root/.cache/pip
