FROM python:3.12-slim

# Set working directory
ENV WORKDIR=/opt/facetorch
WORKDIR $WORKDIR

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install facetorch from PyPI
RUN uv pip install --system facetorch

COPY . .
