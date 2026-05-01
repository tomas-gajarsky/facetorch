FROM python:3.12-slim

# Set working directory
ENV WORKDIR=/opt/facetorch
WORKDIR $WORKDIR

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY . .

# Install facetorch from this checkout so release images match the built commit.
RUN uv pip install --system .
