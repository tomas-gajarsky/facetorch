FROM python:3.9.12-slim

# Set working directory
ENV WORKDIR=/opt/facetorch
WORKDIR $WORKDIR

# Upgrade pip to latest version
RUN pip install --upgrade pip

# Install facetorch from PyPI
RUN pip install facetorch

COPY . .