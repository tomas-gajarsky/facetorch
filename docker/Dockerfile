ARG SOURCE_DATE_EPOCH=315532800

FROM python:3.14-slim@sha256:cae66f2ef0ec51a9891263eeee7f987dacf0a9879e8aa9353d5606e0530619a5 AS builder
ARG SOURCE_DATE_EPOCH

COPY --from=ghcr.io/astral-sh/uv:0.9.14@sha256:ad3f022815822ee5124d1e3d77169ba4420ed8906815ecb486164a48fdba360b /uv /uvx /bin/
WORKDIR /src
COPY pyproject.toml README.md LICENSE MODEL_NOTICE.md ./
COPY facetorch/ facetorch/
RUN uv build --wheel --out-dir /dist

FROM python:3.14-slim@sha256:cae66f2ef0ec51a9891263eeee7f987dacf0a9879e8aa9353d5606e0530619a5
ARG SOURCE_DATE_EPOCH

ENV FACETORCH_CACHE_DIR=/var/cache/facetorch
ENV PATH="/opt/facetorch-profile/.venv/bin:$PATH"
COPY --from=ghcr.io/astral-sh/uv:0.9.14@sha256:ad3f022815822ee5124d1e3d77169ba4420ed8906815ecb486164a48fdba360b /uv /uvx /bin/
COPY environments/torch-2.6-cpu/pyproject.toml /opt/facetorch-profile/pyproject.toml
COPY environments/torch-2.6-cpu/uv.lock /opt/facetorch-profile/uv.lock
COPY --from=builder /dist/*.whl /tmp/
RUN uv sync --project /opt/facetorch-profile --frozen --no-dev && \
    uv pip install --python /opt/facetorch-profile/.venv/bin/python \
        --no-deps /tmp/facetorch-*.whl && \
    useradd --create-home --uid 10001 facetorch && \
    mkdir -p /var/cache/facetorch /workspace/data/output && \
    chown -R facetorch:facetorch /var/cache/facetorch /workspace
RUN cd /tmp && python -c \
    "import facetorch; cfg=facetorch.load_config(offline=True); assert cfg.analyzer.device == 'cpu'"
COPY --chown=10001:10001 scripts/example.py /opt/facetorch/example.py

USER facetorch
WORKDIR /workspace
CMD ["/bin/bash"]
