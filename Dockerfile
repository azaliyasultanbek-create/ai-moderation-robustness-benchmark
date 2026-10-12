# Reproducible CPU-only environment for the benchmark.
#   docker build -t ai-mod-bench .                                   # core (smoke test)
#   docker build -t ai-mod-bench:full --build-arg WITH_TRANSFORMER=1 .  # + RoBERTa model
FROM python:3.13.5-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=4 \
    HF_HOME=/cache/huggingface

WORKDIR /app

ARG WITH_TRANSFORMER=0
COPY requirements.txt requirements-transformer.txt ./
RUN pip install -r requirements.txt \
 && if [ "$WITH_TRANSFORMER" = "1" ]; then \
      pip install --no-deps --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 \
      && pip install -r requirements-transformer.txt ; \
    fi \
 && pip check

# Code and config change more often than dependencies -> copied last (layer cache).
COPY configs/ configs/
COPY scripts/ scripts/
COPY src/ src/
COPY tests/ tests/
COPY data/sample/ data/sample/

ENTRYPOINT ["python", "-m", "src.benchmark"]
CMD ["--config", "configs/config.yaml"]
