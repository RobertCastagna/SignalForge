FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/usr/local \
    HF_HOME=/opt/huggingface

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock* ./
COPY src ./src

RUN uv sync --frozen --group test

RUN python -m playwright install --with-deps chromium
RUN python -c "from transformers import AutoModel, AutoTokenizer; name='BAAI/bge-small-en-v1.5'; AutoTokenizer.from_pretrained(name); AutoModel.from_pretrained(name)"

COPY run_amdc.py streamlit_app.py show_latest.py ./
COPY tests ./tests

RUN mkdir -p \
    /app/data/lakehouse/bronze \
    /app/data/lakehouse/silver \
    /app/data/lakehouse/gold \
    /app/data/lakehouse/_quality \
    /app/data/lakehouse/_pipeline
VOLUME ["/app/data"]
EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
