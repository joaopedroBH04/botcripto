# ============================================================
# BotCripto - Dockerfile (multi-stage, otimizado para Streamlit)
# ============================================================

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /install

# Dependencias de build apenas no stage builder
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --prefix=/install/deps -r requirements.txt


# ============================================================
# Stage final — imagem enxuta
# ============================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    BOTCRIPTO_DB_URL=sqlite:///data/botcripto.db

# Dependencias de runtime minimas (curl para healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash botcripto

WORKDIR /app

# Copia pacotes Python instalados no builder
COPY --from=builder /install/deps /usr/local

# Copia o codigo da aplicacao
COPY --chown=botcripto:botcripto . /app

# Pasta de dados persistentes (SQLite + cache)
RUN mkdir -p /app/data && chown -R botcripto:botcripto /app/data

USER botcripto

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py"]
