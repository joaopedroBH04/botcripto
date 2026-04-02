#!/bin/bash
echo "Iniciando BotCripto..."

if ! command -v python3 &> /dev/null; then
    echo "ERRO: Python não encontrado. Instale em https://python.org"
    exit 1
fi

if ! command -v streamlit &> /dev/null; then
    echo "Instalando dependências..."
    pip install -r requirements.txt
fi

echo "Abrindo dashboard em http://localhost:8501"
streamlit run app.py
