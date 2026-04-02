@echo off
echo Iniciando BotCripto...
echo.
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ERRO: Python nao encontrado. Instale em https://python.org
    pause
    exit /b 1
)
where streamlit >nul 2>&1
if %errorlevel% neq 0 (
    echo Instalando dependencias...
    pip install -r requirements.txt
)
echo Abrindo dashboard em http://localhost:8501
streamlit run app.py
