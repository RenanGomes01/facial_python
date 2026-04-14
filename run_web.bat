@echo off
cd /d "%~dp0"
title Facial Python - servidor web
echo.
echo  Servidor iniciando. NAO FECHE esta janela.
echo  No navegador abra: http://127.0.0.1:8000
echo  (Ou use o IP da rede para o celular na mesma Wi-Fi.)
echo.
python -m uvicorn web_app:app --host 0.0.0.0 --port 8000
if errorlevel 1 pause
