@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
  echo Creando entorno virtual...
  python -m venv .venv
  call .venv\Scripts\activate.bat
  python -m pip install -U pip
  pip install -r requirements.txt
  pip install -e .
) else (
  call .venv\Scripts\activate.bat
)

streamlit run app/streamlit_app.py
