@echo off
REM ============================================
REM Weather Forecaster - Windows Setup Script
REM ============================================
echo.
echo =============================================
echo   Weather Forecaster Setup Script
echo =============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.9 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo [INFO] Python found:
python --version
echo.

REM Create virtual environment
echo [INFO] Creating virtual environment...
if exist "venv" (
    echo [INFO] Virtual environment already exists.
    echo [INFO] To recreate, delete the 'venv' folder first.
) else (
    python -m venv venv
    echo [INFO] Virtual environment created successfully.
)
echo.

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install dependencies
echo [INFO] Installing dependencies...
pip install -r requirements.txt
echo.

REM Create necessary directories
echo [INFO] Creating project directories...
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "models" mkdir models
if not exist "logs" mkdir logs
echo [INFO] Directories created.
echo.

REM Verify installation
echo [INFO] Verifying installation...
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')"
python -c "import xgboost; print(f'XGBoost: {xgboost.__version__}')"
echo.

echo =============================================
echo   Setup Complete!
echo =============================================
echo.
echo To activate the virtual environment, run:
echo   venv\Scripts\activate
echo.
echo Available commands:
echo   python run.py --mode full     Run complete pipeline
echo   python run.py --mode ui       Launch Streamlit UI
echo   streamlit run app.py          Launch UI directly
echo   pytest tests/                 Run all tests
echo.
echo =============================================
pause
