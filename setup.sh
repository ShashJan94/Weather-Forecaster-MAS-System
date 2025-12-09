#!/bin/bash
# ============================================
# Weather Forecaster - Unix Setup Script
# ============================================

echo ""
echo "============================================="
echo "  Weather Forecaster Setup Script"
echo "============================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python is not installed."
    echo "Please install Python 3.9 or higher."
    exit 1
fi

echo "[INFO] Python found:"
python3 --version
echo ""

# Create virtual environment
echo "[INFO] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "[INFO] Virtual environment already exists."
    echo "[INFO] To recreate, delete the 'venv' folder first."
else
    python3 -m venv venv
    echo "[INFO] Virtual environment created successfully."
fi
echo ""

# Activate virtual environment
echo "[INFO] Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "[INFO] Upgrading pip..."
pip install --upgrade pip
echo ""

# Install dependencies
echo "[INFO] Installing dependencies..."
pip install -r requirements.txt
echo ""

# Create necessary directories
echo "[INFO] Creating project directories..."
mkdir -p data/raw data/processed models logs
echo "[INFO] Directories created."
echo ""

# Verify installation
echo "[INFO] Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')"
python -c "import xgboost; print(f'XGBoost: {xgboost.__version__}')"
echo ""

echo "============================================="
echo "  Setup Complete!"
echo "============================================="
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "Available commands:"
echo "  python run.py --mode full     Run complete pipeline"
echo "  python run.py --mode ui       Launch Streamlit UI"
echo "  streamlit run app.py          Launch UI directly"
echo "  pytest tests/                 Run all tests"
echo ""
echo "============================================="
