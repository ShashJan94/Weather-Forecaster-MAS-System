# 🌤️ Weather Forecaster AI

## Tiny Weather Forecaster with Transformer + Multi-Agent System

A lightweight, laptop-friendly weather prediction system that uses a tiny Transformer model and a Multi-Agent System (MAS) architecture to forecast tomorrow's temperature and weather conditions.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Features](#-features)
3. [Quick Start](#-quick-start)
4. [Installation Guide](#-installation-guide)
5. [Usage](#-usage)
6. [Project Structure](#-project-structure)
7. [Multi-Agent System Design](#-multi-agent-system-design)
8. [Model Architecture](#-model-architecture)
9. [Saved Files](#-saved-files)
10. [Presentation Q&A](#-presentation-qa)
11. [Troubleshooting](#-troubleshooting)
12. [License](#-license)

---

## 🎯 Overview

This project implements a **complete weather forecasting pipeline** that:

- **Predicts tomorrow's temperature** (regression task)
- **Classifies weather type** as sunny ☀️, cloudy ☁️, rainy 🌧️, or snowy ❄️ (classification task)
- **Identifies cold days** (temperature < 5°C) (binary classification)

The system uses a **Multi-Agent Architecture** where specialized agents handle different aspects of the ML pipeline:

```
Data Retriever → Data Agent → Baseline Agent  → Evaluation Agent → Narrator Agent
                           → Transformer Agent ↗
```

### Key Highlights

| Feature | Description |
|---------|-------------|
| ✅ **Laptop-Friendly** | Runs on CPU, trains in ~2 minutes |
| ✅ **Tiny Transformer** | ~72,000 parameters, d_model=64 |
| ✅ **Multi-Agent System** | 6 specialized agents |
| ✅ **Live Data** | Real weather data from Open-Meteo API |
| ✅ **Professional UI** | Streamlit dashboard with weather cards |
| ✅ **Model Comparison** | Baseline vs Transformer metrics |
| ✅ **No API Key Needed** | Open-Meteo is free and open |

---

## ✨ Features

### 🤖 Machine Learning

- **Transformer Model**: Tiny encoder-only Transformer for time-series
- **Baseline Models**: RandomForest for comparison
- **Joint Training**: Regression + Classification in single forward pass
- **Early Stopping**: Prevents overfitting
- **Model Persistence**: Save/load trained models

### 📊 Data Pipeline

- **Live Data Retrieval**: Open-Meteo API (no API key required)
- **Time-Series Windowing**: 7-day sliding window sequences
- **Feature Normalization**: StandardScaler for stable training
- **Train/Val/Test Splits**: 70/15/15 split

### 🎨 User Interface

- **Google-style Weather Cards**: Beautiful weather display with icons
- **Probability Bars**: Visual weather type probabilities
- **Natural Language Forecasts**: AI-generated human-readable predictions
- **Model Metrics Dashboard**: Compare Baseline vs Transformer

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Weather-Forecaster-MAS-System.git
cd Weather-Forecaster-MAS-System

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (fetch data, train models, make prediction)
python run.py --mode full --days 180 --epochs 15

# Launch the web UI
python run.py --mode ui
```

Then open http://localhost:8501 in your browser.

---

## 📦 Installation Guide

### Prerequisites

| Requirement | Version | Check Command |
|-------------|---------|---------------|
| Python | 3.9+ | `python --version` |
| pip | Latest | `pip --version` |
| Git | Any | `git --version` |

### Step-by-Step Installation

#### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Weather-Forecaster-MAS-System.git
cd Weather-Forecaster-MAS-System
```

Or download and extract the ZIP file.

#### Step 2: Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv venv
.\venv\Scripts\activate.bat
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `torch` - PyTorch for deep learning
- `scikit-learn` - For baseline models
- `pandas`, `numpy` - Data processing
- `streamlit` - Web UI
- `plotly` - Interactive charts
- `requests` - API calls

#### Step 4: Verify Installation

```bash
python -c "import torch; from src.agents import TransformerAgent; print('✅ Installation successful!')"
```

#### Step 5: Run the Application

```bash
# Option 1: Full pipeline (recommended for first run)
python run.py --mode full --days 180 --epochs 15

# Option 2: Launch UI directly (requires trained models)
python run.py --mode ui
```

---

## 🔧 Usage

### Command Line Interface

The `run.py` script supports multiple modes:

```bash
# Full pipeline: fetch data → train → evaluate → predict
python run.py --mode full --days 180 --epochs 15

# Train models only (uses existing data)
python run.py --mode train --epochs 20

# Make prediction only (uses saved models)
python run.py --mode predict

# Evaluate existing models
python run.py --mode evaluate

# Launch Streamlit UI
python run.py --mode ui
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--mode` | `full` | Execution mode: `full`, `train`, `predict`, `evaluate`, `ui` |
| `--days` | `365` | Days of historical data to fetch |
| `--epochs` | `30` | Number of training epochs |
| `--batch-size` | `32` | Training batch size |
| `--location` | `Warsaw` | City name for weather data |
| `--seq-length` | `7` | Sequence length (days) for prediction |
| `--seed` | `42` | Random seed for reproducibility |

### Examples

```bash
# Train with 1 year of data, 25 epochs
python run.py --mode full --days 365 --epochs 25

# Train for a different city
python run.py --mode full --location "Berlin" --days 180

# Quick training with smaller data
python run.py --mode full --days 90 --epochs 10

# Just make a prediction (models must be trained)
python run.py --mode predict
```

### Web UI Usage

1. Run `python run.py --mode ui`
2. Open http://localhost:8501
3. **Forecast Tab**: Click "Get Tomorrow's Forecast" to see predictions
4. **Analysis Tab**: View model metrics and comparison
5. **Training**: Use sidebar to retrain models or fetch new data

---

## 📁 Project Structure

```
Weather-Forecaster-MAS-System/
│
├── app.py                      # Streamlit web application
├── run.py                      # Main CLI entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
│
├── src/                        # Source code
│   ├── __init__.py
│   │
│   ├── agents/                 # Multi-Agent System (6 agents)
│   │   ├── __init__.py
│   │   ├── data_retriever.py   # Agent 1: Fetches live weather data
│   │   ├── data_agent.py       # Agent 2: Data processing & sequences
│   │   ├── baseline_agent.py   # Agent 3: RandomForest/XGBoost models
│   │   ├── transformer_agent.py # Agent 4: Transformer training
│   │   ├── evaluation_agent.py # Agent 5: Model comparison
│   │   └── narrator_agent.py   # Agent 6: Natural language forecasts
│   │
│   ├── models/                 # Neural network architectures
│   │   ├── __init__.py
│   │   └── transformer_model.py # TinyWeatherTransformer
│   │
│   └── utils/                  # Utilities
│       ├── __init__.py
│       ├── config.py           # Configuration settings
│       └── helpers.py          # Helper functions
│
├── tests/                      # Test suite
│   ├── conftest.py             # pytest fixtures
│   ├── test_data_agent.py
│   ├── test_transformer_agent.py
│   └── ...
│
├── data/                       # Data directory (auto-created)
│   └── raw/                    # Raw weather CSV files
│
└── models/                     # Saved models (auto-created)
    ├── baseline_model.joblib   # Trained RandomForest
    ├── transformer_model.pth   # Trained Transformer
    └── metrics.json            # Model comparison metrics
```

---

## 🤖 Multi-Agent System Design

The system consists of **6 specialized agents**, each with a single responsibility:

### Agent 1: Data Retriever 📡

**Purpose**: Fetch weather data from external sources

```python
from src.agents import DataRetriever

retriever = DataRetriever(latitude=52.23, longitude=21.01)
df = retriever.fetch_historical_data("2024-01-01", "2024-12-31")
```

- Connects to Open-Meteo API (free, no key)
- Fetches: temperature, humidity, pressure, wind, precipitation
- Supports 10+ popular cities
- Auto-saves to CSV

### Agent 2: Data Agent 📊

**Purpose**: Process and prepare data for training

```python
from src.agents import DataAgent

agent = DataAgent(sequence_length=7)
splits, loaders = agent.prepare_data(data=df, batch_size=32)
```

- Cleans missing values
- Creates 7-day sliding window sequences
- Normalizes features with StandardScaler
- Splits into train/val/test (70/15/15)

### Agent 3: Baseline Agent 🌲

**Purpose**: Train traditional ML models for comparison

```python
from src.agents import BaselineAgent

baseline = BaselineAgent(use_xgboost=False)  # RandomForest
baseline.train(X_train, y_temp, y_class, X_val, y_temp_val, y_class_val)
baseline.save_models("models/baseline_model.joblib")
```

- RandomForest or XGBoost regressor
- RandomForest or XGBoost classifier
- Provides baseline performance metrics

### Agent 4: Transformer Agent 🤖

**Purpose**: Train and run the Transformer model

```python
from src.agents import TransformerAgent

transformer = TransformerAgent(num_features=4, d_model=64, num_heads=2)
transformer.train(train_loader, val_loader, num_epochs=20)
prediction = transformer.predict(sequence)
```

- Tiny Transformer (72K parameters)
- Joint regression + classification
- Early stopping support
- Model saving/loading

### Agent 5: Evaluation Agent 📈

**Purpose**: Compare models and create ensembles

```python
from src.agents import EvaluationAgent

evaluator = EvaluationAgent()
comparison = evaluator.compare_models(baseline_results, transformer_results)
print(comparison.summary)
```

- Compares MAE, RMSE, R², Accuracy, F1
- Identifies best model per task
- Creates weighted ensemble predictions

### Agent 6: Narrator Agent 📝

**Purpose**: Generate human-readable forecasts

```python
from src.agents import NarratorAgent

narrator = NarratorAgent(use_emoji=True)
forecast = narrator.generate_forecast(
    temperature=15.5,
    weather_type="cloudy",
    confidence=0.85,
    location="Warsaw"
)
print(forecast['full_narrative'])
```

- Natural language forecast generation
- Weather icons and recommendations
- Confidence assessment

---

## 🧠 Model Architecture

### TinyWeatherTransformer

```
Input (batch, seq_len=7, features=4)
    │
    ▼
┌─────────────────────────────────┐
│ Input Projection (4 → 64)       │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ Positional Encoding             │
│ (Sinusoidal, learned positions) │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ Transformer Encoder Layer 1     │
│ • Multi-Head Attention (2 heads)│
│ • Feed-Forward (64 → 128 → 64)  │
│ • LayerNorm + Dropout           │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ Transformer Encoder Layer 2     │
│ (Same structure)                │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ Global Average Pooling          │
└─────────────────────────────────┘
    │
    ├──────────────┬──────────────┐
    ▼              ▼              ▼
┌────────┐   ┌──────────┐   ┌──────────┐
│Temp Head│   │Class Head│   │Binary Head│
│ 64→1   │   │ 64→4     │   │ 64→1     │
│(Linear)│   │(Linear)  │   │(Linear)  │
└────────┘   └──────────┘   └──────────┘
    │              │              │
    ▼              ▼              ▼
Temperature   Weather Type   Is Cold Day
(Regression)  (Softmax)     (Sigmoid)
```

### Model Configuration

| Parameter | Value |
|-----------|-------|
| d_model | 64 |
| num_heads | 2 |
| num_layers | 2 |
| d_ff | 128 |
| dropout | 0.1 |
| sequence_length | 7 |
| num_features | 4 |
| **Total Parameters** | **~72,000** |

---

## 📊 Saved Files

After training, the following files are created:

| File | Location | Description |
|------|----------|-------------|
| `baseline_model.joblib` | `models/` | Trained RandomForest models |
| `transformer_model.pth` | `models/` | Trained Transformer weights |
| `metrics.json` | `models/` | Model comparison metrics |
| `weather_data_*.csv` | `data/raw/` | Fetched weather data |

### metrics.json Example

```json
{
  "timestamp": "2025-12-09T02:51:23",
  "baseline": {
    "temp_MAE": 6.01,
    "temp_RMSE": 6.60,
    "temp_R2": -4.80,
    "weather_Accuracy": 0.22,
    "weather_F1_macro": 0.12
  },
  "transformer": {
    "temp_MAE": 4.98,
    "temp_RMSE": 5.60,
    "temp_R2": -3.17,
    "weather_Accuracy": 0.22,
    "weather_F1_macro": 0.12
  },
  "comparison": {
    "temp_winner": "Transformer",
    "weather_winner": "Baseline"
  }
}
```

---

## ❓ Presentation Q&A

### Architecture & Design

**Q: Why use a Multi-Agent System architecture?**
> A: MAS provides separation of concerns, making the code modular, testable, and maintainable. Each agent has a single responsibility (data fetching, processing, training, evaluation, narration), which follows the Single Responsibility Principle. This also allows easy swapping of components (e.g., replacing RandomForest with XGBoost).

**Q: Why is the Transformer "tiny"?**
> A: With only ~72,000 parameters, it can train on a laptop CPU in 2-3 minutes. This demonstrates that Transformers can be effective even at small scales for time-series tasks. Larger models would be overkill for this problem size.

**Q: Why use both Baseline and Transformer models?**
> A: The baseline (RandomForest) provides a reference point to evaluate whether the added complexity of a Transformer is justified. This is standard ML practice - always compare against simpler baselines.

### Technical Decisions

**Q: Why use sequence length of 7 days?**
> A: Weather patterns often follow weekly cycles. 7 days captures enough temporal context without making the model too complex. This is also interpretable - "using the last week to predict tomorrow."

**Q: Why joint training for regression and classification?**
> A: Joint training with shared features is more efficient than separate models. The model learns representations useful for both tasks. The shared encoder captures weather patterns, while task-specific heads decode to temperature and weather type.

**Q: How do you handle the 4 weather classes (sunny/cloudy/rainy/snowy)?**
> A: Weather codes from the API are mapped to 4 categories. The model outputs a 4-class softmax distribution, giving probability for each weather type. The class with highest probability is selected.

### Data & Training

**Q: Where does the weather data come from?**
> A: Open-Meteo API - a free, open-source weather API requiring no API key. It provides historical and current weather data globally.

**Q: How do you prevent overfitting?**
> A: Multiple techniques: (1) Early stopping if validation loss doesn't improve for 5 epochs, (2) Dropout (10%), (3) Small model size, (4) Proper train/val/test splits.

**Q: What features are used for prediction?**
> A: 4 core features: temperature, humidity, pressure, wind speed. These are normalized using StandardScaler before training.

### Performance

**Q: What metrics are used for evaluation?**
> A: For regression (temperature): MAE, RMSE, R². For classification (weather type): Accuracy, F1 macro, F1 weighted.

**Q: Why might the R² be negative?**
> A: A negative R² means the model performs worse than simply predicting the mean. This can happen with small test sets or when the model hasn't learned the pattern well. The MAE is often more interpretable.

**Q: How fast does it train?**
> A: On a typical laptop CPU: ~10-15 epochs in 2-3 minutes. GPU is not required.

### Extensions & Improvements

**Q: How could this be improved?**
> A: (1) More features (cloud cover, UV index), (2) Longer sequences, (3) Larger dataset (multiple years), (4) Attention visualization, (5) Uncertainty quantification, (6) Multi-location training.

**Q: Could this use real-time predictions?**
> A: Yes! The DataRetriever fetches live data from Open-Meteo. The UI already supports making predictions with current weather conditions.

**Q: What about GPU acceleration?**
> A: The code automatically detects CUDA if available. But the model is small enough that CPU training is fast enough for practical use.

---

## 🔧 Troubleshooting

### Common Issues

**Issue: `ModuleNotFoundError: No module named 'src'`**
```bash
# Make sure you're in the project root directory
cd Weather-Forecaster-MAS-System
python run.py --mode full
```

**Issue: `No trained models found`**
```bash
# Run the full pipeline first to train models
python run.py --mode full --days 180 --epochs 15
```

**Issue: Streamlit won't start**
```bash
# Check if port 8501 is in use
# Try specifying a different port
streamlit run app.py --server.port 8502
```

**Issue: API fetch fails**
```bash
# Check internet connection
# Open-Meteo might have temporary issues, try again later
# Or use existing data:
python run.py --mode train --epochs 15
```

**Issue: Virtual environment not activating (Windows)**
```powershell
# Windows PowerShell - may need execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 👥 Author

Created for **WSB University** - Machine Learning Course (4th Semester)

---

## 🙏 Acknowledgments

- [Open-Meteo](https://open-meteo.com/) for free weather API
- [PyTorch](https://pytorch.org/) for deep learning framework
- [Streamlit](https://streamlit.io/) for web UI framework
- [Plotly](https://plotly.com/) for interactive visualizations
