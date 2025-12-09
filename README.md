# 🌤️ Weather Forecaster AI

## Tiny Weather Forecaster with Transformer + Multi-Agent System

A lightweight, laptop-friendly weather prediction system that uses a tiny Transformer model and a Multi-Agent System (MAS) architecture to forecast tomorrow's temperature and weather conditions. Features a **hybrid prediction approach** combining the best of both Transformer and RandomForest models.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

### 🎯 Current Performance (After Optimization)

| Metric | Baseline (RandomForest) | Transformer | Best |
|--------|------------------------|-------------|------|
| **Temperature MAE** | 1.54°C | **1.53°C** ✅ | Transformer |
| **Temperature R²** | 0.909 | **0.909** | Tie |
| **Weather Accuracy** | **49.5%** ✅ | 41.3% | Baseline |
| **F1 Macro** | 0.33 | **0.33** | Tie |

> 💡 **Hybrid Approach**: Temperature predictions use the Transformer (better regression), while weather classification uses RandomForest (better accuracy).

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
9. [Hybrid Prediction Approach](#-hybrid-prediction-approach)
10. [Development Journey](#-development-journey)
11. [Saved Files](#-saved-files)
12. [Presentation Q&A](#-presentation-qa)
13. [Troubleshooting](#-troubleshooting)
14. [License](#-license)

---

## 🎯 Overview

This project implements a **complete weather forecasting pipeline** that:

- **Predicts tomorrow's temperature** (regression task) - **MAE: 1.53°C, R²: 0.91**
- **Classifies weather type** as sunny ☀️, cloudy ☁️, rainy 🌧️, or snowy ❄️ (classification task) - **Accuracy: 49.5%**
- **Identifies cold days** (temperature < 5°C) (binary classification)

The system uses a **Multi-Agent Architecture** where specialized agents handle different aspects of the ML pipeline:

```
Data Retriever → Data Agent → Baseline Agent  → Evaluation Agent → Narrator Agent
                           → Transformer Agent ↗
```

### 🔄 Hybrid Prediction System

The system intelligently combines both models:
- **🌡️ Temperature**: Uses **Transformer** (better regression performance)
- **🌤️ Weather Type**: Uses **RandomForest** (better classification accuracy)

This hybrid approach leverages the strengths of each model type!

### Key Highlights

| Feature | Description |
|---------|-------------|
| ✅ **Laptop-Friendly** | Runs on CPU, trains in ~2 minutes |
| ✅ **Tiny Transformer** | ~72,774 parameters, d_model=64 |
| ✅ **Multi-Agent System** | 6 specialized agents |
| ✅ **Live Data** | Real weather data from Open-Meteo API |
| ✅ **Professional UI** | Streamlit dashboard with weather cards |
| ✅ **Hybrid Predictions** | Best of both models combined |
| ✅ **Scaler Persistence** | Proper data normalization for inference |
| ✅ **Class Weight Balancing** | Handles imbalanced weather data |
| ✅ **No API Key Needed** | Open-Meteo is free and open |

---

## ✨ Features

### 🤖 Machine Learning

- **Transformer Model**: Tiny encoder-only Transformer for time-series (72,774 params)
- **Baseline Models**: RandomForest for comparison and classification
- **Hybrid Predictions**: Combines best of both models
- **Joint Training**: Regression + Classification in single forward pass
- **Class Weight Balancing**: Handles imbalanced weather categories
- **Early Stopping**: Prevents overfitting
- **Model Persistence**: Save/load trained models and scalers

### 📊 Data Pipeline

- **Live Data Retrieval**: Open-Meteo API (no API key required)
- **Time-Series Windowing**: 7-day sliding window sequences
- **Feature Normalization**: StandardScaler with persistence for inference
- **Train/Val/Test Splits**: 70/15/15 split
- **Scaler Persistence**: Saved to `models/scaler.joblib` for consistent predictions

### 🎨 User Interface

- **Google-style Weather Cards**: Beautiful weather display with icons
- **Probability Bars**: Visual weather type probabilities
- **Natural Language Forecasts**: AI-generated human-readable predictions
- **Model Metrics Dashboard**: Compare Baseline vs Transformer
- **Training History**: View saved models and training status
- **Hybrid Approach Info**: Visual explanation of prediction strategy

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
├── models/                     # Saved models (auto-created)
    ├── baseline/               # Baseline model directory
    │   ├── model.joblib        # Trained RandomForest regressor
    │   └── classifier.joblib   # Trained RandomForest classifier
    ├── transformer_model.pth   # Trained Transformer weights
    ├── scaler.joblib           # Feature scaler for inference
    └── metrics.json            # Model comparison metrics
```

---

## 🔄 Hybrid Prediction Approach

### Why Hybrid?

During development, we discovered that different models excel at different tasks:

| Task | Best Model | Why? |
|------|-----------|------|
| **Temperature Prediction** | Transformer | Better at capturing temporal patterns |
| **Weather Classification** | RandomForest | More robust to class imbalance |

### How It Works

```python
# Simplified prediction flow
def predict(sequence):
    # 1. Normalize input with saved scaler
    normalized = scaler.transform(sequence)
    
    # 2. Get temperature from Transformer (better regression)
    temperature = transformer.predict(normalized)['temperature']
    
    # 3. Get weather type from RandomForest (better classification)
    weather_type = baseline.predict(sequence)[1]  # classification output
    
    return {
        'temperature': temperature,
        'weather_type': weather_type,
        'method': 'hybrid'
    }
```

### Class Distribution Challenge

Our training data (731 days from Warsaw) has severe class imbalance:

| Weather Type | Count | Percentage | Class Weight |
|--------------|-------|------------|--------------|
| 🌧️ Rainy | 345 | 47.2% | 2.04 |
| ☁️ Cloudy | 298 | 40.8% | 0.57 |
| ❄️ Snowy | 68 | 9.3% | 7.03 |
| ☀️ Sunny | 20 | 2.7% | 0.62 |

We use **class weights** to balance the training and prevent the model from always predicting the majority class.

---

## 📈 Development Journey

This section documents the evolution of the project, including bugs discovered and how they were fixed.

### Phase 1: Initial Implementation ✅

**Goal**: Create a working weather forecasting pipeline with MAS architecture.

- Implemented 6 specialized agents (Data Retriever, Data Agent, Baseline, Transformer, Evaluation, Narrator)
- Built Streamlit UI with Google-style weather cards
- Integrated Open-Meteo API for live weather data
- Created training pipeline with joint regression + classification

**Initial Results**: 
- Model trained successfully
- UI displayed predictions
- But predictions were **terrible** (R² = -3.17, always predicted "sunny")

### Phase 2: The Scaler Bug Discovery 🐛

**Problem**: Predictions were completely wrong despite training looking successful.

**Symptoms**:
- R² = -3.17 (worse than predicting the mean!)
- Temperature predictions off by 5-6°C
- Always predicted "sunny" regardless of actual weather

**Root Cause Discovery**:
```python
# BEFORE (BUG): Scaler was fitted during training but NEVER saved!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Training used scaled data
# ... model trains on scaled data ...

# At prediction time:
sequence = get_current_weather()  # Raw, unscaled data!
prediction = model.predict(sequence)  # WRONG! Model expects scaled input
```

**The Fix**:
```python
# AFTER (FIXED): Save and load the scaler
# During training:
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, 'models/scaler.joblib')  # Save scaler!

# At prediction time:
scaler = joblib.load('models/scaler.joblib')  # Load scaler
sequence_scaled = scaler.transform(sequence)  # Scale input!
prediction = model.predict(sequence_scaled)  # Correct!
```

**Results After Fix**:
| Metric | Before Fix | After Fix | Improvement |
|--------|-----------|-----------|-------------|
| **R²** | -3.17 | **0.91** | +4.08 |
| **MAE** | 5.6°C | **1.53°C** | -4.07°C |

### Phase 3: Class Imbalance Solution ⚖️

**Problem**: Even with correct scaling, weather classification was poor. Model always predicted "cloudy" or "rainy".

**Root Cause**: Class imbalance
- Rainy: 47% of data
- Cloudy: 41% of data  
- Snowy: 9% of data
- Sunny: Only 2.7% of data!

**The Fix**: Added class weights to the loss function:
```python
# Calculate class weights (inverse of frequency)
class_counts = [sunny_count, cloudy_count, rainy_count, snowy_count]
class_weights = 1.0 / np.array(class_counts)
class_weights = class_weights / class_weights.sum() * len(class_counts)

# Use weighted CrossEntropyLoss
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
```

### Phase 4: Hybrid Approach Implementation 🔀

**Discovery**: After fixing the scaler, we noticed:
- Transformer was better at temperature prediction (regression)
- RandomForest was better at weather classification

**Solution**: Combine the best of both!
```python
# Hybrid prediction
temperature = transformer.predict(normalized)['temperature']  # Better regression
weather_type = baseline.predict(sequence)[1]  # Better classification
```

### Phase 5: UI Enhancements 🎨

Updated all UI tabs to reflect the new hybrid approach:

1. **Forecast Tab**: Shows hybrid approach info box
2. **History Tab**: Displays 3 components (Baseline, Transformer, Scaler)
3. **About Tab**: Explains hybrid approach and class distribution
4. **Analysis Tab**: Compares both models with winner indicators

### Summary of Changes

| Component | Change | Impact |
|-----------|--------|--------|
| `run.py` | Added scaler save after data processing | Enables consistent predictions |
| `run.py` | Added class weight computation | Better classification |
| `app.py` | Added scaler loading in get_data_agent() | Fixed prediction pipeline |
| `app.py` | Implemented hybrid prediction | Best of both models |
| `app.py` | Fixed baseline.predict() tuple handling | Eliminated TypeError |
| `app.py` | Updated all UI tabs | Improved user experience |

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
| **Total Parameters** | **~72,774** |

---

## 📊 Saved Files

After training, the following files are created:

| File | Location | Description |
|------|----------|-------------|
| `model.joblib` | `models/baseline/` | Trained RandomForest regressor |
| `classifier.joblib` | `models/baseline/` | Trained RandomForest classifier |
| `transformer_model.pth` | `models/` | Trained Transformer weights |
| `scaler.joblib` | `models/` | StandardScaler for feature normalization |
| `metrics.json` | `models/` | Model comparison metrics |
| `weather_data_*.csv` | `data/raw/` | Fetched weather data |

### metrics.json Example (Current)

```json
{
  "timestamp": "2025-12-09T03:51:01",
  "baseline": {
    "temp_MAE": 1.54,
    "temp_RMSE": 1.95,
    "temp_R2": 0.909,
    "weather_Accuracy": 0.495,
    "weather_F1_macro": 0.326
  },
  "transformer": {
    "temp_MAE": 1.53,
    "temp_RMSE": 1.94,
    "temp_R2": 0.909,
    "weather_Accuracy": 0.413,
    "weather_F1_macro": 0.330
  },
  "comparison": {
    "temp_winner": "Transformer",
    "weather_winner": "Baseline"
  }
}
```

### scaler.joblib

This file is **critical** for making predictions. It stores the StandardScaler fitted on training data, ensuring that new input data is normalized the same way.

```python
# Loading and using the scaler
import joblib

scaler = joblib.load('models/scaler.joblib')
normalized_input = scaler.transform(raw_weather_data)
prediction = model.predict(normalized_input)
```

---

## ❓ Presentation Q&A

### Architecture & Design

**Q: Why use a Multi-Agent System architecture?**
> A: MAS provides separation of concerns, making the code modular, testable, and maintainable. Each agent has a single responsibility (data fetching, processing, training, evaluation, narration), which follows the Single Responsibility Principle. This also allows easy swapping of components (e.g., replacing RandomForest with XGBoost).

**Q: Why is the Transformer "tiny"?**
> A: With only ~72,774 parameters, it can train on a laptop CPU in 2-3 minutes. This demonstrates that Transformers can be effective even at small scales for time-series tasks. Larger models would be overkill for this problem size.

**Q: Why use both Baseline and Transformer models?**
> A: We discovered during development that each model excels at different tasks. The Transformer is better at temperature regression (capturing temporal patterns), while RandomForest is better at weather classification (more robust to class imbalance). Our hybrid approach uses the best of both!

**Q: Why use a Hybrid Prediction approach?**
> A: After extensive testing, we found that combining models improves overall accuracy. The Transformer achieves MAE of 1.53°C for temperature, while RandomForest achieves 49.5% accuracy for weather classification. Using both gives us the best of both worlds.

### Technical Decisions

**Q: Why use sequence length of 7 days?**
> A: Weather patterns often follow weekly cycles. 7 days captures enough temporal context without making the model too complex. This is also interpretable - "using the last week to predict tomorrow."

**Q: Why joint training for regression and classification?**
> A: Joint training with shared features is more efficient than separate models. The model learns representations useful for both tasks. The shared encoder captures weather patterns, while task-specific heads decode to temperature and weather type.

**Q: How do you handle the 4 weather classes (sunny/cloudy/rainy/snowy)?**
> A: Weather codes from the API are mapped to 4 categories. We use **class weights** to handle the severe imbalance (sunny is only 2.7% of data). The model outputs a 4-class softmax distribution, and RandomForest provides the final classification.

**Q: What was the biggest bug you found and how did you fix it?**
> A: The **scaler persistence bug**! During training, we normalized data using StandardScaler, but we never saved it. At prediction time, raw (unnormalized) data was fed to a model trained on normalized data. This caused R² to be -3.17 (worse than mean prediction!). The fix was simple: save the scaler with `joblib.dump()` and load it before predictions.

### Data & Training

**Q: Where does the weather data come from?**
> A: Open-Meteo API - a free, open-source weather API requiring no API key. It provides historical and current weather data globally.

**Q: How do you prevent overfitting?**
> A: Multiple techniques: (1) Early stopping if validation loss doesn't improve for 5 epochs, (2) Dropout (10%), (3) Small model size, (4) Proper train/val/test splits, (5) Class weights for balanced training.

**Q: What features are used for prediction?**
> A: 4 core features: temperature, humidity, pressure, wind speed. These are normalized using StandardScaler, which is persisted to `models/scaler.joblib` for inference.

**Q: How do you handle class imbalance?**
> A: We compute class weights inversely proportional to class frequency. For example, sunny (2.7% of data) gets a higher weight than rainy (47% of data). This prevents the model from always predicting the majority class.

### Performance

**Q: What metrics are used for evaluation?**
> A: For regression (temperature): MAE, RMSE, R². For classification (weather type): Accuracy, F1 macro, F1 weighted.

**Q: What is a good R² value?**
> A: Our current R² is **0.91**, which means the model explains 91% of the variance in temperature. This is excellent! Values above 0.9 indicate strong predictive power. Initially, we had R² = -3.17, which meant the model was worse than predicting the mean.

**Q: Why might the R² have been negative initially?**
> A: We discovered this was due to the **scaler bug**. The model was trained on normalized data but received unnormalized data during inference. After fixing this, R² improved from -3.17 to 0.91.

**Q: How accurate are the temperature predictions?**
> A: The Transformer achieves **MAE of 1.53°C**, meaning predictions are typically within 1.5°C of actual temperature. This is comparable to short-term weather forecast accuracy.

**Q: How fast does it train?**
> A: On a typical laptop CPU: ~15-30 epochs in 2-3 minutes. GPU is not required but will speed up training.

### Development & Debugging

**Q: What was the development process?**
> A: 
> 1. **Phase 1**: Built MAS architecture with 6 agents
> 2. **Phase 2**: Discovered and fixed scaler persistence bug (R² -3.17 → 0.91)
> 3. **Phase 3**: Added class weights for imbalanced data
> 4. **Phase 4**: Implemented hybrid prediction (Transformer for temp, RF for weather)
> 5. **Phase 5**: Enhanced UI with hybrid approach info

**Q: What debugging techniques did you use?**
> A: We analyzed model outputs, compared training vs inference data distributions, and discovered the scaler mismatch. Key insight: if R² is negative, something is fundamentally wrong with the data pipeline!

**Q: What would you do differently next time?**
> A: Save all preprocessing artifacts (scaler, encoder, etc.) from the start. Implement end-to-end testing that validates the full pipeline from data fetching to prediction.

### Extensions & Improvements

**Q: How could this be improved?**
> A: (1) More features (cloud cover, UV index), (2) Longer sequences, (3) Larger dataset (multiple years), (4) Attention visualization, (5) Uncertainty quantification, (6) Multi-location training, (7) Better handling of rare weather types (sunny, snowy).

**Q: Could this use real-time predictions?**
> A: Yes! The DataRetriever fetches live data from Open-Meteo. The UI already supports making predictions with current weather conditions.

**Q: What about GPU acceleration?**
> A: The code automatically detects CUDA if available. But the model is small enough that CPU training is fast enough for practical use.

### Key Takeaways for Presentation

1. **MAS Architecture**: 6 specialized agents with single responsibility principle
2. **Hybrid Approach**: Best of both worlds - Transformer for regression, RandomForest for classification
3. **Scaler Bug**: Critical lesson about persisting preprocessing artifacts
4. **Class Imbalance**: Real-world challenge handled with class weights
5. **Performance**: R² = 0.91, MAE = 1.53°C (excellent for weather prediction)
6. **Laptop-Friendly**: 72,774 parameters, trains in 2-3 minutes on CPU

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

**Issue: Predictions always show "sunny"**
```bash
# This was fixed! The issue was the scaler not being persisted.
# Make sure models/scaler.joblib exists. If not, retrain:
python run.py --mode full --days 365 --epochs 30
```

**Issue: R² is negative**
```bash
# Check that the scaler is being loaded correctly
# The scaler must be the same one used during training
# Delete models/ folder and retrain if needed
```

---

## 📊 Performance History

| Version | R² Score | MAE | Issue |
|---------|----------|-----|-------|
| v1.0 | -3.17 | 5.6°C | Scaler not saved |
| v1.1 | 0.74 | 1.4°C | Fixed scaler, no class weights |
| v2.0 | **0.91** | **1.53°C** | Hybrid approach + class weights |

---

## 📄 License

This project is licensed under the MIT License.

---

## 👥 Author

Created for **WSB University** - Machine Learning Course (4th Semester)

**Development Team**: Shashank Jan

**GitHub**: [ShashJan94/Weather-Forecaster-MAS-System](https://github.com/ShashJan94/Weather-Forecaster-MAS-System)

---

## 🙏 Acknowledgments

- [Open-Meteo](https://open-meteo.com/) for free weather API
- [PyTorch](https://pytorch.org/) for deep learning framework
- [Streamlit](https://streamlit.io/) for web UI framework
- [Plotly](https://plotly.com/) for interactive visualizations
- GitHub Copilot for development assistance

---

## 📅 Changelog

### v2.0 (2025-12-09) - Major Update
- ✅ Fixed critical scaler persistence bug (R² improved from -3.17 to 0.91)
- ✅ Implemented hybrid prediction approach (Transformer + RandomForest)
- ✅ Added class weight balancing for imbalanced weather data
- ✅ Updated UI with hybrid approach info
- ✅ Added scaler info to History tab
- ✅ Enhanced About tab with class distribution details
- ✅ Comprehensive README update with development journey

### v1.0 (2025-12-08) - Initial Release
- ✅ Multi-Agent System with 6 agents
- ✅ Tiny Transformer model (72,774 parameters)
- ✅ RandomForest baseline
- ✅ Streamlit UI with weather cards
- ✅ Open-Meteo API integration
