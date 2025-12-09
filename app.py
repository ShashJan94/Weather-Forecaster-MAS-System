"""
Weather Forecaster UI
Professional Streamlit dashboard with Google-style weather display.
Loads saved models - no need to retrain!
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.data_retriever import DataRetriever, generate_synthetic_data
from src.agents.data_agent import DataAgent
from src.agents.baseline_agent import BaselineAgent
from src.agents.transformer_agent import TransformerAgent
from src.agents.evaluation_agent import EvaluationAgent
from src.agents.narrator_agent import NarratorAgent

# Paths
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Page configuration
st.set_page_config(
    page_title="Weather Forecaster AI",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with Google Weather-inspired styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    
    .weather-main-card {
        background: linear-gradient(135deg, #4A90D9 0%, #67B8DE 100%);
        border-radius: 24px;
        padding: 40px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(74, 144, 217, 0.3);
        margin: 20px 0;
    }
    
    .weather-main-card.sunny {
        background: linear-gradient(135deg, #FFB347 0%, #FFCC33 100%);
    }
    
    .weather-main-card.cloudy {
        background: linear-gradient(135deg, #8E9EAB 0%, #B8C6DB 100%);
    }
    
    .weather-main-card.rainy {
        background: linear-gradient(135deg, #3A7BD5 0%, #00D2FF 100%);
    }
    
    .weather-main-card.snowy {
        background: linear-gradient(135deg, #E6DADA 0%, #274046 100%);
        color: #333;
    }
    
    .weather-icon-large {
        font-size: 7rem;
        line-height: 1;
        margin-bottom: 10px;
        text-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .temp-large {
        font-size: 5rem;
        font-weight: 300;
        line-height: 1;
        margin: 10px 0;
    }
    
    .weather-type-large {
        font-size: 1.8rem;
        font-weight: 400;
        text-transform: capitalize;
        margin-bottom: 10px;
    }
    
    .location-text {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    .confidence-badge {
        display: inline-block;
        background: rgba(255,255,255,0.25);
        padding: 8px 16px;
        border-radius: 20px;
        margin-top: 15px;
        font-size: 0.95rem;
    }
    
    .prob-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin: 10px 0;
    }
    
    .narrative-card {
        background: #f8f9fa;
        border-left: 4px solid #4A90D9;
        padding: 20px 25px;
        border-radius: 0 12px 12px 0;
        margin: 20px 0;
    }
    
    .narrative-card h3 {
        color: #2c3e50;
        margin-top: 0;
    }
    
    .model-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        text-align: center;
        border: 2px solid transparent;
    }
    
    .model-card.transformer {
        border-color: #667eea;
    }
    
    .model-card.current {
        border-color: #11998e;
    }
    
    .model-card h4 {
        margin: 0 0 15px 0;
    }
    
    .model-card .temp {
        font-size: 2.2rem;
        font-weight: 700;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'location' not in st.session_state:
    st.session_state.location = "Warsaw"


def get_weather_icon(weather_type: str) -> str:
    """Get emoji icon for weather type."""
    icons = {
        "sunny": "☀️",
        "cloudy": "☁️",
        "rainy": "🌧️",
        "snowy": "❄️"
    }
    return icons.get(weather_type.lower(), "🌤️")


def get_weather_gradient(weather_type: str) -> str:
    """Get CSS class for weather type."""
    return weather_type.lower() if weather_type.lower() in ['sunny', 'cloudy', 'rainy', 'snowy'] else 'cloudy'


def check_saved_models() -> dict:
    """Check if trained models exist."""
    transformer_path = MODELS_DIR / "transformer_model.pth"
    baseline_dir = MODELS_DIR / "baseline"
    
    # Baseline saves to directory with regressor/classifier files
    baseline_exists = (
        baseline_dir.exists() and 
        (baseline_dir / "randomforest_regressor.joblib").exists() and
        (baseline_dir / "randomforest_classifier.joblib").exists()
    )
    
    return {
        'transformer': transformer_path.exists(),
        'baseline': baseline_exists,
        'transformer_path': transformer_path,
        'baseline_path': baseline_dir
    }


@st.cache_resource
def load_saved_models():
    """Load pre-trained models from disk. Cached for performance."""
    model_status = check_saved_models()
    
    if not model_status['transformer'] or not model_status['baseline']:
        return None, None, False
    
    try:
        baseline_agent = BaselineAgent(use_xgboost=False, random_state=42)
        transformer_agent = TransformerAgent(
            num_features=4,
            d_model=64,
            num_heads=2,
            num_layers=2,
            d_ff=128
        )
        
        baseline_agent.load_models(model_status['baseline_path'])
        transformer_agent.load(model_status['transformer_path'])
        
        return baseline_agent, transformer_agent, True
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None, None, False


@st.cache_resource
def get_data_agent():
    """Cached DataAgent instance with loaded scaler."""
    agent = DataAgent(sequence_length=7)
    
    # Load scaler if available (fitted during training)
    scaler_path = MODELS_DIR / "scaler.joblib"
    if scaler_path.exists():
        agent.load_scaler(scaler_path)
    
    return agent


@st.cache_resource
def get_data_retriever():
    """Cached DataRetriever instance."""
    return DataRetriever()


@st.cache_resource
def get_narrator():
    """Cached NarratorAgent instance."""
    return NarratorAgent(use_emoji=True)


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_prediction(location: str):
    """Cache predictions to avoid redundant API calls.
    
    Uses hybrid approach:
    - Transformer for temperature prediction
    - Baseline for weather classification
    """
    baseline_agent, transformer_agent, success = load_saved_models()
    if not success:
        return None
    
    data_retriever = get_data_retriever()
    data_agent = get_data_agent()
    narrator = get_narrator()
    
    locations = DataRetriever.get_popular_locations()
    if location in locations:
        lat, lon, tz = locations[location]
        data_retriever.set_location(lat, lon, tz)
    
    try:
        sequence_df, current = data_retriever.get_data_for_prediction(sequence_length=7)
        sequence = data_agent.prepare_single_sequence(sequence_df)
        
        # Get Transformer prediction (good for temperature)
        transformer_pred = transformer_agent.predict(sequence)
        temp_pred = transformer_pred['temperature']
        
        # Get Baseline prediction (better for weather classification)
        weather_classes = ['sunny', 'cloudy', 'rainy', 'snowy']
        sequence_flat = sequence.numpy().reshape(1, -1)
        baseline_weather_pred = baseline_agent.predict(sequence_flat)
        weather_pred = weather_classes[int(baseline_weather_pred['weather_class'][0])]
        weather_probs = baseline_weather_pred.get('weather_probs', {weather_pred: 0.7})
        
        if isinstance(weather_probs, np.ndarray):
            weather_probs = {cls: float(p) for cls, p in zip(weather_classes, weather_probs[0])}
            confidence = max(weather_probs.values())
        else:
            confidence = weather_probs.get(weather_pred, 0.6)
        
        is_cold = temp_pred < 10
        
        forecast = narrator.generate_forecast(
            temperature=temp_pred,
            weather_type=weather_pred,
            confidence=confidence,
            is_cold_day=is_cold,
            location=location,
            target_date="Tomorrow"
        )
        
        return {
            'temperature': temp_pred,
            'weather_type': weather_pred,
            'weather_probs': weather_probs,
            'confidence': confidence,
            'is_cold': is_cold,
            'forecast': forecast,
            'current_weather': current,
            'sequence_df': sequence_df
        }
        
    except Exception as e:
        return None


def make_quick_prediction(location: str = "Warsaw"):
    """Make a prediction using saved models.
    
    Uses hybrid approach:
    - Transformer for temperature prediction (better at regression)
    - Baseline (RandomForest) for weather classification (better at imbalanced classification)
    """
    
    baseline_agent, transformer_agent, success = load_saved_models()
    if not success:
        return None
    
    data_retriever = get_data_retriever()
    data_agent = get_data_agent()
    narrator = NarratorAgent(use_emoji=True)
    
    locations = DataRetriever.get_popular_locations()
    if location in locations:
        lat, lon, tz = locations[location]
        data_retriever.set_location(lat, lon, tz)
    
    try:
        sequence_df, current = data_retriever.get_data_for_prediction(sequence_length=7)
        sequence = data_agent.prepare_single_sequence(sequence_df)
        
        # Get Transformer prediction (good for temperature)
        transformer_pred = transformer_agent.predict(sequence)
        temp_pred = transformer_pred['temperature']
        
        # Get Baseline prediction (better for weather classification with imbalanced data)
        sequence_flat = sequence.numpy().reshape(1, -1)
        baseline_weather_pred = baseline_agent.predict(sequence_flat)
        weather_classes = ['sunny', 'cloudy', 'rainy', 'snowy']
        weather_pred = weather_classes[int(baseline_weather_pred['weather_class'][0])]
        weather_probs = baseline_weather_pred.get('weather_probs', {weather_pred: 0.7})
        
        # Use max probability as confidence
        if isinstance(weather_probs, np.ndarray):
            weather_probs = {cls: float(p) for cls, p in zip(weather_classes, weather_probs[0])}
            confidence = max(weather_probs.values())
        else:
            confidence = weather_probs.get(weather_pred, 0.6)
        
        is_cold = temp_pred < 10
        
        forecast = narrator.generate_forecast(
            temperature=temp_pred,
            weather_type=weather_pred,
            confidence=confidence,
            is_cold_day=is_cold,
            location=location,
            target_date="Tomorrow"
        )
        
        return {
            'temperature': temp_pred,
            'weather_type': weather_pred,
            'weather_probs': weather_probs,
            'confidence': confidence,
            'is_cold': is_cold,
            'forecast': forecast,
            'current_weather': current,
            'sequence_df': sequence_df
        }
        
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


def render_weather_card(prediction: dict, location: str):
    """Render Google-style weather card."""
    
    temp = prediction['temperature']
    weather = prediction['weather_type']
    icon = get_weather_icon(weather)
    confidence = prediction['confidence']
    gradient = get_weather_gradient(weather)
    
    st.markdown(f"""
    <div class="weather-main-card {gradient}">
        <div class="weather-icon-large">{icon}</div>
        <div class="temp-large">{temp:.0f}°</div>
        <div class="weather-type-large">{weather}</div>
        <div class="location-text">📍 {location} • Tomorrow</div>
        <div class="confidence-badge">🎯 {confidence:.0%} confidence</div>
    </div>
    """, unsafe_allow_html=True)


def render_probability_bars(weather_probs: dict):
    """Render weather probability bars."""
    
    st.markdown("#### 📊 Weather Probabilities")
    
    weather_order = ['sunny', 'cloudy', 'rainy', 'snowy']
    icons = {'sunny': '☀️ Sunny', 'cloudy': '☁️ Cloudy', 'rainy': '🌧️ Rainy', 'snowy': '❄️ Snowy'}
    
    for weather in weather_order:
        prob = weather_probs.get(weather, 0)
        pct = int(prob * 100)
        
        col1, col2, col3 = st.columns([2, 5, 1])
        with col1:
            st.write(icons[weather])
        with col2:
            st.progress(prob)
        with col3:
            st.write(f"**{pct}%**")


def render_forecast_narrative(forecast: dict):
    """Render the forecast narrative."""
    
    st.markdown(f"""
    <div class="narrative-card">
        <h3>{forecast.get('headline', 'Weather Forecast')}</h3>
        <p>{forecast.get('description', '')}</p>
        <p><strong>💡 {forecast.get('recommendation', '')}</strong></p>
        <p><em>{forecast.get('confidence_note', '')}</em></p>
    </div>
    """, unsafe_allow_html=True)


def render_current_vs_prediction(prediction: dict):
    """Render current weather vs prediction comparison."""
    
    st.markdown("#### 🔮 Prediction Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="model-card transformer">
            <h4>🤖 Tomorrow's Prediction</h4>
            <div class="temp" style="color: #667eea;">{prediction['temperature']:.1f}°C</div>
            <p style="font-size: 2rem; margin: 10px 0;">{get_weather_icon(prediction['weather_type'])}</p>
            <p><strong>{prediction['weather_type'].capitalize()}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if 'current_weather' in prediction and prediction['current_weather']:
            current = prediction['current_weather']
            current_temp = current.get('temp', 'N/A')
            st.markdown(f"""
            <div class="model-card current">
                <h4>📍 Current Weather</h4>
                <div class="temp" style="color: #11998e;">{current_temp}°C</div>
                <p style="font-size: 2rem; margin: 10px 0;">🌡️</p>
                <p><strong>Humidity: {current.get('humidity', 'N/A')}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Current weather data not available")


def train_models_ui(data_source: str, num_epochs: int, location: str, days: int = 365):
    """Train models from UI with specified amount of data."""
    
    progress = st.progress(0)
    status = st.empty()
    
    try:
        # Create fresh agents (not cached) for training
        data_retriever = DataRetriever()
        data_agent = DataAgent(sequence_length=7)
        baseline_agent = BaselineAgent(use_xgboost=False, random_state=42)
        
        locations = DataRetriever.get_popular_locations()
        if location in locations:
            lat, lon, tz = locations[location]
            data_retriever.set_location(lat, lon, tz)
        
        status.text(f"📡 Fetching {days} days of weather data...")
        progress.progress(10)
        
        if data_source == "Fetch Fresh Data (Recommended)":
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            df = data_retriever.fetch_historical_data(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(DATA_DIR / f"weather_data_{location.lower()}.csv", index=False)
            status.text(f"✅ Fetched {len(df)} days of data!")
        else:
            data_path = DATA_DIR / f"weather_data_{location.lower()}.csv"
            if data_path.exists():
                df = pd.read_csv(data_path)
                df['date'] = pd.to_datetime(df['date'])
                status.text(f"📂 Using existing data: {len(df)} days")
            else:
                st.error("No existing data found. Please select 'Fetch Fresh Data'.")
                return False
        
        status.text(f"📊 Processing {len(df)} days of data...")
        progress.progress(20)
        
        splits, loaders = data_agent.prepare_data(data=df, batch_size=32)
        
        status.text("🌲 Training Baseline Model (RandomForest)...")
        progress.progress(30)
        
        aggregated = data_agent.get_aggregated_features(splits)
        X_train, y_temp_train, y_class_train = aggregated["train"]
        X_val, y_temp_val, y_class_val = aggregated["val"]
        
        baseline_agent.train(
            X_train, y_temp_train, y_class_train,
            X_val, y_temp_val, y_class_val
        )
        
        status.text("🤖 Training Transformer Model...")
        progress.progress(50)
        
        # Compute class weights to handle imbalanced data
        train_labels = splits['train'].class_targets.numpy()
        class_counts = np.bincount(train_labels, minlength=4)
        total_samples = len(train_labels)
        class_weights = total_samples / (4 * class_counts + 1e-6)
        class_weights = torch.FloatTensor(class_weights)
        
        transformer_agent = TransformerAgent(
            num_features=data_agent.num_features,
            d_model=64,
            num_heads=2,
            num_layers=2,
            d_ff=128
        )
        
        # Build with class weights for imbalanced data
        transformer_agent.build_training_components(
            learning_rate=1e-3,
            num_epochs=num_epochs,
            class_weights=class_weights
        )
        
        history = transformer_agent.train(
            loaders['train'],
            loaders['val'],
            num_epochs=num_epochs,
            early_stopping_patience=5
        )
        
        status.text("💾 Saving models and scaler...")
        progress.progress(90)
        
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        baseline_agent.save_models(MODELS_DIR / "baseline")
        transformer_agent.save(MODELS_DIR / "transformer_model.pth")
        data_agent.save_scaler(MODELS_DIR / "scaler.joblib")  # Save scaler for inference
        
        progress.progress(100)
        status.text("✅ Training complete!")
        
        st.session_state.models_loaded = True
        st.session_state.training_history = history
        
        return True
        
    except Exception as e:
        st.error(f"Training failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        return False


def main():
    """Main application."""
    
    st.markdown("""
    <div style="text-align: center; padding: 10px 0 20px 0;">
        <h1 style="margin: 0;">🌤️ Weather Forecaster AI</h1>
        <p style="color: #6c757d; margin: 5px 0 0 0;">
            Transformer + Multi-Agent System
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    model_status = check_saved_models()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        locations = list(DataRetriever.get_popular_locations().keys())
        location = st.selectbox("📍 Location", locations, index=locations.index("Warsaw") if "Warsaw" in locations else 0)
        st.session_state.location = location
        
        st.markdown("---")
        
        st.markdown("### 📦 Model Status")
        if model_status['transformer'] and model_status['baseline']:
            st.success("✅ Trained models found!")
            if st.button("🔮 Get Tomorrow's Forecast", type="primary"):
                with st.spinner("Making prediction..."):
                    pred = make_quick_prediction(location)
                    if pred:
                        st.session_state.predictions = pred
                        st.rerun()
        else:
            st.warning("⚠️ No trained models found")
            st.info("Train models or run:\n`python run.py --mode full`")
        
        st.markdown("---")
        
        st.markdown("### 🔧 Training")
        
        with st.expander("Train New Models"):
            training_days = st.slider("Days of training data", 90, 730, 365, help="More data = better model (up to 2 years)")
            data_source = st.radio(
                "Data Source",
                ["Fetch Fresh Data (Recommended)", "Use Existing Data"]
            )
            num_epochs = st.slider("Epochs", 5, 50, 20)
            
            if st.button("🚀 Start Training", type="primary"):
                success = train_models_ui(data_source, num_epochs, location, training_days)
                if success:
                    st.success("Models trained and saved!")
                    # Clear cached models to reload fresh ones
                    load_saved_models.clear()
                    st.rerun()
        
        st.markdown("---")
        
        st.markdown("### 📡 Data")
        
        with st.expander("Fetch New Data"):
            days = st.slider("Days of history", 90, 730, 365, help="Up to 2 years of historical data")
            if st.button("📥 Fetch Data"):
                with st.spinner(f"Fetching {days} days of data..."):
                    try:
                        data_retriever = DataRetriever()
                        locs = DataRetriever.get_popular_locations()
                        if location in locs:
                            lat, lon, tz = locs[location]
                            data_retriever.set_location(lat, lon, tz)
                        
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=days)
                        df = data_retriever.fetch_historical_data(
                            start_date.strftime('%Y-%m-%d'),
                            end_date.strftime('%Y-%m-%d')
                        )
                        DATA_DIR.mkdir(parents=True, exist_ok=True)
                        df.to_csv(DATA_DIR / f"weather_data_{location.lower()}.csv", index=False)
                        st.success(f"✅ Fetched {len(df)} days of data!")
                    except Exception as e:
                        st.error(f"Failed: {e}")
    
    # Main content
    tabs = st.tabs(["🌤️ Forecast", "📊 Analysis", "📈 History", "ℹ️ About"])
    
    # FORECAST TAB
    with tabs[0]:
        if st.session_state.predictions:
            pred = st.session_state.predictions
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                render_weather_card(pred, location)
            
            with col2:
                st.markdown("<div class='prob-card'>", unsafe_allow_html=True)
                render_probability_bars(pred['weather_probs'])
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                render_forecast_narrative(pred['forecast'])
            
            with col2:
                render_current_vs_prediction(pred)
            
            if 'sequence_df' in pred and pred['sequence_df'] is not None:
                st.markdown("### 📈 Weather Analysis (Last 7 Days → Tomorrow's Prediction)")
                df = pred['sequence_df']
                
                # Create comprehensive subplot figure
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=(
                        '<b>🌡️ Temperature Trend</b>',
                        '<b>💧 Humidity & Pressure</b>',
                        '<b>💨 Wind Speed</b>',
                        '<b>📊 Prediction Probabilities</b>'
                    ),
                    specs=[[{"type": "scatter"}, {"type": "scatter"}],
                           [{"type": "scatter"}, {"type": "bar"}]],
                    vertical_spacing=0.18,
                    horizontal_spacing=0.12
                )
                
                x_data = df['date'] if 'date' in df.columns else list(range(1, len(df) + 1))
                
                # Temperature plot with min/max range
                fig.add_trace(go.Scatter(
                    x=x_data, y=df['temp'],
                    mode='lines+markers',
                    name='Avg Temp',
                    line=dict(color='#667eea', width=3),
                    marker=dict(size=8, symbol='circle'),
                    hovertemplate='%{x}<br>Temp: %{y:.1f}°C<extra></extra>',
                    showlegend=False
                ), row=1, col=1)
                
                if 'temp_max' in df.columns and 'temp_min' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=x_data, y=df['temp_max'],
                        mode='lines',
                        name='Max',
                        line=dict(color='#ff6b6b', width=1.5, dash='dot'),
                        hovertemplate='Max: %{y:.1f}°C<extra></extra>',
                        showlegend=False
                    ), row=1, col=1)
                    fig.add_trace(go.Scatter(
                        x=x_data, y=df['temp_min'],
                        mode='lines',
                        name='Min',
                        line=dict(color='#4ecdc4', width=1.5, dash='dot'),
                        fill='tonexty',
                        fillcolor='rgba(78, 205, 196, 0.1)',
                        hovertemplate='Min: %{y:.1f}°C<extra></extra>',
                        showlegend=False
                    ), row=1, col=1)
                
                # Add tomorrow's prediction as a point
                tomorrow_date = pd.to_datetime(x_data.iloc[-1]) + timedelta(days=1) if 'date' in df.columns else len(df) + 1
                fig.add_trace(go.Scatter(
                    x=[tomorrow_date], y=[pred['temperature']],
                    mode='markers',
                    name='⭐ Tomorrow',
                    marker=dict(size=14, color='#ff9500', symbol='star', line=dict(width=2, color='white')),
                    hovertemplate='Tomorrow<br>Predicted: %{y:.1f}°C<extra></extra>',
                    showlegend=False
                ), row=1, col=1)
                
                # Humidity plot
                if 'humidity' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=x_data, y=df['humidity'],
                        mode='lines+markers',
                        name='Humidity %',
                        line=dict(color='#3498db', width=2),
                        marker=dict(size=6),
                        hovertemplate='Humidity: %{y:.0f}%<extra></extra>',
                        showlegend=False
                    ), row=1, col=2)
                
                # Pressure plot
                if 'pressure' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=x_data, y=df['pressure'],
                        mode='lines+markers',
                        name='Pressure hPa',
                        line=dict(color='#9b59b6', width=2),
                        marker=dict(size=6),
                        hovertemplate='Pressure: %{y:.0f} hPa<extra></extra>',
                        showlegend=False
                    ), row=1, col=2)
                
                # Wind speed plot
                if 'wind_speed' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=x_data, y=df['wind_speed'],
                        mode='lines+markers',
                        name='Wind km/h',
                        line=dict(color='#1abc9c', width=2),
                        marker=dict(size=6),
                        fill='tozeroy',
                        fillcolor='rgba(26, 188, 156, 0.2)',
                        hovertemplate='Wind: %{y:.1f} km/h<extra></extra>',
                        showlegend=False
                    ), row=2, col=1)
                
                # Weather probabilities bar chart
                weather_types = list(pred['weather_probs'].keys())
                weather_probs = list(pred['weather_probs'].values())
                weather_colors = ['#FFB347', '#B8C6DB', '#3A7BD5', '#E6DADA']
                weather_labels = ['☀️', '☁️', '🌧️', '❄️']
                
                fig.add_trace(go.Bar(
                    x=weather_labels,
                    y=[p * 100 for p in weather_probs],
                    name='Probability',
                    marker=dict(color=weather_colors, line=dict(width=2, color='#333')),
                    text=[f'{p*100:.0f}%' for p in weather_probs],
                    textposition='outside',
                    textfont=dict(size=12, color='#333'),
                    hovertemplate='%{x}: %{y:.1f}%<extra></extra>',
                    showlegend=False
                ), row=2, col=2)
                
                # Update layout - cleaner without legend
                fig.update_layout(
                    template='plotly_white',
                    height=550,
                    showlegend=False,
                    margin=dict(l=50, r=50, t=80, b=50),
                    font=dict(size=11)
                )
                
                # Update subplot titles font
                for annotation in fig['layout']['annotations']:
                    annotation['font'] = dict(size=14, color='#2c3e50')
                
                # Update axes - cleaner labels
                fig.update_xaxes(tickfont=dict(size=9), row=1, col=1)
                fig.update_yaxes(title=dict(text="°C", font=dict(size=11)), tickfont=dict(size=9), row=1, col=1)
                fig.update_xaxes(tickfont=dict(size=9), row=1, col=2)
                fig.update_yaxes(title=dict(text="Value", font=dict(size=11)), tickfont=dict(size=9), row=1, col=2)
                fig.update_xaxes(tickfont=dict(size=9), row=2, col=1)
                fig.update_yaxes(title=dict(text="km/h", font=dict(size=11)), tickfont=dict(size=9), row=2, col=1)
                fig.update_xaxes(tickfont=dict(size=12), row=2, col=2)
                fig.update_yaxes(title=dict(text="%", font=dict(size=11)), tickfont=dict(size=9), range=[0, 100], row=2, col=2)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add summary statistics
                st.markdown("#### 📊 7-Day Statistics")
                stat_cols = st.columns(5)
                with stat_cols[0]:
                    st.metric("📈 Avg Temp", f"{df['temp'].mean():.1f}°C")
                with stat_cols[1]:
                    st.metric("🔥 Max Temp", f"{df['temp'].max():.1f}°C")
                with stat_cols[2]:
                    st.metric("❄️ Min Temp", f"{df['temp'].min():.1f}°C")
                with stat_cols[3]:
                    if 'humidity' in df.columns:
                        st.metric("💧 Avg Humidity", f"{df['humidity'].mean():.0f}%")
                with stat_cols[4]:
                    if 'wind_speed' in df.columns:
                        st.metric("💨 Avg Wind", f"{df['wind_speed'].mean():.1f} km/h")
        
        else:
            st.markdown("""
            <div style="text-align: center; padding: 60px 20px;">
                <div style="font-size: 6rem; margin-bottom: 20px;">🌤️</div>
                <h2>Welcome to Weather Forecaster AI</h2>
                <p style="color: #6c757d; font-size: 1.1rem; max-width: 600px; margin: 20px auto;">
                    Get accurate weather predictions for tomorrow powered by a tiny Transformer model
                    and multi-agent system.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if model_status['transformer'] and model_status['baseline']:
                st.success("✅ **Trained models detected!** Click **'Get Tomorrow's Forecast'** in the sidebar.")
            else:
                st.warning("⚠️ **No trained models found.** Train models using the sidebar or run:")
                st.code("python run.py --mode full --days 180 --epochs 15", language="bash")
    
    # ANALYSIS TAB
    with tabs[1]:
        st.markdown("### 📊 Model Metrics & Comparison")
        
        # Load saved metrics
        metrics_path = MODELS_DIR / "metrics.json"
        if metrics_path.exists():
            import json
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            st.info(f"📅 Last trained: {metrics.get('timestamp', 'Unknown')[:19]}")
            
            # Winner badges
            col1, col2 = st.columns(2)
            with col1:
                temp_winner = metrics.get('comparison', {}).get('temp_winner', 'Unknown')
                st.success(f"🌡️ **Temperature Winner:** {temp_winner}")
            with col2:
                weather_winner = metrics.get('comparison', {}).get('weather_winner', 'Unknown')
                st.success(f"☁️ **Weather Winner:** {weather_winner}")
            
            st.markdown("---")
            
            # Metrics comparison table
            baseline = metrics.get('baseline', {})
            transformer = metrics.get('transformer', {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🌲 Baseline (RandomForest)")
                st.metric("MAE", f"{baseline.get('temp_MAE', 0):.2f}°C")
                st.metric("RMSE", f"{baseline.get('temp_RMSE', 0):.2f}°C")
                st.metric("R²", f"{baseline.get('temp_R2', 0):.3f}")
                st.metric("Accuracy", f"{baseline.get('weather_Accuracy', 0)*100:.1f}%")
                st.metric("F1 Score", f"{baseline.get('weather_F1_macro', 0)*100:.1f}%")
            
            with col2:
                st.markdown("#### 🤖 Transformer")
                st.metric("MAE", f"{transformer.get('temp_MAE', 0):.2f}°C")
                st.metric("RMSE", f"{transformer.get('temp_RMSE', 0):.2f}°C")
                st.metric("R²", f"{transformer.get('temp_R2', 0):.3f}")
                st.metric("Accuracy", f"{transformer.get('weather_Accuracy', 0)*100:.1f}%")
                st.metric("F1 Score", f"{transformer.get('weather_F1_macro', 0)*100:.1f}%")
            
            # Bar chart comparison
            st.markdown("### 📊 Visual Comparison")
            
            metrics_names = ['MAE (°C)', 'Accuracy (%)', 'F1 (%)']
            baseline_vals = [
                baseline.get('temp_MAE', 0),
                baseline.get('weather_Accuracy', 0) * 100,
                baseline.get('weather_F1_macro', 0) * 100
            ]
            transformer_vals = [
                transformer.get('temp_MAE', 0),
                transformer.get('weather_Accuracy', 0) * 100,
                transformer.get('weather_F1_macro', 0) * 100
            ]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Baseline', x=metrics_names, y=baseline_vals, marker_color='#11998e'))
            fig.add_trace(go.Bar(name='Transformer', x=metrics_names, y=transformer_vals, marker_color='#667eea'))
            fig.update_layout(barmode='group', template='plotly_white', height=350)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("No metrics found. Train models to see comparison.")
            st.code("python run.py --mode full --days 180 --epochs 15", language="bash")
        
        st.markdown("---")
        st.markdown("### 📈 Weather Data")
        
        data_path = DATA_DIR / f"weather_data_{location.lower()}.csv"
        if data_path.exists():
            df = pd.read_csv(data_path)
            df['date'] = pd.to_datetime(df['date'])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📅 Days of Data", len(df))
            with col2:
                st.metric("🌡️ Avg Temp", f"{df['temp'].mean():.1f}°C")
            with col3:
                st.metric("🔥 Max Temp", f"{df['temp'].max():.1f}°C")
            with col4:
                st.metric("❄️ Min Temp", f"{df['temp'].min():.1f}°C")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['date'], y=df['temp'],
                name='Temperature', line=dict(color='#667eea', width=2)
            ))
            if 'temp_max' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['date'], y=df['temp_max'],
                    name='Max', line=dict(color='#ff6b6b', width=1, dash='dot')
                ))
            if 'temp_min' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['date'], y=df['temp_min'],
                    name='Min', line=dict(color='#4ecdc4', width=1, dash='dot')
                ))
            fig.update_layout(
                title=f'Temperature History - {location}',
                xaxis_title='Date',
                yaxis_title='Temperature (°C)',
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No data found for {location}. Use the sidebar to fetch data.")
    
    # TRAINING HISTORY TAB
    with tabs[2]:
        st.markdown("### 📈 Training History")
        
        # First check for saved models trained via CLI
        model_status = check_saved_models()
        metrics_path = MODELS_DIR / "metrics.json"
        
        if model_status['transformer'] or model_status['baseline']:
            st.markdown("#### 💾 Saved Models")
            
            col1, col2 = st.columns(2)
            with col1:
                if model_status['baseline']:
                    import os
                    baseline_dir = model_status['baseline_path']
                    # Calculate total size of files in baseline directory
                    baseline_size = sum(f.stat().st_size for f in baseline_dir.iterdir() if f.is_file())
                    baseline_time = datetime.fromtimestamp(baseline_dir.stat().st_mtime)
                    st.success(f"✅ **Baseline Model** (RandomForest)")
                    st.caption(f"📁 `{baseline_dir.name}/` (2 files)")
                    st.caption(f"📅 Last modified: {baseline_time.strftime('%Y-%m-%d %H:%M')}")
                    st.caption(f"📦 Size: {baseline_size / 1024:.1f} KB")
                else:
                    st.warning("❌ Baseline model not found")
            
            with col2:
                if model_status['transformer']:
                    transformer_stat = os.stat(model_status['transformer_path'])
                    transformer_time = datetime.fromtimestamp(transformer_stat.st_mtime)
                    st.success(f"✅ **Transformer Model**")
                    st.caption(f"📁 `{model_status['transformer_path'].name}`")
                    st.caption(f"📅 Last modified: {transformer_time.strftime('%Y-%m-%d %H:%M')}")
                    st.caption(f"📦 Size: {transformer_stat.st_size / 1024:.1f} KB")
                else:
                    st.warning("❌ Transformer model not found")
            
            # Show metrics if available
            if metrics_path.exists():
                import json
                with open(metrics_path, 'r') as f:
                    saved_metrics = json.load(f)
                
                st.markdown("---")
                st.markdown("#### 📊 Training Metrics (from last training run)")
                st.caption(f"🕐 Trained on: {saved_metrics.get('timestamp', 'Unknown')[:19]}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**🌲 Baseline Performance**")
                    baseline_m = saved_metrics.get('baseline', {})
                    st.write(f"- MAE: {baseline_m.get('temp_MAE', 'N/A'):.2f}°C" if isinstance(baseline_m.get('temp_MAE'), (int, float)) else f"- MAE: {baseline_m.get('temp_MAE', 'N/A')}")
                    st.write(f"- RMSE: {baseline_m.get('temp_RMSE', 'N/A'):.2f}°C" if isinstance(baseline_m.get('temp_RMSE'), (int, float)) else f"- RMSE: {baseline_m.get('temp_RMSE', 'N/A')}")
                    st.write(f"- Accuracy: {baseline_m.get('weather_Accuracy', 0)*100:.1f}%" if isinstance(baseline_m.get('weather_Accuracy'), (int, float)) else f"- Accuracy: {baseline_m.get('weather_Accuracy', 'N/A')}")
                
                with col2:
                    st.markdown("**🤖 Transformer Performance**")
                    trans_m = saved_metrics.get('transformer', {})
                    st.write(f"- MAE: {trans_m.get('temp_MAE', 'N/A'):.2f}°C" if isinstance(trans_m.get('temp_MAE'), (int, float)) else f"- MAE: {trans_m.get('temp_MAE', 'N/A')}")
                    st.write(f"- RMSE: {trans_m.get('temp_RMSE', 'N/A'):.2f}°C" if isinstance(trans_m.get('temp_RMSE'), (int, float)) else f"- RMSE: {trans_m.get('temp_RMSE', 'N/A')}")
                    st.write(f"- Accuracy: {trans_m.get('weather_Accuracy', 0)*100:.1f}%" if isinstance(trans_m.get('weather_Accuracy'), (int, float)) else f"- Accuracy: {trans_m.get('weather_Accuracy', 'N/A')}")
                
                comparison = saved_metrics.get('comparison', {})
                st.info(f"🏆 **Winners:** Temperature → {comparison.get('temp_winner', 'N/A')} | Weather → {comparison.get('weather_winner', 'N/A')}")
        else:
            st.warning("⚠️ No saved models found. Train models first!")
            st.code("python run.py --mode full --days 180 --epochs 15", language="bash")
        
        st.markdown("---")
        
        # Show live training history from current session
        st.markdown("#### 📈 Current Session Training Progress")
        if hasattr(st.session_state, 'training_history') and st.session_state.training_history:
            history = st.session_state.training_history
            
            epochs = list(range(1, len(history.train_losses) + 1))
            
            fig = make_subplots(rows=1, cols=2, subplot_titles=('Loss', 'Validation Metrics'))
            
            fig.add_trace(go.Scatter(x=epochs, y=history.train_losses, name='Train Loss', 
                                    line=dict(color='#667eea')), row=1, col=1)
            fig.add_trace(go.Scatter(x=epochs, y=history.val_losses, name='Val Loss',
                                    line=dict(color='#ff6b6b')), row=1, col=1)
            
            val_acc = [m.get('weather_accuracy', 0) for m in history.val_metrics]
            fig.add_trace(go.Scatter(x=epochs, y=val_acc, name='Val Accuracy',
                                    line=dict(color='#4ecdc4')), row=1, col=2)
            
            fig.update_layout(height=400, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"🏆 **Best epoch:** {history.best_epoch + 1} | **Best val loss:** {history.best_val_loss:.4f}")
        else:
            st.caption("💡 Train models from the sidebar to see live training curves here.")
    
    # ABOUT TAB
    with tabs[3]:
        st.markdown("""
        ### 🌤️ About Weather Forecaster AI
        
        A **Multi-Agent System (MAS)** for weather prediction.
        
        #### 🤖 The 6 Agents
        
        | Agent | Role |
        |-------|------|
        | 📡 **Data Retriever** | Fetches live weather from Open-Meteo API |
        | 📊 **Data Agent** | Preprocesses data, creates sequences |
        | 🌲 **Baseline Agent** | RandomForest/XGBoost models |
        | 🤖 **Transformer Agent** | Tiny transformer (72K params) |
        | 📈 **Evaluation Agent** | Compares models, ensemble predictions |
        | 📝 **Narrator Agent** | Generates human-readable forecasts |
        
        #### 🎯 Predictions
        - **Temperature**: Tomorrow's temperature in °C
        - **Weather Type**: ☀️ Sunny / ☁️ Cloudy / 🌧️ Rainy / ❄️ Snowy
        
        #### 🚀 Commands
        ```bash
        python run.py --mode full --days 180 --epochs 15  # Train
        python run.py --mode predict                       # Predict
        python run.py --mode ui                            # This UI
        ```
        """)


if __name__ == "__main__":
    main()
