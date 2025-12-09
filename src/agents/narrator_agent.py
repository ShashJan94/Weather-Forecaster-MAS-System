"""
Narrator Agent
Generates human-readable weather forecast narratives.

This agent is responsible for:
- Converting predictions to natural language
- Creating engaging forecast descriptions
- Providing weather-appropriate recommendations
"""

from typing import Dict, Any, Optional
from datetime import datetime
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NarratorAgent:
    """
    Agent for generating natural language weather forecasts.
    
    Uses templates and rules to create human-readable forecasts.
    No external LLM required - keeps it laptop-friendly.
    """
    
    # Weather type icons and descriptions
    WEATHER_INFO = {
        "sunny": {
            "icon": "☀️",
            "descriptions": [
                "expect clear skies and sunshine",
                "it will be a bright and sunny day",
                "perfect weather for outdoor activities",
                "clear conditions throughout the day"
            ],
            "tips": [
                "Don't forget your sunscreen!",
                "Great day for a walk in the park.",
                "Perfect weather for outdoor activities.",
                "Stay hydrated in the sunshine."
            ]
        },
        "cloudy": {
            "icon": "☁️",
            "descriptions": [
                "expect overcast skies",
                "clouds will dominate the sky",
                "a grey and cloudy day ahead",
                "partly to mostly cloudy conditions"
            ],
            "tips": [
                "A good day for indoor activities.",
                "Light layers recommended.",
                "Consider bringing a light jacket.",
                "Might be a good day for a coffee shop visit."
            ]
        },
        "rainy": {
            "icon": "🌧️",
            "descriptions": [
                "expect rain showers",
                "rain is in the forecast",
                "wet conditions expected",
                "precipitation likely throughout the day"
            ],
            "tips": [
                "Don't forget your umbrella!",
                "Waterproof jacket recommended.",
                "Watch out for slippery surfaces.",
                "Perfect weather for a cozy day indoors."
            ]
        },
        "snowy": {
            "icon": "❄️",
            "descriptions": [
                "expect snowfall",
                "snow is in the forecast",
                "winter wonderland conditions ahead",
                "prepare for snowy conditions"
            ],
            "tips": [
                "Dress warmly in layers.",
                "Allow extra travel time.",
                "Watch for icy conditions.",
                "Hot cocoa weather!"
            ]
        }
    }
    
    # Temperature descriptors
    TEMP_DESCRIPTORS = {
        "freezing": {"max": -10, "words": ["frigid", "extremely cold", "bitter"]},
        "very_cold": {"max": 0, "words": ["very cold", "freezing", "icy"]},
        "cold": {"max": 5, "words": ["cold", "chilly", "brisk"]},
        "cool": {"max": 12, "words": ["cool", "crisp", "mild"]},
        "mild": {"max": 18, "words": ["mild", "pleasant", "comfortable"]},
        "warm": {"max": 25, "words": ["warm", "nice", "pleasant"]},
        "hot": {"max": 32, "words": ["hot", "warm", "summery"]},
        "very_hot": {"max": float('inf'), "words": ["very hot", "scorching", "sweltering"]}
    }
    
    def __init__(self, use_emoji: bool = True):
        """
        Initialize the Narrator Agent.
        
        Args:
            use_emoji: Whether to include emojis in forecasts
        """
        self.use_emoji = use_emoji
    
    def generate_forecast(
        self,
        temperature: float,
        weather_type: str,
        confidence: float = 0.8,
        is_cold_day: bool = False,
        location: str = "your location",
        target_date: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate a complete weather forecast narrative.
        
        Args:
            temperature: Predicted temperature in Celsius
            weather_type: Weather type (sunny/cloudy/rainy/snowy)
            confidence: Prediction confidence (0-1)
            is_cold_day: Whether it's a cold day
            location: Location name
            target_date: Target date string (optional)
            
        Returns:
            Dictionary with forecast components
        """
        # Get weather info
        weather_info = self.WEATHER_INFO.get(weather_type, self.WEATHER_INFO["cloudy"])
        icon = weather_info["icon"] if self.use_emoji else ""
        
        # Get temperature descriptor
        temp_desc = self._get_temp_descriptor(temperature)
        
        # Build date string
        date_str = target_date or "Tomorrow"
        
        # Generate main headline
        headline = self._generate_headline(
            temperature, weather_type, icon, date_str, location
        )
        
        # Generate detailed description
        description = self._generate_description(
            temperature, weather_type, temp_desc, 
            weather_info["descriptions"], confidence
        )
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            temperature, weather_type, weather_info["tips"], is_cold_day
        )
        
        # Generate confidence note
        confidence_note = self._generate_confidence_note(confidence)
        
        # Full narrative
        full_narrative = f"{headline}\n\n{description}\n\n{recommendation}"
        if confidence < 0.7:
            full_narrative += f"\n\n{confidence_note}"
        
        return {
            "headline": headline,
            "description": description,
            "recommendation": recommendation,
            "confidence_note": confidence_note,
            "full_narrative": full_narrative,
            "icon": icon,
            "temperature_description": temp_desc
        }
    
    def _get_temp_descriptor(self, temperature: float) -> str:
        """Get descriptive word for temperature."""
        for category, info in self.TEMP_DESCRIPTORS.items():
            if temperature < info["max"]:
                return random.choice(info["words"])
        return "very hot"
    
    def _generate_headline(
        self,
        temperature: float,
        weather_type: str,
        icon: str,
        date_str: str,
        location: str
    ) -> str:
        """Generate the forecast headline."""
        temp_rounded = round(temperature)
        
        templates = [
            f"{icon} {date_str}: {weather_type.capitalize()} with {temp_rounded}°C in {location}",
            f"{icon} {temp_rounded}°C and {weather_type} expected {date_str.lower()} in {location}",
            f"{date_str}'s Forecast for {location}: {icon} {weather_type.capitalize()}, {temp_rounded}°C"
        ]
        
        return random.choice(templates)
    
    def _generate_description(
        self,
        temperature: float,
        weather_type: str,
        temp_desc: str,
        weather_descriptions: list,
        confidence: float
    ) -> str:
        """Generate the detailed description."""
        weather_desc = random.choice(weather_descriptions)
        temp_rounded = round(temperature)
        
        # Build sentence
        if confidence > 0.8:
            confidence_word = ""
        elif confidence > 0.6:
            confidence_word = "likely "
        else:
            confidence_word = "possibly "
        
        templates = [
            f"Tomorrow, {confidence_word}{weather_desc} with temperatures around {temp_rounded}°C. "
            f"It will feel {temp_desc} throughout the day.",
            
            f"The forecast shows {confidence_word}{temp_desc} conditions with temperatures "
            f"reaching approximately {temp_rounded}°C. You can {weather_desc.replace('expect ', '')}.",
            
            f"Expect {temp_desc} weather tomorrow with the thermometer showing around {temp_rounded}°C. "
            f"The day will {confidence_word}bring {weather_type} conditions."
        ]
        
        return random.choice(templates)
    
    def _generate_recommendation(
        self,
        temperature: float,
        weather_type: str,
        weather_tips: list,
        is_cold_day: bool
    ) -> str:
        """Generate recommendations based on conditions."""
        recommendations = []
        
        # Add weather-specific tip
        recommendations.append(random.choice(weather_tips))
        
        # Add temperature-based recommendation
        if temperature < 0:
            recommendations.append("Wear warm winter clothing and watch for ice.")
        elif temperature < 10:
            recommendations.append("Layer up to stay comfortable.")
        elif temperature > 30:
            recommendations.append("Stay cool and drink plenty of water.")
        
        # Add cold day note
        if is_cold_day:
            recommendations.append("It's officially a cold day - bundle up!")
        
        return " ".join(recommendations[:2])
    
    def _generate_confidence_note(self, confidence: float) -> str:
        """Generate a note about prediction confidence."""
        if confidence > 0.9:
            return "This forecast has high confidence."
        elif confidence > 0.7:
            return "This forecast has moderate confidence."
        elif confidence > 0.5:
            return "⚠️ This forecast has lower confidence. Conditions may vary."
        else:
            return "⚠️ This forecast is uncertain. Please check for updates."
    
    def generate_comparison_narrative(
        self,
        baseline_pred: Dict[str, Any],
        transformer_pred: Dict[str, Any],
        ensemble_pred: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a narrative comparing model predictions.
        
        Args:
            baseline_pred: Baseline model prediction
            transformer_pred: Transformer model prediction
            ensemble_pred: Optional ensemble prediction
            
        Returns:
            Comparison narrative
        """
        base_temp = baseline_pred["temperature"]
        trans_temp = transformer_pred["temperature"]
        base_weather = baseline_pred["weather_type"]
        trans_weather = transformer_pred["weather_type"]
        
        lines = ["📊 **Model Comparison:**\n"]
        
        # Temperature comparison
        temp_diff = abs(base_temp - trans_temp)
        if temp_diff < 1:
            lines.append(f"Both models agree on temperature: around {round((base_temp + trans_temp) / 2)}°C")
        else:
            lines.append(f"• Baseline predicts: {round(base_temp)}°C")
            lines.append(f"• Transformer predicts: {round(trans_temp)}°C")
            lines.append(f"  (Difference: {temp_diff:.1f}°C)")
        
        lines.append("")
        
        # Weather type comparison
        if base_weather == trans_weather:
            icon = self.WEATHER_INFO[base_weather]["icon"]
            lines.append(f"Both models agree on weather: {icon} {base_weather}")
        else:
            base_icon = self.WEATHER_INFO[base_weather]["icon"]
            trans_icon = self.WEATHER_INFO[trans_weather]["icon"]
            lines.append(f"• Baseline predicts: {base_icon} {base_weather}")
            lines.append(f"• Transformer predicts: {trans_icon} {trans_weather}")
        
        # Ensemble prediction
        if ensemble_pred:
            lines.append("")
            lines.append("**Ensemble Recommendation:**")
            ens_icon = self.WEATHER_INFO[ensemble_pred["weather_type"]]["icon"]
            lines.append(
                f"{ens_icon} {round(ensemble_pred['temperature'])}°C, "
                f"{ensemble_pred['weather_type']} "
                f"(confidence: {ensemble_pred.get('confidence', 0.8):.0%})"
            )
        
        return "\n".join(lines)
    
    def generate_weekly_summary(
        self,
        daily_forecasts: list
    ) -> str:
        """
        Generate a weekly weather summary.
        
        Args:
            daily_forecasts: List of daily forecast dicts
            
        Returns:
            Weekly summary narrative
        """
        if not daily_forecasts:
            return "No forecast data available."
        
        temps = [f["temperature"] for f in daily_forecasts]
        avg_temp = sum(temps) / len(temps)
        min_temp = min(temps)
        max_temp = max(temps)
        
        # Count weather types
        weather_counts: Dict[str, int] = {}
        for f in daily_forecasts:
            wt = f["weather_type"]
            weather_counts[wt] = weather_counts.get(wt, 0) + 1
        
        dominant_weather = max(weather_counts.keys(), key=lambda k: weather_counts[k])
        icon = self.WEATHER_INFO[dominant_weather]["icon"]
        
        summary = f"""📅 **Weekly Weather Summary**

Temperature Range: {round(min_temp)}°C to {round(max_temp)}°C (avg: {round(avg_temp)}°C)

{icon} Predominantly {dominant_weather} conditions expected.

Weather breakdown:
"""
        for wt, count in sorted(weather_counts.items(), key=lambda x: -x[1]):
            wt_icon = self.WEATHER_INFO[wt]["icon"]
            summary += f"  • {wt_icon} {wt.capitalize()}: {count} day(s)\n"
        
        return summary


if __name__ == "__main__":
    # Test the narrator agent
    agent = NarratorAgent(use_emoji=True)
    
    # Test single forecast
    forecast = agent.generate_forecast(
        temperature=12.5,
        weather_type="rainy",
        confidence=0.85,
        is_cold_day=False,
        location="Warsaw",
        target_date="Tomorrow"
    )
    
    print("=" * 60)
    print("SINGLE FORECAST")
    print("=" * 60)
    print(forecast["full_narrative"])
    
    # Test comparison narrative
    baseline_pred = {
        "temperature": 14.2,
        "weather_type": "cloudy"
    }
    transformer_pred = {
        "temperature": 13.8,
        "weather_type": "rainy"
    }
    ensemble_pred = {
        "temperature": 14.0,
        "weather_type": "rainy",
        "confidence": 0.78
    }
    
    print("\n" + "=" * 60)
    print("COMPARISON NARRATIVE")
    print("=" * 60)
    print(agent.generate_comparison_narrative(baseline_pred, transformer_pred, ensemble_pred))
    
    # Test weekly summary
    daily_forecasts = [
        {"temperature": 12, "weather_type": "rainy"},
        {"temperature": 14, "weather_type": "cloudy"},
        {"temperature": 16, "weather_type": "sunny"},
        {"temperature": 18, "weather_type": "sunny"},
        {"temperature": 15, "weather_type": "cloudy"},
        {"temperature": 13, "weather_type": "rainy"},
        {"temperature": 11, "weather_type": "rainy"},
    ]
    
    print("\n" + "=" * 60)
    print("WEEKLY SUMMARY")
    print("=" * 60)
    print(agent.generate_weekly_summary(daily_forecasts))
