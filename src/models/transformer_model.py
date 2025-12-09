"""
Tiny Weather Transformer Model
A lightweight transformer architecture for time-series weather prediction.

The model performs joint regression (temperature) and classification (weather type)
using a shared transformer encoder backbone.
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TransformerModelConfig:
    """Configuration for the Transformer model."""
    num_features: int = 4       # Number of input features
    d_model: int = 64           # Model dimension
    num_heads: int = 2          # Number of attention heads
    num_layers: int = 2         # Number of encoder layers
    d_ff: int = 128             # Feed-forward dimension
    dropout: float = 0.1        # Dropout rate
    max_seq_len: int = 14       # Maximum sequence length
    num_classes: int = 4        # Number of weather classes


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer.
    
    Adds sinusoidal positional embeddings to input embeddings so the model
    can understand the temporal order of the sequence.
    """
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length to pre-compute
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (not a parameter, but should be part of state_dict)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TinyWeatherTransformer(nn.Module):
    """
    Tiny Transformer for Weather Forecasting.
    
    Architecture:
    1. Linear projection: num_features -> d_model
    2. Positional encoding
    3. Transformer encoder (N layers)
    4. Pooling (mean over sequence)
    5. Two output heads:
       - Regression head: predicts temperature
       - Classification head: predicts weather type
    """
    
    def __init__(self, config: Optional[TransformerModelConfig] = None, **kwargs):
        """
        Args:
            config: Model configuration object
            **kwargs: Individual config parameters (override config)
        """
        super().__init__()
        
        # Handle configuration
        if config is None:
            config = TransformerModelConfig()
        
        # Allow kwargs to override config
        self.config = TransformerModelConfig(
            num_features=kwargs.get("num_features", config.num_features),
            d_model=kwargs.get("d_model", config.d_model),
            num_heads=kwargs.get("num_heads", config.num_heads),
            num_layers=kwargs.get("num_layers", config.num_layers),
            d_ff=kwargs.get("d_ff", config.d_ff),
            dropout=kwargs.get("dropout", config.dropout),
            max_seq_len=kwargs.get("max_seq_len", config.max_seq_len),
            num_classes=kwargs.get("num_classes", config.num_classes)
        )
        
        # Input projection
        self.input_projection = nn.Linear(
            self.config.num_features, 
            self.config.d_model
        )
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=self.config.d_model,
            max_len=self.config.max_seq_len,
            dropout=self.config.dropout
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.d_ff,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.num_layers,
            enable_nested_tensor=False
        )
        
        # Layer normalization after encoder
        self.layer_norm = nn.LayerNorm(self.config.d_model)
        
        # Regression head (temperature prediction)
        self.regression_head = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_model // 2, 1)
        )
        
        # Classification head (weather type prediction)
        self.classification_head = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_model // 2, self.config.num_classes)
        )
        
        # Cold day classifier (binary)
        self.cold_day_head = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model // 4),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_model // 4, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)
            mask: Optional attention mask
            
        Returns:
            Dictionary with:
                - temp_pred: Temperature prediction (batch_size, 1)
                - weather_logits: Weather class logits (batch_size, num_classes)
                - cold_day_logit: Cold day prediction logit (batch_size, 1)
                - features: Pooled features for analysis (batch_size, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input features to model dimension
        x = self.input_projection(x)  # (batch, seq, d_model)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x, mask=mask)  # (batch, seq, d_model)
        
        # Apply layer norm
        x = self.layer_norm(x)
        
        # Pool over sequence dimension (mean pooling)
        # Could also use last timestep: x[:, -1, :]
        pooled = x.mean(dim=1)  # (batch, d_model)
        
        # Generate predictions from each head
        temp_pred = self.regression_head(pooled)          # (batch, 1)
        weather_logits = self.classification_head(pooled)  # (batch, num_classes)
        cold_day_logit = self.cold_day_head(pooled)       # (batch, 1)
        
        return {
            "temp_pred": temp_pred,
            "weather_logits": weather_logits,
            "cold_day_logit": cold_day_logit,
            "features": pooled
        }
    
    def predict(
        self, 
        x: torch.Tensor,
        return_probs: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with the model (inference mode).
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)
            return_probs: Whether to return probabilities instead of logits
            
        Returns:
            Dictionary with predictions
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            
            if return_probs:
                outputs["weather_probs"] = torch.softmax(
                    outputs["weather_logits"], dim=-1
                )
                outputs["cold_day_prob"] = torch.sigmoid(
                    outputs["cold_day_logit"]
                )
            
            # Get predicted class
            outputs["weather_pred"] = torch.argmax(
                outputs["weather_logits"], dim=-1
            )
            outputs["cold_day_pred"] = (
                outputs["cold_day_logit"] > 0
            ).long().squeeze(-1)
            
        return outputs
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Get the number of model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


class WeatherLoss(nn.Module):
    """
    Combined loss for weather forecasting.
    
    Combines:
    - MSE loss for temperature regression
    - CrossEntropy loss for weather classification
    - BCE loss for cold day classification
    """
    
    def __init__(
        self,
        lambda_temp: float = 1.0,
        lambda_weather: float = 1.0,
        lambda_cold: float = 0.5,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Args:
            lambda_temp: Weight for temperature loss
            lambda_weather: Weight for weather classification loss
            lambda_cold: Weight for cold day classification loss
            class_weights: Optional class weights for weather classification
        """
        super().__init__()
        self.lambda_temp = lambda_temp
        self.lambda_weather = lambda_weather
        self.lambda_cold = lambda_cold
        
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate combined loss.
        
        Args:
            outputs: Model outputs dictionary
            targets: Dictionary with target values:
                - temp: Target temperature (batch, 1)
                - weather_class: Target weather class (batch,)
                - cold_day: Target cold day flag (batch, 1)
                
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # Temperature regression loss
        temp_loss = self.mse_loss(
            outputs["temp_pred"], 
            targets["temp"].float()
        )
        
        # Weather classification loss
        weather_loss = self.ce_loss(
            outputs["weather_logits"],
            targets["weather_class"].long()
        )
        
        # Cold day classification loss
        cold_loss = self.bce_loss(
            outputs["cold_day_logit"],
            targets["cold_day"].float()
        )
        
        # Combined loss
        total_loss = (
            self.lambda_temp * temp_loss +
            self.lambda_weather * weather_loss +
            self.lambda_cold * cold_loss
        )
        
        # Return both total loss and components for logging
        loss_components = {
            "temp_loss": temp_loss.item(),
            "weather_loss": weather_loss.item(),
            "cold_loss": cold_loss.item(),
            "total_loss": total_loss.item()
        }
        
        return total_loss, loss_components
