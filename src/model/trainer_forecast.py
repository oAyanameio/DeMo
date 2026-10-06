"""Legacy compatibility entry point for the forecasting Lightning module."""

from .forecasting_module import ForecastingLightningModule, Trainer

__all__ = ["ForecastingLightningModule", "Trainer"]
