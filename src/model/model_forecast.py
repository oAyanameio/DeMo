"""Legacy compatibility entry point.

The implementation moved to :mod:`trajectory_forecaster`; the old public name
is retained so historical configs and scripts remain loadable.
"""

from .trajectory_forecaster import ModelForecast, TrajectoryForecaster

__all__ = ["ModelForecast", "TrajectoryForecaster"]
