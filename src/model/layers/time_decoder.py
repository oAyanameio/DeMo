"""Legacy compatibility entry point for the structured future decoder."""

from .structured_future_decoder import (
    CoupledTrajectoryHead,
    FutureDistributionDecoder,
    GMMPredictor,
    GMMPredictor_dense,
    ModeTrajectoryHead,
    TimeDecoder,
)

__all__ = [
    "FutureDistributionDecoder",
    "TimeDecoder",
    "ModeTrajectoryHead",
    "CoupledTrajectoryHead",
    "GMMPredictor",
    "GMMPredictor_dense",
]
