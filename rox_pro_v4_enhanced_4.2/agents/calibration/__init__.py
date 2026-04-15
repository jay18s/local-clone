"""
Calibration Module
==================

Agent conviction calibration for improved prediction accuracy.
"""

from .confidence_calibrator import (
    ConfidenceCalibrator,
    CalibratedResult,
    CalibrationBucket,
    AgentCalibrationData,
)

__all__ = [
    'ConfidenceCalibrator',
    'CalibratedResult',
    'CalibrationBucket',
    'AgentCalibrationData',
]
