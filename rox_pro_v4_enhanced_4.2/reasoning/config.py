"""
ROX Engine v5.0 — Reasoning Configuration
ComplexityLevel enum used by adaptive prompt selection and other reasoning modules.
"""

from enum import Enum


class ComplexityLevel(Enum):
    """Market complexity classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"
