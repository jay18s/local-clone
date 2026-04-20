"""
Quality Module
==============

Pre-trade quality filtering and validation.
"""

from .setup_quality_checklist import (
    SetupQualityChecklist,
    ChecklistResult,
    CheckItem,
    CheckSeverity,
)

__all__ = [
    'SetupQualityChecklist',
    'ChecklistResult',
    'CheckItem',
    'CheckSeverity',
]
