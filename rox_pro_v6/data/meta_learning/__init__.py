"""
Meta Learning Module
====================

Persistent storage for meta-learning recommendations.
"""

from .recommendations_store import (
    RecommendationsStore,
    StoredRecommendation,
)

__all__ = [
    'RecommendationsStore',
    'StoredRecommendation',
]
