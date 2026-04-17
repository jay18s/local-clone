"""
ROX Proven Edge Engine v3.0 - Execution Package
==============================================
Order management and execution engine.
"""

from .order_manager import OrderManager, Order, OrderType, OrderStatus
from .execution_algorithms import (
    ExecutionAlgorithm, TWAP, VWAP, ImplementationShortfall
)
from .slippage_control import SlippageController

# ── v6.0 Execution Modules ─────────────────────────────────────────────────
from .short_executor import ShortExecutor, ShortOrder, ShortStrategy
from .directional_router import DirectionalRouter, ExecutionResult

__all__ = [
    "OrderManager", "Order", "OrderType", "OrderStatus",
    "ExecutionAlgorithm", "TWAP", "VWAP", "ImplementationShortfall",
    "SlippageController",
    "ShortExecutor", "ShortOrder", "ShortStrategy",
    "DirectionalRouter", "ExecutionResult",
]
