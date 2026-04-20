"""
Directional Router — Routes validated signals to LONG or SHORT executor.
Acts as a single entry point for all trade execution, applying circuit
breaker checks and position sizing before delegation.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger("rox.execution.router")


@dataclass
class ExecutionResult:
    """Result of an execution routing attempt."""
    executed: bool
    reason: str
    direction: str = ""
    order_id: Optional[str] = None
    strategy: Optional[str] = None
    details: Optional[dict] = None


class CircuitBreakerProtocol:
    """Protocol/interface for circuit breaker — implement actual class separately."""
    def can_trade(self) -> Tuple[bool, str]:
        return True, "OK"
    def get_size_multiplier(self) -> float:
        return 1.0


class DirectionalRouter:
    """
    Routes validated signals to the appropriate executor.

    Flow:
    1. Circuit breaker check → halt if triggered
    2. Direction check:
       - LONG  → existing equity executor
       - SHORT → ShortExecutor (F&O-based)
       - NEUTRAL → skip
    3. Apply position sizing from circuit breaker multiplier
    4. Log execution path
    """

    def __init__(self, circuit_breaker: Optional[CircuitBreakerProtocol] = None):
        self.circuit_breaker = circuit_breaker or CircuitBreakerProtocol()

    def route_long(self, signal_data: dict, execute_fn) -> ExecutionResult:
        """
        Route a LONG signal to the existing equity execution path.

        Args:
            signal_data: Dict with symbol and other signal metadata.
            execute_fn: Callable that performs the actual execution.

        Returns:
            ExecutionResult with success/failure details.
        """
        can_trade, reason = self.circuit_breaker.can_trade()
        if not can_trade:
            logger.warning(f"LONG blocked by circuit breaker: {reason}")
            return ExecutionResult(executed=False, reason=f"CIRCUIT_BREAKER:{reason}",
                                   direction="LONG")

        size_mult = self.circuit_breaker.get_size_multiplier()
        if size_mult < 1.0:
            logger.info(f"LONG position size reduced to {size_mult:.0%}")
            signal_data = dict(signal_data)
            signal_data["size_multiplier"] = size_mult

        try:
            result = execute_fn(signal_data)
            logger.info(f"LONG executed: {signal_data.get('symbol', '?')}")
            return ExecutionResult(executed=True, reason="OK", direction="LONG",
                                   details=result if isinstance(result, dict) else None)
        except Exception as e:
            logger.error(f"LONG execution failed: {e}")
            return ExecutionResult(executed=False, reason=f"ERROR:{e}", direction="LONG")

    def route_short(self, short_order, execute_fn) -> ExecutionResult:
        """
        Route a SHORT signal via F&O short executor.

        Args:
            short_order: ShortOrder object from ShortExecutor.
            execute_fn: Callable that performs the actual F&O execution.

        Returns:
            ExecutionResult with success/failure details.
        """
        can_trade, reason = self.circuit_breaker.can_trade()
        if not can_trade:
            logger.warning(f"SHORT blocked by circuit breaker: {reason}")
            return ExecutionResult(executed=False, reason=f"CIRCUIT_BREAKER:{reason}",
                                   direction="SHORT")

        if short_order is None:
            return ExecutionResult(executed=False, reason="NO_SHORT_ORDER_CONSTRUCTED",
                                   direction="SHORT")

        size_mult = self.circuit_breaker.get_size_multiplier()
        if size_mult < 1.0:
            short_order.lots = max(1, int(short_order.lots * size_mult))
            short_order.quantity = short_order.lots * short_order.lot_size
            logger.info(f"SHORT lots reduced to {short_order.lots} (multiplier={size_mult:.0%})")

        try:
            result = execute_fn(short_order)
            logger.info(
                f"SHORT executed: {short_order.underlying} via {short_order.strategy.value} "
                f"strike={short_order.strike} lots={short_order.lots}"
            )
            return ExecutionResult(
                executed=True, reason="OK", direction="SHORT",
                strategy=short_order.strategy.value,
                details={"strike": short_order.strike, "lots": short_order.lots,
                         "premium": short_order.premium},
            )
        except Exception as e:
            logger.error(f"SHORT execution failed: {e}")
            return ExecutionResult(executed=False, reason=f"ERROR:{e}", direction="SHORT")
