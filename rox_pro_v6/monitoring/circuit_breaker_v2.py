"""
Circuit Breaker v2 — Multi-layer capital preservation.
Layers: consecutive loss, daily loss, drawdown, regime accuracy.
Prevents catastrophic capital erosion during adverse sequences.
"""

import logging
from dataclasses import dataclass
from typing import Tuple
from datetime import datetime

logger = logging.getLogger("rox.monitoring.circuit_breaker")


@dataclass
class CircuitBreakerState:
    """Current state of the circuit breaker for diagnostics."""
    halted: bool
    halt_reason: str
    consecutive_losses: int
    daily_pnl: float
    current_capital: float
    peak_capital: float
    size_multiplier: float
    drawdown_pct: float


class CircuitBreakerV2:
    """
    Multi-layer circuit breaker for capital preservation.

    Layers:
    1. CONSECUTIVE_LOSS: 3 losses in a row → 50% position size for next 3 trades
    2. DAILY_LOSS: Portfolio down >3% in a day → halt for session
    3. DRAWDOWN: Portfolio down >8% from peak → halt, require manual restart
    4. WIN_RATE: Rolling win rate < 25% after 20+ trades → ALERT + size cut (FIX-CB-WINRATE)
    5. SIZE_RESET: After 3 wins in reduced-size mode, restore to 100%
    """

    def __init__(
        self,
        initial_capital: float,
        consecutive_loss_threshold: int = 3,
        daily_loss_limit_pct: float = 3.0,
        max_drawdown_pct: float = 8.0,
        reduced_size_pct: float = 50.0,
        wins_to_reset_size: int = 3,
        win_rate_min_trades: int = 20,
        win_rate_alert_threshold: float = 25.0,
    ):
        """
        Initialize circuit breaker with capital and threshold settings.

        Args:
            initial_capital: Starting portfolio capital.
            consecutive_loss_threshold: Number of consecutive losses to trigger size reduction.
            daily_loss_limit_pct: Daily loss limit as percentage of peak capital.
            max_drawdown_pct: Maximum drawdown from peak before halting.
            reduced_size_pct: Position size percentage during reduced mode.
            wins_to_reset_size: Number of wins in reduced mode to restore full size.
            win_rate_min_trades: Minimum trades before win-rate layer activates.
            win_rate_alert_threshold: Win rate % below which Layer-4 fires (default 25%).
        """
        self.consecutive_losses = 0
        self.consecutive_wins_in_reduced = 0
        self.daily_pnl = 0.0
        self.peak_capital = initial_capital
        self.current_capital = initial_capital
        self.size_multiplier = 1.0
        self.halted = False
        self.halt_reason = ""

        self._consecutive_threshold = consecutive_loss_threshold
        self._daily_limit_pct = daily_loss_limit_pct / 100.0
        self._max_dd_pct = max_drawdown_pct / 100.0
        self._reduced_size = reduced_size_pct / 100.0
        self._wins_to_reset = wins_to_reset_size

        # FIX-CB-WINRATE: Layer-4 rolling win-rate tracking
        self._win_rate_min_trades = win_rate_min_trades
        self._win_rate_alert_pct = win_rate_alert_threshold / 100.0
        self._trade_results: list = []   # rolling list of True(win)/False(loss)
        self._win_rate_alerted: bool = False  # suppress repeated log spam

    def on_trade_close(self, pnl: float) -> None:
        """
        Update circuit breaker state after a trade closes.

        Applies all protection layers: consecutive loss tracking,
        size reduction, daily loss halts, max drawdown halts, and
        rolling win-rate alert (Layer 4).

        Args:
            pnl: Profit/loss from the closed trade.
        """
        self.daily_pnl += pnl
        self.current_capital += pnl

        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital

        _won = pnl > 0
        if _won:
            self.consecutive_losses = 0
            if self.size_multiplier < 1.0:
                self.consecutive_wins_in_reduced += 1
        else:
            self.consecutive_losses += 1
            self.consecutive_wins_in_reduced = 0

        # FIX-CB-WINRATE: Layer-4 rolling win rate (keep last 30 results)
        self._trade_results.append(_won)
        self._trade_results = self._trade_results[-30:]
        _n = len(self._trade_results)
        _wr = sum(self._trade_results) / _n if _n > 0 else 1.0
        if _n >= self._win_rate_min_trades and _wr < self._win_rate_alert_pct:
            if not self._win_rate_alerted:
                self._win_rate_alerted = True
                logger.critical(
                    f"CIRCUIT BREAKER LAYER-4 WIN-RATE: {_wr:.1%} over last {_n} trades "
                    f"is below {self._win_rate_alert_pct:.0%} threshold. "
                    f"Reducing size to 50% until win rate recovers above 35%."
                )
            # Apply size reduction (on top of any consecutive-loss reduction)
            self.size_multiplier = min(self.size_multiplier, self._reduced_size)
        elif _n >= self._win_rate_min_trades and _wr >= 0.35 and self._win_rate_alerted:
            # Win rate recovered above 35% — clear the alert
            self._win_rate_alerted = False
            logger.info(
                f"CIRCUIT BREAKER LAYER-4: Win rate recovered to {_wr:.1%} — "
                f"Layer-4 alert cleared."
            )

        # Layer 1: Consecutive losses → size reduction
        if self.consecutive_losses >= self._consecutive_threshold:
            self.size_multiplier = self._reduced_size
            logger.warning(
                f"CIRCUIT BREAKER: {self.consecutive_losses} consecutive losses → "
                f"size reduced to {self._reduced_size:.0%}"
            )

        # Size reset after wins in reduced mode
        if self.consecutive_wins_in_reduced >= self._wins_to_reset and not self._win_rate_alerted:
            self.size_multiplier = 1.0
            self.consecutive_wins_in_reduced = 0
            logger.info("CIRCUIT BREAKER: Size restored to 100% after recovery wins")

        # Layer 2: Daily loss limit
        if self.daily_pnl < -self.peak_capital * self._daily_limit_pct:
            self.halted = True
            self.halt_reason = "DAILY_LOSS_LIMIT"
            logger.critical(
                f"CIRCUIT BREAKER: Daily P&L {self.daily_pnl:.0f} exceeds "
                f"-{self._daily_limit_pct:.1%} of peak → HALTED"
            )

        # Layer 3: Max drawdown
        drawdown_pct = (self.peak_capital - self.current_capital) / self.peak_capital
        if drawdown_pct > self._max_dd_pct:
            self.halted = True
            self.halt_reason = "MAX_DRAWDOWN"
            logger.critical(
                f"CIRCUIT BREAKER: Drawdown {drawdown_pct:.1%} exceeds "
                f"{self._max_dd_pct:.1%} → HALTED (manual restart required)"
            )

        # Always log state after every close for audit visibility
        logger.info(
            f"CIRCUIT BREAKER STATE | halted={self.halted} | "
            f"consecutive_losses={self.consecutive_losses} | "
            f"size={self.size_multiplier:.0%} | "
            f"daily_pnl={self.daily_pnl:+.0f} | "
            f"drawdown={drawdown_pct:.1%} | "
            f"rolling_wr={_wr:.1%} ({_n} trades)"
        )

    def can_trade(self) -> Tuple[bool, str]:
        """
        Check whether trading is currently allowed.

        Returns:
            Tuple of (can_trade: bool, reason: str).
        """
        if self.halted:
            return False, self.halt_reason
        return True, "OK"

    def get_size_multiplier(self) -> float:
        """
        Get the current position size multiplier.

        Returns:
            Float between 0.0 and 1.0 representing size scaling.
        """
        return self.size_multiplier

    def reset_daily(self) -> None:
        """Call at market open (09:15 IST). Resets daily counters only."""
        self.daily_pnl = 0.0
        if self.halted and self.halt_reason == "DAILY_LOSS_LIMIT":
            self.halted = False
            self.halt_reason = ""
            logger.info("CIRCUIT BREAKER: Daily halt cleared for new session")

    def manual_restart(self) -> None:
        """Call manually to clear drawdown halt after review."""
        self.halted = False
        self.halt_reason = ""
        logger.info("CIRCUIT BREAKER: Manual restart executed")

    def get_state(self) -> CircuitBreakerState:
        """
        Get the current circuit breaker state for diagnostics.

        Returns:
            CircuitBreakerState with all current metrics.
        """
        dd = (self.peak_capital - self.current_capital) / self.peak_capital if self.peak_capital > 0 else 0
        return CircuitBreakerState(
            halted=self.halted,
            halt_reason=self.halt_reason,
            consecutive_losses=self.consecutive_losses,
            daily_pnl=self.daily_pnl,
            current_capital=self.current_capital,
            peak_capital=self.peak_capital,
            size_multiplier=self.size_multiplier,
            drawdown_pct=round(dd, 4),
        )
