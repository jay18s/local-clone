"""
FIX 7 — Theta Time Stop Module
===============================
Tracks open straddle/strangle positions and generates exit signals
when theta decay exceeds thresholds or hold time exceeds limits.

Exit rules:
  - MAX_HOLD_DAYS:    trading days since entry >= max_hold_days (default 3)
  - THETA_DECAY_THRESHOLD: abs(theta_eaten) >= entry_cost * 0.40
  - EXPIRY_RISK:      dte <= 2 (gamma risk explodes)
  Priority: EXPIRY_RISK > THETA_DECAY > MAX_HOLD_DAYS
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger("rox.theta_time_stop")


class ThetaExitReason(Enum):
    MAX_HOLD_DAYS = "MAX_HOLD_DAYS"
    THETA_DECAY_THRESHOLD = "THETA_DECAY_THRESHOLD"
    EXPIRY_RISK = "EXPIRY_RISK"


@dataclass
class ThetaPosition:
    index: str
    strategy: str  # "LONG_STRADDLE" | "LONG_STRANGLE"
    entry_time: datetime
    entry_cost_per_unit: float
    lot_size: int
    daily_theta: float  # negative number (decay per day)
    breakeven_low: float
    breakeven_high: float
    max_hold_days: int = 3
    strike: float = 0.0
    expiry: Optional[date] = None
    dte_at_entry: int = 0
    # Internal tracking
    _trading_days_held: int = field(default=0, init=False, repr=False)
    _last_check_date: Optional[date] = field(default=None, init=False, repr=False)
    _total_theta_eaten: float = field(default=0.0, init=False, repr=False)


@dataclass
class ThetaExitSignal:
    position: ThetaPosition
    reason: str  # "MAX_HOLD_DAYS" | "THETA_DECAY_THRESHOLD" | "EXPIRY_RISK"
    current_spot: float
    hold_days: int
    theta_eaten: float  # total theta decay since entry
    unrealized_pnl: float
    urgency: str  # "HIGH" | "MEDIUM" | "LOW"


# Indian market trading calendar — simplified (skip weekends + major holidays)
_HOLIDAYS_2026 = {
    date(2026, 1, 26),   # Republic Day
    date(2026, 3, 10),   # Holi
    date(2026, 3, 29),   # Mahavir Jayanti
    date(2026, 4, 2),    # Good Friday
    date(2026, 4, 14),   # Dr. Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 8, 15),   # Independence Day
    date(2026, 10, 2),   # Gandhi Jayanti
    date(2026, 11, 5),   # Diwali
    date(2026, 11, 6),   # Diwali Balipratipada
    date(2026, 12, 25),  # Christmas
}


def _is_trading_day(d: date) -> bool:
    """Check if a date is a trading day (not weekend, not holiday)."""
    if d.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    if d in _HOLIDAYS_2026:
        return False
    return True


def _count_trading_days(start: datetime, end: datetime) -> int:
    """Count trading days between two datetimes."""
    start_date = start.date()
    end_date = end.date()
    if end_date <= start_date:
        return 0
    count = 0
    d = start_date + timedelta(days=1)  # exclude entry day
    while d <= end_date:
        if _is_trading_day(d):
            count += 1
        d += timedelta(days=1)
    return count


class ThetaTimeStop:
    """
    Tracks straddle/strangle positions and generates exit signals
    based on theta decay and time-based rules.
    """

    def __init__(
        self,
        default_max_hold_days: int = 3,
        theta_decay_threshold: float = 0.40,  # 40% of entry cost
        expiry_risk_dte: int = 2,
    ):
        self.default_max_hold_days = default_max_hold_days
        self.theta_decay_threshold = theta_decay_threshold
        self.expiry_risk_dte = expiry_risk_dte
        self._positions: Dict[str, ThetaPosition] = {}  # key: index:strategy

    def register_position(
        self,
        index: str,
        strategy: str,
        entry_cost_per_unit: float,
        lot_size: int,
        daily_theta: float,
        breakeven_low: float,
        breakeven_high: float,
        strike: float = 0.0,
        expiry: Optional[date] = None,
        dte_at_entry: int = 0,
        entry_time: Optional[datetime] = None,
        max_hold_days: Optional[int] = None,
    ) -> None:
        """Register a new straddle/strangle position for tracking."""
        pos = ThetaPosition(
            index=index,
            strategy=strategy,
            entry_time=entry_time or datetime.now(),
            entry_cost_per_unit=entry_cost_per_unit,
            lot_size=lot_size,
            daily_theta=daily_theta,
            breakeven_low=breakeven_low,
            breakeven_high=breakeven_high,
            max_hold_days=max_hold_days or self.default_max_hold_days,
            strike=strike,
            expiry=expiry,
            dte_at_entry=dte_at_entry,
        )
        key = f"{index}:{strategy}"
        self._positions[key] = pos
        logger.info(
            f"[THETA-TRACK] Registered {key} | cost=₹{entry_cost_per_unit:,.0f}/unit | "
            f"theta=₹{daily_theta:,.0f}/day | max_hold={pos.max_hold_days}d"
        )

    def register_from_suggestion(self, suggestion, market_data: dict) -> None:
        """Register a position from a DirectionalOptionAdvisor suggestion."""
        try:
            idx = getattr(suggestion, "index", "UNKNOWN")
            strat = getattr(suggestion, "strategy", "LONG_STRADDLE")
            entry_cost = float(getattr(suggestion, "cost_per_lot", 0))
            lot = int(getattr(suggestion, "lot_size", 1))
            greeks = getattr(suggestion, "greeks", None)
            theta = float(getattr(greeks, "theta", -25)) if greeks else -25.0
            spot = float(market_data.get(f"{idx.lower()}_price", 0))
            strike = float(getattr(suggestion, "strike", spot))
            exp = getattr(suggestion, "expiry", None)
            dte = int(getattr(suggestion, "dte", 10))

            # Estimate breakevens: spot ± straddle cost
            be_low = spot - entry_cost / lot if spot > 0 else 0
            be_high = spot + entry_cost / lot if spot > 0 else 0

            self.register_position(
                index=idx,
                strategy=strat,
                entry_cost_per_unit=entry_cost,
                lot_size=lot,
                daily_theta=theta,
                breakeven_low=be_low,
                breakeven_high=be_high,
                strike=strike,
                expiry=exp if isinstance(exp, date) else None,
                dte_at_entry=dte,
            )
        except Exception as e:
            logger.debug(f"[THETA-TRACK] Failed to register position: {e}")

    def check_exits(
        self,
        spot_prices: Dict[str, float],
        current_time: Optional[datetime] = None,
        current_dte: Optional[Dict[str, int]] = None,
    ) -> List[ThetaExitSignal]:
        """
        Check all tracked positions for exit signals.

        Args:
            spot_prices: Dict of index -> current spot price
            current_time: Current datetime (default: now)
            current_dte: Optional dict of index -> current DTE

        Returns:
            List of exit signals, sorted by urgency (HIGH first)
        """
        now = current_time or datetime.now()
        signals: List[ThetaExitSignal] = []
        to_remove: List[str] = []

        for key, pos in self._positions.items():
            # Update trading days held
            trading_days = _count_trading_days(pos.entry_time, now)
            pos._trading_days_held = trading_days

            # Calculate total theta eaten
            days_elapsed = max(0, (now - pos.entry_time).days)
            pos._total_theta_eaten = pos.daily_theta * days_elapsed

            current_spot = spot_prices.get(pos.index, 0.0)
            current_dte_val = current_dte.get(pos.index, pos.dte_at_entry - days_elapsed) if current_dte else pos.dte_at_entry - days_elapsed

            # Calculate unrealized P&L (simplified: distance from breakeven)
            if current_spot > 0 and pos.breakeven_high > 0 and pos.breakeven_low > 0:
                if current_spot > pos.breakeven_high:
                    unrealized = (current_spot - pos.breakeven_high) * pos.lot_size
                elif current_spot < pos.breakeven_low:
                    unrealized = (pos.breakeven_low - current_spot) * pos.lot_size
                else:
                    # Inside breakeven range — P&L is negative (theta decay)
                    unrealized = pos._total_theta_eaten
            else:
                unrealized = pos._total_theta_eaten

            # Check exit rules (in priority order)
            signal = None

            # 1. EXPIRY_RISK (highest priority)
            if current_dte_val <= self.expiry_risk_dte:
                signal = ThetaExitSignal(
                    position=pos,
                    reason=ThetaExitReason.EXPIRY_RISK.value,
                    current_spot=current_spot,
                    hold_days=trading_days,
                    theta_eaten=abs(pos._total_theta_eaten),
                    unrealized_pnl=unrealized,
                    urgency="HIGH",
                )

            # 2. THETA_DECAY_THRESHOLD
            elif abs(pos._total_theta_eaten) >= pos.entry_cost_per_unit * self.theta_decay_threshold:
                signal = ThetaExitSignal(
                    position=pos,
                    reason=ThetaExitReason.THETA_DECAY_THRESHOLD.value,
                    current_spot=current_spot,
                    hold_days=trading_days,
                    theta_eaten=abs(pos._total_theta_eaten),
                    unrealized_pnl=unrealized,
                    urgency="HIGH" if abs(pos._total_theta_eaten) >= pos.entry_cost_per_unit * 0.50 else "MEDIUM",
                )

            # 3. MAX_HOLD_DAYS
            elif trading_days >= pos.max_hold_days:
                signal = ThetaExitSignal(
                    position=pos,
                    reason=ThetaExitReason.MAX_HOLD_DAYS.value,
                    current_spot=current_spot,
                    hold_days=trading_days,
                    theta_eaten=abs(pos._total_theta_eaten),
                    unrealized_pnl=unrealized,
                    urgency="MEDIUM",
                )

            if signal:
                signals.append(signal)
                to_remove.append(key)
                logger.warning(
                    f"[THETA-EXIT] {pos.index} {pos.strategy} → {signal.reason} | "
                    f"hold={signal.hold_days}d | theta_eaten=₹{signal.theta_eaten:,.0f} | "
                    f"P&L=₹{signal.unrealized_pnl:,.0f} | urgency={signal.urgency}"
                )

        # Remove exited positions
        for key in to_remove:
            del self._positions[key]

        # Sort by urgency
        urgency_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        signals.sort(key=lambda s: urgency_order.get(s.urgency, 9))

        return signals

    def get_active_positions(self) -> List[ThetaPosition]:
        """Return list of currently tracked positions."""
        return list(self._positions.values())

    def get_position_count(self) -> int:
        """Return number of tracked positions."""
        return len(self._positions)
