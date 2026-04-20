"""
Iron Condor Trigger Monitor — FIX-IC-TRIGGER
=============================================
Monitors market conditions and auto-triggers Iron Condor entries
when price sustains in the expected range for a configurable duration.

Integration:
    ic_monitor = ICTriggerMonitor(portfolio_value=1_000_000)
    ic_monitor.add_trigger(
        index_name="NIFTY",
        spot_range=(24150, 24400),
        sustain_minutes=15,
        suggestion=<OptionSuggestion>,
        strikes={"sell_ce": 24600, "buy_ce": 24800, "sell_pe": 23900, "buy_pe": 24100},
    )
    # Each cycle:
    result = ic_monitor.check(spot_prices={"NIFTY": 24300})
    if result:
        # Execute trade
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger("rox.execution.ic_trigger")


@dataclass
class ICTrigger:
    """One Iron Condor trigger condition."""
    index_name: str
    spot_low: float
    spot_high: float
    sustain_minutes: int
    suggestion: Any  # OptionSuggestion or dict
    strikes: Dict[str, float]  # {"sell_ce": X, "buy_ce": Y, "sell_pe": A, "buy_pe": B}
    max_loss_per_lot: float
    credit_per_unit: float
    conviction: int
    created_at: datetime = field(default_factory=datetime.now)
    # Internal tracking
    _range_entered_at: Optional[datetime] = None
    _consecutive_in_range: int = 0  # cycles in range
    _triggered: bool = False
    _triggered_at: Optional[datetime] = None
    _cancelled: bool = False
    _cancel_reason: str = ""

    @property
    def is_active(self) -> bool:
        return not self._triggered and not self._cancelled

    @property
    def description(self) -> str:
        return (
            f"{self.index_name} IC: spot in [{self.spot_low:.0f}, {self.spot_high:.0f}] "
            f"for {self.sustain_minutes}min | credit=₹{self.credit_per_unit:.0f}/unit "
            f"max_loss=₹{self.max_loss_per_lot:.0f}"
        )


@dataclass
class ICTriggerResult:
    """Result when a trigger fires."""
    trigger: ICTrigger
    spot_at_trigger: float
    triggered_at: datetime
    wait_duration_seconds: float
    strikes: Dict[str, float]
    credit_per_unit: float
    max_loss_per_lot: float
    conviction: int


class ICTriggerMonitor:
    """
    Monitors spot prices against Iron Condor trigger conditions.
    Auto-fires when spot sustains in the configured range for the required duration.
    """

    def __init__(self, portfolio_value: float = 1_000_000, cycle_interval_seconds: int = 300):
        self.portfolio_value = portfolio_value
        self.cycle_interval_seconds = cycle_interval_seconds  # time between checks
        self._triggers: List[ICTrigger] = []
        self._fired_triggers: List[ICTriggerResult] = []

    def add_trigger(
        self,
        index_name: str,
        spot_range: Tuple[float, float],
        sustain_minutes: int,
        suggestion: Any,
        strikes: Dict[str, float],
        max_loss_per_lot: float,
        credit_per_unit: float,
        conviction: int,
    ) -> ICTrigger:
        """Register a new IC trigger condition."""
        trigger = ICTrigger(
            index_name=index_name,
            spot_low=spot_range[0],
            spot_high=spot_range[1],
            sustain_minutes=sustain_minutes,
            suggestion=suggestion,
            strikes=strikes,
            max_loss_per_lot=max_loss_per_lot,
            credit_per_unit=credit_per_unit,
            conviction=conviction,
        )
        self._triggers.append(trigger)
        logger.info(f"[IC-TRIGGER] Added: {trigger.description}")
        return trigger

    def check(self, spot_prices: Dict[str, float], current_time: Optional[datetime] = None) -> Optional[ICTriggerResult]:
        """
        Check all active triggers against current spot prices.
        Returns the first fired trigger result, or None.
        """
        now = current_time or datetime.now()
        fired = None

        for trigger in self._triggers:
            if not trigger.is_active:
                continue

            spot = spot_prices.get(trigger.index_name)
            if spot is None:
                continue

            in_range = trigger.spot_low <= spot <= trigger.spot_high

            if in_range:
                if trigger._range_entered_at is None:
                    trigger._range_entered_at = now
                    trigger._consecutive_in_range = 1
                else:
                    trigger._consecutive_in_range += 1

                elapsed = (now - trigger._range_entered_at).total_seconds()
                required = trigger.sustain_minutes * 60

                if elapsed >= required:
                    # TRIGGER FIRED
                    trigger._triggered = True
                    trigger._triggered_at = now
                    wait_secs = (now - trigger.created_at).total_seconds()

                    result = ICTriggerResult(
                        trigger=trigger,
                        spot_at_trigger=spot,
                        triggered_at=now,
                        wait_duration_seconds=wait_secs,
                        strikes=trigger.strikes,
                        credit_per_unit=trigger.credit_per_unit,
                        max_loss_per_lot=trigger.max_loss_per_lot,
                        conviction=trigger.conviction,
                    )
                    self._fired_triggers.append(result)
                    fired = result
                    logger.info(
                        f"[IC-TRIGGER] ✅ FIRED: {trigger.index_name} spot={spot:.0f} "
                        f"sustained {elapsed/60:.0f}min in [{trigger.spot_low:.0f}, {trigger.spot_high:.0f}] | "
                        f"strikes={trigger.strikes} credit=₹{trigger.credit_per_unit:.0f}"
                    )
            else:
                # Spot left range — reset timer
                if trigger._range_entered_at is not None:
                    logger.debug(
                        f"[IC-TRIGGER] {trigger.index_name} spot={spot:.0f} left range "
                        f"[{trigger.spot_low:.0f}, {trigger.spot_high:.0f}] — timer reset"
                    )
                trigger._range_entered_at = None
                trigger._consecutive_in_range = 0

                # Cancel if spot breaks invalidation level (significantly outside range)
                range_width = trigger.spot_high - trigger.spot_low
                if spot < trigger.spot_low - range_width * 0.5 or spot > trigger.spot_high + range_width * 0.5:
                    trigger._cancelled = True
                    trigger._cancel_reason = f"spot={spot:.0f} broke far outside range"
                    logger.warning(
                        f"[IC-TRIGGER] ❌ CANCELLED: {trigger.index_name} {trigger._cancel_reason}"
                    )

        return fired

    def add_from_trading_plan(self, plan, market_data: Dict) -> int:
        """
        Extract IC triggers from the trading plan's fno_ready_trades
        and register them automatically.
        Returns number of triggers added.
        """
        added = 0
        fno_trades = getattr(plan, 'fno_ready_trades', []) or []
        for trade in fno_trades:
            if getattr(trade, 'strategy', '') != 'IRON_CONDOR':
                continue
            if getattr(trade, 'status', '') != 'WAIT_FOR_TRIGGER':
                continue

            # Parse the entry trigger condition
            entry_trigger = getattr(trade, 'entry_trigger', '')
            spot = market_data.get('nifty_price', 0)
            if spot <= 0:
                continue

            # Default: spot range ±1.5% from current with 15min sustain
            range_pct = 0.015
            spot_low = spot * (1 - range_pct)
            spot_high = spot * (1 + range_pct)

            self.add_trigger(
                index_name=getattr(trade, 'instrument', 'NIFTY'),
                spot_range=(spot_low, spot_high),
                sustain_minutes=15,
                suggestion=trade,
                strikes=getattr(trade, 'strikes', {}),
                max_loss_per_lot=getattr(trade, 'max_loss_per_lot', 0),
                credit_per_unit=0,  # parsed from trade if available
                conviction=getattr(trade, 'confidence', 60),
            )
            added += 1

        if added:
            logger.info(f"[IC-TRIGGER] Auto-registered {added} IC trigger(s) from trading plan")
        return added

    def get_status(self) -> List[Dict]:
        """Get status of all triggers for logging/display."""
        status = []
        for t in self._triggers:
            if t._triggered:
                state = "FIRED"
            elif t._cancelled:
                state = f"CANCELLED ({t._cancel_reason})"
            elif t._range_entered_at:
                elapsed = (datetime.now() - t._range_entered_at).total_seconds()
                state = f"IN_RANGE ({elapsed/60:.0f}/{t.sustain_minutes}min)"
            else:
                state = "WAITING"
            status.append({
                "index": t.index_name,
                "range": f"[{t.spot_low:.0f}, {t.spot_high:.0f}]",
                "state": state,
                "sustain": f"{t.sustain_minutes}min",
            })
        return status

    def clear_expired(self):
        """Remove triggers that fired or were cancelled more than 1 hour ago."""
        cutoff = datetime.now() - timedelta(hours=1)
        self._triggers = [
            t for t in self._triggers
            if t.is_active or (t._triggered_at and t._triggered_at > cutoff)
        ]
