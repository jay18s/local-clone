"""
ROX Proven Edge Engine v3.0 - Circuit Breakers
=============================================
Automatic trading halts for risk protection.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum
import json


class BreakerType(Enum):
    """Types of circuit breakers"""
    INTRADAY_LOSS = "INTRADAY_LOSS"
    CONSECUTIVE_LOSS = "CONSECUTIVE_LOSS"
    VOLATILITY = "VOLATILITY"
    LIQUIDITY = "LIQUIDITY"
    NEWS = "NEWS"
    DRAWDOWN = "DRAWDOWN"
    HEAT = "HEAT"
    CORRELATION = "CORRELATION"


class BreakerStatus(Enum):
    """Circuit breaker status"""
    ARMED = "ARMED"
    TRIPPED = "TRIPPED"
    COOLING_DOWN = "COOLING_DOWN"
    DISABLED = "DISABLED"


@dataclass
class BreakerEvent:
    """Circuit breaker trip event"""
    breaker_type: BreakerType
    timestamp: datetime
    trigger_value: float
    threshold: float
    reason: str
    action_taken: str
    cooldown_until: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        return {
            "breaker_type": self.breaker_type.value,
            "timestamp": self.timestamp.isoformat(),
            "trigger_value": self.trigger_value,
            "threshold": self.threshold,
            "reason": self.reason,
            "action_taken": self.action_taken,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None
        }


@dataclass
class CircuitBreaker:
    """Single circuit breaker configuration"""
    name: str
    breaker_type: BreakerType
    threshold: float
    cooldown_minutes: int = 60
    status: BreakerStatus = BreakerStatus.ARMED
    trip_count: int = 0
    last_trip: Optional[datetime] = None
    
    # Callbacks
    on_trip: Optional[Callable] = None
    on_reset: Optional[Callable] = None
    
    def check(self, value: float) -> Optional[BreakerEvent]:
        """Check if breaker should trip"""
        if self.status != BreakerStatus.ARMED:
            return None
        
        if self._should_trip(value):
            return self._trip(value)
        
        return None
    
    def _should_trip(self, value: float) -> bool:
        """Determine if value exceeds threshold"""
        # Lower-is-worse: trip when value drops BELOW threshold (negative thresholds)
        if self.breaker_type in [BreakerType.INTRADAY_LOSS, BreakerType.DRAWDOWN,
                                 BreakerType.CONSECUTIVE_LOSS]:
            return value < self.threshold
        # Higher-is-worse: trip when value rises ABOVE threshold (positive thresholds)
        elif self.breaker_type in [BreakerType.VOLATILITY, BreakerType.LIQUIDITY,
                                   BreakerType.HEAT, BreakerType.CORRELATION]:
            return value > self.threshold
        else:
            return value >= self.threshold
    
    def _trip(self, value: float) -> BreakerEvent:
        """Trip the circuit breaker"""
        self.status = BreakerStatus.TRIPPED
        self.trip_count += 1
        self.last_trip = datetime.now()
        
        event = BreakerEvent(
            breaker_type=self.breaker_type,
            timestamp=self.last_trip,
            trigger_value=value,
            threshold=self.threshold,
            reason=self._get_reason(value),
            action_taken=self._get_action(),
            cooldown_until=datetime.now() + timedelta(minutes=self.cooldown_minutes)
        )
        
        if self.on_trip:
            self.on_trip(event)
        
        return event
    
    def _get_reason(self, value: float) -> str:
        """Get trip reason"""
        reasons = {
            BreakerType.INTRADAY_LOSS: f"Intraday loss {value:.2%} exceeds limit {self.threshold:.2%}",
            BreakerType.CONSECUTIVE_LOSS: f"{int(value)} consecutive losses exceed limit {int(self.threshold)}",
            BreakerType.VOLATILITY: f"VIX {value:.1f} exceeds threshold {self.threshold:.1f}",
            BreakerType.LIQUIDITY: f"Bid-ask spread {value:.2%} exceeds threshold {self.threshold:.2%}",
            BreakerType.NEWS: f"Major news event detected",
            BreakerType.DRAWDOWN: f"Drawdown {value:.2%} exceeds limit {self.threshold:.2%}",
            BreakerType.HEAT: f"Portfolio heat {value:.2%} exceeds limit {self.threshold:.2%}",
            BreakerType.CORRELATION: f"Position correlation {value:.2f} exceeds threshold {self.threshold:.2f}"
        }
        return reasons.get(self.breaker_type, "Threshold exceeded")
    
    def _get_action(self) -> str:
        """Get action to take"""
        actions = {
            BreakerType.INTRADAY_LOSS: "Pause all new trades for the day",
            BreakerType.CONSECUTIVE_LOSS: "Pause trading, review strategy",
            BreakerType.VOLATILITY: "Reduce position sizes by 50%",
            BreakerType.LIQUIDITY: "Use limit orders only, widen stops",
            BreakerType.NEWS: "Exit discretionary positions, pause new entries",
            BreakerType.DRAWDOWN: "Exit 50% of positions, preserve capital",
            BreakerType.HEAT: "No new positions until heat reduces",
            BreakerType.CORRELATION: "Reduce concentrated positions"
        }
        return actions.get(self.breaker_type, "Review positions")
    
    def reset(self) -> bool:
        """Reset the breaker"""
        if self.status == BreakerStatus.TRIPPED:
            if datetime.now() >= (self.last_trip + timedelta(minutes=self.cooldown_minutes)):
                self.status = BreakerStatus.ARMED
                if self.on_reset:
                    self.on_reset()
                return True
        return False
    
    def force_reset(self):
        """Force reset regardless of cooldown"""
        self.status = BreakerStatus.ARMED
        if self.on_reset:
            self.on_reset()
    
    def disable(self):
        """Disable the breaker"""
        self.status = BreakerStatus.DISABLED
    
    def enable(self):
        """Enable the breaker"""
        self.status = BreakerStatus.ARMED


class CircuitBreakerManager:
    """
    Manages all circuit breakers.
    
    Features:
    - Multiple breaker types
    - Cooldown periods
    - Automatic reset
    - Event logging
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger("CircuitBreakerManager")
        
        # Breakers storage
        self.breakers: Dict[str, CircuitBreaker] = {}
        
        # Event log
        self.event_log: List[BreakerEvent] = []
        
        # Global trading halt
        self.trading_halted = False
        self.halt_reason: Optional[str] = None
        
        # Callbacks
        self._on_halt: Optional[Callable] = None
        self._on_resume: Optional[Callable] = None
        
        # Initialize default breakers
        self._initialize_default_breakers()
    
    def _initialize_default_breakers(self):
        """Initialize default set of circuit breakers"""
        defaults = [
            ("intraday_loss", BreakerType.INTRADAY_LOSS, -0.05, 390),  # -5%, halt for day
            ("consecutive_loss", BreakerType.CONSECUTIVE_LOSS, 4, 120),  # 4 losses, 2hr cooldown
            ("volatility", BreakerType.VOLATILITY, 25, 60),  # VIX > 25, 1hr cooldown
            ("liquidity", BreakerType.LIQUIDITY, 0.02, 30),  # 2% spread, 30min cooldown
            ("drawdown", BreakerType.DRAWDOWN, -0.15, 1440),  # -15%, 24hr cooldown
            ("heat", BreakerType.HEAT, 0.10, 60),  # 10% heat, 1hr cooldown
        ]
        
        for name, btype, threshold, cooldown in defaults:
            self.add_breaker(name, btype, threshold, cooldown)
    
    def add_breaker(self, name: str, breaker_type: BreakerType,
                   threshold: float, cooldown_minutes: int = 60,
                   on_trip: Callable = None, on_reset: Callable = None):
        """Add a new circuit breaker"""
        breaker = CircuitBreaker(
            name=name,
            breaker_type=breaker_type,
            threshold=threshold,
            cooldown_minutes=cooldown_minutes,
            on_trip=on_trip,
            on_reset=on_reset
        )
        
        self.breakers[name] = breaker
        self.logger.info(f"Added circuit breaker: {name} ({breaker_type.value})")
    
    def check_all(self, values: Dict[str, float]) -> List[BreakerEvent]:
        """
        Check all breakers with current values.
        
        Args:
            values: Dictionary of breaker_name -> current_value
            
        Returns:
            List of trip events
        """
        events = []
        
        for name, breaker in self.breakers.items():
            if name in values:
                event = breaker.check(values[name])
                if event:
                    events.append(event)
                    self.event_log.append(event)
                    self._handle_trip(event)
        
        return events
    
    def _handle_trip(self, event: BreakerEvent):
        """Handle circuit breaker trip"""
        self.logger.warning(
            f"CIRCUIT BREAKER TRIPPED: {event.breaker_type.value} | {event.reason}"
        )
        
        # Check if trading should halt
        critical_types = [
            BreakerType.INTRADAY_LOSS,
            BreakerType.DRAWDOWN,
            BreakerType.CONSECUTIVE_LOSS
        ]
        
        if event.breaker_type in critical_types:
            self._halt_trading(event.reason)
    
    def _halt_trading(self, reason: str):
        """Halt all trading"""
        self.trading_halted = True
        self.halt_reason = reason
        
        self.logger.error(f"TRADING HALTED: {reason}")
        
        if self._on_halt:
            self._on_halt(reason)
    
    def resume_trading(self) -> bool:
        """Attempt to resume trading"""
        if not self.trading_halted:
            return True
        
        # Check if all breakers are reset
        for breaker in self.breakers.values():
            if breaker.status == BreakerStatus.TRIPPED:
                if not breaker.reset():
                    self.logger.info("Cannot resume - breaker still in cooldown")
                    return False
        
        self.trading_halted = False
        self.halt_reason = None
        
        self.logger.info("Trading resumed")
        
        if self._on_resume:
            self._on_resume()
        
        return True
    
    def force_resume_trading(self):
        """Force resume trading"""
        for breaker in self.breakers.values():
            breaker.force_reset()
        
        self.trading_halted = False
        self.halt_reason = None
        
        self.logger.warning("Trading force resumed")
    
    def update_cooldowns(self):
        """Update breaker cooldowns"""
        for breaker in self.breakers.values():
            if breaker.status == BreakerStatus.TRIPPED:
                breaker.reset()
    
    def get_status(self) -> Dict:
        """Get all breaker statuses"""
        return {
            "trading_halted": self.trading_halted,
            "halt_reason": self.halt_reason,
            "breakers": {
                name: {
                    "type": breaker.breaker_type.value,
                    "status": breaker.status.value,
                    "threshold": breaker.threshold,
                    "trip_count": breaker.trip_count,
                    "last_trip": breaker.last_trip.isoformat() if breaker.last_trip else None
                }
                for name, breaker in self.breakers.items()
            }
        }
    
    def get_active_breakers(self) -> List[CircuitBreaker]:
        """Get all tripped breakers"""
        return [b for b in self.breakers.values() if b.status == BreakerStatus.TRIPPED]
    
    def get_event_history(self, limit: int = 50) -> List[Dict]:
        """Get recent trip events"""
        return [e.to_dict() for e in self.event_log[-limit:]]
    
    def register_halt_callback(self, callback: Callable):
        """Register callback for trading halt"""
        self._on_halt = callback
    
    def register_resume_callback(self, callback: Callable):
        """Register callback for trading resume"""
        self._on_resume = callback
    
    def should_allow_trade(self) -> Tuple[bool, Optional[str]]:
        """Check if trading is allowed"""
        if self.trading_halted:
            return False, self.halt_reason
        
        # Check for any tripped breakers that would block
        for name, breaker in self.breakers.items():
            if breaker.status == BreakerStatus.TRIPPED:
                return False, f"Circuit breaker '{name}' is tripped"
        
        return True, None
    
    async def monitor_loop(self, check_interval: int = 30):
        """Background monitoring loop"""
        while True:
            self.update_cooldowns()
            
            # Try to resume if halted
            if self.trading_halted:
                self.resume_trading()
            
            await asyncio.sleep(check_interval)


# ===========================================================================
# ENHANCEMENT: Compliance Engine
# Circuit Filter, SEBI Rules, Corporate Actions, Max Drawdown Auto-Halt
# ===========================================================================

@dataclass
class CorporateAction:
    """Represents a corporate action for a stock."""
    symbol: str
    action_type: str          # DIVIDEND, BONUS, SPLIT, RIGHTS, MERGER
    ex_date: str              # YYYY-MM-DD format
    ratio_or_amount: float    # e.g. 1.5 for split 3:2, 5.0 for Rs 5 dividend
    record_date: str = ""
    notes: str = ""

    def days_until_ex(self) -> int:
        from datetime import date
        try:
            ex = datetime.strptime(self.ex_date, "%Y-%m-%d").date()
            return (ex - date.today()).days
        except ValueError:
            return 999

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "action_type": self.action_type,
            "ex_date": self.ex_date,
            "ratio_or_amount": self.ratio_or_amount,
            "record_date": self.record_date,
            "notes": self.notes,
            "days_until_ex": self.days_until_ex(),
        }


@dataclass
class ComplianceCheckResult:
    """Result of a compliance check for a stock/trade."""
    symbol: str
    allowed: bool
    reasons: List[str]          # Why blocked (if not allowed)
    warnings: List[str]         # Non-blocking alerts
    sebi_flags: List[str]       # SEBI-specific flags
    corporate_actions: List[str] # Upcoming corporate actions
    circuit_status: str         # NORMAL / UPPER_CIRCUIT / LOWER_CIRCUIT / NEAR_CIRCUIT

    def summary(self) -> str:
        lines = [f"[{self.symbol}] Compliance: {'✅ ALLOWED' if self.allowed else '❌ BLOCKED'}"]
        if self.reasons:
            lines += [f"  ✗ {r}" for r in self.reasons]
        if self.warnings:
            lines += [f"  ⚠ {w}" for w in self.warnings]
        if self.sebi_flags:
            lines += [f"  📋 SEBI: {s}" for s in self.sebi_flags]
        if self.corporate_actions:
            lines += [f"  🏷 Corp Action: {c}" for c in self.corporate_actions]
        lines.append(f"  Circuit: {self.circuit_status}")
        return "\n".join(lines)


class ComplianceEngine:
    """
    ROX Compliance & Risk Controls Engine
    ======================================
    Enforces SEBI regulations, circuit filter checks, corporate action
    awareness, and automatic portfolio drawdown halt.

    Key Features
    ------------
    1. Circuit Filter Detection  – blocks trades in locked upper/lower circuit stocks
    2. Corporate Action Awareness – flags stocks near ex-dividend / split / bonus dates
    3. SEBI Regulation Checks    – peak margin, intraday restrictions, F&O ban checks
    4. Max Drawdown Auto-Halt    – halts ALL trading if portfolio drops >10% from peak
    """

    # NSE/BSE circuit limits (percentage)
    CIRCUIT_LIMITS = {
        "DEFAULT": 20.0,     # Standard circuit limit
        "FNOSTOCK": 20.0,
        "ILLIQUID": 10.0,    # Illiquid/SME stocks
        "INDEX": 10.0,
    }

    # SEBI peak margin requirement (as fraction of trade value)
    SEBI_PEAK_MARGIN = 0.20  # 20% upfront margin

    # Days before ex-date to flag corporate action risk
    CORP_ACTION_ALERT_DAYS = 5
    CORP_ACTION_BLOCK_DAYS = 2  # Block trades within 2 days of ex-date

    def __init__(self, portfolio_value: float = 1_000_000,
                 max_drawdown_pct: float = 0.10,
                 config: Dict = None):
        self.logger = logging.getLogger("ComplianceEngine")
        self.portfolio_value = portfolio_value
        self.peak_portfolio_value = portfolio_value
        self.max_drawdown_pct = max_drawdown_pct  # Default 10%
        self.config = config or {}

        # State
        self.trading_halted_drawdown = False
        self.drawdown_halt_reason = ""
        self.corporate_actions: List[CorporateAction] = []
        self.fno_ban_stocks: List[str] = []
        self.circuit_locked_stocks: Dict[str, str] = {}  # symbol -> UPPER/LOWER
        self.intraday_restricted_stocks: List[str] = []

        # Audit trail
        self.compliance_log: List[Dict] = []

    # ------------------------------------------------------------------
    # Corporate Action Registry
    # ------------------------------------------------------------------

    def register_corporate_action(self, action: CorporateAction):
        """Add/update a corporate action."""
        self.corporate_actions = [a for a in self.corporate_actions
                                  if not (a.symbol == action.symbol
                                          and a.action_type == action.action_type
                                          and a.ex_date == action.ex_date)]
        self.corporate_actions.append(action)
        self.logger.info(f"Corporate action registered: {action.symbol} "
                         f"{action.action_type} ex {action.ex_date}")

    def load_corporate_actions_from_dict(self, actions: List[Dict]):
        """Bulk-load corporate actions from list of dicts."""
        for d in actions:
            try:
                self.register_corporate_action(CorporateAction(**d))
            except Exception as e:
                self.logger.warning(f"Bad corporate action record: {d} — {e}")

    def get_corporate_actions_for(self, symbol: str) -> List[CorporateAction]:
        return [a for a in self.corporate_actions if a.symbol == symbol]

    # ------------------------------------------------------------------
    # Circuit Filter Management
    # ------------------------------------------------------------------

    def update_circuit_status(self, symbol: str, change_pct: float,
                              circuit_type: str = "DEFAULT"):
        """
        Update circuit status for a symbol based on its daily % change.
        NSE/BSE locks a stock when it hits ±20% (or ±10% for illiquid).
        """
        limit = self.CIRCUIT_LIMITS.get(circuit_type, 20.0)
        if change_pct >= limit:
            self.circuit_locked_stocks[symbol] = "UPPER"
            self.logger.warning(f"CIRCUIT LOCKED (UPPER): {symbol} +{change_pct:.1f}%")
        elif change_pct <= -limit:
            self.circuit_locked_stocks[symbol] = "LOWER"
            self.logger.warning(f"CIRCUIT LOCKED (LOWER): {symbol} {change_pct:.1f}%")
        elif symbol in self.circuit_locked_stocks:
            del self.circuit_locked_stocks[symbol]

    def bulk_update_circuits(self, price_data: Dict[str, Dict]):
        """
        Process live price data dict: {symbol: {change_pct: float, ...}}
        and update circuit statuses for all symbols.
        """
        for symbol, data in price_data.items():
            change_pct = data.get("change_pct", data.get("day_change_pct", 0.0))
            self.update_circuit_status(symbol, change_pct)

    def is_circuit_locked(self, symbol: str) -> Optional[str]:
        """Returns 'UPPER', 'LOWER', or None."""
        return self.circuit_locked_stocks.get(symbol)

    def get_circuit_status(self, symbol: str, change_pct: float,
                           circuit_limit: float = 20.0) -> str:
        """
        Return descriptive circuit status for a symbol.
        NORMAL / NEAR_UPPER / NEAR_LOWER / UPPER_CIRCUIT / LOWER_CIRCUIT
        """
        locked = self.circuit_locked_stocks.get(symbol)
        if locked == "UPPER":
            return "UPPER_CIRCUIT"
        if locked == "LOWER":
            return "LOWER_CIRCUIT"
        near_threshold = circuit_limit * 0.80  # within 80% of limit
        if change_pct >= near_threshold:
            return "NEAR_UPPER"
        if change_pct <= -near_threshold:
            return "NEAR_LOWER"
        return "NORMAL"

    # ------------------------------------------------------------------
    # SEBI Compliance
    # ------------------------------------------------------------------

    def update_fno_ban_list(self, symbols: List[str]):
        """Update the F&O ban-period stock list (SEBI imposed)."""
        self.fno_ban_stocks = [s.upper() for s in symbols]
        self.logger.info(f"F&O ban list updated: {self.fno_ban_stocks}")

    def update_intraday_restricted(self, symbols: List[str]):
        """Stocks restricted from intraday trading (exchange directive)."""
        self.intraday_restricted_stocks = [s.upper() for s in symbols]

    def check_sebi_margin_compliance(self, trade_value: float,
                                    available_margin: float) -> Tuple[bool, str]:
        """
        Check SEBI peak margin requirement (20% upfront).
        Returns (compliant: bool, message: str).
        """
        required_margin = trade_value * self.SEBI_PEAK_MARGIN
        if available_margin < required_margin:
            msg = (f"Insufficient margin: need ₹{required_margin:,.0f} "
                   f"(20% of ₹{trade_value:,.0f}), have ₹{available_margin:,.0f}")
            return False, msg
        return True, "Margin compliant"

    # ------------------------------------------------------------------
    # Max Drawdown Auto-Halt
    # ------------------------------------------------------------------

    def update_portfolio_value(self, current_value: float):
        """
        Update current portfolio value and check max-drawdown breach.
        Automatically halts trading if drawdown > threshold (default 10%).
        """
        # Update peak
        if current_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_value

        drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
        self.portfolio_value = current_value

        if drawdown >= self.max_drawdown_pct and not self.trading_halted_drawdown:
            self.trading_halted_drawdown = True
            self.drawdown_halt_reason = (
                f"Portfolio drawdown {drawdown*100:.1f}% exceeds "
                f"limit of {self.max_drawdown_pct*100:.1f}% "
                f"(Peak: ₹{self.peak_portfolio_value:,.0f}, "
                f"Current: ₹{current_value:,.0f})"
            )
            self.logger.critical(f"🚨 TRADING HALTED – MAX DRAWDOWN: {self.drawdown_halt_reason}")
            self._log_event("DRAWDOWN_HALT", self.drawdown_halt_reason)
        elif drawdown < self.max_drawdown_pct * 0.5 and self.trading_halted_drawdown:
            # Auto-resume when recovered to 50% of threshold
            self.trading_halted_drawdown = False
            self.drawdown_halt_reason = ""
            self.logger.info("Trading resumed – drawdown recovered below 50% of threshold")

        return drawdown

    def get_drawdown_status(self) -> Dict:
        drawdown = (self.peak_portfolio_value - self.portfolio_value) / max(self.peak_portfolio_value, 1)
        return {
            "peak_value": self.peak_portfolio_value,
            "current_value": self.portfolio_value,
            "drawdown_pct": round(drawdown * 100, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct * 100, 2),
            "halt_active": self.trading_halted_drawdown,
            "halt_reason": self.drawdown_halt_reason,
        }

    # ------------------------------------------------------------------
    # Master Compliance Check
    # ------------------------------------------------------------------

    def check_trade(self, symbol: str, trade_value: float,
                    available_margin: float, is_intraday: bool = False,
                    change_pct: float = 0.0) -> ComplianceCheckResult:
        """
        Run all compliance checks for a proposed trade.

        Returns ComplianceCheckResult with allowed flag and full reason breakdown.
        """
        reasons: List[str] = []
        warnings: List[str] = []
        sebi_flags: List[str] = []
        corp_action_msgs: List[str] = []

        sym_upper = symbol.upper()

        # 1. Drawdown auto-halt
        if self.trading_halted_drawdown:
            reasons.append(f"DRAWDOWN HALT: {self.drawdown_halt_reason}")

        # 2. Circuit filter check
        circuit_status = self.get_circuit_status(sym_upper, change_pct)
        if circuit_status in ("UPPER_CIRCUIT", "LOWER_CIRCUIT"):
            reasons.append(f"Stock locked in {circuit_status.replace('_', ' ')} – no trading possible")
        elif circuit_status in ("NEAR_UPPER", "NEAR_LOWER"):
            warnings.append(f"Approaching {circuit_status.replace('_', ' ')} – high gap-risk")

        # 3. F&O ban period (SEBI)
        if sym_upper in self.fno_ban_stocks:
            sebi_flags.append(f"{sym_upper} is in F&O ban period – only squaring off permitted")
            if is_intraday:
                reasons.append("New F&O positions blocked during ban period")

        # 4. Intraday restriction
        if is_intraday and sym_upper in self.intraday_restricted_stocks:
            reasons.append(f"{sym_upper} restricted from intraday trading today (exchange directive)")
            sebi_flags.append("Intraday restriction active")

        # 5. SEBI peak margin
        margin_ok, margin_msg = self.check_sebi_margin_compliance(trade_value, available_margin)
        if not margin_ok:
            reasons.append(margin_msg)
            sebi_flags.append("Peak margin deficiency (SEBI Circular SEBI/HO/MRD2/DCAP/CIR/2021/0589)")

        # 6. Corporate action proximity
        actions = self.get_corporate_actions_for(sym_upper)
        for action in actions:
            days = action.days_until_ex()
            if days < 0:
                continue  # Ex-date passed
            if days <= self.CORP_ACTION_BLOCK_DAYS:
                reasons.append(
                    f"Ex-{action.action_type} in {days} day(s) – trading blocked "
                    f"(price will adjust by ~{action.ratio_or_amount})"
                )
                corp_action_msgs.append(f"{action.action_type} ex {action.ex_date} — BLOCK zone")
            elif days <= self.CORP_ACTION_ALERT_DAYS:
                warnings.append(
                    f"Upcoming {action.action_type} ex-date in {days} day(s) – "
                    f"expect price distortion ({action.notes})"
                )
                corp_action_msgs.append(f"{action.action_type} ex {action.ex_date} — ALERT zone")

        allowed = len(reasons) == 0

        result = ComplianceCheckResult(
            symbol=sym_upper,
            allowed=allowed,
            reasons=reasons,
            warnings=warnings,
            sebi_flags=sebi_flags,
            corporate_actions=corp_action_msgs,
            circuit_status=circuit_status,
        )

        self._log_event("TRADE_CHECK", result.summary())
        return result

    def bulk_screen_watchlist(self, watchlist: List[str],
                              price_data: Dict[str, Dict],
                              available_margin: float = 200000) -> Dict[str, ComplianceCheckResult]:
        """
        Run compliance checks on all symbols in watchlist.
        price_data: {symbol: {change_pct: float, price: float, ...}}
        Returns dict of symbol -> ComplianceCheckResult.
        """
        results = {}
        for symbol in watchlist:
            data = price_data.get(symbol, {})
            change_pct = data.get("change_pct", 0.0)
            price = data.get("price", data.get("close", 0.0))
            trade_value = price * 50  # Assume 50 shares as reference
            results[symbol] = self.check_trade(
                symbol=symbol,
                trade_value=trade_value,
                available_margin=available_margin,
                change_pct=change_pct,
            )
        return results

    def get_compliant_symbols(self, screening_results: Dict[str, ComplianceCheckResult]) -> List[str]:
        """Return only symbols that passed all compliance checks."""
        return [sym for sym, result in screening_results.items() if result.allowed]

    def generate_compliance_report(self) -> str:
        """Generate a formatted compliance status report."""
        lines = ["=" * 60, "COMPLIANCE & RISK CONTROLS REPORT",
                 f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                 "=" * 60, ""]

        # Drawdown status
        dd = self.get_drawdown_status()
        lines.append("DRAWDOWN MONITOR")
        lines.append(f"  Peak Portfolio: ₹{dd['peak_value']:,.0f}")
        lines.append(f"  Current:        ₹{dd['current_value']:,.0f}")
        lines.append(f"  Drawdown:       {dd['drawdown_pct']:.2f}% (limit: {dd['max_drawdown_pct']}%)")
        status_icon = "🚨 HALTED" if dd['halt_active'] else "✅ ACTIVE"
        lines.append(f"  Trading Status: {status_icon}")
        lines.append("")

        # Circuit-locked stocks
        lines.append("CIRCUIT-LOCKED STOCKS")
        if self.circuit_locked_stocks:
            for sym, direction in self.circuit_locked_stocks.items():
                lines.append(f"  🔒 {sym}: {direction} CIRCUIT")
        else:
            lines.append("  ✅ No stocks in circuit lock")
        lines.append("")

        # F&O ban list
        lines.append("F&O BAN LIST (SEBI)")
        if self.fno_ban_stocks:
            lines.append(f"  ⛔ {', '.join(self.fno_ban_stocks)}")
        else:
            lines.append("  ✅ No stocks in F&O ban")
        lines.append("")

        # Corporate actions
        lines.append("UPCOMING CORPORATE ACTIONS")
        upcoming = sorted(
            [a for a in self.corporate_actions if 0 <= a.days_until_ex() <= 15],
            key=lambda a: a.days_until_ex()
        )
        if upcoming:
            for action in upcoming:
                days = action.days_until_ex()
                alert = "🚫 BLOCK" if days <= self.CORP_ACTION_BLOCK_DAYS else "⚠️  ALERT"
                lines.append(
                    f"  {alert} {action.symbol}: {action.action_type} "
                    f"ex-{action.ex_date} (T-{days}d)"
                )
        else:
            lines.append("  ✅ No corporate actions within 15 days")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_event(self, event_type: str, detail: str):
        self.compliance_log.append({
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            "detail": detail,
        })
        # Keep last 500 events
        if len(self.compliance_log) > 500:
            self.compliance_log = self.compliance_log[-500:]
