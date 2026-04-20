"""
ROX Virtual Broker — File-Based Live Execution Simulator
=========================================================
Treats every predicted trade as a LIVE order in a virtual account.
Tracks full F&O lifecycle: entry → mark-to-market → exit/expiry → PnL.

Design Principle: Engine behaves AS IF these are live trades.
  - Orders are queued, filled, tracked, and closed just like a real broker.
  - MTM is applied every cycle using latest market prices.
  - Theta decay is simulated daily.
  - Expiry is handled automatically.
  - Self-evaluation triggers after each trade closes.

All state persists to: data/virtual_trades/
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger("rox.virtual_broker")


# ---------------------------------------------------------------------------
# Enums & Constants
# ---------------------------------------------------------------------------

class OrderStatus(Enum):
    PENDING    = "PENDING"      # Queued, awaiting fill
    OPEN       = "OPEN"         # Filled, position live
    CLOSED     = "CLOSED"       # Exited normally (SL/target/manual)
    EXPIRED    = "EXPIRED"      # Held to expiry (options)
    CANCELLED  = "CANCELLED"    # Cancelled before fill

class OptionType(Enum):
    CE = "CE"   # Call
    PE = "PE"   # Put

class StrategyType(Enum):
    BUY_CE          = "BUY_CE"
    BUY_PE          = "BUY_PE"
    LONG_STRADDLE   = "LONG_STRADDLE"
    LONG_STRANGLE   = "LONG_STRANGLE"
    BULL_SPREAD     = "BULL_SPREAD"
    BEAR_SPREAD     = "BEAR_SPREAD"
    IRON_CONDOR     = "IRON_CONDOR"
    DIRECTIONAL     = "DIRECTIONAL"   # Futures/stock directional

# NSE lot sizes (2025-26)
LOT_SIZES = {
    "NIFTY": 75, "BANKNIFTY": 15, "FINNIFTY": 40,
    "MIDCPNIFTY": 50, "SENSEX": 10, "BANKEX": 15,
    "RELIANCE": 250, "TCS": 175, "HDFCBANK": 550,
    "ICICIBANK": 700, "INFY": 400, "SBIN": 750,
}

EXIT_REASONS = {
    "SL_HIT": "Stop loss triggered",
    "TARGET_HIT": "Target achieved",
    "EXPIRY": "Held to expiry / worthless",
    "THETA_STOP": "Theta time-stop (max hold days exceeded)",
    "MANUAL": "Manual exit by engine decision",
    "REGIME_CHANGE": "Regime flip — exit signal",
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class OptionLeg:
    """Single option leg in a trade."""
    symbol: str          # e.g. NIFTY
    option_type: str     # CE or PE
    strike: float
    expiry: str          # ISO date string
    entry_price: float   # Premium paid/received
    current_price: float
    lot_size: int
    lots: int            # Number of lots
    is_buy: bool         # True=buy, False=sell
    iv_at_entry: float = 0.0
    delta_at_entry: float = 0.0
    theta_at_entry: float = 0.0
    vega_at_entry: float = 0.0

    @property
    def cost(self) -> float:
        """Total premium cost (positive=paid, negative=received)."""
        direction = 1 if self.is_buy else -1
        return direction * self.entry_price * self.lot_size * self.lots

    @property
    def current_value(self) -> float:
        """Current MTM value."""
        direction = 1 if self.is_buy else -1
        return direction * self.current_price * self.lot_size * self.lots

    @property
    def unrealised_pnl(self) -> float:
        return self.current_value - self.cost


@dataclass
class VirtualTrade:
    """Full trade record — engine's virtual execution."""
    trade_id: str
    strategy: str               # StrategyType value
    underlying: str             # NIFTY, BANKNIFTY, etc.
    status: str                 # OrderStatus value

    # Entry context
    entry_timestamp: str
    entry_regime: str
    regime_confidence: float
    engine_conviction: int
    agent_consensus: str        # STRONG/MODERATE/WEAK/NO_CONSENSUS
    vix_at_entry: float
    spot_at_entry: float
    iv_rank_at_entry: int
    expiry_date: str
    dte_at_entry: int

    # Legs
    legs: List[Dict]            # Serialised OptionLeg dicts

    # Risk params
    max_loss: float             # Rs — maximum loss this trade can incur
    stop_loss_pct: float        # % of premium as SL (e.g. 0.4 = 40%)
    target_pct: float           # % return target (e.g. 0.8 = 80%)
    lot_size: int
    lots: int
    cost_per_lot: float         # Premium per lot
    total_cost: float           # Total premium deployed

    # SL/Target absolute levels
    sl_trigger: float           # Portfolio-level max_loss trigger
    target_trigger: float       # Portfolio-level target trigger

    # Computed at entry
    greeks: Dict = field(default_factory=dict)

    # Live tracking
    current_premium: float = 0.0
    current_spot: float = 0.0
    unrealised_pnl: float = 0.0
    mtm_history: List[Dict] = field(default_factory=list)
    theta_accrued: float = 0.0
    days_held: int = 0
    max_hold_days: int = 5

    # Exit
    exit_timestamp: Optional[str] = None
    exit_reason: Optional[str] = None
    exit_premium: Optional[float] = None
    exit_spot: Optional[float] = None
    realised_pnl: Optional[float] = None
    pnl_pct: Optional[float] = None

    # Self-evaluation
    prediction_correct: Optional[bool] = None
    prediction_verdict: Optional[str] = None  # HIT_TARGET/HIT_SL/EXPIRED/PARTIAL
    self_eval_notes: Optional[str] = None

    # Meta
    source_cycle: int = 0
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    mode: str = "LIVE"   # LIVE = real execution | SHADOW = watch-only tracking

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "VirtualTrade":
        return cls(**d)

    @property
    def is_open(self) -> bool:
        return self.status == OrderStatus.OPEN.value

    @property
    def is_closed(self) -> bool:
        return self.status in (
            OrderStatus.CLOSED.value,
            OrderStatus.EXPIRED.value,
        )


# ---------------------------------------------------------------------------
# Virtual Broker
# ---------------------------------------------------------------------------

class VirtualBroker:
    """
    File-based live-like order management system for ROX Engine.

    Simulates a real broker:
      - execute_order()   → Places a "live" virtual order
      - mark_to_market()  → Updates all open positions with latest prices
      - check_exits()     → Fires SL/target/theta/expiry auto-exits
      - close_trade()     → Manual or auto exit
      - portfolio_state() → Full account snapshot
    """

    TRADES_FILE = "data/virtual_trades/trades.jsonl"
    PORTFOLIO_FILE = "data/virtual_trades/portfolio.json"
    EVAL_FILE = "data/virtual_trades/self_eval.jsonl"
    PERF_FILE = "data/virtual_trades/performance_summary.json"

    def __init__(self, initial_capital: float = 1_000_000.0, data_dir: str = "data"):
        self._data_dir = Path(data_dir)
        self._trades_path = Path(self.TRADES_FILE)
        self._portfolio_path = Path(self.PORTFOLIO_FILE)
        self._eval_path = Path(self.EVAL_FILE)
        self._perf_path = Path(self.PERF_FILE)

        # Create dirs
        self._trades_path.parent.mkdir(parents=True, exist_ok=True)

        # Portfolio state
        self._initial_capital = initial_capital
        self._portfolio = self._load_portfolio()
        if not self._portfolio:
            self._portfolio = {
                "initial_capital": initial_capital,
                "available_capital": initial_capital,
                "deployed_capital": 0.0,
                "total_unrealised_pnl": 0.0,
                "total_realised_pnl": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_win_pct": 0.0,
                "avg_loss_pct": 0.0,
                "profit_factor": 0.0,
                "max_drawdown_pct": 0.0,
                "peak_capital": initial_capital,
                "last_updated": datetime.now().isoformat(),
            }
            self._save_portfolio()

        # Startup cleanup: remove trades with zero cost (corrupted from field-name bug)
        self._cleanup_broken_trades()
        logger.info(
            f"[VIRTUAL-BROKER] Initialised | capital=₹{initial_capital:,.0f} | "
            f"trades={self._count_open_trades()} open"
        )

    # ------------------------------------------------------------------ #
    #  Order Execution                                                     #
    # ------------------------------------------------------------------ #

    def execute_fno_trade(
        self,
        *,
        underlying: str,
        strategy: str,
        legs: List[Dict],              # List of OptionLeg-compatible dicts
        expiry_date: str,
        dte: int,
        entry_regime: str,
        regime_confidence: float,
        conviction: int,
        agent_consensus: str,
        vix: float,
        spot: float,
        iv_rank: int,
        cost_per_lot: float,
        lot_size: Optional[int] = None,
        lots: int = 1,
        greeks: Optional[Dict] = None,
        stop_loss_pct: float = 0.40,
        target_pct: float = 0.80,
        max_hold_days: int = 5,
        cycle_number: int = 0,
        notes: str = "",
        mode: str = "LIVE",   # "LIVE" or "SHADOW" (watch-only tracking)
    ) -> str:
        """
        Place a virtual F&O order. Behaves as if sent to a live broker.
        SHADOW mode tracks watch-only suggestions to evaluate the engine's
        conservatism — these do NOT affect capital but DO affect self-eval.

        Returns trade_id.
        """
        lot_size = lot_size or LOT_SIZES.get(underlying.upper(), 50)
        # cost_per_lot from OptionSuggestion = premium × lot_size (already per-lot total)
        # total_cost = cost_per_lot × lots  (NOT × lot_size again)
        total_cost     = cost_per_lot * lots
        max_loss       = total_cost
        sl_trigger     = total_cost * (1 - stop_loss_pct)
        target_trigger = total_cost * (1 + target_pct)

        trade_id = str(uuid.uuid4())[:8].upper()
        now = datetime.now().isoformat()

        trade = VirtualTrade(
            trade_id=trade_id,
            strategy=strategy,
            underlying=underlying,
            status=OrderStatus.OPEN.value,
            entry_timestamp=now,
            entry_regime=entry_regime,
            regime_confidence=regime_confidence,
            engine_conviction=conviction,
            agent_consensus=agent_consensus,
            vix_at_entry=vix,
            spot_at_entry=spot,
            iv_rank_at_entry=iv_rank,
            expiry_date=expiry_date,
            dte_at_entry=dte,
            legs=legs,
            max_loss=max_loss,
            stop_loss_pct=stop_loss_pct,
            target_pct=target_pct,
            lot_size=lot_size,
            lots=lots,
            cost_per_lot=cost_per_lot,
            total_cost=total_cost,
            sl_trigger=sl_trigger,
            target_trigger=target_trigger,
            greeks=greeks or {},
            current_premium=cost_per_lot,
            current_spot=spot,
            unrealised_pnl=0.0,
            max_hold_days=max_hold_days,
            source_cycle=cycle_number,
            notes=notes,
            tags=[strategy, underlying, entry_regime],
            mode=mode,
        )

        self._append_trade(trade)

        # Update portfolio (SHADOW trades don't use real capital)
        if mode == "LIVE":
            self._portfolio["available_capital"] -= total_cost
            self._portfolio["deployed_capital"] += total_cost
            self._portfolio["total_trades"] += 1
        else:
            self._portfolio["shadow_trades"] = self._portfolio.get("shadow_trades", 0) + 1
        self._portfolio["last_updated"] = now
        self._save_portfolio()

        _mode_tag = "" if mode == "LIVE" else " [SHADOW]"
        logger.info(
            f"[VIRTUAL-BROKER] ✅ ORDER EXECUTED{_mode_tag} | id={trade_id} | "
            f"{underlying} {strategy} | lots={lots} | "
            f"cost/lot=₹{cost_per_lot:,.0f} | total=₹{total_cost:,.0f} | "
            f"expiry={expiry_date} | SL=₹{sl_trigger:,.0f} | TGT=₹{target_trigger:,.0f}"
        )
        return trade_id

    # ------------------------------------------------------------------ #
    #  Mark-to-Market                                                      #
    # ------------------------------------------------------------------ #

    def mark_to_market(
        self,
        price_updates: Dict[str, float],    # {underlying: current_spot}
        option_price_updates: Optional[Dict[str, float]] = None,  # {trade_id: current_premium}
        theta_decay_factor: float = 1.0,    # days elapsed since last MTM
        vix_premium_mult: float = 1.0,      # from MarketImpactMonitor — expands/contracts premium
    ) -> Dict[str, float]:
        """
        Update all open positions with current market prices.
        Applies theta decay + VIX premium multiplier if exact option prices
        not available.  vix_premium_mult > 1 means vol expansion (e.g. VIX spike).
        Returns {trade_id: unrealised_pnl}.
        """
        open_trades = self.get_open_trades()
        pnl_map = {}
        total_unrealised = 0.0

        for trade in open_trades:
            underlying = trade.underlying
            current_spot = price_updates.get(underlying, trade.current_spot)
            trade.current_spot = current_spot
            trade.days_held += theta_decay_factor

            # Get current premium
            if option_price_updates and trade.trade_id in option_price_updates:
                new_premium = option_price_updates[trade.trade_id]
            else:
                # Simulate: theta decay + VIX vol-expansion adjustment
                # When VIX spikes, remaining premium expands even as theta burns
                daily_theta  = trade.greeks.get("theta", 0.0)
                theta_decay  = abs(daily_theta) * theta_decay_factor
                # Apply vol expansion: if VIX mult > 1, vega partially offsets theta
                vega_val     = trade.greeks.get("vega", 0.0)
                vol_expansion = 0.0
                if vix_premium_mult > 1.0 and vega_val > 0:
                    # Each +1% IV adds vega*lot_size to premium — approximate 1 IV pt per 0.1x mult
                    iv_pts_approx = (vix_premium_mult - 1.0) * 10.0
                    vol_expansion = vega_val * iv_pts_approx * 0.3   # dampened: not full expansion
                new_premium = max(
                    trade.current_premium - theta_decay + vol_expansion,
                    0.0
                )
                trade.theta_accrued += theta_decay

            trade.current_premium = new_premium
            # cost_per_lot and current_premium are already per-lot totals
            unrealised = (new_premium - trade.cost_per_lot) * trade.lots
            trade.unrealised_pnl = unrealised
            total_unrealised += unrealised
            pnl_map[trade.trade_id] = unrealised

            # Log MTM snapshot
            trade.mtm_history.append({
                "ts": datetime.now().isoformat(),
                "spot": current_spot,
                "premium": new_premium,
                "unrealised_pnl": unrealised,
                "theta_accrued": trade.theta_accrued,
            })

            self._update_trade(trade)

        # Update portfolio
        self._portfolio["total_unrealised_pnl"] = total_unrealised
        self._portfolio["last_updated"] = datetime.now().isoformat()
        self._save_portfolio()

        if open_trades:
            logger.info(
                f"[VIRTUAL-BROKER] MTM | {len(open_trades)} open | "
                f"unrealised=₹{total_unrealised:+,.0f}"
            )
        return pnl_map

    # ------------------------------------------------------------------ #
    #  Auto-Exit Checks                                                    #
    # ------------------------------------------------------------------ #

    def check_exits(
        self,
        current_date: Optional[date] = None,
    ) -> List[Dict]:
        """
        Check all open trades for exit conditions.
        Returns list of exit events.
        """
        today = current_date or date.today()
        exits = []

        for trade in self.get_open_trades():
            exit_reason = None
            exit_premium = trade.current_premium

            # 1. EXPIRY check
            try:
                expiry = VirtualBroker._parse_expiry_date(trade.expiry_date)
                if expiry and today >= expiry:
                    exit_reason = "EXPIRY"
                    exit_premium = 0.0
            except Exception:
                pass

            # 2. THETA TIME-STOP
            if exit_reason is None and trade.days_held >= trade.max_hold_days:
                exit_reason = "THETA_STOP"

            # 3. STOP-LOSS (current_premium is per-lot total, sl_trigger is total×lots)
            if exit_reason is None:
                current_value = trade.current_premium * trade.lots
                if current_value <= trade.sl_trigger:
                    exit_reason = "SL_HIT"

            # 4. TARGET HIT
            if exit_reason is None:
                current_value = trade.current_premium * trade.lots
                if current_value >= trade.target_trigger:
                    exit_reason = "TARGET_HIT"

            if exit_reason:
                result = self.close_trade(
                    trade_id=trade.trade_id,
                    exit_premium=exit_premium,
                    exit_spot=trade.current_spot,
                    reason=exit_reason,
                )
                exits.append(result)
                logger.info(
                    f"[VIRTUAL-BROKER] AUTO-EXIT | {trade.trade_id} | "
                    f"{trade.underlying} {trade.strategy} | reason={exit_reason} | "
                    f"pnl=₹{result.get('realised_pnl', 0):+,.0f}"
                )

        return exits

    # ------------------------------------------------------------------ #
    #  Trade Close                                                         #
    # ------------------------------------------------------------------ #

    def close_trade(
        self,
        trade_id: str,
        exit_premium: float,
        exit_spot: float,
        reason: str = "MANUAL",
    ) -> Dict:
        """Close a trade and compute final PnL."""
        all_trades = self._load_all_trades()
        trade = None
        idx = None

        for i, t in enumerate(all_trades):
            if t.trade_id == trade_id and t.is_open:
                trade = t
                idx = i
                break

        if trade is None:
            logger.warning(f"[VIRTUAL-BROKER] close_trade: {trade_id} not found or not open")
            return {}

        now = datetime.now().isoformat()
        realised_pnl = (exit_premium - trade.cost_per_lot) * trade.lots
        pnl_pct = (realised_pnl / trade.total_cost * 100) if trade.total_cost > 0 else 0.0

        trade.exit_timestamp = now
        trade.exit_reason = reason
        trade.exit_premium = exit_premium
        trade.exit_spot = exit_spot
        trade.realised_pnl = realised_pnl
        trade.pnl_pct = pnl_pct
        trade.status = OrderStatus.EXPIRED.value if reason == "EXPIRY" else OrderStatus.CLOSED.value
        trade.unrealised_pnl = 0.0

        # Self-evaluation
        trade = self._self_evaluate(trade)

        all_trades[idx] = trade
        self._rewrite_trades(all_trades)

        # Update portfolio (SHADOW trades tracked separately — no real capital impact)
        if getattr(trade, "mode", "LIVE") == "LIVE":
            self._portfolio["available_capital"] += (trade.total_cost + realised_pnl)
            self._portfolio["deployed_capital"] = max(
                0.0, self._portfolio["deployed_capital"] - trade.total_cost
            )
            self._portfolio["total_realised_pnl"] += realised_pnl
            if realised_pnl > 0:
                self._portfolio["winning_trades"] += 1
            else:
                self._portfolio["losing_trades"] += 1
        else:
            # SHADOW mode: track separately for conservatism evaluation
            self._portfolio["shadow_realised_pnl"] = (
                self._portfolio.get("shadow_realised_pnl", 0.0) + realised_pnl
            )
            shadow_wins = self._portfolio.get("shadow_wins", 0)
            shadow_losses = self._portfolio.get("shadow_losses", 0)
            if realised_pnl > 0:
                self._portfolio["shadow_wins"] = shadow_wins + 1
            else:
                self._portfolio["shadow_losses"] = shadow_losses + 1

        closed = self._portfolio["winning_trades"] + self._portfolio["losing_trades"]
        self._portfolio["win_rate"] = (
            self._portfolio["winning_trades"] / closed * 100 if closed > 0 else 0.0
        )

        # Update peak and drawdown
        current_equity = self._portfolio["available_capital"] + self._portfolio["deployed_capital"]
        if current_equity > self._portfolio["peak_capital"]:
            self._portfolio["peak_capital"] = current_equity
        drawdown = (self._portfolio["peak_capital"] - current_equity) / self._portfolio["peak_capital"] * 100
        if drawdown > self._portfolio["max_drawdown_pct"]:
            self._portfolio["max_drawdown_pct"] = drawdown

        self._portfolio["last_updated"] = now
        self._save_portfolio()

        # Log to self-eval file
        self._log_eval(trade)

        # Update performance summary
        self._update_performance_summary()

        return {
            "trade_id": trade_id,
            "underlying": trade.underlying,
            "strategy": trade.strategy,
            "exit_reason": reason,
            "realised_pnl": realised_pnl,
            "pnl_pct": pnl_pct,
            "prediction_correct": trade.prediction_correct,
            "verdict": trade.prediction_verdict,
        }

    # ------------------------------------------------------------------ #
    #  Self Evaluation                                                     #
    # ------------------------------------------------------------------ #

    def _self_evaluate(self, trade: VirtualTrade) -> VirtualTrade:
        """
        Evaluate engine's prediction accuracy after trade closes.
        Fills prediction_correct, prediction_verdict, self_eval_notes.
        """
        pnl = trade.realised_pnl or 0.0
        reason = trade.exit_reason or ""

        if reason == "TARGET_HIT":
            trade.prediction_correct = True
            trade.prediction_verdict = "HIT_TARGET"
            trade.self_eval_notes = (
                f"Engine correctly predicted {trade.strategy} opportunity. "
                f"PnL=₹{pnl:+,.0f} ({trade.pnl_pct:.1f}%)"
            )
        elif reason == "SL_HIT":
            trade.prediction_correct = False
            trade.prediction_verdict = "HIT_SL"
            trade.self_eval_notes = (
                f"Engine prediction failed — SL triggered. "
                f"PnL=₹{pnl:+,.0f} ({trade.pnl_pct:.1f}%). "
                f"Regime was {trade.entry_regime}, conviction={trade.engine_conviction}."
            )
        elif reason == "EXPIRY":
            trade.prediction_correct = pnl >= 0
            trade.prediction_verdict = "EXPIRED_PROFIT" if pnl >= 0 else "EXPIRED_WORTHLESS"
            trade.self_eval_notes = (
                f"Held to expiry. {'Profitable' if pnl >= 0 else 'Expired worthless'}. "
                f"PnL=₹{pnl:+,.0f}"
            )
        elif reason == "THETA_STOP":
            trade.prediction_correct = pnl >= 0
            trade.prediction_verdict = "THETA_EXIT"
            trade.self_eval_notes = (
                f"Time-stop triggered after {trade.days_held:.0f} days. "
                f"PnL=₹{pnl:+,.0f} — theta eroded premium without expected move."
            )
        else:
            trade.prediction_correct = pnl >= 0
            trade.prediction_verdict = "MANUAL_EXIT"
            trade.self_eval_notes = f"Manual exit. PnL=₹{pnl:+,.0f}"

        logger.info(
            f"[SELF-EVAL] {trade.trade_id} | {trade.prediction_verdict} | "
            f"correct={trade.prediction_correct} | pnl=₹{pnl:+,.0f} ({trade.pnl_pct:.1f}%)"
        )
        return trade

    def _log_eval(self, trade: VirtualTrade):
        """Append evaluation record to self-eval JSONL."""
        record = {
            "ts": datetime.now().isoformat(),
            "trade_id": trade.trade_id,
            "underlying": trade.underlying,
            "strategy": trade.strategy,
            "entry_regime": trade.entry_regime,
            "regime_confidence": trade.regime_confidence,
            "conviction": trade.engine_conviction,
            "agent_consensus": trade.agent_consensus,
            "vix_at_entry": trade.vix_at_entry,
            "iv_rank": trade.iv_rank_at_entry,
            "dte_at_entry": trade.dte_at_entry,
            "days_held": trade.days_held,
            "exit_reason": trade.exit_reason,
            "realised_pnl": trade.realised_pnl,
            "pnl_pct": trade.pnl_pct,
            "prediction_correct": trade.prediction_correct,
            "prediction_verdict": trade.prediction_verdict,
            "notes": trade.self_eval_notes,
        }
        try:
            with open(self._eval_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.warning(f"[VIRTUAL-BROKER] eval log failed: {e}")

    # ------------------------------------------------------------------ #
    #  Performance Summary (Self-Improvement Feed)                        #
    # ------------------------------------------------------------------ #

    def _update_performance_summary(self):
        """Rebuild performance summary from all closed trades."""
        all_trades = self._load_all_trades()
        closed = [t for t in all_trades if t.is_closed]

        if not closed:
            return

        wins = [t for t in closed if (t.realised_pnl or 0) > 0]
        losses = [t for t in closed if (t.realised_pnl or 0) <= 0]

        total_pnl = sum(t.realised_pnl or 0 for t in closed)
        gross_profit = sum(t.realised_pnl or 0 for t in wins)
        gross_loss = abs(sum(t.realised_pnl or 0 for t in losses))

        by_strategy = {}
        by_regime = {}
        by_conviction = {}

        for t in closed:
            # By strategy
            s = t.strategy
            if s not in by_strategy:
                by_strategy[s] = {"trades": 0, "wins": 0, "pnl": 0.0}
            by_strategy[s]["trades"] += 1
            if (t.realised_pnl or 0) > 0:
                by_strategy[s]["wins"] += 1
            by_strategy[s]["pnl"] += (t.realised_pnl or 0)

            # By regime
            r = t.entry_regime
            if r not in by_regime:
                by_regime[r] = {"trades": 0, "wins": 0, "pnl": 0.0}
            by_regime[r]["trades"] += 1
            if (t.realised_pnl or 0) > 0:
                by_regime[r]["wins"] += 1
            by_regime[r]["pnl"] += (t.realised_pnl or 0)

            # By conviction bucket
            cb = self._conviction_bucket(t.engine_conviction)
            if cb not in by_conviction:
                by_conviction[cb] = {"trades": 0, "wins": 0, "pnl": 0.0}
            by_conviction[cb]["trades"] += 1
            if (t.realised_pnl or 0) > 0:
                by_conviction[cb]["wins"] += 1
            by_conviction[cb]["pnl"] += (t.realised_pnl or 0)

        summary = {
            "generated_at": datetime.now().isoformat(),
            "total_closed_trades": len(closed),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate_pct": len(wins) / len(closed) * 100,
            "total_pnl": total_pnl,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else float("inf"),
            "avg_win_pnl": gross_profit / len(wins) if wins else 0.0,
            "avg_loss_pnl": -(gross_loss / len(losses)) if losses else 0.0,
            "avg_win_pct": sum(t.pnl_pct or 0 for t in wins) / len(wins) if wins else 0.0,
            "avg_loss_pct": sum(t.pnl_pct or 0 for t in losses) / len(losses) if losses else 0.0,
            "portfolio": self._portfolio,
            "by_strategy": by_strategy,
            "by_regime": by_regime,
            "by_conviction": by_conviction,
            # Self-improvement insights
            "improvement_signals": self._generate_improvement_signals(
                closed, by_strategy, by_regime, by_conviction
            ),
        }

        try:
            with open(self._perf_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"[VIRTUAL-BROKER] perf summary write failed: {e}")

    def _generate_improvement_signals(
        self,
        closed: List[VirtualTrade],
        by_strategy: Dict,
        by_regime: Dict,
        by_conviction: Dict,
    ) -> List[Dict]:
        """
        Generate self-improvement recommendations from trade history.
        This feeds back into the engine's calibration loop.
        """
        signals = []

        # Signal 1: Strategy accuracy
        for strategy, stats in by_strategy.items():
            if stats["trades"] >= 3:
                wr = stats["wins"] / stats["trades"] * 100
                if wr < 40:
                    signals.append({
                        "type": "STRATEGY_UNDERPERFORMING",
                        "signal": f"{strategy} win rate {wr:.0f}% is below 40% threshold",
                        "action": f"Reduce position size for {strategy} or raise conviction threshold",
                        "severity": "HIGH" if wr < 30 else "MEDIUM",
                    })
                elif wr > 70:
                    signals.append({
                        "type": "STRATEGY_OUTPERFORMING",
                        "signal": f"{strategy} win rate {wr:.0f}% — high-performing setup",
                        "action": f"Consider increasing lot allocation for {strategy}",
                        "severity": "INFO",
                    })

        # Signal 2: Regime accuracy
        for regime, stats in by_regime.items():
            if stats["trades"] >= 3:
                wr = stats["wins"] / stats["trades"] * 100
                if wr < 40:
                    signals.append({
                        "type": "REGIME_MISMATCH",
                        "signal": f"Trades in {regime} regime have {wr:.0f}% win rate",
                        "action": f"Avoid trading or ultra-reduce size in {regime} regime",
                        "severity": "HIGH",
                    })

        # Signal 3: Conviction calibration
        for cb, stats in by_conviction.items():
            if stats["trades"] >= 3:
                wr = stats["wins"] / stats["trades"] * 100
                if cb in ("55-65", "65-75") and wr < 40:
                    signals.append({
                        "type": "LOW_CONVICTION_LOSING",
                        "signal": f"Conviction {cb} trades: {wr:.0f}% win rate",
                        "action": "Raise minimum conviction threshold to 75+",
                        "severity": "MEDIUM",
                    })

        # Signal 4: Theta decay patterns
        theta_stops = [t for t in closed if t.exit_reason == "THETA_STOP"]
        if len(theta_stops) > len(closed) * 0.3:
            signals.append({
                "type": "THETA_DECAY_PROBLEM",
                "signal": f"{len(theta_stops)}/{len(closed)} trades exited via theta-stop",
                "action": "Reduce max_hold_days or buy closer-to-expiry options",
                "severity": "MEDIUM",
            })

        # Signal 5: SL hit rate
        sl_hits = [t for t in closed if t.exit_reason == "SL_HIT"]
        if len(sl_hits) > len(closed) * 0.5:
            signals.append({
                "type": "SL_HIT_RATE_HIGH",
                "signal": f"{len(sl_hits)}/{len(closed)} trades hit stop loss",
                "action": "Widen SL or reduce position size; review entry timing",
                "severity": "HIGH",
            })

        return signals

    def get_improvement_signals(self) -> List[Dict]:
        """Return latest improvement signals from performance file."""
        try:
            if self._perf_path.exists():
                with open(self._perf_path) as f:
                    perf = json.load(f)
                return perf.get("improvement_signals", [])
        except Exception:
            pass
        return []

    # ------------------------------------------------------------------ #
    #  Portfolio & Queries                                                 #
    # ------------------------------------------------------------------ #

    def get_open_trades(self) -> List[VirtualTrade]:
        return [t for t in self._load_all_trades() if t.is_open]

    def get_closed_trades(self, last_n: Optional[int] = None) -> List[VirtualTrade]:
        closed = [t for t in self._load_all_trades() if t.is_closed]
        if last_n:
            return closed[-last_n:]
        return closed

    def get_shadow_trades(self, status: Optional[str] = None) -> List["VirtualTrade"]:
        """Return shadow (watch-only) trades, optionally filtered by status."""
        all_t = self._load_all_trades()
        shadow = [t for t in all_t if getattr(t, "mode", "LIVE") == "SHADOW"]
        if status:
            shadow = [t for t in shadow if t.status == status]
        return shadow

    def get_live_trades(self, status: Optional[str] = None) -> List["VirtualTrade"]:
        """Return live (real) trades, optionally filtered by status."""
        all_t = self._load_all_trades()
        live = [t for t in all_t if getattr(t, "mode", "LIVE") == "LIVE"]
        if status:
            live = [t for t in live if t.status == status]
        return live

    def shadow_conservatism_report(self) -> Dict:
        """
        Compares LIVE vs SHADOW trade outcomes to evaluate the engine's
        WAIT decisions. Answers: 'Were we right to wait?'
        """
        closed_shadow = self.get_shadow_trades("CLOSED") + self.get_shadow_trades("EXPIRED")
        if not closed_shadow:
            return {"trades": 0, "verdict": "No shadow trades closed yet"}

        shadow_wins   = [t for t in closed_shadow if (t.realised_pnl or 0) > 0]
        shadow_losses = [t for t in closed_shadow if (t.realised_pnl or 0) <= 0]
        shadow_total_pnl = sum(t.realised_pnl or 0 for t in closed_shadow)
        shadow_wr = len(shadow_wins) / len(closed_shadow) * 100 if closed_shadow else 0

        live_closed = self.get_live_trades("CLOSED") + self.get_live_trades("EXPIRED")
        live_pnl = sum(t.realised_pnl or 0 for t in live_closed)
        live_wr = (len([t for t in live_closed if (t.realised_pnl or 0) > 0])
                   / len(live_closed) * 100) if live_closed else 0

        # Was WAIT a good decision?
        # If shadow trades would have LOST money → engine was RIGHT to wait
        # If shadow trades would have MADE money → engine was TOO conservative
        wait_was_correct = shadow_total_pnl < 0
        opportunity_cost = shadow_total_pnl  # What we missed (negative = we saved losses)

        verdict = "WAIT_WAS_CORRECT" if wait_was_correct else "TOO_CONSERVATIVE"
        if abs(shadow_total_pnl) < 5000:
            verdict = "WAIT_WAS_NEUTRAL"

        return {
            "trades": len(closed_shadow),
            "shadow_win_rate": shadow_wr,
            "shadow_total_pnl": shadow_total_pnl,
            "shadow_wins": len(shadow_wins),
            "shadow_losses": len(shadow_losses),
            "live_win_rate": live_wr,
            "live_total_pnl": live_pnl,
            "opportunity_cost": opportunity_cost,
            "verdict": verdict,
            "message": (
                f"WAIT decisions saved ₹{abs(opportunity_cost):,.0f} in losses"
                if wait_was_correct else
                f"WAIT decisions missed ₹{opportunity_cost:,.0f} in potential profit"
            ),
        }

    def portfolio_state(self) -> Dict:
        """Full portfolio snapshot."""
        self._portfolio = self._load_portfolio() or self._portfolio
        open_trades = self.get_open_trades()
        total_unrealised = sum(t.unrealised_pnl for t in open_trades)
        self._portfolio["total_unrealised_pnl"] = total_unrealised

        return {
            **self._portfolio,
            "open_positions": len(open_trades),
            "open_trades_detail": [
                {
                    "id": t.trade_id,
                    "underlying": t.underlying,
                    "strategy": t.strategy,
                    "entry_ts": t.entry_timestamp,
                    "expiry": t.expiry_date,
                    "days_held": t.days_held,
                    "total_cost": t.total_cost,
                    "unrealised_pnl": t.unrealised_pnl,
                    "regime": t.entry_regime,
                }
                for t in open_trades
            ],
        }

    def win_rate(self) -> float:
        return self._portfolio.get("win_rate", 0.0)

    def total_realised_pnl(self) -> float:
        return self._portfolio.get("total_realised_pnl", 0.0)

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def _append_trade(self, trade: VirtualTrade):
        try:
            with open(self._trades_path, "a") as f:
                f.write(json.dumps(trade.to_dict(), default=str) + "\n")
        except Exception as e:
            logger.error(f"[VIRTUAL-BROKER] append_trade failed: {e}")

    def _update_trade(self, trade: VirtualTrade):
        """Update a single trade record in the file."""
        all_trades = self._load_all_trades()
        for i, t in enumerate(all_trades):
            if t.trade_id == trade.trade_id:
                all_trades[i] = trade
                break
        self._rewrite_trades(all_trades)

    def _rewrite_trades(self, trades: List[VirtualTrade]):
        try:
            with open(self._trades_path, "w") as f:
                for t in trades:
                    f.write(json.dumps(t.to_dict(), default=str) + "\n")
        except Exception as e:
            logger.error(f"[VIRTUAL-BROKER] rewrite_trades failed: {e}")

    def _load_all_trades(self) -> List[VirtualTrade]:
        if not self._trades_path.exists():
            return []
        trades = []
        try:
            with open(self._trades_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            d = json.loads(line)
                            trades.append(VirtualTrade(**d))
                        except Exception:
                            pass
        except Exception as e:
            logger.error(f"[VIRTUAL-BROKER] load_all_trades failed: {e}")
        return trades

    def _count_open_trades(self) -> int:
        return len(self.get_open_trades())

    def _cleanup_broken_trades(self):
        """
        Remove trades with zero cost_per_lot (corrupted from field-name bug).
        Also reconciles portfolio capital for removed trades.
        Called once at startup.
        """
        all_trades = self._load_all_trades()
        broken = [t for t in all_trades if t.cost_per_lot <= 0 and t.is_open]
        if not broken:
            return
        clean = [t for t in all_trades if not (t.cost_per_lot <= 0 and t.is_open)]
        self._rewrite_trades(clean)
        logger.info(
            f"[VIRTUAL-BROKER] Startup cleanup: removed {len(broken)} "
            f"broken zero-cost trade(s)"
        )

    def _load_portfolio(self) -> Optional[Dict]:
        if not self._portfolio_path.exists():
            return None
        try:
            with open(self._portfolio_path) as f:
                return json.load(f)
        except Exception:
            return None

    def _save_portfolio(self):
        try:
            with open(self._portfolio_path, "w") as f:
                json.dump(self._portfolio, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"[VIRTUAL-BROKER] save_portfolio failed: {e}")

    @staticmethod
    def _parse_expiry_date(expiry_str: str):
        """Parse expiry string in multiple formats: ISO, DD-MMM-YYYY, DD/MM/YYYY."""
        import re
        from datetime import date
        s = str(expiry_str).strip()
        # ISO: 2026-04-29
        try:
            return date.fromisoformat(s)
        except Exception:
            pass
        # DD-MMM-YYYY: 29-APR-2026 or 29-Apr-2026
        _months = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
                   "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}
        m = re.match(r"(\d{1,2})[-/]([A-Za-z]{3})[-/](\d{4})", s)
        if m:
            d,mn,y = int(m.group(1)),m.group(2).upper(),int(m.group(3))
            if mn in _months:
                return date(_months[mn] and y, _months[mn], d)
        # DD/MM/YYYY
        m2 = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", s)
        if m2:
            return date(int(m2.group(3)), int(m2.group(2)), int(m2.group(1)))
        return None

    @staticmethod
    def _conviction_bucket(conviction: int) -> str:
        if conviction < 55: return "<55"
        if conviction < 65: return "55-65"
        if conviction < 75: return "65-75"
        if conviction < 85: return "75-85"
        return "85+"


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_broker_instance: Optional[VirtualBroker] = None

def get_virtual_broker(capital: float = 1_000_000.0) -> VirtualBroker:
    global _broker_instance
    if _broker_instance is None:
        _broker_instance = VirtualBroker(initial_capital=capital)
    return _broker_instance
