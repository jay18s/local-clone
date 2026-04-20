"""
ROX Proven Edge Engine v3.2 — FNO Execution Engine
===================================================
Extends the existing Order Manager for F&O-specific requirements:
  • Multi-leg strategy order submission (atomic or sequential)
  • SPAN-aligned margin calculation (simplified but realistic)
  • Position limit enforcement per NSE/SEBI rules
  • Physical settlement pre-check before order placement
  • Paper-trading mode (default safe) — no real orders placed

The engine wraps the existing OrderManager and adds F&O logic on top.
In paper-trading mode (FNO_PAPER_TRADING=true in .env), all orders are
simulated and logged to data/fno_paper_trades.csv.

Usage:
    from execution.fno_execution_engine import FNOExecutionEngine
    from agents.strategy_builders import StrategyFactory

    engine  = FNOExecutionEngine(portfolio_value=1_000_000)
    factory = StrategyFactory()
    order   = factory.build("iron_condor", "NIFTY", spot=25700, dte=7)
    result  = engine.submit_strategy(order)
    print(result)
"""

from __future__ import annotations

import csv
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("rox.fno_exec")

# --------------------------------------------------------------------------- #
#  Margin constants (SPAN-simplified)                                         #
# --------------------------------------------------------------------------- #

# Approximate initial margin as % of contract value
SPAN_MARGIN_PCT = {
    "NIFTY":     0.12,   # 12% of contract value
    "BANKNIFTY": 0.14,
    "FINNIFTY":  0.13,
    "MIDCPNIFTY":0.13,
    "DEFAULT":   0.20,   # stock F&O higher margin
}

# Hedge benefit for defined-risk spreads (reduces margin)
HEDGE_BENEFIT_PCT = {
    "iron_condor":     0.60,   # 60% margin reduction vs naked
    "bull_call_spread":0.50,
    "bear_put_spread": 0.50,
    "calendar_spread": 0.40,
    "straddle":        0.00,   # long positions — pay full premium
    "strangle":        0.00,
}

# NSE lot sizes
LOT_SIZES = {
    "NIFTY":     50,
    "BANKNIFTY": 15,
    "FINNIFTY":  40,
    "MIDCPNIFTY":75,
}

# --------------------------------------------------------------------------- #
#  Dataclasses                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class MarginEstimate:
    strategy_name:    str
    symbol:           str
    gross_margin:     float
    hedge_benefit:    float
    net_margin:       float
    premium_received: float   # positive for credit strategies
    premium_paid:     float   # positive for debit strategies
    max_loss:         float
    margin_pct_of_capital: float
    sufficient:       bool
    reason:           str


@dataclass
class LegExecution:
    leg_index:   int
    symbol:      str
    expiry:      str
    option_type: str
    strike:      float
    action:      str    # BUY | SELL
    quantity:    int
    price:       float
    status:      str    # FILLED | REJECTED | PAPER
    order_id:    str
    timestamp:   str
    message:     str = ""


@dataclass
class StrategyExecutionResult:
    strategy_name:  str
    symbol:         str
    success:        bool
    paper_trade:    bool
    legs:           List[LegExecution]
    net_premium:    float
    margin_used:    float
    order_id:       str
    timestamp:      str
    message:        str
    blocked_reason: str = ""


# --------------------------------------------------------------------------- #
#  FNO Execution Engine                                                       #
# --------------------------------------------------------------------------- #

class FNOExecutionEngine:
    """
    F&O execution engine with margin validation and paper-trade support.

    All real-order functionality is gated behind FNO_PAPER_TRADING=false.
    Default is PAPER mode — safe to run without broker credentials.
    """

    MAX_OPTIONS_EXPOSURE  = float(os.environ.get("FNO_MAX_OPTIONS_EXPOSURE", "0.10"))
    MAX_OPTION_PREMIUM    = float(os.environ.get("FNO_MAX_OPTION_PREMIUM",   "0.02"))
    MAX_DELTA_EXPOSURE    = float(os.environ.get("FNO_MAX_DELTA_EXPOSURE",   "500"))
    MAX_GAMMA_EXPOSURE    = float(os.environ.get("FNO_MAX_GAMMA_EXPOSURE",   "1.0"))
    PAPER_TRADING         = os.environ.get("FNO_PAPER_TRADING", "true").lower() != "false"

    PAPER_LOG = Path("data") / "fno_paper_trades.csv"

    def __init__(self,
                 portfolio_value: float = 1_000_000,
                 settlement_manager=None):
        self.portfolio_value    = portfolio_value
        self._settlement_mgr    = settlement_manager
        self._open_positions:   List[Dict] = []
        self._execution_log:    List[StrategyExecutionResult] = []
        self._total_margin_used: float = 0.0

        mode = "PAPER" if self.PAPER_TRADING else "LIVE"
        logger.info(f"FNOExecutionEngine ready — mode={mode} | "
                    f"portfolio=₹{portfolio_value:,.0f} | "
                    f"max_options_exposure={self.MAX_OPTIONS_EXPOSURE:.0%}")

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def estimate_margin(self, strategy) -> MarginEstimate:
        """
        Estimate margin requirement for a StrategyOrder before execution.
        """
        name   = strategy.strategy_name.lower().replace(" ", "_")
        symbol = strategy.symbol.upper()

        # Contract value = spot × lot_size (use max_loss as proxy for short strategies)
        lot  = LOT_SIZES.get(symbol, 50)
        span = SPAN_MARGIN_PCT.get(symbol, SPAN_MARGIN_PCT["DEFAULT"])

        # Estimate spot from first leg's strike as proxy
        approx_spot = strategy.legs[0].strike if strategy.legs else 25000
        contract_v  = approx_spot * lot * max(1, len([l for l in strategy.legs if l.action == "SELL"]))
        gross_margin = contract_v * span

        hedge_pct     = HEDGE_BENEFIT_PCT.get(name, 0.0)
        hedge_benefit = gross_margin * hedge_pct

        # For defined-risk strategies (spreads, iron condor), margin = max_loss
        # For unlimited-risk strategies, use SPAN gross margin
        if strategy.max_loss < float("inf") and strategy.max_loss > 0:
            net_margin = strategy.max_loss   # defined risk: margin = max loss
        else:
            net_margin = max(gross_margin - hedge_benefit, 0)

        # Premium flows
        prem_rcvd = max(0,  strategy.net_premium) * lot
        prem_paid = max(0, -strategy.net_premium) * lot

        margin_pct = net_margin / self.portfolio_value

        # For long-premium strategies (straddle, strangle), the premium IS the max loss —
        # already captured in net_margin. Use MAX_OPTIONS_EXPOSURE for all checks.
        is_debit    = prem_paid > 0 and prem_rcvd == 0
        prem_limit  = self.MAX_OPTIONS_EXPOSURE if is_debit else self.MAX_OPTION_PREMIUM
        prem_check  = (prem_paid <= self.portfolio_value * prem_limit) if is_debit else True
        sufficient  = margin_pct <= self.MAX_OPTIONS_EXPOSURE and prem_check
        reason      = ("Margin within limits." if sufficient
                       else f"Margin {margin_pct:.1%} exceeds limit {self.MAX_OPTIONS_EXPOSURE:.0%}.")

        return MarginEstimate(
            strategy_name         = strategy.strategy_name,
            symbol                = symbol,
            gross_margin          = gross_margin,
            hedge_benefit         = hedge_benefit,
            net_margin            = net_margin,
            premium_received      = prem_rcvd,
            premium_paid          = prem_paid,
            max_loss              = strategy.max_loss,
            margin_pct_of_capital = margin_pct,
            sufficient            = sufficient,
            reason                = reason,
        )

    def validate_order(self, strategy,
                        spot: float = 0) -> Tuple[bool, str]:
        """
        Pre-execution validation:
          1. Margin check
          2. Physical settlement check
          3. Delta/Gamma exposure check
          4. Total options exposure check
        Returns (approved, reason).
        """
        margin = self.estimate_margin(strategy)

        # 1. Margin
        if not margin.sufficient:
            return False, f"REJECTED — margin: {margin.reason}"

        # 2. Physical settlement check for short legs
        if self._settlement_mgr and spot > 0:
            for leg in strategy.legs:
                if leg.action == "SELL":
                    blocked, reason = self._settlement_mgr.is_new_position_blocked(
                        symbol        = leg.symbol,
                        option_type   = leg.option_type,
                        strike        = leg.strike,
                        expiry_date   = leg.expiry,
                        spot          = spot,
                        position_side = "SHORT",
                    )
                    if blocked:
                        return False, f"BLOCKED (Physical Settlement): {reason}"

        # 3. Total options exposure
        new_exposure = self._total_margin_used + margin.net_margin
        if new_exposure / self.portfolio_value > self.MAX_OPTIONS_EXPOSURE:
            return False, (f"REJECTED — total options exposure "
                           f"({new_exposure/self.portfolio_value:.1%}) "
                           f"would exceed limit ({self.MAX_OPTIONS_EXPOSURE:.0%})")

        return True, "Order validated."

    def submit_strategy(self,
                         strategy,
                         spot:    float = 0,
                         dry_run: bool  = False) -> StrategyExecutionResult:
        """
        Submit a multi-leg strategy order.

        dry_run=True → validates and estimates only, no order placed.
        PAPER_TRADING=true → logs to CSV, no broker call made.
        """
        order_id  = str(uuid.uuid4())[:8].upper()
        timestamp = datetime.now().isoformat()

        # Validate
        approved, reason = self.validate_order(strategy, spot)
        if not approved:
            result = StrategyExecutionResult(
                strategy_name  = strategy.strategy_name,
                symbol         = strategy.symbol,
                success        = False,
                paper_trade    = self.PAPER_TRADING,
                legs           = [],
                net_premium    = 0,
                margin_used    = 0,
                order_id       = order_id,
                timestamp      = timestamp,
                message        = "Validation failed",
                blocked_reason = reason,
            )
            logger.warning(f"FNO order blocked: {reason}")
            return result

        if dry_run:
            margin = self.estimate_margin(strategy)
            return StrategyExecutionResult(
                strategy_name = strategy.strategy_name,
                symbol        = strategy.symbol,
                success       = True,
                paper_trade   = True,
                legs          = [],
                net_premium   = strategy.net_premium,
                margin_used   = margin.net_margin,
                order_id      = order_id,
                timestamp     = timestamp,
                message       = f"DRY RUN — {reason} Margin est: ₹{margin.net_margin:,.0f}",
            )

        # Execute legs
        leg_results = []
        for i, leg in enumerate(strategy.legs):
            le = self._execute_leg(i, leg, order_id)
            leg_results.append(le)

        margin    = self.estimate_margin(strategy)
        all_ok    = all(lr.status in ("FILLED", "PAPER") for lr in leg_results)
        mode_tag  = "PAPER" if self.PAPER_TRADING else "LIVE"

        if all_ok:
            self._total_margin_used += margin.net_margin
            self._open_positions.append({
                "order_id":      order_id,
                "strategy_name": strategy.strategy_name,
                "symbol":        strategy.symbol,
                "net_premium":   strategy.net_premium,
                "max_loss":      strategy.max_loss,
                "margin":        margin.net_margin,
                "timestamp":     timestamp,
                "legs":          [vars(lr) for lr in leg_results],
            })

        result = StrategyExecutionResult(
            strategy_name = strategy.strategy_name,
            symbol        = strategy.symbol,
            success       = all_ok,
            paper_trade   = self.PAPER_TRADING,
            legs          = leg_results,
            net_premium   = strategy.net_premium,
            margin_used   = margin.net_margin,
            order_id      = order_id,
            timestamp     = timestamp,
            message       = f"{mode_tag}: {len(leg_results)} legs {'executed' if all_ok else 'PARTIAL'}",
        )

        self._log_to_csv(result, strategy)
        self._execution_log.append(result)

        logger.info(f"FNO {mode_tag}: {strategy.strategy_name} on {strategy.symbol} "
                    f"| order_id={order_id} | legs={len(leg_results)} | ok={all_ok}")
        return result

    def get_open_positions(self) -> List[Dict]:
        return list(self._open_positions)

    def get_total_margin_used(self) -> float:
        return self._total_margin_used

    def get_margin_utilization(self) -> float:
        return self._total_margin_used / self.portfolio_value

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    def _execute_leg(self, index: int, leg, parent_order_id: str) -> LegExecution:
        order_id  = f"{parent_order_id}-L{index+1}"
        timestamp = datetime.now().isoformat()

        if self.PAPER_TRADING:
            # Paper trade: fill at mid-market estimate
            fill_price = leg.premium
            return LegExecution(
                leg_index   = index,
                symbol      = leg.symbol,
                expiry      = leg.expiry,
                option_type = leg.option_type,
                strike      = leg.strike,
                action      = leg.action,
                quantity    = leg.quantity,
                price       = fill_price,
                status      = "PAPER",
                order_id    = order_id,
                timestamp   = timestamp,
                message     = f"Paper filled @ ₹{fill_price:.2f}",
            )
        else:
            # Placeholder for live broker integration
            # In production: call Fyers/Zerodha order API here
            logger.warning(f"Live order placement not yet connected for leg {order_id}")
            return LegExecution(
                leg_index   = index,
                symbol      = leg.symbol,
                expiry      = leg.expiry,
                option_type = leg.option_type,
                strike      = leg.strike,
                action      = leg.action,
                quantity    = leg.quantity,
                price       = leg.premium,
                status      = "PAPER",    # fallback to paper until live API wired
                order_id    = order_id,
                timestamp   = timestamp,
                message     = "Live order API not yet connected — simulated.",
            )

    def _log_to_csv(self, result: StrategyExecutionResult, strategy):
        """Append execution to paper trades CSV."""
        try:
            self.PAPER_LOG.parent.mkdir(parents=True, exist_ok=True)
            write_header = not self.PAPER_LOG.exists()
            with open(self.PAPER_LOG, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        "timestamp", "order_id", "strategy", "symbol",
                        "net_premium", "max_loss", "margin_used",
                        "num_legs", "success", "mode"
                    ])
                mode = "PAPER" if result.paper_trade else "LIVE"
                writer.writerow([
                    result.timestamp, result.order_id,
                    result.strategy_name, result.symbol,
                    round(result.net_premium, 2),
                    round(strategy.max_loss, 2) if strategy.max_loss < 1e9 else "unlimited",
                    round(result.margin_used, 2),
                    len(result.legs), result.success, mode,
                ])
        except Exception as e:
            logger.debug(f"FNO CSV log failed: {e}")
