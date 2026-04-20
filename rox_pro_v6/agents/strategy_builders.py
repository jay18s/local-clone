"""
ROX Proven Edge Engine v3.2 — Strategy Builders
================================================
Pre-built option strategy templates invoked by the Lead Coordinator
when the AI Brain recommends a specific strategy type.

Each builder returns a StrategyOrder with concrete legs, premium,
max-profit, max-loss, breakevens, and risk management parameters.

Supported strategies:
  - Iron Condor      (market-neutral income)
  - Calendar Spread  (volatility expansion / time decay)
  - Straddle         (ATM buy both sides — event driven)
  - Strangle         (OTM buy both sides — cheaper straddle)
  - Bull Call Spread (defined-risk directional)
  - Bear Put Spread  (defined-risk directional)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import List, Optional, Tuple

from infrastructure.greeks_calculator import GreeksCalculator, OptionsLeg, PortfolioGreeks

logger = logging.getLogger("rox.strategy")


# --------------------------------------------------------------------------- #
#  Shared dataclasses                                                         #
# --------------------------------------------------------------------------- #

@dataclass
class StrategyLeg:
    """One leg of a multi-leg options strategy."""
    symbol:       str
    expiry:       str      # ISO date
    option_type:  str      # CE | PE
    strike:       float
    action:       str      # BUY | SELL
    quantity:     int      # lots
    premium:      float    # estimated mid-market price
    delta:        float = 0.0
    lot_size:     int   = 50


@dataclass
class StrategyOrder:
    """Complete multi-leg strategy order ready for execution."""
    strategy_name:  str
    symbol:         str
    legs:           List[StrategyLeg]
    net_premium:    float       # positive = credit, negative = debit
    max_profit:     float
    max_loss:       float
    breakeven_low:  float
    breakeven_high: float
    risk_reward:    float       # max_profit / max_loss
    portfolio_greeks: Optional[PortfolioGreeks] = None

    # Risk management
    profit_target_pct: float = 50.0   # close at 50% of max profit
    stop_loss_pct:     float = 100.0  # close at 2× premium received (credit)
    adjustment_trigger: str = ""      # condition that triggers leg adjustment

    # Metadata
    regime_fit:   str = ""    # which market regime this suits
    iv_fit:       str = ""    # HIGH_IV | LOW_IV | ANY
    rationale:    str = ""
    conviction:   int = 65    # 0-100


# --------------------------------------------------------------------------- #
#  Base builder                                                               #
# --------------------------------------------------------------------------- #

class _BaseBuilder:
    def __init__(self, calc: Optional[GreeksCalculator] = None):
        self._calc = calc or GreeksCalculator()

    def _atm_strike(self, spot: float, step: float = 50.0) -> float:
        return round(spot / step) * step

    def _otm_strike(self, spot: float, side: str,
                    steps: int = 1, step: float = 50.0) -> float:
        atm = self._atm_strike(spot, step)
        direction = 1 if side.upper() == "CE" else -1
        return atm + direction * steps * step

    def _nearest_expiry(self, offset_weeks: int = 0) -> str:
        today = date.today()
        days_to_thu = (3 - today.weekday()) % 7
        if days_to_thu == 0:
            days_to_thu = 7
        expiry = today + timedelta(days=days_to_thu + offset_weeks * 7)
        return expiry.isoformat()

    def _mock_premium(self, option_type: str, spot: float,
                      strike: float, dte: int, iv: float = 0.15) -> float:
        """
        Rough BSM price for premium estimation when live chain not available.
        """
        try:
            g = self._calc.calculate(option_type, spot, strike, dte, iv)
            return max(0.5, round(g.theoretical_price, 1))
        except Exception:
            intrinsic = max(0.0, spot - strike if option_type == "CE" else strike - spot)
            time_val  = spot * iv * math.sqrt(dte / 365) * 0.4
            return max(0.5, round(intrinsic + time_val, 1))


# --------------------------------------------------------------------------- #
#  Iron Condor Builder                                                        #
# --------------------------------------------------------------------------- #

class IronCondorBuilder(_BaseBuilder):
    """
    Market-neutral income strategy for range-bound / high-IV markets.

    Structure:
      SELL OTM Call  (upper short strike)
      BUY  OTM Call  (upper long strike  — wing)
      SELL OTM Put   (lower short strike)
      BUY  OTM Put   (lower long strike  — wing)

    Default delta for short strikes: 0.16–0.20 (approximately 1-2 OTM)
    Wing width: configurable (default 2 strikes)
    """

    def build(self,
              symbol:         str,
              spot:           float,
              expiry:         Optional[str] = None,
              short_delta:    float = 0.18,
              wing_width:     int   = 2,
              quantity:       int   = 1,
              lot_size:       int   = 50,
              iv:             float = 0.15,
              dte:            int   = 7) -> StrategyOrder:

        expiry  = expiry or self._nearest_expiry()
        step    = 50.0
        atm     = self._atm_strike(spot, step)

        # Short strikes (approximately 0.16-0.20 delta = ~1-2 OTM strikes)
        short_call = atm + 2 * step
        short_put  = atm - 2 * step
        long_call  = short_call + wing_width * step
        long_put   = short_put  - wing_width * step

        # Premiums
        sc_prem = self._mock_premium("CE", spot, short_call, dte, iv)
        sp_prem = self._mock_premium("PE", spot, short_put,  dte, iv)
        lc_prem = self._mock_premium("CE", spot, long_call,  dte, iv)
        lp_prem = self._mock_premium("PE", spot, long_put,   dte, iv)

        net_credit = (sc_prem + sp_prem) - (lc_prem + lp_prem)
        wing_w     = wing_width * step
        max_loss   = (wing_w - net_credit) * lot_size * quantity
        max_profit = net_credit * lot_size * quantity

        legs = [
            StrategyLeg(symbol, expiry, "CE", short_call, "SELL", quantity, sc_prem, lot_size=lot_size),
            StrategyLeg(symbol, expiry, "CE", long_call,  "BUY",  quantity, lc_prem, lot_size=lot_size),
            StrategyLeg(symbol, expiry, "PE", short_put,  "SELL", quantity, sp_prem, lot_size=lot_size),
            StrategyLeg(symbol, expiry, "PE", long_put,   "BUY",  quantity, lp_prem, lot_size=lot_size),
        ]

        pg = self._calc.portfolio_greeks([
            OptionsLeg("CE", spot, short_call, dte, iv, -quantity, lot_size=lot_size),
            OptionsLeg("CE", spot, long_call,  dte, iv,  quantity, lot_size=lot_size),
            OptionsLeg("PE", spot, short_put,  dte, iv, -quantity, lot_size=lot_size),
            OptionsLeg("PE", spot, long_put,   dte, iv,  quantity, lot_size=lot_size),
        ])

        return StrategyOrder(
            strategy_name   = "Iron Condor",
            symbol          = symbol,
            legs            = legs,
            net_premium     = net_credit,
            max_profit      = max_profit,
            max_loss        = max_loss,
            breakeven_low   = short_put  - net_credit,
            breakeven_high  = short_call + net_credit,
            risk_reward     = round(max_profit / max(max_loss, 1), 2),
            portfolio_greeks = pg,
            profit_target_pct = 50.0,
            stop_loss_pct     = 200.0,
            adjustment_trigger = f"Adjust if delta of short leg exceeds 0.30",
            regime_fit      = "CONSOLIDATION | BULL (low vol)",
            iv_fit          = "HIGH_IV",
            rationale       = (f"Iron Condor on {symbol}: "
                                f"sell {short_put}PE/{short_call}CE, "
                                f"buy {long_put}PE/{long_call}CE. "
                                f"Net credit ₹{net_credit:.1f}. "
                                f"Profit if {symbol} stays between "
                                f"{short_put:.0f}–{short_call:.0f}."),
            conviction      = 70,
        )


# --------------------------------------------------------------------------- #
#  Calendar Spread Builder                                                    #
# --------------------------------------------------------------------------- #

class CalendarSpreadBuilder(_BaseBuilder):
    """
    Sell near-term option, buy longer-dated option at same strike.
    Profits from differential time decay and volatility expansion.
    """

    def build(self,
              symbol:    str,
              spot:      float,
              option_type: str = "CE",
              near_dte:  int   = 7,
              far_dte:   int   = 30,
              quantity:  int   = 1,
              lot_size:  int   = 50,
              iv_near:   float = 0.18,
              iv_far:    float = 0.16) -> StrategyOrder:

        near_expiry = self._nearest_expiry(0)
        far_expiry  = self._nearest_expiry(3)
        atm         = self._atm_strike(spot)

        near_prem = self._mock_premium(option_type, spot, atm, near_dte, iv_near)
        far_prem  = self._mock_premium(option_type, spot, atm, far_dte,  iv_far)
        net_debit = far_prem - near_prem

        max_loss   = net_debit * lot_size * quantity
        max_profit = net_debit * 1.5 * lot_size * quantity  # rough estimate

        legs = [
            StrategyLeg(symbol, near_expiry, option_type, atm, "SELL", quantity, near_prem, lot_size=lot_size),
            StrategyLeg(symbol, far_expiry,  option_type, atm, "BUY",  quantity, far_prem,  lot_size=lot_size),
        ]

        return StrategyOrder(
            strategy_name   = f"Calendar Spread ({option_type})",
            symbol          = symbol,
            legs            = legs,
            net_premium     = -net_debit,
            max_profit      = max_profit,
            max_loss        = max_loss,
            breakeven_low   = atm * 0.97,
            breakeven_high  = atm * 1.03,
            risk_reward     = round(max_profit / max(max_loss, 1), 2),
            profit_target_pct = 75.0,
            stop_loss_pct     = 50.0,
            regime_fit      = "CONSOLIDATION",
            iv_fit          = "LOW_IV",
            rationale       = (f"Calendar on {symbol} ATM {atm:.0f}{option_type}: "
                                f"sell {near_expiry}, buy {far_expiry}. "
                                f"Net debit ₹{net_debit:.1f}/lot. "
                                f"Profit if IV expands or near-term theta decays faster."),
            conviction      = 65,
        )


# --------------------------------------------------------------------------- #
#  Straddle / Strangle Builder                                                #
# --------------------------------------------------------------------------- #

class StraddleStrangleBuilder(_BaseBuilder):
    """
    Long volatility strategy for event-driven or high-uncertainty setups.
    Straddle: buy ATM CE + PE at same strike.
    Strangle: buy OTM CE + OTM PE (cheaper, wider breakevens).
    """

    def build(self,
              symbol:     str,
              spot:       float,
              style:      str   = "straddle",   # 'straddle' | 'strangle'
              dte:        int   = 7,
              quantity:   int   = 1,
              lot_size:   int   = 50,
              iv:         float = 0.15,
              otm_steps:  int   = 1) -> StrategyOrder:

        expiry = self._nearest_expiry()
        atm    = self._atm_strike(spot)

        if style == "straddle":
            call_strike = atm
            put_strike  = atm
        else:
            call_strike = self._otm_strike(spot, "CE", otm_steps)
            put_strike  = self._otm_strike(spot, "PE", otm_steps)

        ce_prem = self._mock_premium("CE", spot, call_strike, dte, iv)
        pe_prem = self._mock_premium("PE", spot, put_strike,  dte, iv)
        total   = ce_prem + pe_prem

        max_loss = total * lot_size * quantity

        # Practical max_profit target: 3× premium paid (200% ROI).
        # Theoretically a long straddle/strangle has unlimited upside, but
        # float("inf") is meaningless for risk reporting, margin calculations,
        # and the execution engine log.  3× is the standard profit-target used
        # in discretionary vol trading and maps to a clean risk_reward = 3.0.
        max_profit = 3.0 * total * lot_size * quantity

        legs = [
            StrategyLeg(symbol, expiry, "CE", call_strike, "BUY", quantity, ce_prem, lot_size=lot_size),
            StrategyLeg(symbol, expiry, "PE", put_strike,  "BUY", quantity, pe_prem, lot_size=lot_size),
        ]

        be_up  = (call_strike + total) if style == "straddle" else (call_strike + total)
        be_dn  = (put_strike  - total) if style == "straddle" else (put_strike  - total)

        return StrategyOrder(
            strategy_name   = style.title(),
            symbol          = symbol,
            legs            = legs,
            net_premium     = -total,
            max_profit      = max_profit,
            max_loss        = max_loss,
            breakeven_low   = be_dn,
            breakeven_high  = be_up,
            risk_reward     = round(max_profit / max(max_loss, 1), 2),
            profit_target_pct = 200.0,   # 3× premium = 200% return on debit paid
            stop_loss_pct     = 50.0,
            regime_fit      = "ANY (event-driven)",
            iv_fit          = "LOW_IV",
            rationale       = (f"{style.title()} on {symbol}: "
                                f"buy {put_strike:.0f}PE + {call_strike:.0f}CE. "
                                f"Total debit ₹{total:.1f}/lot. "
                                f"Max loss ₹{max_loss:,.0f} | Target ₹{max_profit:,.0f} (3×). "
                                f"Breakeven: {be_dn:.0f} ↔ {be_up:.0f}."),
            conviction      = 60,
        )


# --------------------------------------------------------------------------- #
#  Vertical Spread Builder (Bull Call / Bear Put)                             #
# --------------------------------------------------------------------------- #

class VerticalSpreadBuilder(_BaseBuilder):
    """
    Defined-risk directional strategies.
    Bull Call Spread: buy lower CE, sell higher CE.
    Bear Put Spread:  buy higher PE, sell lower PE.
    """

    def build(self,
              symbol:     str,
              spot:       float,
              direction:  str   = "BULL",  # 'BULL' | 'BEAR'
              dte:        int   = 7,
              quantity:   int   = 1,
              lot_size:   int   = 50,
              iv:         float = 0.15,
              width:      int   = 2) -> StrategyOrder:

        expiry  = self._nearest_expiry()
        step    = 50.0
        atm     = self._atm_strike(spot, step)

        if direction.upper() == "BULL":
            long_strike  = atm
            short_strike = atm + width * step
            ot           = "CE"
            name         = "Bull Call Spread"
        else:
            long_strike  = atm
            short_strike = atm - width * step
            ot           = "PE"
            name         = "Bear Put Spread"

        long_prem  = self._mock_premium(ot, spot, long_strike,  dte, iv)
        short_prem = self._mock_premium(ot, spot, short_strike, dte, iv)
        net_debit  = long_prem - short_prem
        max_profit = (abs(long_strike - short_strike) - net_debit) * lot_size * quantity
        max_loss   = net_debit * lot_size * quantity

        legs = [
            StrategyLeg(symbol, expiry, ot, long_strike,  "BUY",  quantity, long_prem,  lot_size=lot_size),
            StrategyLeg(symbol, expiry, ot, short_strike, "SELL", quantity, short_prem, lot_size=lot_size),
        ]

        if direction.upper() == "BULL":
            be = long_strike + net_debit
            return StrategyOrder(
                strategy_name=name, symbol=symbol, legs=legs,
                net_premium=-net_debit, max_profit=max_profit, max_loss=max_loss,
                breakeven_low=be, breakeven_high=short_strike,
                risk_reward=round(max_profit / max(max_loss, 1), 2),
                regime_fit="BULL", iv_fit="ANY",
                rationale=(f"Bull Call {atm:.0f}/{short_strike:.0f}: "
                           f"debit ₹{net_debit:.1f}, max profit ₹{max_profit:.0f}, "
                           f"breakeven {be:.0f}."),
                conviction=68,
            )
        else:
            be = long_strike - net_debit
            return StrategyOrder(
                strategy_name=name, symbol=symbol, legs=legs,
                net_premium=-net_debit, max_profit=max_profit, max_loss=max_loss,
                breakeven_low=short_strike, breakeven_high=be,
                risk_reward=round(max_profit / max(max_loss, 1), 2),
                regime_fit="BEAR", iv_fit="ANY",
                rationale=(f"Bear Put {atm:.0f}/{short_strike:.0f}: "
                           f"debit ₹{net_debit:.1f}, max profit ₹{max_profit:.0f}, "
                           f"breakeven {be:.0f}."),
                conviction=68,
            )


# --------------------------------------------------------------------------- #
#  Strategy Factory                                                           #
# --------------------------------------------------------------------------- #

class StrategyFactory:
    """
    Single entry point — picks and builds the right strategy given market context.
    Called by the Lead Coordinator based on AI Brain recommendation.
    """

    STRATEGY_MAP = {
        "iron_condor":      IronCondorBuilder,
        "calendar_spread":  CalendarSpreadBuilder,
        "straddle":         StraddleStrangleBuilder,
        "strangle":         StraddleStrangleBuilder,
        "bull_call_spread": VerticalSpreadBuilder,
        "bear_put_spread":  VerticalSpreadBuilder,
    }

    def __init__(self):
        self._calc = GreeksCalculator()

    def build(self,
              strategy_name: str,
              symbol:        str,
              spot:          float,
              dte:           int   = 7,
              quantity:      int   = 1,
              lot_size:      int   = 50,
              iv:            float = 0.15,
              **kwargs) -> Optional[StrategyOrder]:
        """
        Build a strategy by name.

        strategy_name  one of: iron_condor, calendar_spread, straddle,
                                strangle, bull_call_spread, bear_put_spread
        """
        key = strategy_name.lower().replace(" ", "_")
        builder_cls = self.STRATEGY_MAP.get(key)

        if builder_cls is None:
            logger.warning(f"StrategyFactory: unknown strategy '{strategy_name}'")
            return None

        builder = builder_cls(self._calc)

        if key == "iron_condor":
            return builder.build(symbol, spot, dte=dte, quantity=quantity,
                                  lot_size=lot_size, iv=iv, **kwargs)
        elif key == "calendar_spread":
            near_dte_val = kwargs.pop("near_dte", dte)
            return builder.build(symbol, spot, near_dte=near_dte_val, quantity=quantity,
                                  lot_size=lot_size, **kwargs)
        elif key == "straddle":
            return builder.build(symbol, spot, style="straddle", dte=dte,
                                  quantity=quantity, lot_size=lot_size, iv=iv, **kwargs)
        elif key == "strangle":
            return builder.build(symbol, spot, style="strangle", dte=dte,
                                  quantity=quantity, lot_size=lot_size, iv=iv, **kwargs)
        elif key == "bull_call_spread":
            return builder.build(symbol, spot, direction="BULL", dte=dte,
                                  quantity=quantity, lot_size=lot_size, iv=iv, **kwargs)
        elif key == "bear_put_spread":
            return builder.build(symbol, spot, direction="BEAR", dte=dte,
                                  quantity=quantity, lot_size=lot_size, iv=iv, **kwargs)
        return None
