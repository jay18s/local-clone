"""
ROX Proven Edge Engine v3.2 — Physical Settlement Manager
==========================================================
Enforces SEBI's physical settlement mandate for stock F&O contracts.

Since 2019, all in-the-money stock options/futures that expire must be
settled by physical delivery of underlying shares. This manager:
  1. Identifies positions at risk of triggering physical settlement
  2. Calculates days-to-expiry thresholds (T-4, T-2, T-1)
  3. Generates auto-exit or roll recommendations before the final window
  4. Blocks new ITM short positions in the settlement danger zone
  5. Estimates capital required for delivery obligations

SEBI Physical Settlement Timeline:
  T-4 : Last recommended day for new short ITM positions
  T-2 : Capital requirement finalisation; simulation recommended
  T-1 : Final opportunity to close/roll ITM positions
  T   : Expiry day — options cease trading at 3:30 PM
  T+1 : Settlement obligation confirmation
  T+2 : Share delivery / acceptance completed
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger("rox.settlement")


# --------------------------------------------------------------------------- #
#  Enums and dataclasses                                                      #
# --------------------------------------------------------------------------- #

class SettlementRisk(Enum):
    NONE     = "NONE"
    LOW      = "LOW"
    MODERATE = "MODERATE"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


class SettlementAction(Enum):
    HOLD           = "HOLD"
    MONITOR        = "MONITOR"
    PREPARE_EXIT   = "PREPARE_EXIT"
    EXIT_TODAY     = "EXIT_TODAY"
    EXIT_IMMEDIATE = "EXIT_IMMEDIATE"
    BLOCK_NEW      = "BLOCK_NEW"


@dataclass
class SettlementCheck:
    """Result of a settlement risk assessment for one position."""
    symbol:          str
    option_type:     str   # CE | PE
    strike:          float
    expiry_date:     str   # ISO
    spot:            float
    position_side:   str   # LONG | SHORT
    quantity_lots:   int

    # Computed
    days_to_expiry:  int   = 0
    itm_probability: float = 0.0   # 0-100%
    intrinsic_value: float = 0.0
    is_itm:          bool  = False
    settlement_risk: SettlementRisk  = SettlementRisk.NONE
    recommended_action: SettlementAction = SettlementAction.HOLD

    # Delivery obligation estimate (for short positions)
    delivery_shares_required: int   = 0
    delivery_capital_required: float = 0.0

    # Human-readable
    reason:   str = ""
    deadline: str = ""   # ISO date of recommended action


@dataclass
class PortfolioSettlementReport:
    """Settlement risk across all F&O positions."""
    generated_at:     str
    positions_checked: int
    critical_count:   int
    high_count:       int
    total_delivery_capital: float
    checks:           List[SettlementCheck] = field(default_factory=list)
    immediate_actions: List[str]            = field(default_factory=list)
    summary:          str = ""


# --------------------------------------------------------------------------- #
#  Physical Settlement Manager                                                #
# --------------------------------------------------------------------------- #

class PhysicalSettlementManager:
    """
    Monitors F&O positions for physical settlement risk and generates
    automated exit/roll recommendations well within SEBI's deadlines.

    Index options (Nifty, BankNifty, FinNifty, MidcapNifty) are
    cash-settled — no physical delivery required.
    Stock options and single-stock futures ARE physically settled.
    """

    # Cash-settled instruments — exempt from physical settlement
    CASH_SETTLED_INDICES = {
        "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY",
        "SENSEX", "BANKEX",
    }

    # ITM probability thresholds that trigger action
    ITM_PROB_THRESHOLDS = {
        SettlementRisk.LOW:      5,
        SettlementRisk.MODERATE: 20,
        SettlementRisk.HIGH:     40,
        SettlementRisk.CRITICAL: 60,
    }

    # Days-to-expiry action triggers
    DTE_EXIT_TODAY     = 1   # T-1 → exit today
    DTE_PREPARE        = 2   # T-2 → prepare exit
    DTE_WARN           = 4   # T-4 → start monitoring

    def __init__(self,
                 itm_probability_trigger: float = 30.0,
                 block_new_dte: int = 2):
        """
        Parameters
        ----------
        itm_probability_trigger
            ITM probability % above which auto-exit is triggered
        block_new_dte
            Block new short ITM positions within this many days of expiry
        """
        self.itm_trigger   = itm_probability_trigger
        self.block_new_dte = block_new_dte

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def check_position(self,
                       symbol:        str,
                       option_type:   str,
                       strike:        float,
                       expiry_date:   str,
                       spot:          float,
                       position_side: str,
                       quantity_lots: int,
                       lot_size:      int = 1,
                       volatility:    float = 0.20) -> SettlementCheck:
        """
        Assess physical settlement risk for one position.

        Parameters
        ----------
        symbol          Stock/index ticker
        option_type     'CE' or 'PE'
        strike          Option strike price
        expiry_date     ISO date 'YYYY-MM-DD'
        spot            Current underlying price
        position_side   'LONG' or 'SHORT'
        quantity_lots   Number of lots
        lot_size        Shares per lot
        volatility      Annualised IV for probability estimation
        """
        chk = SettlementCheck(
            symbol=symbol, option_type=option_type.upper(),
            strike=strike, expiry_date=expiry_date,
            spot=spot, position_side=position_side.upper(),
            quantity_lots=quantity_lots,
        )

        # Index options → cash settled, skip
        if symbol.upper() in self.CASH_SETTLED_INDICES:
            chk.settlement_risk     = SettlementRisk.NONE
            chk.recommended_action  = SettlementAction.HOLD
            chk.reason = "Index option — cash settled, no delivery risk."
            return chk

        # Days to expiry
        try:
            expiry = date.fromisoformat(expiry_date)
            dte    = (expiry - date.today()).days
        except ValueError:
            dte = 30

        chk.days_to_expiry = dte

        # ITM check
        if option_type.upper() == "CE":
            intrinsic = max(0.0, spot - strike)
            is_itm    = spot > strike
        else:
            intrinsic = max(0.0, strike - spot)
            is_itm    = spot < strike

        chk.intrinsic_value = intrinsic
        chk.is_itm          = is_itm

        # ITM probability estimate (simplified log-normal approximation)
        chk.itm_probability = self._itm_probability(
            option_type, spot, strike, dte, volatility
        )

        # Delivery obligation for SHORT positions
        if position_side.upper() == "SHORT" and is_itm:
            shares_req = quantity_lots * lot_size
            chk.delivery_shares_required  = shares_req
            chk.delivery_capital_required = shares_req * spot

        # Risk level
        chk.settlement_risk = self._risk_level(
            chk.itm_probability, dte, is_itm, position_side
        )

        # Action and deadline
        chk.recommended_action, chk.deadline, chk.reason = self._recommend_action(
            chk.settlement_risk, dte, position_side, is_itm, expiry_date
        )

        return chk

    def check_portfolio(self,
                        positions: List[dict],
                        lot_sizes: Optional[Dict[str, int]] = None) -> PortfolioSettlementReport:
        """
        Check all F&O positions for settlement risk.

        positions  list of dicts with keys:
                   symbol, option_type, strike, expiry_date, spot,
                   position_side, quantity_lots, [volatility]
        """
        lot_sizes = lot_sizes or {}
        checks = []
        for pos in positions:
            sym      = pos.get("symbol", "")
            lot_size = lot_sizes.get(sym.upper(), pos.get("lot_size", 50))
            chk = self.check_position(
                symbol        = sym,
                option_type   = pos.get("option_type", "CE"),
                strike        = float(pos.get("strike", 0)),
                expiry_date   = pos.get("expiry_date", ""),
                spot          = float(pos.get("spot", 0)),
                position_side = pos.get("position_side", "LONG"),
                quantity_lots = int(pos.get("quantity_lots", 1)),
                lot_size      = lot_size,
                volatility    = float(pos.get("volatility", 0.20)),
            )
            checks.append(chk)

        critical = sum(1 for c in checks
                       if c.settlement_risk == SettlementRisk.CRITICAL)
        high     = sum(1 for c in checks
                       if c.settlement_risk == SettlementRisk.HIGH)
        total_cap = sum(c.delivery_capital_required for c in checks)

        immediate = [
            f"{c.symbol} {c.option_type} {c.strike} — {c.reason}"
            for c in checks
            if c.recommended_action in (SettlementAction.EXIT_IMMEDIATE,
                                         SettlementAction.EXIT_TODAY)
        ]

        summary_parts = []
        if critical > 0:
            summary_parts.append(f"⚠️ {critical} CRITICAL positions require immediate action!")
        if high > 0:
            summary_parts.append(f"{high} HIGH-risk positions — exit by T-1.")
        if total_cap > 0:
            summary_parts.append(
                f"Estimated delivery capital: ₹{total_cap:,.0f}"
            )
        if not summary_parts:
            summary_parts.append("All positions within safe settlement parameters.")

        return PortfolioSettlementReport(
            generated_at     = datetime.now().isoformat(),
            positions_checked = len(checks),
            critical_count   = critical,
            high_count       = high,
            total_delivery_capital = total_cap,
            checks           = checks,
            immediate_actions = immediate,
            summary          = " ".join(summary_parts),
        )

    def is_new_position_blocked(self,
                                 symbol:        str,
                                 option_type:   str,
                                 strike:        float,
                                 expiry_date:   str,
                                 spot:          float,
                                 position_side: str) -> Tuple[bool, str]:
        """
        Returns (blocked, reason).
        Blocks new short ITM stock positions within block_new_dte days.
        """
        from typing import Tuple  # avoid top-level circular

        if symbol.upper() in self.CASH_SETTLED_INDICES:
            return False, ""

        try:
            expiry = date.fromisoformat(expiry_date)
            dte    = (expiry - date.today()).days
        except ValueError:
            dte = 30

        ot = option_type.upper()
        is_itm = (ot == "CE" and spot > strike) or (ot == "PE" and spot < strike)

        if (position_side.upper() == "SHORT"
                and is_itm
                and dte <= self.block_new_dte):
            reason = (
                f"BLOCKED: Short ITM {ot} on {symbol} with only {dte} days "
                f"to expiry — physical settlement risk. "
                f"Physical Settlement Manager policy: no new short ITM "
                f"positions within {self.block_new_dte} days of expiry."
            )
            return True, reason

        return False, ""

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _itm_probability(self,
                          option_type: str,
                          spot:        float,
                          strike:      float,
                          dte:         int,
                          vol:         float) -> float:
        """
        Simplified ITM probability using log-normal model.
        Returns probability 0-100.
        """
        import math
        T = max(dte / 365.0, 1e-6)
        σ = max(vol, 0.01)
        rfr = 0.065

        try:
            d2 = (math.log(spot / strike) + (rfr - 0.5 * σ**2) * T) / (σ * math.sqrt(T))
            prob_nd2 = 0.5 * (1 + math.erf(d2 / math.sqrt(2)))
        except (ValueError, ZeroDivisionError):
            prob_nd2 = 0.5

        if option_type.upper() == "CE":
            return round(prob_nd2 * 100, 1)
        else:
            return round((1 - prob_nd2) * 100, 1)

    def _risk_level(self,
                    itm_prob:      float,
                    dte:           int,
                    is_itm:        bool,
                    position_side: str) -> SettlementRisk:
        """
        Risk is only material for SHORT positions that could expire ITM.
        """
        if position_side.upper() == "LONG":
            return SettlementRisk.NONE

        if dte <= 1 and is_itm:
            return SettlementRisk.CRITICAL
        elif dte <= 2 and itm_prob >= self.ITM_PROB_THRESHOLDS[SettlementRisk.HIGH]:
            return SettlementRisk.HIGH
        elif dte <= 4 and itm_prob >= self.ITM_PROB_THRESHOLDS[SettlementRisk.MODERATE]:
            return SettlementRisk.MODERATE
        elif itm_prob >= self.ITM_PROB_THRESHOLDS[SettlementRisk.LOW]:
            return SettlementRisk.LOW
        else:
            return SettlementRisk.NONE

    def _recommend_action(self,
                           risk:          SettlementRisk,
                           dte:           int,
                           position_side: str,
                           is_itm:        bool,
                           expiry_date:   str):
        """Returns (action, deadline_iso, reason_text)."""
        expiry     = date.fromisoformat(expiry_date) if expiry_date else date.today()
        t_minus_1  = (expiry - timedelta(days=1)).isoformat()
        t_minus_2  = (expiry - timedelta(days=2)).isoformat()
        today_iso  = date.today().isoformat()

        if risk == SettlementRisk.CRITICAL:
            return (SettlementAction.EXIT_IMMEDIATE, today_iso,
                    f"CRITICAL: ITM short expires in {dte}d — exit immediately to avoid physical settlement.")

        if risk == SettlementRisk.HIGH:
            return (SettlementAction.EXIT_TODAY, t_minus_1,
                    f"HIGH: {dte}d to expiry, high ITM probability — exit by {t_minus_1}.")

        if risk == SettlementRisk.MODERATE:
            return (SettlementAction.PREPARE_EXIT, t_minus_2,
                    f"MODERATE: Prepare exit plan. Target close by {t_minus_2}.")

        if risk == SettlementRisk.LOW:
            return (SettlementAction.MONITOR, t_minus_2,
                    f"LOW: Monitor daily. Exit if moves further ITM.")

        return (SettlementAction.HOLD, "", "No settlement risk — hold.")
