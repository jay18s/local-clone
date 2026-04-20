"""
ROX Proven Edge Engine v3.2 — Greeks Calculator
================================================
Black-Scholes-Merton implementation for Delta, Gamma, Theta, Vega, Rho.
Supports European-style index options (Nifty, BankNifty) and single-stock options.

Usage:
    from infrastructure.greeks_calculator import GreeksCalculator, OptionType

    calc = GreeksCalculator()
    g = calc.calculate("CE", spot=25700, strike=25700, days_to_expiry=7,
                        volatility=0.142, risk_free_rate=0.065)
    print(g.delta, g.gamma, g.theta, g.vega, g.rho)

    # Multi-leg portfolio Greeks
    portfolio = calc.portfolio_greeks([leg1, leg2, leg3])
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger("rox.greeks")


# --------------------------------------------------------------------------- #
#  Dataclasses                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class Greeks:
    """Complete Greeks for a single option position."""
    option_type:     str    # CE | PE
    spot:            float
    strike:          float
    days_to_expiry:  float
    volatility:      float  # annualised decimal (e.g. 0.142 = 14.2%)
    risk_free_rate:  float  # annualised decimal (e.g. 0.065 = 6.5%)
    dividend_yield:  float  = 0.0

    # Computed fields
    theoretical_price: float = 0.0
    delta:  float = 0.0   # ∂V/∂S
    gamma:  float = 0.0   # ∂²V/∂S²
    theta:  float = 0.0   # ∂V/∂t  (per calendar day)
    vega:   float = 0.0   # ∂V/∂σ  (per 1% move in IV)
    rho:    float = 0.0   # ∂V/∂r  (per 1% move in rate)

    # Derived risk metrics
    charm:           float = 0.0   # ∂delta/∂t (delta decay per day)
    dollar_delta:    float = 0.0   # delta × spot (share-equivalent exposure)
    daily_theta_pct: float = 0.0   # theta as % of option price per day
    iv_percentile:   float = 0.0   # 0-100, set externally by caller

    # Human-readable context
    moneyness:  str = ""   # ATM | 1-OTM | 2-OTM | 1-ITM | 2-ITM …
    risk_label: str = ""   # LOW | MODERATE | HIGH | EXTREME


@dataclass
class PortfolioGreeks:
    """Aggregated Greeks for a multi-leg options position."""
    net_delta:  float = 0.0
    net_gamma:  float = 0.0
    net_theta:  float = 0.0
    net_vega:   float = 0.0
    net_rho:    float = 0.0
    dollar_delta: float = 0.0
    legs:       int   = 0
    risk_label: str   = ""


@dataclass
class OptionsLeg:
    """Single leg for portfolio Greeks aggregation."""
    option_type:    str    # CE | PE
    spot:           float
    strike:         float
    days_to_expiry: float
    volatility:     float
    quantity:       int    # positive = long, negative = short
    risk_free_rate: float = 0.065
    dividend_yield: float = 0.0
    lot_size:       int   = 50    # NSE lot size (default Nifty)


# --------------------------------------------------------------------------- #
#  Main calculator                                                            #
# --------------------------------------------------------------------------- #

class GreeksCalculator:
    """
    BSM Greeks calculator.

    All angles / rates are annualised decimals.
    Theta is returned as per-calendar-day (divide by 365 internally).
    Vega is returned per 1% move in implied volatility.
    """

    # NSE lot sizes (override via lot_sizes dict)
    DEFAULT_LOT_SIZES = {
        "NIFTY":     50,
        "BANKNIFTY": 15,
        "FINNIFTY":  40,
        "MIDCPNIFTY":75,
    }

    def __init__(self,
                 risk_free_rate: float = 0.065,
                 lot_sizes: Optional[dict] = None):
        self.default_rfr   = risk_free_rate
        self._lot_sizes    = {**self.DEFAULT_LOT_SIZES, **(lot_sizes or {})}

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def calculate(self,
                  option_type:    str,
                  spot:           float,
                  strike:         float,
                  days_to_expiry: float,
                  volatility:     float,
                  risk_free_rate: Optional[float] = None,
                  dividend_yield: float = 0.0,
                  quantity:       int = 1) -> Greeks:
        """
        Calculate all Greeks for one option.

        Parameters
        ----------
        option_type     'CE' or 'PE'
        spot            Current underlying price
        strike          Option strike price
        days_to_expiry  Calendar days until expiry (can be fractional)
        volatility      Annualised implied volatility (0.142 = 14.2%)
        risk_free_rate  Annualised risk-free rate (None = use default 6.5%)
        dividend_yield  Continuous dividend yield (0 for index futures)
        quantity        Signed position size (positive=long, negative=short)
        """
        rfr = risk_free_rate if risk_free_rate is not None else self.default_rfr
        ot  = option_type.upper()

        # FIX-GREEKS-SIGMA-01: Detect callers accidentally passing raw VIX (e.g. 17.21)
        # instead of decimal IV (e.g. 0.1721).  A raw VIX value causes sigma to be ~100x
        # too large in d1/d2, collapsing delta to near-zero (observed: delta=0.044 on an
        # ATM BANKNIFTY straddle that should be ~0.50).  Auto-convert and warn so the bug
        # is visible in logs without crashing production.
        if volatility > 1.5:
            logger.warning(
                f"GreeksCalculator.calculate(): volatility={volatility:.4f} appears to be a "
                f"raw VIX/percentage value, not a decimal fraction. "
                f"Auto-converting: {volatility:.4f} → {volatility / 100:.4f}. "
                f"Caller should pass e.g. 0.172 not 17.2."
            )
            volatility = volatility / 100.0

        # Guard: invalid inputs return zero Greeks
        if spot <= 0 or strike <= 0 or volatility <= 0:
            return Greeks(
                option_type=option_type, spot=spot, strike=strike,
                days_to_expiry=days_to_expiry, volatility=volatility,
                risk_free_rate=rfr, theoretical_price=0.0,
            )
        T = max(days_to_expiry / 365.0, 1e-6)
        σ = max(volatility, 1e-6)
        S = spot
        K = strike
        q = dividend_yield

        # BSM d1, d2
        d1 = (math.log(S / K) + (rfr - q + 0.5 * σ**2) * T) / (σ * math.sqrt(T))
        d2 = d1 - σ * math.sqrt(T)

        Nd1  = self._N(d1)
        Nd2  = self._N(d2)
        Nd1n = self._N(-d1)
        Nd2n = self._N(-d2)
        n_d1 = self._n(d1)          # standard normal PDF

        # Option price
        if ot == "CE":
            price = (S * math.exp(-q * T) * Nd1
                     - K * math.exp(-rfr * T) * Nd2)
        else:
            price = (K * math.exp(-rfr * T) * Nd2n
                     - S * math.exp(-q * T) * Nd1n)

        # Delta
        if ot == "CE":
            delta = math.exp(-q * T) * Nd1
        else:
            delta = math.exp(-q * T) * (Nd1 - 1)

        # Gamma (same for CE and PE)
        gamma = (math.exp(-q * T) * n_d1) / (S * σ * math.sqrt(T))

        # Theta (per calendar day)
        theta_base = (-(S * math.exp(-q * T) * n_d1 * σ) / (2 * math.sqrt(T)))
        if ot == "CE":
            theta = (theta_base
                     - rfr * K * math.exp(-rfr * T) * Nd2
                     + q * S * math.exp(-q * T) * Nd1) / 365.0
        else:
            theta = (theta_base
                     + rfr * K * math.exp(-rfr * T) * Nd2n
                     - q * S * math.exp(-q * T) * Nd1n) / 365.0

        # Vega (per 1% IV move — divide by 100)
        vega = (S * math.exp(-q * T) * n_d1 * math.sqrt(T)) / 100.0

        # Rho (per 1% rate move — divide by 100)
        if ot == "CE":
            rho = (K * T * math.exp(-rfr * T) * Nd2) / 100.0
        else:
            rho = (-K * T * math.exp(-rfr * T) * Nd2n) / 100.0

        # Charm (delta decay per day)
        if ot == "CE":
            charm = -math.exp(-q * T) * (n_d1 * (
                (rfr - q) / (σ * math.sqrt(T)) - d2 / (2 * T)
            )) / 365.0
        else:
            charm = math.exp(-q * T) * (n_d1 * (
                (rfr - q) / (σ * math.sqrt(T)) - d2 / (2 * T)
            )) / 365.0

        # Apply quantity sign
        q_sign = 1 if quantity >= 0 else -1
        price *= q_sign

        # Derived
        dollar_delta    = delta * S * abs(quantity)
        daily_theta_pct = (theta / price * 100) if price != 0 else 0.0

        g = Greeks(
            option_type    = ot,
            spot           = S,
            strike         = K,
            days_to_expiry = days_to_expiry,
            volatility     = σ,
            risk_free_rate = rfr,
            dividend_yield = q,
            theoretical_price = price,
            delta          = delta * quantity,
            gamma          = gamma * quantity,
            theta          = theta * quantity,
            vega           = vega  * quantity,
            rho            = rho   * quantity,
            charm          = charm * quantity,
            dollar_delta   = dollar_delta,
            daily_theta_pct= daily_theta_pct,
            moneyness      = self._moneyness_label(ot, S, K),
            risk_label     = self._risk_label(delta, gamma, days_to_expiry),
        )

        # FIX-GREEKS-ATM-ASSERT: Sanity check for near-ATM options.
        # ATM delta for a CE should be ~+0.50, PE ~-0.50 (before quantity sign).
        # If an ATM option returns |delta| < 0.10 it almost certainly means sigma
        # was wrong (raw VIX not caught above, or some other upstream error).
        # Log a WARNING — do not raise, to avoid crashing live production.
        _moneyness_ratio = abs(S - K) / S
        if _moneyness_ratio < 0.005 and days_to_expiry >= 1:  # within 0.5% of ATM, not expiry
            _unsigned_delta = abs(delta)  # un-signed (before quantity)
            if _unsigned_delta < 0.10:
                logger.warning(
                    f"GreeksCalculator: ATM delta sanity FAIL — "
                    f"{ot} spot={S:.0f} strike={K:.0f} dte={days_to_expiry:.1f} "
                    f"sigma={σ:.4f} → delta={_unsigned_delta:.4f} (expected ~0.50). "
                    f"Check that volatility was passed as decimal fraction (e.g. 0.172 not 17.2)."
                )

        return g

    def portfolio_greeks(self, legs: List[OptionsLeg]) -> PortfolioGreeks:
        """
        Aggregate Greeks across a multi-leg position.
        Each leg's Greeks are multiplied by its lot_size × quantity.
        """
        pg = PortfolioGreeks(legs=len(legs))
        for leg in legs:
            g = self.calculate(
                option_type    = leg.option_type,
                spot           = leg.spot,
                strike         = leg.strike,
                days_to_expiry = leg.days_to_expiry,
                volatility     = leg.volatility,
                risk_free_rate = leg.risk_free_rate,
                dividend_yield = leg.dividend_yield,
                quantity       = leg.quantity,
            )
            multiplier = abs(leg.quantity) * leg.lot_size
            pg.net_delta  += g.delta  * multiplier
            pg.net_gamma  += g.gamma  * multiplier
            pg.net_theta  += g.theta  * multiplier
            pg.net_vega   += g.vega   * multiplier
            pg.net_rho    += g.rho    * multiplier

        pg.dollar_delta = pg.net_delta * (legs[0].spot if legs else 0)
        pg.risk_label   = self._portfolio_risk_label(pg)
        return pg

    def implied_volatility(self,
                           option_type:    str,
                           market_price:   float,
                           spot:           float,
                           strike:         float,
                           days_to_expiry: float,
                           risk_free_rate: Optional[float] = None,
                           tol:            float = 1e-5,
                           max_iter:       int   = 100) -> float:
        """
        Newton-Raphson IV solver.
        Returns annualised IV (e.g. 0.142) or 0.0 if no convergence.
        """
        rfr = risk_free_rate if risk_free_rate is not None else self.default_rfr
        σ = 0.3  # initial guess 30%
        for _ in range(max_iter):
            g     = self.calculate(option_type, spot, strike,
                                   days_to_expiry, σ, rfr)
            price = g.theoretical_price
            vega  = g.vega * 100    # undo the /100 we applied
            diff  = price - market_price
            if abs(diff) < tol:
                return σ
            if abs(vega) < 1e-10:
                break
            σ -= diff / vega
            σ = max(1e-4, min(σ, 20.0))
        return σ

    def lot_size(self, symbol: str) -> int:
        return self._lot_sizes.get(symbol.upper(), 1)

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _N(x: float) -> float:
        """Standard normal CDF."""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

    @staticmethod
    def _n(x: float) -> float:
        """Standard normal PDF."""
        return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    @staticmethod
    def _moneyness_label(option_type: str, spot: float, strike: float) -> str:
        diff_pct = (spot - strike) / spot * 100
        if option_type == "CE":
            if abs(diff_pct) < 0.3:      return "ATM"
            elif diff_pct > 2.0:         return "2-ITM"
            elif diff_pct > 0.5:         return "1-ITM"
            elif diff_pct < -2.0:        return "2-OTM"
            else:                        return "1-OTM"
        else:  # PE
            if abs(diff_pct) < 0.3:      return "ATM"
            elif diff_pct < -2.0:        return "2-ITM"
            elif diff_pct < -0.5:        return "1-ITM"
            elif diff_pct > 2.0:         return "2-OTM"
            else:                        return "1-OTM"

    @staticmethod
    def _risk_label(delta: float, gamma: float, dte: float) -> str:
        abs_delta = abs(delta)
        if dte <= 2 and abs_delta > 0.4:
            return "EXTREME"
        elif abs_delta > 0.7 or (gamma > 0.005 and dte <= 5):
            return "HIGH"
        elif abs_delta > 0.4:
            return "MODERATE"
        else:
            return "LOW"

    @staticmethod
    def _portfolio_risk_label(pg: PortfolioGreeks) -> str:
        abs_delta = abs(pg.net_delta)
        abs_gamma = abs(pg.net_gamma)
        if abs_delta > 500 or abs_gamma > 1.0:
            return "EXTREME"
        elif abs_delta > 200 or abs_gamma > 0.5:
            return "HIGH"
        elif abs_delta > 50:
            return "MODERATE"
        else:
            return "LOW"
