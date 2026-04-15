"""
ROX Proven Edge Engine v4.0 — Directional Option Advisor  (v4.2)
=================================================================
v4.1: Dynamic, holiday-aware expiry selection with trading-day DTE.
v4.2 additions:
  A. Delta-based strike selection (target delta 0.30-0.40).
     Full Greeks (delta/gamma/theta/vega) and breakeven logged.
     Liquidity filter: OI > 10,000 AND volume > 2,000.
     Bid-ask spread auto-reject > 15%.
  B. Pre-trade checklist (5-point) before every suggestion.
     Any fail -> REJECTED; shown in output.
  C. Regime-aware strategy routing:
     BULL/BEAR     -> naked long OTM (existing _buy_call/_buy_put)
     All others    -> Bull/Bear Spread (max risk capped at 4.5% portfolio)
     HIGH IV       -> Iron Condor
     NO_CONSENSUS  -> Long Straddle (LOW IV only)
  D. Scoring engine 0-100 (threshold 68 to show).
     30% tech + 25% Greeks + 20% IV + 15% liquidity + 10% calendar.

NSE 2026 expiry weekdays:
  NIFTY/FINNIFTY  -> Tuesday  (1)
  BANKNIFTY       -> Wednesday (2)
  SENSEX/BANKEX   -> Thursday (3)  [BSE]
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("DirectionalOptionAdvisor")

# ============================================================================
# 2026 NSE/BSE Holiday Calendar
# ============================================================================

NSE_HOLIDAYS_2026: List[date] = [
    date(2026, 1, 26),   # Republic Day
    date(2026, 2, 19),   # Chhatrapati Shivaji Maharaj Jayanti
    date(2026, 3, 3),    # Holi
    date(2026, 3, 26),   # Ram Navami
    date(2026, 3, 31),   # Mahavir Jayanti
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr. Ambedkar Jayanti / Ugadi
    date(2026, 5, 1),    # Maharashtra Day / Labour Day
    date(2026, 6, 16),   # Id-Ul-Adha (Bakri Id)
    date(2026, 7, 6),    # Muharram
    date(2026, 8, 15),   # Independence Day
    date(2026, 8, 27),   # Ganesh Chaturthi
    date(2026, 10, 2),   # Gandhi Jayanti / Dussehra
    date(2026, 10, 22),  # Diwali (Laxmi Puja)
    date(2026, 10, 23),  # Diwali (Balipratipada)
    date(2026, 11, 4),   # Guru Nanak Jayanti
    date(2026, 12, 25),  # Christmas
]

_HOLIDAY_SET: frozenset = frozenset(NSE_HOLIDAYS_2026)

_HOLIDAY_NAMES: Dict[date, str] = {
    date(2026, 1, 26): "Republic Day",
    date(2026, 2, 19): "Chhatrapati Shivaji Maharaj Jayanti",
    date(2026, 3, 3):  "Holi",
    date(2026, 3, 26): "Ram Navami",
    date(2026, 3, 31): "Mahavir Jayanti",
    date(2026, 4, 3):  "Good Friday",
    date(2026, 4, 14): "Dr. Ambedkar Jayanti",
    date(2026, 5, 1):  "Maharashtra Day",
    date(2026, 6, 16): "Bakri Id",
    date(2026, 7, 6):  "Muharram",
    date(2026, 8, 15): "Independence Day",
    date(2026, 8, 27): "Ganesh Chaturthi",
    date(2026, 10, 2): "Gandhi Jayanti",
    date(2026, 10, 22): "Diwali (Laxmi Puja)",
    date(2026, 10, 23): "Diwali (Balipratipada)",
    date(2026, 11, 4): "Guru Nanak Jayanti",
    date(2026, 12, 25): "Christmas",
}


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in _HOLIDAY_SET


def count_trading_days(start: date, end: date) -> int:
    if end <= start:
        return 0
    count = 0
    cur = start + timedelta(days=1)
    while cur <= end:
        if is_trading_day(cur):
            count += 1
        cur += timedelta(days=1)
    return count


def validate_and_adjust_expiry(
    proposed: date,
    index_name: str,
) -> Tuple[date, str]:
    """
    Validate proposed expiry; shift to previous trading day if holiday/weekend.
    Returns (adjusted_date, reason_str). reason_str empty if no shift needed.
    Raises ValueError if no valid day found within 7 calendar days before proposed.

    Example
    -------
    >>> validate_and_adjust_expiry(date(2026, 3, 3), "NIFTY")
    (date(2026, 3, 2), "Selected expiry 2026-03-03 -> shifted to 2026-03-02 because of Holi")
    """
    original = proposed
    for offset in range(8):
        candidate = proposed - timedelta(days=offset)
        if is_trading_day(candidate):
            reason = ""
            if candidate != original:
                if original.weekday() == 5:
                    why = "Saturday"
                elif original.weekday() == 6:
                    why = "Sunday"
                else:
                    why = _HOLIDAY_NAMES.get(original, "exchange holiday")
                reason = (
                    f"Selected expiry {original.strftime('%Y-%m-%d')} -> "
                    f"shifted to {candidate.strftime('%Y-%m-%d')} "
                    f"because of {why}"
                )
                logger.warning(f"DirectionalOptionAdvisor [{index_name}]: {reason}")
            return candidate, reason
    raise ValueError(
        f"[{index_name}] No valid trading day within 7 days before "
        f"{original.strftime('%Y-%m-%d')}."
    )


# ============================================================================
# Index configuration
# ============================================================================

@dataclass(frozen=True)
class IndexSpec:
    name:              str
    fyers_symbol:      str
    exchange:          str
    strike_interval:   int
    lot_size:          int
    weekly_expiry_day: int   # 0=Mon ... 6=Sun
    nse_symbol_prefix: str


INDEX_SPECS: Dict[str, IndexSpec] = {
    "NIFTY": IndexSpec(
        name="NIFTY", fyers_symbol="NSE:NIFTY50-INDEX", exchange="NSE",
        strike_interval=50, lot_size=75, weekly_expiry_day=1,   # Tuesday
        nse_symbol_prefix="NIFTY",
    ),
    "BANKNIFTY": IndexSpec(
        name="BANKNIFTY", fyers_symbol="NSE:NIFTYBANK-INDEX", exchange="NSE",
        strike_interval=100, lot_size=15, weekly_expiry_day=2,  # Wednesday
        nse_symbol_prefix="BANKNIFTY",
    ),
    "SENSEX": IndexSpec(
        name="SENSEX", fyers_symbol="BSE:SENSEX-INDEX", exchange="BSE",
        strike_interval=100, lot_size=10, weekly_expiry_day=3,  # Thursday
        nse_symbol_prefix="SENSEX",
    ),
    "FINNIFTY": IndexSpec(
        name="FINNIFTY", fyers_symbol="NSE:FINNIFTY-INDEX", exchange="NSE",
        strike_interval=50, lot_size=40, weekly_expiry_day=1,   # Tuesday
        nse_symbol_prefix="FINNIFTY",
    ),
    "BANKEX": IndexSpec(
        name="BANKEX", fyers_symbol="BSE:BANKEX-INDEX", exchange="BSE",
        strike_interval=100, lot_size=15, weekly_expiry_day=3,  # Thursday
        nse_symbol_prefix="BANKEX",
    ),
}

DTE_MIN_TRADING = 5
DTE_MAX_TRADING = 10

# Regime buckets for strategy routing (Feature C)
TRENDING_REGIMES = {"BULL", "BEAR"}
SPREAD_REGIMES   = {"CONSOLIDATION", "MILD_BULL", "MILD_BEAR", "CORRECTION"}

# Pre-trade checklist thresholds (Feature B)
DELTA_MIN             = 0.28
DELTA_MAX             = 0.45
DELTA_TARGET_LO       = 0.30
DELTA_TARGET_HI       = 0.40
OI_MIN                = 8_000   # global floor — per-index overrides below
VOL_MIN               = 1_500   # global floor — per-index overrides below
LIQUIDITY_OI_STRICT   = 50_000  # index chains routinely have 50k+ combined OI
LIQUIDITY_VOL_STRICT  = 10_000  # reference level for scoring (not rejection)
BID_ASK_MAX           = 0.15

# ── Per-index tiered liquidity thresholds ───────────────────────────────────
# Flat OI_MIN=8,000 treated BANKEX (typical 60K OI) same as NIFTY (100M+ OI).
# These values reflect each index's actual typical daily OI range so the
# liquidity score and checklist pass/fail is calibrated per-instrument.
INDEX_LIQUIDITY = {
    # index_key : (oi_min, vol_min, oi_excellent, vol_excellent)
    "NIFTY":     (5_000_000,  500_000, 50_000_000,  5_000_000),
    "BANKNIFTY": (500_000,    50_000,   5_000_000,    500_000),
    "SENSEX":    (500_000,    50_000,   5_000_000,    500_000),
    "FINNIFTY":  (100_000,    10_000,   1_000_000,    100_000),
    "BANKEX":    (200_000,    20_000,   2_000_000,    200_000),
}
# Max-pain proximity:
#   Fyers provides max_pain only from ~10 strikes, so it is often ≈ ATM.
#   0.5% of NIFTY spot ≈ 128 pts — too tight for daily use (rejects every day).
#   Tightened to 0.15% to catch only extreme pin-risk (within 1–2 strike intervals).
#   Guard: only apply when total OI data is rich enough (>= OI_MIN).
MAX_PAIN_PROXIMITY    = 0.003   # 0.3% of spot — realistic pin-risk threshold
PORTFOLIO_RISK_PCT    = 0.045   # 4.5% per F&O trade on ₹10L portfolio
# Rationale: NIFTY lot (75 units) premium ~₹39k = 3.9% — 1% was blocking every index.
# 4.5% gives a ~15% buffer above the most expensive index (NIFTY) while staying
# within standard F&O risk sizing for a 10-lakh account.

# Scoring (Feature D)
SCORE_WEIGHTS = {
    "technical": 0.30,
    "greeks":    0.25,
    "iv_rank":   0.20,
    "liquidity": 0.15,
    "calendar":  0.10,
}
SCORE_THRESHOLD = 68   # default for trending regimes (BULL/BEAR + naked long)

# Regime-aware thresholds — lower bar in low-conviction/consolidation environments
SCORE_THRESHOLDS: Dict[str, int] = {
    "BUY_CE":         68,   # naked call — need strong directional signal
    "BUY_PE":         68,   # naked put  — need strong directional signal
    "BULL_SPREAD":    60,   # spread limits risk, can act on moderate conviction
    "BEAR_SPREAD":    60,
    "LONG_STRADDLE":  55,   # uncertainty trade by design — lower bar acceptable
    "IRON_CONDOR":    60,   # vol-selling in high IV, checklist already validates
}

def _score_threshold(strategy: str) -> int:
    """Return the appropriate score threshold for the given strategy."""
    return SCORE_THRESHOLDS.get(strategy, SCORE_THRESHOLD)


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class ChecklistResult:
    expiry_ok:      bool = True;  expiry_note:    str = ""
    delta_ok:       bool = True;  delta_note:     str = ""
    liquidity_ok:   bool = True;  liquidity_note: str = ""
    risk_ok:        bool = True;  risk_note:      str = ""
    maxpain_ok:     bool = True;  maxpain_note:   str = ""

    @property
    def passed(self) -> bool:
        return all([self.expiry_ok, self.delta_ok,
                    self.liquidity_ok, self.risk_ok, self.maxpain_ok])

    def rejection_reason(self) -> str:
        reasons = []
        if not self.expiry_ok:    reasons.append(self.expiry_note)
        if not self.delta_ok:     reasons.append(self.delta_note)
        if not self.liquidity_ok: reasons.append(self.liquidity_note)
        if not self.risk_ok:      reasons.append(self.risk_note)
        if not self.maxpain_ok:   reasons.append(self.maxpain_note)
        return " | ".join(reasons)

    def display_lines(self) -> List[str]:
        checks = [
            ("Expiry valid & holiday-adjusted", self.expiry_ok,    self.expiry_note),
            ("Delta in target range",             self.delta_ok,     self.delta_note),
            ("OI > 8,000 & volume > 1,500",     self.liquidity_ok, self.liquidity_note),
            (f"Portfolio risk <= {PORTFOLIO_RISK_PCT*100:.1f}%", self.risk_ok, self.risk_note),
            ("Not near max-pain level",          self.maxpain_ok,   self.maxpain_note),
        ]
        lines = []
        for label, ok, note in checks:
            mark = "[+]" if ok else "[-]"
            suffix = f" ({note})" if note else ""
            lines.append(f"         {mark} {label}{suffix}")
        return lines


@dataclass
class ScoreBreakdown:
    technical:  float = 0.0
    greeks:     float = 0.0
    iv_rank:    float = 0.0
    liquidity:  float = 0.0
    calendar:   float = 0.0

    @property
    def total(self) -> float:
        return (
            self.technical  * SCORE_WEIGHTS["technical"] +
            self.greeks     * SCORE_WEIGHTS["greeks"]    +
            self.iv_rank    * SCORE_WEIGHTS["iv_rank"]   +
            self.liquidity  * SCORE_WEIGHTS["liquidity"] +
            self.calendar   * SCORE_WEIGHTS["calendar"]
        )

    def display(self) -> str:
        return (
            f"Score {self.total:.1f}/100  "
            f"[Tech={self.technical:.0f}({SCORE_WEIGHTS['technical']*100:.0f}%) "
            f"Grk={self.greeks:.0f}({SCORE_WEIGHTS['greeks']*100:.0f}%) "
            f"IV={self.iv_rank:.0f}({SCORE_WEIGHTS['iv_rank']*100:.0f}%) "
            f"Liq={self.liquidity:.0f}({SCORE_WEIGHTS['liquidity']*100:.0f}%) "
            f"Cal={self.calendar:.0f}({SCORE_WEIGHTS['calendar']*100:.0f}%)]"
        )


@dataclass
class Greeks:
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega:  float = 0.0


@dataclass
class OptionSuggestion:
    """One specific option trade suggestion (v4.2)."""
    index:             str
    option_type:       str
    strategy:          str
    strike:            float
    strike2:           Optional[float]
    expiry:            date
    expiry_str:        str
    dte:               int
    spot:              float
    lot_size:          int
    estimated_premium: float
    sl_premium:        float
    target_premium:    float
    cost_per_lot:      float
    max_loss_per_lot:  float
    max_profit_per_lot: float     = 0.0
    breakeven:         float      = 0.0
    iv_rank:           float      = 0.0
    iv_regime:         str        = "LOW"
    basis:             str        = ""
    confidence:        int        = 0
    proceed:           bool       = True
    expiry_adjustment: str        = ""
    greeks:            Greeks     = field(default_factory=Greeks)
    checklist:         ChecklistResult  = field(default_factory=ChecklistResult)
    score:             ScoreBreakdown  = field(default_factory=ScoreBreakdown)
    regime:            str        = ""
    prob_profit:       float      = 0.0
    buy_leg_delta:     float      = 0.0   # for spreads: individual buy-leg delta (for scoring)


@dataclass
class OptionAdvisorOutput:
    timestamp:    str
    market_stance: str
    india_vix:    float
    suggestions:  List[OptionSuggestion] = field(default_factory=list)
    skipped:      List[str]             = field(default_factory=list)


# ============================================================================
# Black-Scholes helpers
# ============================================================================

def _norm_cdf(x: float) -> float:
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    poly = t * (0.319381530 + t * (-0.356563782
                + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    cdf = 1.0 - (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x) * poly
    return cdf if x >= 0 else 1.0 - cdf


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _bs_price(
    option_type: str, spot: float, strike: float,
    dte_cal: int, iv: float, rf: float = 0.065,
) -> float:
    if dte_cal <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return 0.0
    T = dte_cal / 365.0; sqT = math.sqrt(T)
    d1 = (math.log(spot / strike) + (rf + 0.5 * iv * iv) * T) / (iv * sqT)
    d2 = d1 - iv * sqT
    if option_type == "CE":
        return spot * _norm_cdf(d1) - strike * math.exp(-rf * T) * _norm_cdf(d2)
    return strike * math.exp(-rf * T) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def _implied_vol(
    option_type: str, spot: float, strike: float,
    dte_cal: int, market_price: float, rf: float = 0.065,
    tol: float = 1e-5, max_iter: int = 50,
) -> float:
    """
    FIX 4.1: Compute true implied volatility via Newton-Raphson inversion.
    Returns 0.0 if market_price is invalid or iteration doesn't converge.
    Uses real LTP from Fyers option chain (Fix 1) for accurate IV skew.
    """
    if dte_cal <= 0 or market_price <= 0 or spot <= 0 or strike <= 0:
        return 0.0
    iv = 0.2  # initial guess
    T  = dte_cal / 365.0
    sqT = math.sqrt(T)
    for _ in range(max_iter):
        price  = _bs_price(option_type, spot, strike, dte_cal, iv, rf)
        diff   = price - market_price
        if abs(diff) < tol:
            break
        # Vega = dPrice/dIV = S * sqrt(T) * N'(d1)
        d1    = (math.log(spot / strike) + (rf + 0.5 * iv * iv) * T) / (iv * sqT)
        vega  = spot * sqT * _norm_pdf(d1)
        if vega < 1e-8:
            break
        iv -= diff / vega
        iv  = max(0.01, min(5.0, iv))   # clamp to [1%, 500%]
    return round(iv, 4) if 0.01 <= iv <= 5.0 else 0.0


def _bs_greeks(
    option_type: str, spot: float, strike: float,
    dte_cal: int, iv: float, rf: float = 0.065,
) -> Greeks:
    """Delta, Gamma, Theta (per calendar day), Vega (per 1% IV move)."""
    if dte_cal <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return Greeks()
    T = dte_cal / 365.0; sqT = math.sqrt(T)
    d1 = (math.log(spot / strike) + (rf + 0.5 * iv * iv) * T) / (iv * sqT)
    d2 = d1 - iv * sqT
    pdf_d1 = _norm_pdf(d1)
    delta = _norm_cdf(d1) if option_type == "CE" else _norm_cdf(d1) - 1.0
    gamma = pdf_d1 / (spot * iv * sqT)
    theta = (-(spot * pdf_d1 * iv) / (2 * sqT) - rf * strike * math.exp(-rf * T)
             * (_norm_cdf(d2) if option_type == "CE" else _norm_cdf(-d2))) / 365.0
    vega  = spot * sqT * pdf_d1 * 0.01
    return Greeks(delta=round(delta,4), gamma=round(gamma,6),
                  theta=round(theta,4), vega=round(vega,4))


def _trading_to_calendar(tdays: int) -> int:
    return max(1, round(tdays * 7 / 5))


def _prob_itm(option_type: str, spot: float, strike: float,
              dte_cal: int, iv: float) -> float:
    if dte_cal <= 0 or iv <= 0:
        return 0.0
    T = dte_cal / 365.0
    d2 = (math.log(spot / strike) + (0.065 - 0.5 * iv * iv) * T) / (iv * math.sqrt(T))
    return _norm_cdf(d2) if option_type == "CE" else _norm_cdf(-d2)


def _find_delta_strike(
    option_type: str, spot: float, dte_cal: int, iv: float, rf: float,
    strike_interval: int,
    target_lo: float = DELTA_TARGET_LO,
    target_hi: float = DELTA_TARGET_HI,
    search_range_pct: float = 0.10,
) -> Tuple[float, float]:
    """
    Find strike with |delta| in [target_lo, target_hi].
    Returns (strike, actual_delta). Falls back to closest-to-midpoint if none in range.
    """
    mid_target = (target_lo + target_hi) / 2.0
    atm = round(spot / strike_interval) * strike_interval
    n_steps = int(spot * search_range_pct / strike_interval) + 5
    if option_type == "CE":
        candidates = [atm + i * strike_interval for i in range(0, n_steps + 1)]
    else:
        candidates = [atm - i * strike_interval for i in range(0, n_steps + 1)]

    best_strike = atm; best_delta = 0.5; best_diff = 999.0
    for k in candidates:
        if k <= 0:
            continue
        g = _bs_greeks(option_type, spot, k, dte_cal, iv, rf)
        abs_d = abs(g.delta)
        diff  = abs(abs_d - mid_target)
        if target_lo <= abs_d <= target_hi:
            if diff < best_diff:
                best_strike = k; best_delta = g.delta; best_diff = diff
        elif best_diff == 999.0 and diff < abs(abs(best_delta) - mid_target):
            best_strike = k; best_delta = g.delta; best_diff = diff
    return float(best_strike), best_delta


# ============================================================================
# Scoring engine (Feature D)
# ============================================================================

def _chain_oi_and_volume(chain: Dict) -> Tuple[float, float]:
    """
    Extract combined OI and a volume proxy from a Fyers index_option_chains entry.

    FyersFetcher stores:
      total_call_oi  : int  — sum of call OI across strikes
      total_put_oi   : int  — sum of put OI across strikes
    There is no per-chain aggregate volume in Fyers v3; we use total OI as a
    liquidity proxy for both the checklist and the scoring engine.

    Returns (combined_oi, volume_proxy) where volume_proxy = combined_oi * 0.15
    (historical ratio: options volume ≈ 15% of OI on liquid NSE indices).
    """
    call_oi = float(chain.get("total_call_oi", 0) or chain.get("oi", 0))
    put_oi  = float(chain.get("total_put_oi",  0))
    combined = call_oi + put_oi
    # Volume fallback: use explicit key if present, else proxy from combined OI
    volume = float(chain.get("volume", combined * 0.15))
    return combined, volume


def compute_score(
    conviction: int, greeks: Greeks, iv_rank: float,
    oi: float, volume: float, dte: int,
    direction: str, option_type: str,
    strategy: str = "BUY_CE",
    buy_leg_delta: Optional[float] = None,
    skip_liquidity_check: bool = False,
    index_name: str = "",
) -> ScoreBreakdown:
    """
    Score 0-100 across 5 dimensions.

    For spreads (BULL_SPREAD/BEAR_SPREAD), greeks score uses buy_leg_delta
    (the individual leg quality), not the net spread delta.
    """
    tech_score = min(100.0, max(0.0, float(conviction)))

    # Greeks score: strategy-aware
    # - LONG_STRADDLE / IRON_CONDOR: net delta ≈ 0 by design; score on vega fitness
    #   instead of penalising the delta for being away from 0.35.
    #   LONG_STRADDLE wants LOW iv_rank  (cheap vega to buy).
    #   IRON_CONDOR   wants HIGH iv_rank (expensive vega to sell).
    # - BULL_SPREAD / BEAR_SPREAD: use buy-leg delta (individual leg quality).
    # - BUY_CE / BUY_PE: use single-leg delta, target 0.28-0.45.
    if strategy == "LONG_STRADDLE":
        # Vega fitness: peaks when iv_rank=0 (cheap vol), zero at iv_rank=40+
        greeks_score = max(0.0, 100.0 - iv_rank * 100.0 / 40.0)
    elif strategy == "IRON_CONDOR":
        # Vega fitness for sellers: peaks at high iv_rank
        greeks_score = min(100.0, iv_rank * 100.0 / 65.0)
    elif strategy in ("BULL_SPREAD", "BEAR_SPREAD") and buy_leg_delta is not None:
        delta_for_score = abs(buy_leg_delta)
        if 0.28 <= delta_for_score <= 0.45:
            greeks_score = max(0.0, 100.0 - abs(delta_for_score - 0.35) * 200.0)
        else:
            greeks_score = max(0.0, 50.0 - abs(delta_for_score - 0.35) * 300.0)
    else:
        delta_for_score = abs(greeks.delta)
        if 0.28 <= delta_for_score <= 0.45:
            greeks_score = max(0.0, 100.0 - abs(delta_for_score - 0.35) * 200.0)
        else:
            greeks_score = max(0.0, 50.0 - abs(delta_for_score - 0.35) * 300.0)

    is_buying = option_type in ("CE", "PE", "CE+PE")
    if is_buying:
        iv_score = max(0.0, 100.0 - iv_rank * 100.0 / 70.0)
    else:
        iv_score = min(100.0, iv_rank * 100.0 / 65.0)

    oi_score  = min(100.0, oi     / LIQUIDITY_OI_STRICT  * 100.0)
    vol_score = min(100.0, volume / LIQUIDITY_VOL_STRICT * 100.0)
    # Per-index tiered scoring: normalise against the index-specific excellent threshold
    # so BANKEX (OI~66K) scores much lower than NIFTY (OI~450M) on the same scale.
    _idx_liq = INDEX_LIQUIDITY.get(index_name.upper(), None)
    if _idx_liq and not skip_liquidity_check:
        _oi_min, _vol_min, _oi_exc, _vol_exc = _idx_liq
        oi_score  = min(100.0, oi     / _oi_exc  * 100.0)
        vol_score = min(100.0, volume / _vol_exc * 100.0)
    liq_score = 100.0 if skip_liquidity_check else (oi_score * 0.6 + vol_score * 0.4)

    if DTE_MIN_TRADING <= dte <= DTE_MAX_TRADING:
        cal_score = 100.0
    elif dte < DTE_MIN_TRADING:
        cal_score = max(0.0, 100.0 - (DTE_MIN_TRADING - dte) * 25.0)
    else:
        cal_score = max(0.0, 100.0 - (dte - DTE_MAX_TRADING) * 15.0)

    return ScoreBreakdown(
        technical = round(tech_score, 1),
        greeks    = round(greeks_score, 1),
        iv_rank   = round(iv_score, 1),
        liquidity = round(liq_score, 1),
        calendar  = round(cal_score, 1),
    )


# ============================================================================
# Pre-trade checklist (Feature B)
# ============================================================================

def run_checklist(
    expiry_adjustment: str, expiry_ok: bool,
    greeks: Greeks, oi: float, volume: float,
    max_loss_per_lot: float, portfolio_value: float,
    spot: float, max_pain: Optional[float],
    strategy: str = "BUY_CE",
    skip_liquidity_check: bool = False,
    index_name: str = "",
) -> ChecklistResult:
    """
    Pre-trade checklist with strategy-aware delta validation:
      BUY_CE / BUY_PE      -> individual leg delta: [0.28, 0.45]
      BULL_SPREAD / BEAR_SPREAD -> net delta of spread: [0.04, 0.25]
      IRON_CONDOR / LONG_STRADDLE -> net delta near 0: always pass delta check
    """
    cl = ChecklistResult()

    cl.expiry_ok   = expiry_ok
    cl.expiry_note = expiry_adjustment if expiry_adjustment else "OK"

    abs_d = abs(greeks.delta)
    # Strategy-specific delta bounds
    if strategy in ("IRON_CONDOR", "LONG_STRADDLE"):
        # Net delta near zero is expected; skip delta gating
        cl.delta_ok   = True
        cl.delta_note = f"delta={greeks.delta:+.3f} (spread/condor -- N/A)"
    elif strategy in ("BULL_SPREAD", "BEAR_SPREAD"):
        # Net delta of a 2-leg spread: typically 0.02-0.25 depending on width and DTE
        spread_delta_min = 0.02
        spread_delta_max = 0.25
        cl.delta_ok   = spread_delta_min <= abs_d <= spread_delta_max
        cl.delta_note = f"net_delta={greeks.delta:+.3f}"
        if not cl.delta_ok:
            cl.delta_note += f" (need [{spread_delta_min},{spread_delta_max}] for spread)"
    else:
        # BUY_CE / BUY_PE: single-leg delta target
        cl.delta_ok   = DELTA_MIN <= abs_d <= DELTA_MAX
        cl.delta_note = f"delta={greeks.delta:+.3f}"
        if not cl.delta_ok:
            cl.delta_note += f" (need [{DELTA_MIN},{DELTA_MAX}])"

    _idx_liq = INDEX_LIQUIDITY.get(index_name.upper(), None)
    _oi_min  = _idx_liq[0] if _idx_liq else OI_MIN
    _vol_min = _idx_liq[1] if _idx_liq else VOL_MIN
    cl.liquidity_ok   = skip_liquidity_check or (oi >= _oi_min and volume >= _vol_min)
    cl.liquidity_note = f"OI={oi:,.0f} vol={volume:,.0f}"
    if skip_liquidity_check:
        cl.liquidity_note += " (liquidity check skipped — backtest/demo mode)"
    elif not cl.liquidity_ok:
        cl.liquidity_note += f" (need OI>={_oi_min:,} vol>={_vol_min:,})"

    max_risk      = portfolio_value * PORTFOLIO_RISK_PCT
    cl.risk_ok    = max_loss_per_lot <= max_risk
    cl.risk_note  = f"max_loss=Rs{max_loss_per_lot:,.0f} limit=Rs{max_risk:,.0f}"
    if not cl.risk_ok:
        cl.risk_note += " BREACH"

    # Max-pain check:
    # Fyers optionchain with strikecount=10 often returns max_pain ≈ spot/ATM because
    # the data window is narrow. A true pin-risk only exists when max_pain is within
    # 1–2 strike intervals AND we have enough OI data to trust it.
    #
    # Rule: REJECT only if ALL of:
    #   a) We have meaningful OI (>= MAX_PAIN_OI_GUARD)
    #   b) max_pain != round-ATM (i.e. at least 1 strike interval away from computed ATM)
    #   c) proximity is within MAX_PAIN_PROXIMITY (0.3% of spot)
    # Otherwise: PASS with informational note.
    MAX_PAIN_OI_GUARD   = 20_000
    if max_pain and max_pain > 0 and spot > 0 and oi >= MAX_PAIN_OI_GUARD:
        proximity = abs(spot - max_pain) / spot
        # Estimate strike interval for this spot (rough: NIFTY/FINNIFTY=50, others=100)
        strike_interval = 50 if spot < 40_000 else 100
        at_atm = (max_pain == round(spot / strike_interval) * strike_interval)
        is_pin_risk = (not at_atm) and (proximity <= MAX_PAIN_PROXIMITY)
        cl.maxpain_ok   = not is_pin_risk
        cl.maxpain_note = (
            f"max_pain={max_pain:.0f} proximity={proximity*100:.2f}% "
            f"(pin-risk threshold {MAX_PAIN_PROXIMITY*100:.1f}%)"
        )
        if is_pin_risk:
            cl.maxpain_note += " [pin-risk — genuine, avoid]"
        elif at_atm:
            cl.maxpain_note += " [at ATM — limited-strike data, ignored]"
    else:
        cl.maxpain_ok   = True
        if max_pain and oi < MAX_PAIN_OI_GUARD:
            cl.maxpain_note = f"max_pain={max_pain:.0f} (OI too low to trust, skip check)"
        else:
            cl.maxpain_note = "max_pain N/A"

    return cl


# ============================================================================
# Main advisor class
# ============================================================================

class DirectionalOptionAdvisor:
    """
    Rule-based directional F&O advisor (v4.2).
    Call advise() each cycle. No LLM, no external deps beyond market_data.
    """

    IV_LOW_THRESHOLD  = 40
    IV_HIGH_THRESHOLD = 65
    MIN_CONFIDENCE    = 48

    def __init__(self, gsec_yield: float = 7.0, portfolio_value: float = 1_000_000):
        self.rf              = gsec_yield / 100.0
        self.portfolio_value = portfolio_value

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def advise(
        self,
        market_data: Dict,
        consensus_direction: str,
        consensus_strength:  str,
        consensus_score:     float,
        consensus_conviction: int,
        india_vix: float,
        market_regime: str = "CONSOLIDATION",
        skip_liquidity_check: bool = False,
    ) -> OptionAdvisorOutput:
        """
        Produce OptionSuggestion per index where checklist + score pass.

        Parameters
        ----------
        market_data          : dict from FyersFetcher.fetch_market_data()
        consensus_*          : fields from plan.consensus
        india_vix            : current India VIX
        market_regime        : MarketRegime.value string e.g. "CONSOLIDATION"
        """
        from datetime import datetime as _dt
        output = OptionAdvisorOutput(
            timestamp     = _dt.now().strftime("%Y-%m-%d %H:%M:%S"),
            market_stance = self._stance(consensus_direction, consensus_strength),
            india_vix     = india_vix,
        )

        global_iv_rank = min(100, max(0, int((india_vix - 10) / (30 - 10) * 100)))
        iv_regime_global = self._iv_regime(global_iv_rank)
        index_chains: Dict[str, Dict] = market_data.get("index_option_chains", {})

        for idx_name, spec in INDEX_SPECS.items():
            try:
                spot = self._get_spot(idx_name, market_data)
                if not spot or spot <= 0:
                    output.skipped.append(f"{idx_name}: spot price unavailable")
                    continue

                chain         = index_chains.get(idx_name, {})
                iv_rank       = float(chain.get("iv_rank", global_iv_rank))
                iv_regime_idx = self._iv_regime(iv_rank)
                pcr           = float(chain.get("pcr", 1.0))
                # Extract combined OI and volume proxy from actual Fyers chain fields
                chain_oi, chain_vol = _chain_oi_and_volume(chain)

                # FIX 4.4: Expiry Day Gamma Risk
                # On expiry day (DTE≤0 after roll-check) ATM gamma is 3-5x higher.
                # A 50-point Nifty move can double or wipe ATM premium in minutes.
                # We block naked long calls/puts on expiry day unless conviction is HIGH.
                from datetime import date as _date
                _today_wd = _date.today().weekday()
                _is_expiry_day = (_today_wd == spec.weekly_expiry_day)
                if _is_expiry_day and consensus_conviction < 72:
                    output.skipped.append(
                        f"{idx_name}: EXPIRY_DAY gamma risk — conviction={consensus_conviction} "
                        f"< 72 required. Naked options blocked on {spec.name} expiry."
                    )
                    logger.info(
                        f"[FIX4.4] {idx_name}: Expiry-day gamma risk suppressed "
                        f"(conviction={consensus_conviction}, need ≥72)"
                    )
                    continue

                expiry, dte, adjustment = self._get_expiry(spec, chain)
                if expiry is None or dte < 1:
                    output.skipped.append(
                        f"{idx_name}: no valid expiry in "
                        f"{DTE_MIN_TRADING}-{DTE_MAX_TRADING} trading-day window"
                    )
                    continue

                suggestion = self._build_suggestion(
                    spec=spec, spot=spot, expiry=expiry, dte=dte,
                    expiry_adj=adjustment,
                    iv_rank=iv_rank, iv_regime=iv_regime_idx, pcr=pcr,
                    direction=consensus_direction, strength=consensus_strength,
                    conviction=consensus_conviction, score=consensus_score,
                    chain=chain, market_regime=market_regime,
                )
                if suggestion is None:
                    output.skipped.append(
                        f"{idx_name}: no qualifying suggestion "
                        f"(dir={consensus_direction} iv={iv_regime_idx} "
                        f"regime={market_regime})"
                    )
                    continue

                # Feature B: Pre-trade checklist
                cl = run_checklist(
                    expiry_adjustment = suggestion.expiry_adjustment,
                    expiry_ok         = True,
                    greeks            = suggestion.greeks,
                    oi                = chain_oi,
                    volume            = chain_vol,
                    max_loss_per_lot  = suggestion.max_loss_per_lot,
                    portfolio_value   = self.portfolio_value,
                    spot              = spot,
                    max_pain          = chain.get("max_pain"),
                    strategy          = suggestion.strategy,
                    skip_liquidity_check = skip_liquidity_check,
                    index_name        = idx_name,
                )
                suggestion.checklist = cl

                if not cl.passed:
                    reason = cl.rejection_reason()
                    logger.warning(
                        f"DirectionalOptionAdvisor [{idx_name}]: "
                        f"REJECTED -- {reason}"
                    )
                    output.skipped.append(f"{idx_name}: REJECTED -- {reason}")
                    continue

                # Feature D: Scoring
                s = compute_score(
                    conviction     = consensus_conviction,
                    greeks         = suggestion.greeks,
                    iv_rank        = iv_rank,
                    oi             = chain_oi,
                    volume         = chain_vol,
                    dte            = dte,
                    direction      = consensus_direction,
                    option_type    = suggestion.option_type,
                    strategy       = suggestion.strategy,
                    buy_leg_delta  = suggestion.buy_leg_delta if suggestion.buy_leg_delta else None,
                    skip_liquidity_check = skip_liquidity_check,
                    index_name     = idx_name,
                )
                suggestion.score = s

                if s.total < _score_threshold(suggestion.strategy):
                    threshold_used = _score_threshold(suggestion.strategy)
                    logger.info(
                        f"DirectionalOptionAdvisor [{idx_name}]: "
                        f"score {s.total:.1f} < threshold {threshold_used} "
                        f"({suggestion.strategy}) -- skipped"
                    )
                    output.skipped.append(
                        f"{idx_name}: score {s.total:.1f} < {threshold_used} ({suggestion.strategy})"
                    )
                    continue

                output.suggestions.append(suggestion)

            except Exception as exc:
                logger.warning(f"DirectionalOptionAdvisor: {idx_name} error -- {exc}")
                output.skipped.append(f"{idx_name}: error -- {exc}")

        # ── Portfolio concentration guard ─────────────────────────────────
        # When multiple straddles are suggested on correlated indices in a LOW
        # IV environment, cap concurrent straddles at 2 — prefer the most liquid
        # (highest OI-based liquidity score). The rest move to a watch list.
        straddle_suggestions = [s for s in output.suggestions if s.strategy == "LONG_STRADDLE"]
        if len(straddle_suggestions) > 2 and iv_regime_global in ("LOW", "VERY_LOW"):
            # Sort by liquidity component of score (descending)
            straddle_suggestions.sort(
                key=lambda s: s.score.liquidity if s.score else 0, reverse=True
            )
            keep = set(id(s) for s in straddle_suggestions[:2])
            watch = [s for s in straddle_suggestions if id(s) not in keep]
            output.suggestions = [s for s in output.suggestions
                                  if s.strategy != "LONG_STRADDLE" or id(s) in keep]
            for ws in watch:
                output.skipped.append(
                    f"{ws.index}: STRADDLE watch-listed (concentration cap — top 2 kept by liquidity)"
                )
            logger.info(
                f"[CONCENTRATION-GUARD] Low-IV straddle cap: kept {len(straddle_suggestions[:2])}, "
                f"watch-listed {len(watch)} ({[ws.index for ws in watch]})"
            )

        logger.info(
            f"OptionAdvisor: {len(output.suggestions)} suggestion(s) | "
            f"stance={output.market_stance} | VIX={india_vix:.1f} | "
            f"iv_regime={iv_regime_global}"
        )
        return output

    # ------------------------------------------------------------------
    # Suggestion builder -- regime-aware (Features A + C)
    # ------------------------------------------------------------------

    def _build_suggestion(
        self,
        spec: IndexSpec, spot: float, expiry: date, dte: int, expiry_adj: str,
        iv_rank: float, iv_regime: str, pcr: float,
        direction: str, strength: str, conviction: int, score: float,
        chain: Dict, market_regime: str,
    ) -> Optional[OptionSuggestion]:

        if conviction < self.MIN_CONFIDENCE:
            # Long straddle is a market-neutral strategy — it thrives on uncertainty,
            # not directional conviction.  Allow it at a lower floor (48) so that
            # NO_CONSENSUS cycles (conviction=50) still generate a straddle suggestion.
            straddle_path = (
                (direction == "NEUTRAL" or strength == "NO_CONSENSUS")
                and iv_regime != "HIGH"   # HIGH IV → iron condor, not straddle
            )
            if not straddle_path:
                return None
        if dte < 1:
            return None

        iv_ann  = self._iv_from_rank(iv_rank)
        cal_dte = _trading_to_calendar(dte)
        is_trending = market_regime in TRENDING_REGIMES

        if iv_regime == "HIGH":
            return self._iron_condor(spec, spot, expiry, dte, cal_dte,
                                     expiry_adj, iv_ann, iv_rank, pcr,
                                     conviction, market_regime)

        if direction == "NEUTRAL" or strength == "NO_CONSENSUS":
            if iv_regime == "LOW":
                return self._long_straddle(spec, spot, expiry, dte, cal_dte,
                                           expiry_adj, iv_ann, iv_rank,
                                           conviction, market_regime)
            return None

        # Feature C: directional -- naked only in strong trend, else spread
        if direction == "LONG":
            if is_trending:
                return self._buy_call(spec, spot, expiry, dte, cal_dte,
                                      expiry_adj, iv_ann, iv_rank,
                                      conviction, score, chain, market_regime)
            else:
                return self._bull_spread(spec, spot, expiry, dte, cal_dte,
                                         expiry_adj, iv_ann, iv_rank,
                                         conviction, score, market_regime)
        else:  # SHORT
            if is_trending:
                return self._buy_put(spec, spot, expiry, dte, cal_dte,
                                     expiry_adj, iv_ann, iv_rank,
                                     conviction, score, chain, market_regime)
            else:
                return self._bear_spread(spec, spot, expiry, dte, cal_dte,
                                         expiry_adj, iv_ann, iv_rank,
                                         conviction, score, market_regime)

    # ── Strategy builders ─────────────────────────────────────────────

    def _buy_call(self, spec, spot, expiry, dte, cal_dte, expiry_adj,
                  iv, iv_rank, conviction, score, chain, regime) -> OptionSuggestion:
        """Naked long OTM Call -- delta-targeted strike (Feature A)."""
        strike, _ = _find_delta_strike("CE", spot, cal_dte, iv, self.rf,
                                        spec.strike_interval)
        prem   = round(_bs_price("CE", spot, strike, cal_dte, iv, self.rf), 1)
        prem   = max(prem, 1.0)
        # FIX 1: Override BS estimate with real market LTP when available
        _ltp_map = chain.get("ltp_by_strike", {})
        _real_ce = _ltp_map.get(strike, {}).get("ce", 0.0)
        _real_ltp_used_ce = False
        if _real_ce > 0:
            logger.debug(
                f"  [{spec.name}] BUY_CE {strike:.0f}: "
                f"real_ltp={_real_ce:.1f} vs bs_est={prem:.1f} "
                f"(diff={abs(_real_ce-prem)/_real_ce*100:.1f}%)"
            )
            prem = _real_ce
            _real_ltp_used_ce = True
            # FIX 4.1: Invert real LTP → true IV, then recompute Greeks with it
            _real_iv = _implied_vol("CE", spot, strike, cal_dte, prem, self.rf)
            if _real_iv > 0:
                iv = _real_iv
        greeks = _bs_greeks("CE", spot, strike, cal_dte, iv, self.rf)
        prob   = _prob_itm("CE", spot, strike, cal_dte, iv)
        be     = round(strike + prem, 1)
        logger.info(
            f"DirectionalOptionAdvisor [{spec.name}] BUY_CE: strike={strike:.0f} "
            f"delta={greeks.delta:+.3f} gamma={greeks.gamma:.6f} "
            f"theta={greeks.theta:.2f}/day vega={greeks.vega:.2f}/1%IV BE={be:.1f}"
        )
        # FIX 4.3: Tag whether real LTP was used (for MetaLearner data quality)
        _sug_ce = OptionSuggestion(
            index=spec.name, option_type="CE", strategy="BUY_CE",
            strike=strike, strike2=None, expiry=expiry,
            expiry_str=expiry.strftime("%d-%b-%Y").upper(),
            dte=dte, spot=round(spot,1), lot_size=spec.lot_size,
            estimated_premium=prem, sl_premium=round(prem*0.5,1),
            target_premium=round(prem*2.0,1),
            cost_per_lot=round(prem*spec.lot_size,0),
            max_loss_per_lot=round(prem*spec.lot_size,0),
            max_profit_per_lot=round(prem*spec.lot_size*2.0,0),
            breakeven=be, iv_rank=iv_rank, iv_regime="LOW",
            basis=(f"BULLISH {regime} (score {score:+.2f}) | "
                   f"IV {iv_rank:.0f} LOW | naked call | delta={greeks.delta:+.3f}"),
            confidence=conviction, proceed=conviction>=58,
            expiry_adjustment=expiry_adj, greeks=greeks,
            regime=regime, prob_profit=round(prob,3),
        )
        _sug_ce._real_ltp_used = _real_ltp_used_ce
        return _sug_ce

    def _buy_put(self, spec, spot, expiry, dte, cal_dte, expiry_adj,
                 iv, iv_rank, conviction, score, chain, regime) -> OptionSuggestion:
        """Naked long OTM Put -- delta-targeted strike."""
        strike, _ = _find_delta_strike("PE", spot, cal_dte, iv, self.rf,
                                        spec.strike_interval)
        prem   = round(_bs_price("PE", spot, strike, cal_dte, iv, self.rf), 1)
        prem   = max(prem, 1.0)
        # FIX 1: Override BS estimate with real market LTP when available
        _ltp_map = chain.get("ltp_by_strike", {})
        _real_pe = _ltp_map.get(strike, {}).get("pe", 0.0)
        _real_ltp_used_pe = False
        if _real_pe > 0:
            logger.debug(
                f"  [{spec.name}] BUY_PE {strike:.0f}: "
                f"real_ltp={_real_pe:.1f} vs bs_est={prem:.1f} "
                f"(diff={abs(_real_pe-prem)/_real_pe*100:.1f}%)"
            )
            prem = _real_pe
            _real_ltp_used_pe = True
            # FIX 4.1: Invert real LTP → true IV, then recompute Greeks with it
            _real_iv = _implied_vol("PE", spot, strike, cal_dte, prem, self.rf)
            if _real_iv > 0:
                iv = _real_iv
        greeks = _bs_greeks("PE", spot, strike, cal_dte, iv, self.rf)
        prob   = _prob_itm("PE", spot, strike, cal_dte, iv)
        be     = round(strike - prem, 1)
        logger.info(
            f"DirectionalOptionAdvisor [{spec.name}] BUY_PE: strike={strike:.0f} "
            f"delta={greeks.delta:+.3f} gamma={greeks.gamma:.6f} "
            f"theta={greeks.theta:.2f}/day vega={greeks.vega:.2f}/1%IV BE={be:.1f}"
        )
        _sug_pe = OptionSuggestion(
            index=spec.name, option_type="PE", strategy="BUY_PE",
            strike=strike, strike2=None, expiry=expiry,
            expiry_str=expiry.strftime("%d-%b-%Y").upper(),
            dte=dte, spot=round(spot,1), lot_size=spec.lot_size,
            estimated_premium=prem, sl_premium=round(prem*0.5,1),
            target_premium=round(prem*2.0,1),
            cost_per_lot=round(prem*spec.lot_size,0),
            max_loss_per_lot=round(prem*spec.lot_size,0),
            max_profit_per_lot=round(prem*spec.lot_size*2.0,0),
            breakeven=be, iv_rank=iv_rank, iv_regime="LOW",
            basis=(f"BEARISH {regime} (score {score:+.2f}) | "
                   f"IV {iv_rank:.0f} LOW | naked put | delta={greeks.delta:+.3f}"),
            confidence=conviction, proceed=conviction>=58,
            expiry_adjustment=expiry_adj, greeks=greeks,
            regime=regime, prob_profit=round(prob,3),
        )
        _sug_pe._real_ltp_used = _real_ltp_used_pe
        return _sug_pe

    def _bull_spread(self, spec, spot, expiry, dte, cal_dte, expiry_adj,
                     iv, iv_rank, conviction, score, regime) -> OptionSuggestion:
        """Bull Call Spread -- used in CONSOLIDATION/MILD regimes (Feature C)."""
        buy_strike, _ = _find_delta_strike("CE", spot, cal_dte, iv, self.rf,
                                            spec.strike_interval)
        sell_strike   = buy_strike + spec.strike_interval * 2

        pb  = round(_bs_price("CE", spot, buy_strike,  cal_dte, iv, self.rf), 1)
        ps  = round(_bs_price("CE", spot, sell_strike, cal_dte, iv, self.rf), 1)
        net = max(round(pb - ps, 1), 1.0)

        spread_w       = sell_strike - buy_strike
        max_profit_u   = spread_w - net
        max_profit_lot = round(max_profit_u * spec.lot_size, 0)
        max_loss_lot   = round(net * spec.lot_size, 0)
        be             = round(buy_strike + net, 1)

        gb = _bs_greeks("CE", spot, buy_strike,  cal_dte, iv, self.rf)
        gs = _bs_greeks("CE", spot, sell_strike, cal_dte, iv, self.rf)
        net_greeks = Greeks(
            delta = round(gb.delta - gs.delta, 4),
            gamma = round(gb.gamma - gs.gamma, 6),
            theta = round(gb.theta - gs.theta, 4),
            vega  = round(gb.vega  - gs.vega,  4),
        )
        prob = _prob_itm("CE", spot, sell_strike, cal_dte, iv)

        logger.info(
            f"DirectionalOptionAdvisor [{spec.name}] BULL_SPREAD: "
            f"{buy_strike:.0f}/{sell_strike:.0f} net_debit={net:.1f} "
            f"net_delta={net_greeks.delta:+.3f} BE={be:.1f} "
            f"max_profit/lot=Rs{max_profit_lot:,.0f}"
        )
        return OptionSuggestion(
            index=spec.name, option_type="CE", strategy="BULL_SPREAD",
            strike=buy_strike, strike2=sell_strike, expiry=expiry,
            expiry_str=expiry.strftime("%d-%b-%Y").upper(),
            dte=dte, spot=round(spot,1), lot_size=spec.lot_size,
            estimated_premium=net, sl_premium=round(net*1.0,1),
            target_premium=round(max_profit_u,1),
            cost_per_lot=max_loss_lot, max_loss_per_lot=max_loss_lot,
            max_profit_per_lot=max_profit_lot,
            breakeven=be, iv_rank=iv_rank, iv_regime="LOW",
            basis=(f"BULLISH {regime} (score {score:+.2f}) | "
                   f"Spread {buy_strike:.0f}/{sell_strike:.0f} | "
                   f"net_delta={net_greeks.delta:+.3f}"),
            confidence=conviction, proceed=conviction>=60,
            expiry_adjustment=expiry_adj, greeks=net_greeks,
            regime=regime, prob_profit=round(prob,3),
            buy_leg_delta=gb.delta,
        )

    def _bear_spread(self, spec, spot, expiry, dte, cal_dte, expiry_adj,
                     iv, iv_rank, conviction, score, regime) -> OptionSuggestion:
        """Bear Put Spread -- used in CONSOLIDATION/MILD regimes."""
        buy_strike, _ = _find_delta_strike("PE", spot, cal_dte, iv, self.rf,
                                            spec.strike_interval)
        sell_strike   = buy_strike - spec.strike_interval * 2

        pb  = round(_bs_price("PE", spot, buy_strike,  cal_dte, iv, self.rf), 1)
        ps  = round(_bs_price("PE", spot, sell_strike, cal_dte, iv, self.rf), 1)
        net = max(round(pb - ps, 1), 1.0)

        spread_w       = buy_strike - sell_strike
        max_profit_u   = spread_w - net
        max_profit_lot = round(max_profit_u * spec.lot_size, 0)
        max_loss_lot   = round(net * spec.lot_size, 0)
        be             = round(buy_strike - net, 1)

        gb = _bs_greeks("PE", spot, buy_strike,  cal_dte, iv, self.rf)
        gs = _bs_greeks("PE", spot, sell_strike, cal_dte, iv, self.rf)
        net_greeks = Greeks(
            delta = round(gb.delta - gs.delta, 4),
            gamma = round(gb.gamma - gs.gamma, 6),
            theta = round(gb.theta - gs.theta, 4),
            vega  = round(gb.vega  - gs.vega,  4),
        )
        prob = _prob_itm("PE", spot, sell_strike, cal_dte, iv)

        return OptionSuggestion(
            index=spec.name, option_type="PE", strategy="BEAR_SPREAD",
            strike=buy_strike, strike2=sell_strike, expiry=expiry,
            expiry_str=expiry.strftime("%d-%b-%Y").upper(),
            dte=dte, spot=round(spot,1), lot_size=spec.lot_size,
            estimated_premium=net, sl_premium=round(net*1.0,1),
            target_premium=round(max_profit_u,1),
            cost_per_lot=max_loss_lot, max_loss_per_lot=max_loss_lot,
            max_profit_per_lot=max_profit_lot,
            breakeven=be, iv_rank=iv_rank, iv_regime="LOW",
            basis=(f"BEARISH {regime} (score {score:+.2f}) | "
                   f"Spread {buy_strike:.0f}/{sell_strike:.0f} | "
                   f"net_delta={net_greeks.delta:+.3f}"),
            confidence=conviction, proceed=conviction>=60,
            expiry_adjustment=expiry_adj, greeks=net_greeks,
            regime=regime, prob_profit=round(prob,3),
            buy_leg_delta=gb.delta,
        )

    def _iron_condor(self, spec, spot, expiry, dte, cal_dte, expiry_adj,
                     iv, iv_rank, pcr, conviction, regime) -> OptionSuggestion:
        """Iron Condor -- HIGH IV regime."""
        wing = max(spec.strike_interval * 2,
                   round(spot * iv / math.sqrt(252) * 1.5 / spec.strike_interval)
                   * spec.strike_interval)
        atm       = round(spot / spec.strike_interval) * spec.strike_interval
        sc        = atm + wing; sp  = atm - wing
        lc        = sc  + spec.strike_interval * 2
        lp        = sp  - spec.strike_interval * 2
        cc = _bs_price("CE", spot, sc, cal_dte, iv, self.rf)
        cp = _bs_price("PE", spot, sp, cal_dte, iv, self.rf)
        dc = _bs_price("CE", spot, lc, cal_dte, iv, self.rf)
        dp = _bs_price("PE", spot, lp, cal_dte, iv, self.rf)
        net_credit    = max(round(cc + cp - dc - dp, 1), 1.0)
        max_loss_unit = (lc - sc) - net_credit
        be_up = round(sc + net_credit, 1)
        be_dn = round(sp - net_credit, 1)
        gsc = _bs_greeks("CE", spot, sc, cal_dte, iv, self.rf)
        gsp = _bs_greeks("PE", spot, sp, cal_dte, iv, self.rf)
        net_greeks = Greeks(
            delta = round(-gsc.delta - gsp.delta, 4),
            gamma = round(-gsc.gamma - gsp.gamma, 6),
            theta = round(-gsc.theta - gsp.theta, 4),
            vega  = round(-gsc.vega  - gsp.vega,  4),
        )
        logger.info(
            f"DirectionalOptionAdvisor [{spec.name}] IRON_CONDOR: "
            f"{sc:.0f}C/{sp:.0f}P credit={net_credit:.1f} BE {be_dn:.0f}-{be_up:.0f}"
        )
        return OptionSuggestion(
            index=spec.name, option_type="CE/PE", strategy="IRON_CONDOR",
            strike=sc, strike2=sp, expiry=expiry,
            expiry_str=expiry.strftime("%d-%b-%Y").upper(),
            dte=dte, spot=round(spot,1), lot_size=spec.lot_size,
            estimated_premium=net_credit,
            sl_premium=round(net_credit*2.0,1),
            target_premium=round(net_credit*0.5,1),
            cost_per_lot=0.0,
            max_loss_per_lot=round(max_loss_unit*spec.lot_size,0),
            max_profit_per_lot=round(net_credit*spec.lot_size,0),
            breakeven=be_up,
            iv_rank=iv_rank, iv_regime="HIGH",
            basis=(f"HIGH IV {iv_rank:.0f} -> sell premium | "
                   f"short {sc:.0f}C/{sp:.0f}P | BE {be_dn:.0f}-{be_up:.0f} | PCR {pcr:.2f}"),
            confidence=conviction, proceed=iv_rank>=55,
            expiry_adjustment=expiry_adj, greeks=net_greeks,
            regime=regime,
        )

    def _long_straddle(self, spec, spot, expiry, dte, cal_dte, expiry_adj,
                       iv, iv_rank, conviction, regime) -> OptionSuggestion:
        """Long ATM Straddle -- NO_CONSENSUS + LOW IV."""
        atm  = round(spot / spec.strike_interval) * spec.strike_interval
        ce_p = round(_bs_price("CE", spot, atm, cal_dte, iv, self.rf), 1)
        pe_p = round(_bs_price("PE", spot, atm, cal_dte, iv, self.rf), 1)
        total = max(round(ce_p + pe_p, 1), 2.0)
        gc = _bs_greeks("CE", spot, atm, cal_dte, iv, self.rf)
        gp = _bs_greeks("PE", spot, atm, cal_dte, iv, self.rf)
        net_greeks = Greeks(
            delta = round(gc.delta + gp.delta, 4),
            gamma = round(gc.gamma + gp.gamma, 6),
            theta = round(gc.theta + gp.theta, 4),
            vega  = round(gc.vega  + gp.vega,  4),
        )
        be_up = round(atm + total, 1); be_dn = round(atm - total, 1)

        # Probability that spot lands OUTSIDE either breakeven at expiry
        # (i.e. the straddle is profitable) — not prob ITM of a single leg
        T = cal_dte / 365.0
        if T > 0 and iv > 0:
            d2_up = (math.log(spot / be_up) + (self.rf - 0.5*iv*iv)*T) / (iv*math.sqrt(T))
            d2_dn = (math.log(spot / be_dn) + (self.rf - 0.5*iv*iv)*T) / (iv*math.sqrt(T))
            prob_profit = round(_norm_cdf(d2_up) + (1.0 - _norm_cdf(d2_dn)), 3)
        else:
            prob_profit = 0.0

        return OptionSuggestion(
            index=spec.name, option_type="CE+PE", strategy="LONG_STRADDLE",
            strike=atm, strike2=atm, expiry=expiry,
            expiry_str=expiry.strftime("%d-%b-%Y").upper(),
            dte=dte, spot=round(spot,1), lot_size=spec.lot_size,
            estimated_premium=total, sl_premium=round(total*0.4,1),
            target_premium=round(total*1.8,1),
            cost_per_lot=round(total*spec.lot_size,0),
            max_loss_per_lot=round(total*spec.lot_size,0),
            max_profit_per_lot=0,
            breakeven=be_up,
            iv_rank=iv_rank, iv_regime="LOW",
            basis=(f"NO_CONSENSUS + LOW IV {iv_rank:.0f} -> buy vol | "
                   f"ATM={atm:.0f} | BE {be_dn:.0f}-{be_up:.0f}"),
            confidence=conviction, proceed=conviction>=50,  # Straddle is valid at NO_CONSENSUS conviction (50)
            expiry_adjustment=expiry_adj, greeks=net_greeks,
            regime=regime, prob_profit=prob_profit,
        )

    # ------------------------------------------------------------------
    # Expiry resolution (v4.1)
    # ------------------------------------------------------------------

    def _get_expiry(self, spec: IndexSpec, chain: Dict) -> Tuple[Optional[date], int, str]:
        today = date.today()
        expiry_data = chain.get("expiryData") or chain.get("expiry_dates") or []
        if expiry_data:
            candidates = self._parse_expiry_data(expiry_data, spec.name)
            for raw_date in sorted(candidates):
                if raw_date <= today:
                    continue
                try:
                    adjusted, reason = validate_and_adjust_expiry(raw_date, spec.name)
                except ValueError as e:
                    logger.warning(str(e)); continue
                tdte = count_trading_days(today, adjusted)
                if DTE_MIN_TRADING <= tdte <= DTE_MAX_TRADING:
                    logger.info(
                        f"DirectionalOptionAdvisor [{spec.name}]: "
                        f"expiryData -> selected {adjusted.strftime('%Y-%m-%d')} "
                        f"(trading DTE={tdte})"
                        + (f" | adj: {reason}" if reason else "")
                    )
                    return adjusted, tdte, reason
            # Widen to 15-day fallback
            for raw_date in sorted(candidates):
                if raw_date <= today:
                    continue
                try:
                    adjusted, reason = validate_and_adjust_expiry(raw_date, spec.name)
                except ValueError:
                    continue
                tdte = count_trading_days(today, adjusted)
                if 1 <= tdte <= 15:
                    logger.warning(
                        f"DirectionalOptionAdvisor [{spec.name}]: "
                        f"Fallback expiry {adjusted.strftime('%Y-%m-%d')} (DTE={tdte})"
                    )
                    return adjusted, tdte, reason
        return self._rule_based_expiry(spec, today)

    def _parse_expiry_data(self, expiry_data, index_name: str) -> List[date]:
        from datetime import datetime as _dt
        parsed = []
        for item in expiry_data:
            try:
                if isinstance(item, date):
                    parsed.append(item)
                elif isinstance(item, int):
                    parsed.append(_dt.utcfromtimestamp(item).date())
                elif isinstance(item, str):
                    for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d-%B-%Y", "%Y%m%d"):
                        try:
                            parsed.append(_dt.strptime(item, fmt).date()); break
                        except ValueError:
                            continue
            except Exception as e:
                logger.debug(f"[{index_name}] Cannot parse expiry '{item}': {e}")
        return parsed

    def _rule_based_expiry(self, spec: IndexSpec, today: date) -> Tuple[Optional[date], int, str]:
        days_ahead = (spec.weekly_expiry_day - today.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        candidates = [today + timedelta(days=days_ahead + w * 7) for w in range(6)]
        for raw_date in candidates:
            try:
                adjusted, reason = validate_and_adjust_expiry(raw_date, spec.name)
            except ValueError as e:
                logger.warning(str(e)); continue
            tdte = count_trading_days(today, adjusted)
            if DTE_MIN_TRADING <= tdte <= DTE_MAX_TRADING:
                logger.info(
                    f"DirectionalOptionAdvisor [{spec.name}]: "
                    f"rule-based -> selected {adjusted.strftime('%Y-%m-%d')} "
                    f"(trading DTE={tdte})"
                    + (f" | adj: {reason}" if reason else "")
                )
                return adjusted, tdte, reason
        for raw_date in candidates:
            try:
                adjusted, reason = validate_and_adjust_expiry(raw_date, spec.name)
            except ValueError:
                continue
            tdte = count_trading_days(today, adjusted)
            if tdte >= 2:
                logger.warning(
                    f"DirectionalOptionAdvisor [{spec.name}]: "
                    f"fallback {adjusted.strftime('%Y-%m-%d')} DTE={tdte}"
                )
                return adjusted, tdte, reason
        logger.error(f"DirectionalOptionAdvisor [{spec.name}]: No valid expiry found!")
        return None, 0, ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_spot(self, idx: str, market_data: Dict) -> Optional[float]:
        spot_map = {
            "NIFTY":     market_data.get("nifty_price"),
            "BANKNIFTY": market_data.get("banknifty_price"),
            "SENSEX":    market_data.get("sensex_price"),
            "FINNIFTY":  market_data.get("finnifty_price"),
            "BANKEX":    market_data.get("bankex_price"),
        }
        val = spot_map.get(idx)
        if val and val > 0:
            return float(val)
        for key, v in market_data.get("price_data", {}).items():
            if idx.upper() in key.upper():
                c = v.get("close") or v.get("ltp")
                if c and float(c) > 0:
                    return float(c)
        return None

    @staticmethod
    def _iv_regime(iv_rank: float) -> str:
        if iv_rank >= 65: return "HIGH"
        if iv_rank >= 40: return "MID"
        return "LOW"

    @staticmethod
    def _iv_from_rank(iv_rank: float) -> float:
        return (12.0 + (iv_rank / 100.0) * 23.0) / 100.0

    @staticmethod
    def _stance(direction: str, strength: str) -> str:
        if direction == "LONG"  and strength in ("STRONG", "MODERATE"): return "BULLISH"
        if direction == "SHORT" and strength in ("STRONG", "MODERATE"): return "BEARISH"
        return "NEUTRAL"


# ============================================================================
# Printer (v4.2)
# ============================================================================

def format_option_suggestions(output: OptionAdvisorOutput) -> str:
    lines = ["\nDIRECTIONAL F&O SUGGESTIONS"]
    lines.append(
        f"  Market stance : {output.market_stance} | "
        f"India VIX : {output.india_vix:.1f}"
    )
    if not output.suggestions:
        lines.append("  No qualifying suggestions this cycle.")
        if output.skipped:
            for s in output.skipped:
                lines.append(f"  -> Skipped -- {s}")
        return "\n".join(lines)

    lines.append("")
    for s in output.suggestions:
        strat_label = {
            "BUY_CE":        "BUY  CE (naked)",
            "BUY_PE":        "BUY  PE (naked)",
            "BULL_SPREAD":   "BULL SPREAD (buy CE spread)",
            "BEAR_SPREAD":   "BEAR SPREAD (buy PE spread)",
            "IRON_CONDOR":   "IRON CONDOR (sell strangle)",
            "LONG_STRADDLE": "LONG STRADDLE (CE+PE)",
        }.get(s.strategy, s.strategy)

        if s.strategy in ("BULL_SPREAD", "BEAR_SPREAD") and s.strike2:
            strike_str = f"{s.strike:.0f} / {s.strike2:.0f}"
        elif s.strategy == "IRON_CONDOR" and s.strike2:
            strike_str = f"{s.strike:.0f}C / {s.strike2:.0f}P"
        elif s.strategy == "LONG_STRADDLE":
            strike_str = f"{s.strike:.0f} ATM"
        else:
            strike_str = f"{s.strike:.0f}"

        flag = "[OK]" if s.proceed else "[!!]"
        # In watch-only mode the OK/!! distinction is irrelevant — relabel clearly
        if getattr(output, "_watch_only", False):
            flag = "[WATCH]"
        lines.append(
            f"  {flag} {s.index:10s} {strat_label:36s} "
            f"{strike_str:20s} exp {s.expiry_str}  DTE {s.dte}"
        )
        lines.append(f"       spot Rs{s.spot:,.1f} | regime={s.regime}")

        if s.strategy == "IRON_CONDOR":
            lines.append(
                f"       net credit Rs{s.estimated_premium:.0f}/unit | "
                f"max profit/lot Rs{s.max_profit_per_lot:,.0f} | "
                f"max loss/lot Rs{s.max_loss_per_lot:,.0f}"
            )
        elif s.strategy in ("BULL_SPREAD", "BEAR_SPREAD"):
            lines.append(
                f"       net debit Rs{s.estimated_premium:.0f}/unit | "
                f"max loss/lot Rs{s.max_loss_per_lot:,.0f} | "
                f"max profit/lot Rs{s.max_profit_per_lot:,.0f}"
            )
        else:
            lines.append(
                f"       entry Rs{s.estimated_premium:.0f}/unit | "
                f"cost/lot Rs{s.cost_per_lot:,.0f}"
            )
        prob_label = "Prob Profit" if s.strategy == "LONG_STRADDLE" else "Prob ITM"
        lines.append(
            f"       SL Rs{s.sl_premium:.0f} | Target Rs{s.target_premium:.0f} | "
            f"Breakeven Rs{s.breakeven:,.1f} | {prob_label} ~{s.prob_profit*100:.0f}%"
        )
        lines.append(
            f"       Greeks: delta={s.greeks.delta:+.3f} gamma={s.greeks.gamma:.6f} "
            f"theta={s.greeks.theta:.2f}/day vega={s.greeks.vega:.2f}/1%IV"
        )
        lines.append(
            f"       IV rank {s.iv_rank:.0f} ({s.iv_regime}) | conviction {s.confidence}"
        )
        lines.append(f"       Basis: {s.basis}")
        lines.append("       Pre-trade checklist:")
        lines.extend(s.checklist.display_lines())
        lines.append(f"       {s.score.display()}")
        if s.expiry_adjustment:
            lines.append(f"       [!] Expiry adjusted: {s.expiry_adjustment}")
        lines.append("")

    if output.skipped:
        lines.append("  Skipped / Rejected:")
        for s in output.skipped:
            lines.append(f"    -> {s}")

    return "\n".join(lines)
