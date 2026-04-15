"""
ROX Proven Edge Engine v3.0 - OPTIMUS Agent
===========================================
Futures & Options Weekly Expiry Analysis Agent.

Generates directional signals (CALL/PUT) for weekly expiry options on
indices (Nifty, BankNifty) and F&O stocks by analysing options-chain
metrics, IV skew, PCR trends, futures basis, and price-action context
supplied by the other agents.

Baseline weight: 0.15 (15%)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .base_agent import AgentReport, AgentVerdict, BaseAgent

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ConvictionLevel, MarketRegime, TradeDirection


# ---------------------------------------------------------------------------
# Domain enums & dataclasses
# ---------------------------------------------------------------------------

class OptionType(str, Enum):
    CE = "CE"   # Call
    PE = "PE"   # Put


class OptionsStrategy(str, Enum):
    LONG_CALL  = "LONG_CALL"
    LONG_PUT   = "LONG_PUT"
    SHORT_CALL = "SHORT_CALL"
    SHORT_PUT  = "SHORT_PUT"
    NEUTRAL    = "NEUTRAL"


@dataclass
class OptionsSignal:
    """
    Complete options trade signal for a single weekly expiry.
    """
    symbol: str                        # e.g. NIFTY, BANKNIFTY, RELIANCE
    expiry_date: str                   # ISO format yyyy-mm-dd
    option_type: OptionType            # CE / PE
    strategy: OptionsStrategy          # LONG_CALL / LONG_PUT / etc.
    strike: float                      # Recommended strike price
    strike_label: str                  # "ATM", "1 OTM", "2 OTM", etc.
    entry_low: float                   # Entry price range lower bound
    entry_high: float                  # Entry price range upper bound
    stop_loss: float                   # Option price stop-loss
    target_1: float                    # First target (option price)
    target_2: float                    # Second target (option price)
    conviction: int                    # 0-100
    rationale: str                     # Brief explanation
    iv_context: str                    # e.g. "IV rank 35 – moderate"
    pcr_context: str                   # e.g. "PCR 0.85 – bearish bias"
    futures_basis: str                 # e.g. "Premium ₹42 – bullish"
    risk_per_lot: float                # Approx max loss per lot (₹)
    suggested_lots: int = 1


@dataclass
class WeeklyOptionsData:
    """Parsed options-chain context for OPTIMUS analysis."""
    symbol: str = "NIFTY"
    current_price: float = 0.0
    weekly_expiry: str = ""            # ISO date of nearest weekly expiry

    # PCR metrics
    pcr: float = 1.0
    pcr_trend: str = "stable"         # "rising" | "falling" | "stable"

    # Pain / OI walls
    max_pain: float = 0.0
    call_oi_max_strike: float = 0.0   # Highest CE OI strike (resistance)
    put_oi_max_strike: float = 0.0    # Highest PE OI strike (support)

    # OI change signals
    ce_oi_change_pct: float = 0.0     # % change in total CE OI
    pe_oi_change_pct: float = 0.0     # % change in total PE OI

    # Volatility
    india_vix: float = 15.0
    iv_rank: float = 50.0             # 0-100
    iv_skew: float = 0.0              # put IV – call IV (positive = put skew)

    # Futures
    futures_premium: float = 0.0      # positive = contango, negative = backwardation
    oi_signal: str = "NEUTRAL"        # LONG_BUILDUP | SHORT_BUILDUP | etc.

    # Price action
    price_change_pct: float = 0.0
    support_level: float = 0.0
    resistance_level: float = 0.0

    # Strike spacing (lot-size independent)
    strike_gap: float = 50.0          # e.g. 50 for Nifty, 100 for BankNifty


# ---------------------------------------------------------------------------
# OPTIMUS Agent
# ---------------------------------------------------------------------------

class OptimusAgent(BaseAgent):
    """
    OPTIMUS – Futures & Options Weekly Expiry Agent

    Analyses weekly options-chain data to produce directional CALL/PUT
    signals.  Works best alongside ORION (price action) and SENTINEL
    (OI structure) which feed context via the shared data dict.

    Baseline weight: 15%
    """

    # PCR interpretation thresholds (same as SENTINEL for consistency)
    PCR_STRONG_BULLISH  = 1.30   # Contrarian: extreme fear → rally likely
    PCR_MILD_BULLISH    = 1.10
    PCR_NEUTRAL_HI      = 1.10
    PCR_NEUTRAL_LO      = 0.80
    PCR_MILD_BEARISH    = 0.80
    PCR_STRONG_BEARISH  = 0.60   # Contrarian: complacency → fall likely

    # IV rank boundaries
    IV_LOW   = 30   # Prefer buying options
    IV_HIGH  = 70   # Prefer selling options

    # Futures basis threshold
    BASIS_SIGNIFICANT = 20  # ₹ points – meaningful premium/discount

    def __init__(self) -> None:
        super().__init__(
            name="OPTIMUS",
            domain="F&O Weekly Expiry Analysis",
            baseline_weight=0.15,
        )

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def analyze(self, data: Dict[str, Any], regime: MarketRegime) -> AgentReport:
        """
        Perform F&O weekly expiry analysis.

        Expected keys in *data* (all optional – graceful defaults apply):
            pcr                 : float   – Put-Call Ratio
            pcr_trend           : str     – "rising" | "falling" | "stable"
            max_pain            : float   – Max-pain strike
            current_price       : float   – Underlying spot price
            india_vix           : float   – India VIX
            iv_rank             : float   – IV Rank 0-100
            iv_skew             : float   – Put IV minus Call IV
            call_oi_walls       : list    – [{strike, oi}]
            put_oi_walls        : list    – [{strike, oi}]
            ce_oi_change_pct    : float   – % change in call OI
            pe_oi_change_pct    : float   – % change in put OI
            futures_premium     : float   – Futures premium/discount
            oi_signal           : str     – OI direction signal
            price_change_pct    : float   – Today's % price move
            support_level       : float   – Key support from ORION
            resistance_level    : float   – Key resistance from ORION
            symbol              : str     – Underlying symbol
            weekly_expiry       : str     – Nearest weekly expiry (ISO date)
            strike_gap          : float   – Strike interval

        Returns:
            AgentReport with verdict and options-signal details.
        """
        opts = self._parse_options_data(data)

        # --- Core analysis sub-routines ---
        pcr_bias, pcr_score, pcr_ctx = self._analyse_pcr(opts)
        oi_bias, oi_score, oi_ctx    = self._analyse_oi_structure(opts)
        iv_bias, iv_score, iv_ctx    = self._analyse_iv(opts)
        basis_bias, basis_score, basis_ctx = self._analyse_futures_basis(opts)
        pa_bias, pa_score, pa_ctx    = self._analyse_price_action(opts, regime)

        # --- Weighted conviction ---
        # Weights: PCR 25%, OI walls 25%, IV 15%, Futures 15%, Price action 20%
        WEIGHTS = (0.25, 0.25, 0.15, 0.15, 0.20)
        scores  = (pcr_score, oi_score, iv_score, basis_score, pa_score)
        net_score = sum(w * s for w, s in zip(WEIGHTS, scores))  # -100 to +100

        direction, conviction, strategy = self._decide_direction(net_score, opts)

        # --- Build options signal ---
        signal = self._build_signal(opts, strategy, conviction)

        # --- Compile rationale ---
        observations = [pcr_ctx, oi_ctx, iv_ctx, basis_ctx, pa_ctx]
        rationale = f"Net score {net_score:+.1f} → {strategy.value}. " \
                    + " | ".join(observations)
        signal.rationale = rationale

        # --- Build AgentReport ---
        verdict = AgentVerdict(
            direction=direction,
            conviction=conviction,
            weight=self.baseline_weight,
            reason=rationale,
            risks=self._assess_risks(opts, strategy),
        )

        report = AgentReport(
            agent_name=self.name,
            verdict=verdict,
            analysis_details={
                "options_signal": self._signal_to_dict(signal),
                "net_score": net_score,
                "component_scores": {
                    "pcr": pcr_score,
                    "oi_structure": oi_score,
                    "iv": iv_score,
                    "futures_basis": basis_score,
                    "price_action": pa_score,
                },
            },
            key_observations=observations,
            metrics={
                "pcr": opts.pcr,
                "india_vix": opts.india_vix,
                "iv_rank": opts.iv_rank,
                "futures_premium": opts.futures_premium,
                "net_conviction_score": net_score,
            },
            raw_data={
                "symbol": opts.symbol,
                "weekly_expiry": opts.weekly_expiry,
                "strategy": strategy.value,
                "strike": signal.strike,
            },
        )
        return report

    # ------------------------------------------------------------------
    # Enhancement 4: Tree-of-Thoughts for ambiguous F&O signals
    # ------------------------------------------------------------------

    def analyze_with_tree_of_thoughts(self, data: Dict[str, Any],
                                      regime: MarketRegime) -> AgentReport:
        """
        Tree-of-Thoughts for OPTIMUS: branches over PCR/OI interpretation.
        Activates when initial conviction is outside the 40-60 comfort zone.
        """
        initial_report = self.analyze(data, regime)
        if 40 <= initial_report.verdict.conviction <= 60:
            return initial_report

        branches = [
            {
                "name": "gamma_squeeze_up",
                "description": "Options market forces short covering – rally expected",
                "probability": 0.40,
                "pcr_override": 1.5,
                "vix_override": -2,
            },
            {
                "name": "put_buying_panic",
                "description": "Protective put buying signals institutional exit",
                "probability": 0.35,
                "pcr_override": 0.6,
                "vix_override": 3,
            },
            {
                "name": "theta_decay_neutral",
                "description": "Near expiry theta decay dominates; range-bound likely",
                "probability": 0.25,
                "pcr_override": None,
                "vix_override": 0,
            },
        ]

        branch_scores = []
        for branch in branches:
            bd = dict(data)
            if branch["pcr_override"] is not None:
                bd["pcr"] = branch["pcr_override"]
            bd["india_vix"] = bd.get("india_vix", 15) + branch["vix_override"]
            branch_report = self.analyze(bd, regime)
            branch_scores.append({
                "hypothesis": branch["name"],
                "description": branch["description"],
                "conviction": branch_report.verdict.conviction,
                "direction": branch_report.verdict.direction,
                "probability": branch["probability"],
            })

        weighted_conviction = sum(b["conviction"] * b["probability"] for b in branch_scores)
        initial_report.verdict.conviction = round(weighted_conviction, 1)
        initial_report.verdict.reason += (
            f" | [ToT: {len(branches)} F&O branches, "
            f"weighted conviction={weighted_conviction:.0f}]"
        )
        initial_report.analysis_details["tree_branches"] = branch_scores
        initial_report.verdict.__post_init__()
        return initial_report

    # ------------------------------------------------------------------
    # Data parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_strike(w) -> float:
        """Return strike from a wall entry that is either a plain number or a dict."""
        if isinstance(w, dict):
            return float(w.get("strike", 0))
        return float(w)

    def _parse_options_data(self, data: Dict[str, Any]) -> WeeklyOptionsData:
        """Extract and normalise options data from the shared data dict."""
        call_walls = data.get("call_oi_walls", [])
        put_walls  = data.get("put_oi_walls", [])

        call_oi_max_strike = max(
            (self._extract_strike(w) for w in call_walls), default=0.0
        ) if call_walls else 0.0

        put_oi_max_strike = min(
            (self._extract_strike(w) for w in put_walls), default=0.0
        ) if put_walls else 0.0

        # Nearest weekly expiry: default to next Thursday
        weekly_expiry = data.get("weekly_expiry") or self._next_thursday()

        symbol = data.get("symbol", "NIFTY")
        strike_gap = 100.0 if "BANK" in symbol.upper() else 50.0

        return WeeklyOptionsData(
            symbol=symbol,
            current_price=float(data.get("current_price", 0)),
            weekly_expiry=weekly_expiry,
            pcr=float(data.get("pcr", 1.0)),
            pcr_trend=str(data.get("pcr_trend", "stable")),
            max_pain=float(data.get("max_pain", 0)),
            call_oi_max_strike=float(call_oi_max_strike),
            put_oi_max_strike=float(put_oi_max_strike),
            ce_oi_change_pct=float(data.get("ce_oi_change_pct", 0)),
            pe_oi_change_pct=float(data.get("pe_oi_change_pct", 0)),
            india_vix=float(data.get("india_vix", 15)),
            iv_rank=float(data.get("iv_rank", 50)),
            iv_skew=float(data.get("iv_skew", 0)),
            futures_premium=float(data.get("futures_premium", 0)),
            oi_signal=str(data.get("oi_signal", "NEUTRAL")),
            price_change_pct=float(data.get("price_change", data.get("price_change_pct", 0))),
            support_level=float(data.get("support_level", 0)),
            resistance_level=float(data.get("resistance_level", 0)),
            strike_gap=strike_gap,
        )

    # ------------------------------------------------------------------
    # Sub-analysis methods
    # ------------------------------------------------------------------

    def _analyse_pcr(self, opts: WeeklyOptionsData) -> Tuple[str, float, str]:
        """
        Interpret Put-Call Ratio.

        Returns (bias, score, context_string).
        Score is –100 (bearish) to +100 (bullish).
        """
        pcr = opts.pcr
        trend = opts.pcr_trend

        if pcr >= self.PCR_STRONG_BULLISH:
            # Extreme put buying → contrarian bullish
            score, bias = 70, "bullish"
            ctx = f"PCR {pcr:.2f} (extreme) – contrarian BULLISH signal"
        elif pcr >= self.PCR_MILD_BULLISH:
            score, bias = 40, "bullish"
            ctx = f"PCR {pcr:.2f} – moderate bullish tilt"
        elif pcr <= self.PCR_STRONG_BEARISH:
            # Extreme call buying → contrarian bearish
            score, bias = -70, "bearish"
            ctx = f"PCR {pcr:.2f} (extreme low) – contrarian BEARISH signal"
        elif pcr <= self.PCR_MILD_BEARISH:
            score, bias = -40, "bearish"
            ctx = f"PCR {pcr:.2f} – moderate bearish tilt"
        else:
            score, bias = 0, "neutral"
            ctx = f"PCR {pcr:.2f} – neutral zone"

        # Trend adjustments
        if trend == "rising" and score >= 0:
            score = min(100, score + 10)
            ctx += "; PCR rising (increased hedging)"
        elif trend == "falling" and score <= 0:
            score = max(-100, score - 10)
            ctx += "; PCR falling (put unwinding)"

        return bias, float(score), ctx

    def _analyse_oi_structure(self, opts: WeeklyOptionsData) -> Tuple[str, float, str]:
        """
        Evaluate OI walls and OI-change data.
        Score: –100 to +100.
        """
        score = 0.0
        ctx_parts = []

        price = opts.current_price or 1  # avoid div-by-zero

        # Distance from OI walls
        if opts.call_oi_max_strike and opts.put_oi_max_strike:
            call_dist = (opts.call_oi_max_strike - price) / price * 100
            put_dist  = (price - opts.put_oi_max_strike)  / price * 100

            if call_dist < put_dist:
                # Price closer to call wall → ceiling pressure
                score -= 25
                ctx_parts.append(
                    f"Call wall {opts.call_oi_max_strike:.0f} is near "
                    f"({call_dist:.1f}% above) – resistance"
                )
            elif put_dist < call_dist:
                score += 25
                ctx_parts.append(
                    f"Put wall {opts.put_oi_max_strike:.0f} is near "
                    f"({put_dist:.1f}% below) – support"
                )

        # OI change signals
        ce_chg = opts.ce_oi_change_pct
        pe_chg = opts.pe_oi_change_pct
        if ce_chg > 5 and pe_chg < 0:
            score -= 30
            ctx_parts.append(f"CE OI +{ce_chg:.1f}%, PE OI {pe_chg:.1f}% – bearish build")
        elif pe_chg > 5 and ce_chg < 0:
            score += 30
            ctx_parts.append(f"PE OI +{pe_chg:.1f}%, CE OI {ce_chg:.1f}% – bullish build")

        # Existing OI signal from SENTINEL
        oi_sig = opts.oi_signal.upper()
        if oi_sig == "LONG_BUILDUP":
            score += 20
        elif oi_sig == "SHORT_BUILDUP":
            score -= 20
        elif oi_sig == "SHORT_COVERING":
            score += 15
        elif oi_sig == "LONG_UNWINDING":
            score -= 15

        if oi_sig != "NEUTRAL":
            ctx_parts.append(f"OI signal: {oi_sig}")

        bias = "bullish" if score > 0 else ("bearish" if score < 0 else "neutral")
        ctx = " | ".join(ctx_parts) if ctx_parts else "OI structure neutral"
        return bias, float(score), ctx

    def _analyse_iv(self, opts: WeeklyOptionsData) -> Tuple[str, float, str]:
        """
        IV rank and skew analysis.
        Score: –100 to +100 (high IV favours sellers → directionally neutral,
        low IV favours buyers with conviction).
        """
        iv_rank = opts.iv_rank
        skew    = opts.iv_skew

        # IV rank: low → buying options is cheap
        if iv_rank <= self.IV_LOW:
            iv_score = 30  # biased toward buying
            ctx = f"IV rank {iv_rank:.0f} – LOW, favour option buyers"
        elif iv_rank >= self.IV_HIGH:
            iv_score = -20  # selling more attractive; lower net directional score
            ctx = f"IV rank {iv_rank:.0f} – HIGH, favour option sellers/spreads"
        else:
            iv_score = 0
            ctx = f"IV rank {iv_rank:.0f} – moderate"

        # Skew: positive = put IV > call IV → hedging demand → bearish
        if skew > 2:
            iv_score -= 15
            ctx += f"; put skew {skew:.1f} – elevated hedge demand"
        elif skew < -2:
            iv_score += 15
            ctx += f"; call skew {abs(skew):.1f} – call buying"

        ctx += f" (VIX {opts.india_vix:.1f})"
        bias = "bullish" if iv_score > 0 else ("bearish" if iv_score < 0 else "neutral")
        return bias, float(iv_score), ctx

    def _analyse_futures_basis(self, opts: WeeklyOptionsData) -> Tuple[str, float, str]:
        """
        Futures premium/discount as proxy for institutional directional bias.
        Score: –100 to +100.
        """
        basis = opts.futures_premium
        threshold = self.BASIS_SIGNIFICANT

        if basis > threshold * 2:
            score, bias = 40, "bullish"
            ctx = f"Futures basis +₹{basis:.0f} – strong contango, bullish"
        elif basis > threshold:
            score, bias = 20, "bullish"
            ctx = f"Futures basis +₹{basis:.0f} – mild contango"
        elif basis < -threshold * 2:
            score, bias = -40, "bearish"
            ctx = f"Futures basis –₹{abs(basis):.0f} – strong backwardation, bearish"
        elif basis < -threshold:
            score, bias = -20, "bearish"
            ctx = f"Futures basis –₹{abs(basis):.0f} – mild backwardation"
        else:
            score, bias = 0, "neutral"
            ctx = f"Futures basis ₹{basis:+.0f} – near fair value"

        return bias, float(score), ctx

    def _analyse_price_action(
        self, opts: WeeklyOptionsData, regime: MarketRegime
    ) -> Tuple[str, float, str]:
        """
        Combine price-change momentum and market regime.
        Score: –100 to +100.
        """
        score = 0.0
        ctx_parts = []

        pct = opts.price_change_pct
        if pct > 1.0:
            score += 30
            ctx_parts.append(f"Spot up {pct:.2f}% – bullish momentum")
        elif pct < -1.0:
            score -= 30
            ctx_parts.append(f"Spot down {abs(pct):.2f}% – bearish momentum")

        # Max-pain gravity
        if opts.max_pain and opts.current_price:
            to_pain = opts.max_pain - opts.current_price
            if abs(to_pain) < 100:
                ctx_parts.append(
                    f"Near max pain {opts.max_pain:.0f} – range-bound risk"
                )
                score *= 0.7  # dampen directional signal

        # Regime
        regime_adj = {
            MarketRegime.BULL:         20,
            MarketRegime.MILD_BULL:    10,
            MarketRegime.CONSOLIDATION: 0,
            MarketRegime.CORRECTION:  -10,
            MarketRegime.MILD_BEAR:   -10,
            MarketRegime.BEAR:        -20,
        }
        adj = regime_adj.get(regime, 0)
        score += adj
        if adj:
            ctx_parts.append(f"Regime {regime.value} ({'+' if adj>=0 else ''}{adj})")

        bias = "bullish" if score > 0 else ("bearish" if score < 0 else "neutral")
        ctx = " | ".join(ctx_parts) if ctx_parts else "Price action neutral"
        return bias, float(score), ctx

    # ------------------------------------------------------------------
    # Decision making
    # ------------------------------------------------------------------

    def _decide_direction(
        self,
        net_score: float,
        opts: WeeklyOptionsData,
    ) -> Tuple[TradeDirection, int, OptionsStrategy]:
        """
        Convert net score into trade direction, conviction, and strategy.
        """
        iv_rank = opts.iv_rank

        if net_score >= 35:
            direction  = TradeDirection.LONG
            conviction = min(95, int(50 + net_score))
            strategy   = OptionsStrategy.LONG_CALL if iv_rank < self.IV_HIGH \
                         else OptionsStrategy.SHORT_PUT
        elif net_score >= 15:
            direction  = TradeDirection.LONG
            conviction = min(80, int(40 + net_score))
            strategy   = OptionsStrategy.LONG_CALL
        elif net_score <= -35:
            direction  = TradeDirection.SHORT
            conviction = min(95, int(50 + abs(net_score)))
            strategy   = OptionsStrategy.LONG_PUT if iv_rank < self.IV_HIGH \
                         else OptionsStrategy.SHORT_CALL
        elif net_score <= -15:
            direction  = TradeDirection.SHORT
            conviction = min(80, int(40 + abs(net_score)))
            strategy   = OptionsStrategy.LONG_PUT
        else:
            direction  = TradeDirection.NEUTRAL
            conviction = 20
            strategy   = OptionsStrategy.NEUTRAL

        # OPTIMUS FIX: On expiry day (DTE=0 or 1), buying options (LONG_CALL / LONG_PUT)
        # is extremely high-risk due to theta decay. Cap conviction to 45 and
        # prefer short/neutral strategies. The cross-examiner flags OPTIMUS as defective
        # when it suggests high-conviction longs on expiry day.
        _expiry = getattr(opts, "weekly_expiry", "") or ""
        try:
            from datetime import datetime as _dt
            _today = _dt.now().date()
            _exp_date = _dt.strptime(_expiry, "%Y-%m-%d").date() if _expiry else None
            _dte = (_exp_date - _today).days if _exp_date else 99
        except Exception:
            _dte = 99
        if _dte <= 1 and strategy in (OptionsStrategy.LONG_CALL, OptionsStrategy.LONG_PUT):
            # Cap conviction — buying options with ≤1 day to expiry is speculative
            conviction = min(conviction, 45)
            # Prefer selling premium on expiry day if IV is elevated
            if iv_rank >= self.IV_HIGH:
                strategy = (OptionsStrategy.SHORT_PUT if direction == TradeDirection.LONG
                            else OptionsStrategy.SHORT_CALL)

        return direction, conviction, strategy

    # ------------------------------------------------------------------
    # Signal construction
    # ------------------------------------------------------------------

    def _build_signal(
        self,
        opts: WeeklyOptionsData,
        strategy: OptionsStrategy,
        conviction: int,
    ) -> OptionsSignal:
        """
        Build a complete OptionsSignal with entry / SL / targets.

        Option pricing is estimated as % of spot using a simplified rule-of-
        thumb since we may not have live premium data in simulation mode.
        """
        price       = opts.current_price or 20000
        gap         = opts.strike_gap
        iv          = opts.india_vix or 15
        expiry_str  = opts.weekly_expiry

        # Decide option type
        is_call = strategy in (OptionsStrategy.LONG_CALL, OptionsStrategy.SHORT_CALL)
        opt_type = OptionType.CE if is_call else OptionType.PE

        # ATM strike
        atm = round(price / gap) * gap

        # Recommended strike: 1 OTM for long, ATM for short
        if strategy in (OptionsStrategy.LONG_CALL, OptionsStrategy.SHORT_PUT):
            strike = atm + gap      # 1 OTM call / ITM put
            strike_label = "1-OTM CE" if is_call else "1-ITM PE"
        elif strategy in (OptionsStrategy.LONG_PUT, OptionsStrategy.SHORT_CALL):
            strike = atm - gap      # 1 OTM put / ITM call
            strike_label = "1-OTM PE" if not is_call else "1-ITM CE"
        else:
            strike = atm
            strike_label = "ATM"

        # Simplified premium estimation
        # Uses ~0.5% of spot for ATM weekly option (very rough, for fallback only)
        daily_move_pct = iv / (16 * 100)          # 1-sigma daily move
        days_to_expiry = max(1, self._days_to_expiry(expiry_str))
        approx_premium = round(price * daily_move_pct * (days_to_expiry ** 0.5), 1)

        # Entry range ±10%
        entry_low  = round(approx_premium * 0.90, 1)
        entry_high = round(approx_premium * 1.10, 1)

        # Stop loss: 40% of entry premium
        stop_loss = round(approx_premium * 0.60, 1)

        # Targets
        target_1 = round(approx_premium * 1.50, 1)
        target_2 = round(approx_premium * 2.20, 1)

        # Risk per lot (assume lot size ~75 for Nifty, 15 for BankNifty, 50 default)
        lot_size = self._default_lot_size(opts.symbol)
        risk_per_lot = round((approx_premium - stop_loss) * lot_size, 2)

        # PCR/IV/Basis context strings
        pcr_ctx  = f"PCR {opts.pcr:.2f}"
        iv_ctx   = f"IV rank {opts.iv_rank:.0f}"
        fut_ctx  = f"Basis {opts.futures_premium:+.0f}"

        return OptionsSignal(
            symbol=opts.symbol,
            expiry_date=expiry_str,
            option_type=opt_type,
            strategy=strategy,
            strike=strike,
            strike_label=strike_label,
            entry_low=entry_low,
            entry_high=entry_high,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            conviction=conviction,
            rationale="",                  # Filled by caller
            iv_context=iv_ctx,
            pcr_context=pcr_ctx,
            futures_basis=fut_ctx,
            risk_per_lot=risk_per_lot,
            suggested_lots=1,
        )

    # ------------------------------------------------------------------
    # Risk assessment
    # ------------------------------------------------------------------

    def _assess_risks(
        self, opts: WeeklyOptionsData, strategy: OptionsStrategy
    ) -> List[str]:
        risks: List[str] = []

        if opts.india_vix > 25:
            risks.append(f"High VIX {opts.india_vix:.1f} – elevated option premiums")

        if opts.iv_rank > self.IV_HIGH and strategy in (
            OptionsStrategy.LONG_CALL, OptionsStrategy.LONG_PUT
        ):
            risks.append("Buying options in high-IV environment – premium decay risk")

        if opts.weekly_expiry:
            d = self._days_to_expiry(opts.weekly_expiry)
            if d <= 2:
                risks.append(f"Only {d} day(s) to expiry – extreme theta decay")

        if opts.max_pain and opts.current_price:
            gap_pct = abs(opts.max_pain - opts.current_price) / opts.current_price * 100
            if gap_pct < 0.5:
                risks.append("Price very close to max-pain – pin risk on expiry")

        if abs(opts.futures_premium) > self.BASIS_SIGNIFICANT * 3:
            risks.append("Extreme futures basis – possible roll/expiry distortion")

        return risks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _next_thursday() -> str:
        """Return the date of next Thursday (weekly expiry for NSE index options)."""
        today = date.today()
        days_ahead = (3 - today.weekday()) % 7   # Thursday is weekday 3
        if days_ahead == 0:
            days_ahead = 7  # Already Thursday → next Thursday
        return (today + timedelta(days=days_ahead)).isoformat()

    @staticmethod
    def _days_to_expiry(expiry_str: str) -> int:
        try:
            exp = date.fromisoformat(expiry_str)
            return max(0, (exp - date.today()).days)
        except Exception:
            return 3  # Default to 3 days

    @staticmethod
    def _default_lot_size(symbol: str) -> int:
        sym = symbol.upper()
        if "BANKNIFTY" in sym or "BANKNI" in sym:
            return 15
        if "FINNIFTY" in sym:
            return 40
        if "MIDCAP" in sym:
            return 50
        if "NIFTY" in sym:
            return 75
        return 50  # Generic stock default

    @staticmethod
    def _signal_to_dict(signal: OptionsSignal) -> Dict:
        return {
            "symbol": signal.symbol,
            "expiry_date": signal.expiry_date,
            "option_type": signal.option_type.value,
            "strategy": signal.strategy.value,
            "strike": signal.strike,
            "strike_label": signal.strike_label,
            "entry_range": [signal.entry_low, signal.entry_high],
            "stop_loss": signal.stop_loss,
            "target_1": signal.target_1,
            "target_2": signal.target_2,
            "conviction": signal.conviction,
            "iv_context": signal.iv_context,
            "pcr_context": signal.pcr_context,
            "futures_basis": signal.futures_basis,
            "risk_per_lot": signal.risk_per_lot,
            "suggested_lots": signal.suggested_lots,
            "rationale": signal.rationale,
        }
