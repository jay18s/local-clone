"""
LLM Trading Planner - Converts engine signals into a concrete, actionable trading plan
=======================================================================================

Takes all ROX engine outputs (regime, consensus, Phoenix, F&O suggestions, technicals,
macro data) and produces a structured plan a trader can actually execute:

  • KEY LEVELS     — support/resistance, 200DMA, ATR bands, max pain
  • SCENARIO PLAN  — IF market does X → do Y (conditional, not just "WAIT")
  • F&O READY      — exact iron condor / spread params with entry trigger
  • EQUITY WATCH   — top 2-3 stocks with trigger prices and stop levels
  • RISK PARAMS    — lot size / position size for current VIX environment

Uses gemini-2.0-flash (fast, cheap) — this runs every cycle.
Falls back to a rule-based plan if LLM unavailable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Any

from .base_llm_agent import BaseLLMAgent, LLMConfig, LLMResponse

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import MarketRegime, TradeDirection, get_current_stt, days_to_stt_hike


# ── Prompt ────────────────────────────────────────────────────────────────────

TRADING_PLAN_PROMPT = """You are a senior proprietary trader generating a concrete trading plan for the Indian market.

Your job is NOT to analyse — the analysis is already done. Your job is to convert it into
SPECIFIC, NUMBERED, EXECUTABLE instructions a trader can follow when the market opens.

MARKET DATA (from last session):
- Nifty spot       : {nifty_spot}
- 200 DMA          : {dma200}  ({dma_dist:+.2f}% from spot)
- SMA20            : {sma20}
- SMA50            : {sma50}
- ATR (daily)      : {atr:.0f} pts  ({atr_pct:.1f}% of spot)
- BB Upper         : {bb_upper:.0f}  |  BB Lower : {bb_lower:.0f}
- RSI              : {rsi:.0f}
- ADX              : {adx:.0f}  ({adx_label})
- India VIX        : {vix}
- Regime           : {regime}  (confidence {conf:.0f}%)
- Consensus        : {consensus_dir}  ({consensus_str})
- Cross-examiner   : {exam_rec}
- NIFTY PCR        : {pcr:.2f}  ({pcr_label})
- FII 5d flow      : ₹{fii_flow:+,.0f} Cr
- USD/INR          : ₹{usd_inr:.2f}
- Crude oil        : ${crude:.1f}/bbl {crude_note}
- G-sec yield      : {gsec:.2f}%

PREVIOUS SESSION HIGH/LOW:
- High : {prev_high:.0f}   Low : {prev_low:.0f}   Close : {nifty_spot}

PHOENIX SCORE: {phoenix_score:.0f}/100 | Tier: {phoenix_tier}
KEY PHOENIX SIGNALS: {phoenix_signals}

F&O CONTEXT:
- IV rank          : {iv_rank:.0f}  ({iv_regime})
- FNO Brain stance : {fno_stance}  |  Risk score : {fno_risk}/10
- FNO narrative    : {fno_narrative}
- Max pain Nifty   : {max_pain:.0f}
- STT note         : {stt_note}

TOP EQUITY SIGNALS (pre-gate, not yet approved):
{equity_signals}

LIVE STOCK PRICES (use ONLY these prices for entry/SL/target — do NOT invent prices):
{stock_prices}

ALLOWED EQUITY UNIVERSE (equity watchlist MUST only use these stocks — no others):
{allowed_stocks}

AGENT CONVICTION SUMMARY:
{agent_convictions}

INSTRUCTIONS — produce a JSON plan with EXACTLY this schema:
{{
  "key_levels": {{
    "strong_support":  <int>,
    "immediate_support": <int>,
    "immediate_resistance": <int>,
    "strong_resistance": <int>,
    "invalidation_bull": <int>,
    "invalidation_bear": <int>
  }},
  "scenarios": [
    {{
      "condition": "<20-word IF condition with specific price level>",
      "bias": "BULLISH|BEARISH|NEUTRAL",
      "action": "<20-word concrete action — instrument, direction, rough strikes/levels>",
      "target": "<price or range>",
      "stop_loss": "<price>"
    }}
  ],
  "fno_ready_trades": [
    {{
      "instrument": "NIFTY|BANKNIFTY|SENSEX|FINNIFTY",
      "strategy": "iron_condor|bull_call_spread|bear_put_spread|straddle|short_straddle",
      "direction_bias": "NEUTRAL|BULLISH|BEARISH",
      "entry_trigger": "<specific price/condition before entering>",
      "approx_strikes": "<e.g. sell 23000CE / 22000PE>",
      "target_credit_or_debit": "<e.g. ₹80-100 credit>",
      "stop_loss_rule": "<e.g. exit if premium doubles>",
      "max_loss_per_lot": <int>,
      "confidence": <int 0-100>,
      "status": "READY_TO_ENTER|WAIT_FOR_TRIGGER|WATCH_ONLY",
      "rationale": "<one sentence>"
    }}
  ],
  "equity_watchlist": [
    {{
      "stock": "<symbol>",
      "direction": "LONG|SHORT",
      "entry_trigger": "<specific price>",
      "stop_loss": "<price>",
      "target_1": "<price>",
      "target_2": "<price>",
      "conviction": <int 0-100>,
      "sector": "<sector>",
      "reason": "<one sentence catalyst>"
    }}
  ],
  "risk_parameters": {{
    "recommended_position_size_pct": <float>,
    "max_positions": <int>,
    "vix_sizing_note": "<one sentence on position sizing given VIX>",
    "capital_at_risk_per_trade_pct": <float>
  }},
  "overall_stance": "AGGRESSIVE_LONG|MODERATE_LONG|NEUTRAL|MODERATE_SHORT|AGGRESSIVE_SHORT|CASH",
  "market_open_checklist": [
    "<item 1 — specific action at open>",
    "<item 2>",
    "<item 3>"
  ],
  "plan_summary": "<2-3 sentence plain-English summary a trader can read in 10 seconds>"
}}

RULES:
- All prices must be specific integers or tight ranges (not "around 22800")
- Scenarios must be conditional: "IF Nifty opens above X AND holds for 15 min → ..."
- F&O trades: if cross-examiner said AVOID or WAIT, set status=WAIT_FOR_TRIGGER or WATCH_ONLY
- Equity watchlist: include even if no setups qualified — use shadow scan data above
- If VIX > 25, reduce recommended_position_size_pct to ≤ 2% per trade
- If VIX > 20 and IV regime is HIGH, prefer premium-selling strategies (iron_condor, short_straddle)
- Market open checklist must be actionable within the first 30 minutes of trading
- Respond ONLY with valid JSON — no markdown, no preamble
"""


@dataclass
class KeyLevels:
    strong_support: int
    immediate_support: int
    immediate_resistance: int
    strong_resistance: int
    invalidation_bull: int
    invalidation_bear: int


@dataclass
class TradeScenario:
    condition: str
    bias: str
    action: str
    target: str
    stop_loss: str


@dataclass
class FnoReadyTrade:
    instrument: str
    strategy: str
    direction_bias: str
    entry_trigger: str
    approx_strikes: str
    target_credit_or_debit: str
    stop_loss_rule: str
    max_loss_per_lot: int
    confidence: int
    status: str        # READY_TO_ENTER | WAIT_FOR_TRIGGER | WATCH_ONLY
    rationale: str


@dataclass
class EquityWatchItem:
    stock: str
    direction: str
    entry_trigger: str
    stop_loss: str
    target_1: str
    target_2: str
    conviction: int
    sector: str
    reason: str


@dataclass
class RiskParameters:
    recommended_position_size_pct: float
    max_positions: int
    vix_sizing_note: str
    capital_at_risk_per_trade_pct: float


@dataclass
class TradingPlanOutput:
    key_levels: KeyLevels
    scenarios: List[TradeScenario]
    fno_ready_trades: List[FnoReadyTrade]
    equity_watchlist: List[EquityWatchItem]
    risk_parameters: RiskParameters
    overall_stance: str
    market_open_checklist: List[str]
    plan_summary: str
    source: str = "LLM"
    timestamp: datetime = field(default_factory=datetime.now)
    raw_response: Optional[str] = None


class LLMTradingPlanner(BaseLLMAgent):
    """
    Converts all ROX engine signals into a concrete, actionable Monday trading plan.
    Uses gemini-2.0-flash for speed. Falls back to rule-based plan if LLM unavailable.
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config, logger_name="LLMTradingPlanner")
        self._last_plan: Optional[TradingPlanOutput] = None
        self._last_nifty_spot: float = 0.0
        self._last_exam_rec: str = "PROCEED"
        self._last_shadow_stocks: List[str] = []
        self._last_stock_prices: Dict[str, Any] = {}

    def generate_plan(
        self,
        market_data: Dict[str, Any],
        plan_context: Dict[str, Any],   # DailyTradingPlan fields as plain dict
    ) -> TradingPlanOutput:
        """
        Main entry point. Called by coordinator after all agents have run.
        """
        # Store context values needed by _parse_response post-processing
        self._last_nifty_spot    = float(market_data.get("nifty_price", 0) or 0)
        self._last_exam_rec      = plan_context.get("exam_recommendation", "PROCEED")
        self._last_stock_prices  = plan_context.get("stock_prices", {})
        # Allowed stocks = shadow scan stocks first, fall back to full watchlist
        _shadow = [
            s.get("stock", "") for s in plan_context.get("equity_shadow_signals", [])
            if s.get("stock")
        ]
        _universe = plan_context.get("watchlist_stocks", [])
        self._last_shadow_stocks = _shadow if _shadow else _universe

        prompt = self._build_prompt(market_data, plan_context)
        response = self.generate(prompt=prompt, expect_json=True, fallback_handler=None)

        if response.source == "LLM" and response.parsed_json:
            result = self._parse_response(response.parsed_json, response.content)
        else:
            result = self._fallback_plan(market_data, plan_context)

        self._last_plan = result
        return result

    def _build_prompt(self, md: Dict[str, Any], ctx: Dict[str, Any]) -> str:
        ni         = md.get("nifty_indicators", {})
        nifty      = md.get("nifty_price", 0)
        dma200     = md.get("nifty_200dma", nifty)
        vix        = md.get("india_vix", 15)
        flow       = md.get("flow_data", {})
        fii_flow   = flow.get("fii_cash_5day", 0)
        usd_inr    = md.get("usd_inr", 0)
        crude      = md.get("crude_brent_usd", 0)
        gsec       = md.get("gsec_yield", 0)
        pcr        = md.get("derivatives_data", {}).get("pcr", 1.0)

        sma20      = ni.get("sma20",  nifty * 0.99)  or nifty * 0.99
        sma50      = ni.get("sma50",  nifty * 0.975) or nifty * 0.975
        sma200     = ni.get("sma200", dma200)         or dma200
        atr        = ni.get("atr",    nifty * 0.012)  or nifty * 0.012
        rsi        = ni.get("rsi",    50)
        adx        = ni.get("adx",    md.get("adx", 25))
        bb_upper   = ni.get("bb_upper",  nifty * 1.02)
        bb_lower   = ni.get("bb_lower",  nifty * 0.98)

        intraday   = md.get("nifty_intraday", {})
        prev_high  = intraday.get("high", nifty * 1.005)
        prev_low   = intraday.get("low",  nifty * 0.993)

        dma_dist   = ((nifty - dma200) / dma200 * 100) if dma200 > 0 else 0.0
        atr_pct    = (atr / nifty * 100) if nifty > 0 else 1.2

        crude_note = "(data unavailable)" if crude == 0 else ""

        # Pre-compute conditional labels — inline conditionals inside {} crash .format()
        adx_label = "trending" if float(adx or 0) > 25 else "ranging"
        pcr_label = ("bearish" if float(pcr or 1) < 0.9
                     else "bullish" if float(pcr or 1) > 1.1
                     else "neutral")

        # Regime + consensus
        regime     = ctx.get("regime", "CONSOLIDATION")
        conf       = ctx.get("regime_confidence", 80)
        cons_dir   = ctx.get("consensus_direction", "NEUTRAL")
        cons_str   = ctx.get("consensus_strength", "NO_CONSENSUS")
        exam_rec   = ctx.get("exam_recommendation", "WAIT")

        # Phoenix
        px         = ctx.get("phoenix", {})
        px_score   = px.get("score", 0)
        px_tier    = px.get("tier", "DORMANT")
        px_sigs    = px.get("signals_summary", "No active signals")

        # F&O context
        fno        = ctx.get("fno_brain", {})
        iv_rank    = ctx.get("iv_rank", 50)
        iv_regime  = ctx.get("iv_regime", "NORMAL")
        fno_stance = fno.get("stance", "NEUTRAL")
        fno_risk   = fno.get("risk_score", 5)
        fno_narr   = fno.get("narrative", "")[:200]
        max_pain   = ctx.get("max_pain", nifty)

        # STT context
        days_hike  = days_to_stt_hike()
        if days_hike > 0:
            stt_note = f"STT hike in {days_hike} days (Apr 1) — prefer fewer-leg strategies"
        elif days_hike == 0:
            stt_note = "STT hike ACTIVE TODAY — factor new rates into P&L"
        else:
            stt_note = f"Post-STT hike (was {-days_hike} days ago) — new rates apply"

        # Equity signals
        eq_signals = ctx.get("equity_shadow_signals", [])
        if eq_signals:
            eq_lines = "\n".join(
                f"  {s.get('stock','?'):12s} {s.get('direction','?'):5s} "
                f"conviction={int(s.get('conviction',0)):3d}  "
                f"entry≈{s.get('entry_price',0):.0f}  SL≈{s.get('stop_loss',0):.0f}  "
                f"T1≈{s.get('target_1',0):.0f}"
                for s in eq_signals[:5]
            )
        else:
            eq_lines = "  (No shadow scan data — use watchlist universe below for ideas)"

        # Live stock prices — LLM MUST use these for entry/SL/target, not guesses
        _prices = ctx.get("stock_prices", {})
        _wl = ctx.get("watchlist_stocks", [])
        if _prices:
            price_lines = "\n".join(
                f"  {sym:12s}  CMP=₹{v['price']:.0f}  ATR=₹{v['atr']:.0f}  "
                f"52w_low=₹{v['low52']:.0f}  52w_high=₹{v['high52']:.0f}"
                for sym, v in list(_prices.items())[:15]
            )
            allowed_stocks = "  " + ", ".join(_wl) if _wl else "  (All Nifty 50 stocks)"
        else:
            price_lines = "  (Live prices not available — infer from Nifty context)"
            allowed_stocks = ("  " + ", ".join(_wl)) if _wl else "  (All Nifty 50 stocks)"

        # Agent convictions — conviction comes from agent as int, but guard with int()
        agents = ctx.get("agent_convictions", {})
        ag_lines = "\n".join(
            f"  {name:12s}: {v.get('direction','?'):7s} conviction={int(v.get('conviction',0)):3d}%  weight={v.get('weight',0):.2f}"
            for name, v in agents.items()
        ) or "  (Not available)"

        return TRADING_PLAN_PROMPT.format(
            nifty_spot=nifty, dma200=dma200, dma_dist=dma_dist,
            sma20=sma20, sma50=sma50, atr=atr, atr_pct=atr_pct,
            bb_upper=bb_upper, bb_lower=bb_lower, rsi=rsi, adx=adx,
            adx_label=adx_label,
            vix=vix, regime=regime, conf=conf,
            consensus_dir=cons_dir, consensus_str=cons_str,
            exam_rec=exam_rec, pcr=pcr, pcr_label=pcr_label, fii_flow=fii_flow,
            usd_inr=usd_inr if usd_inr else 84.0,
            crude=crude, crude_note=crude_note, gsec=gsec,
            prev_high=prev_high, prev_low=prev_low,
            phoenix_score=px_score, phoenix_tier=px_tier,
            phoenix_signals=px_sigs,
            iv_rank=float(iv_rank or 50), iv_regime=iv_regime,
            fno_stance=fno_stance, fno_risk=fno_risk,
            fno_narrative=fno_narr, max_pain=float(max_pain or nifty), stt_note=stt_note,
            equity_signals=eq_lines, allowed_stocks=allowed_stocks,
            stock_prices=price_lines, agent_convictions=ag_lines,
        )

    def _parse_response(self, parsed: Dict, raw: str) -> TradingPlanOutput:
        try:
            kl_raw = parsed.get("key_levels", {})
            kl = KeyLevels(
                strong_support        = int(kl_raw.get("strong_support", 0)),
                immediate_support     = int(kl_raw.get("immediate_support", 0)),
                immediate_resistance  = int(kl_raw.get("immediate_resistance", 0)),
                strong_resistance     = int(kl_raw.get("strong_resistance", 0)),
                invalidation_bull     = int(kl_raw.get("invalidation_bull", 0)),
                invalidation_bear     = int(kl_raw.get("invalidation_bear", 0)),
            )

            # FIX-PLAN-03: Validate invalidation levels aren't inverted.
            # Bull invalidation must be BELOW current price (it's where the bull case dies).
            # Bear invalidation must be ABOVE current price (it's where the bear case dies).
            # Correct by swapping if the LLM got them backwards.
            _spot = self._last_nifty_spot or 0
            if _spot > 0 and kl.invalidation_bull > 0 and kl.invalidation_bear > 0:
                if kl.invalidation_bull > _spot and kl.invalidation_bear < _spot:
                    # Inverted — swap
                    kl = KeyLevels(
                        strong_support       = kl.strong_support,
                        immediate_support    = kl.immediate_support,
                        immediate_resistance = kl.immediate_resistance,
                        strong_resistance    = kl.strong_resistance,
                        invalidation_bull    = kl.invalidation_bear,   # was bear, now bull
                        invalidation_bear    = kl.invalidation_bull,   # was bull, now bear
                    )
                    self.logger.debug("[TRADING-PLAN] Invalidation levels were inverted — corrected")

            scenarios = [
                TradeScenario(
                    condition = s.get("condition", ""),
                    bias      = s.get("bias", "NEUTRAL"),
                    action    = s.get("action", ""),
                    target    = str(s.get("target", "")),
                    stop_loss = str(s.get("stop_loss", "")),
                )
                for s in parsed.get("scenarios", [])[:3]
            ]

            # FIX-PLAN-04: Enforce WATCH_ONLY status when cross-examiner says AVOID/WAIT.
            # The LLM sometimes marks a trade READY_TO_ENTER despite being told AVOID.
            _exam = self._last_exam_rec or "PROCEED"
            _force_watch = _exam in ("AVOID", "WAIT")

            fno_trades = [
                FnoReadyTrade(
                    instrument           = t.get("instrument", "NIFTY"),
                    strategy             = t.get("strategy", "iron_condor"),
                    direction_bias       = t.get("direction_bias", "NEUTRAL"),
                    entry_trigger        = t.get("entry_trigger", ""),
                    approx_strikes       = t.get("approx_strikes", ""),
                    target_credit_or_debit = t.get("target_credit_or_debit", ""),
                    stop_loss_rule       = t.get("stop_loss_rule", ""),
                    max_loss_per_lot     = int(t.get("max_loss_per_lot", 0)),
                    confidence           = int(t.get("confidence", 0)),
                    # Force WATCH_ONLY when examiner says AVOID; WAIT_FOR_TRIGGER when WAIT
                    status               = (
                        "WATCH_ONLY"        if _force_watch and _exam == "AVOID" else
                        "WAIT_FOR_TRIGGER"  if _force_watch and _exam == "WAIT"  else
                        t.get("status", "WATCH_ONLY")
                    ),
                    rationale            = t.get("rationale", ""),
                )
                for t in parsed.get("fno_ready_trades", [])[:3]
            ]

            # FIX-PLAN-02: Equity watchlist — filter to only stocks from shadow scan.
            # If we have shadow scan stocks, only keep watchlist items that appear in it.
            # This prevents hallucinated stocks that aren't grounded in agent analysis.
            _shadow_stocks = {s.upper() for s in (self._last_shadow_stocks or [])}
            watchlist_raw = parsed.get("equity_watchlist", [])[:5]
            if _shadow_stocks:
                watchlist_raw = [
                    w for w in watchlist_raw
                    if w.get("stock", "").upper() in _shadow_stocks
                ] or watchlist_raw[:2]   # fallback: keep top 2 if nothing matches

            # FIX-PLAN-05: Validate equity watchlist entry prices against live CMP.
            # The LLM sometimes invents prices (e.g. ₹1,550 for HDFCBANK at ₹750).
            # Reject any watchlist item whose entry_trigger price is more than 50%
            # away from the actual live price we passed in the context.
            _live_prices = getattr(self, "_last_stock_prices", {})
            validated_raw = []
            for w in watchlist_raw:
                _sym = w.get("stock", "").upper()
                _entry_str = str(w.get("entry_trigger", "0")).replace("₹", "").replace(",", "").strip()
                try:
                    _entry_val = float(_entry_str)
                except ValueError:
                    validated_raw.append(w)
                    continue
                if _sym in _live_prices and _entry_val > 0:
                    _cmp = _live_prices[_sym].get("price", 0)
                    if _cmp > 0:
                        _ratio = _entry_val / _cmp
                        if not (0.5 < _ratio < 1.5):
                            self.logger.warning(
                                f"[TRADING-PLAN] Rejected {_sym} watchlist entry: "
                                f"entry={_entry_val} vs CMP={_cmp} (ratio={_ratio:.2f}, >50% off)"
                            )
                            continue  # skip this item
                validated_raw.append(w)

            watchlist = [
                EquityWatchItem(
                    stock         = w.get("stock", ""),
                    direction     = w.get("direction", "LONG"),
                    entry_trigger = str(w.get("entry_trigger", "")),
                    stop_loss     = str(w.get("stop_loss", "")),
                    target_1      = str(w.get("target_1", "")),
                    target_2      = str(w.get("target_2", "")),
                    conviction    = int(w.get("conviction", 0)),
                    sector        = w.get("sector", ""),
                    reason        = w.get("reason", ""),
                )
                for w in validated_raw
            ]

            rp_raw = parsed.get("risk_parameters", {})
            rp = RiskParameters(
                recommended_position_size_pct  = float(rp_raw.get("recommended_position_size_pct", 2.0)),
                max_positions                  = int(rp_raw.get("max_positions", 2)),
                vix_sizing_note                = rp_raw.get("vix_sizing_note", ""),
                capital_at_risk_per_trade_pct  = float(rp_raw.get("capital_at_risk_per_trade_pct", 1.0)),
            )

            return TradingPlanOutput(
                key_levels          = kl,
                scenarios           = scenarios,
                fno_ready_trades    = fno_trades,
                equity_watchlist    = watchlist,
                risk_parameters     = rp,
                overall_stance      = parsed.get("overall_stance", "NEUTRAL"),
                market_open_checklist = parsed.get("market_open_checklist", [])[:5],
                plan_summary        = parsed.get("plan_summary", ""),
                source              = "LLM",
                raw_response        = raw,
            )
        except Exception as e:
            self.logger.error(f"TradingPlanner parse error: {e}")
            return self._empty_plan()

    def _fallback_plan(self, md: Dict, ctx: Dict) -> TradingPlanOutput:
        """Rule-based fallback when LLM is unavailable."""
        nifty  = md.get("nifty_price", 22000)
        ni     = md.get("nifty_indicators", {})
        atr    = ni.get("atr", nifty * 0.012) or nifty * 0.012
        dma200 = md.get("nifty_200dma", nifty)
        vix    = md.get("india_vix", 20)

        sup1  = int(dma200 - atr * 0.5)
        sup2  = int(dma200 - atr * 1.5)
        res1  = int(dma200 + atr * 0.5)
        res2  = int(dma200 + atr * 1.5)

        size_pct = 1.5 if vix > 25 else 2.0 if vix > 20 else 3.0

        return TradingPlanOutput(
            key_levels = KeyLevels(
                strong_support       = sup2,
                immediate_support    = sup1,
                immediate_resistance = res1,
                strong_resistance    = res2,
                invalidation_bull    = int(dma200 - atr * 2),
                invalidation_bear    = int(dma200 + atr * 2),
            ),
            scenarios = [
                TradeScenario(
                    condition = f"Nifty opens and holds above {int(dma200 + 50)} for 15 min",
                    bias      = "BULLISH",
                    action    = f"Buy Nifty CE or go long near {int(dma200 + 50)}",
                    target    = str(res2),
                    stop_loss = str(sup1),
                ),
                TradeScenario(
                    condition = f"Nifty breaks below {int(dma200 - 50)} on volume",
                    bias      = "BEARISH",
                    action    = f"Buy Nifty PE or go short near {int(dma200 - 50)}",
                    target    = str(sup2),
                    stop_loss = str(res1),
                ),
            ],
            fno_ready_trades  = [],
            equity_watchlist  = [],
            risk_parameters   = RiskParameters(
                recommended_position_size_pct  = size_pct,
                max_positions                  = 2,
                vix_sizing_note                = f"VIX={vix:.0f} — use {size_pct:.0f}% per trade",
                capital_at_risk_per_trade_pct  = size_pct * 0.5,
            ),
            overall_stance      = "NEUTRAL",
            market_open_checklist = [
                f"Check if Nifty holds 200 DMA ({int(dma200)}) at open",
                "Wait 15 minutes before entering any position",
                "Do NOT trade in first 15 minutes regardless of direction",
            ],
            plan_summary = (
                f"Rule-based plan (LLM unavailable). "
                f"Key level: 200 DMA at {int(dma200)}. "
                f"Trade only after 15-min confirmation. "
                f"Position size capped at {size_pct:.0f}% given VIX {vix:.0f}."
            ),
            source = "FALLBACK",
        )

    def _empty_plan(self) -> TradingPlanOutput:
        return self._fallback_plan({}, {})

    def get_last_plan(self) -> Optional[TradingPlanOutput]:
        return self._last_plan
