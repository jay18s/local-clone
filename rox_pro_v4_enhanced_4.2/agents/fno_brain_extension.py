"""
ROX Proven Edge Engine v4.0 — FNO Brain Extension (Gemini SDK)
===============================================================
Extends the base AI Brain with options-specific reasoning:
  - IV regime assessment  (HIGH / NORMAL / LOW -> strategy selection)
  - Conviction adjustment (IV percentile, earnings proximity, liquidity, settlement)
  - Strategy recommendation  (iron_condor, straddle, bull_call_spread ...)
  - Options post-mortem     (learns from closed option trades)

Uses the Google Gemini SDK (google.genai) directly, with FIX-QUOTA-01
fallback to flash model when gemini-3.1-pro-preview hits 429 quota.

Usage:
    from agents.fno_brain_extension import FNOBrainExtension
    fno_brain = FNOBrainExtension()
    result    = fno_brain.fno_synthesize_sync(consensus, setups,
                                              market_context, fno_context)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("rox.fno_brain")

# --------------------------------------------------------------------------- #
#  Dataclasses                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class StrategyRecommendation:
    strategy_name:  str        # iron_condor | straddle | bull_call_spread ...
    symbol:         str
    conviction:     int        # 0-100
    rationale:      str
    iv_fit:         str        # HIGH_IV | LOW_IV | ANY
    regime_fit:     str        # BULL | BEAR | CONSOLIDATION | ANY
    adjusted_conviction: int   # after IV/earnings/liquidity adjustments
    proceed:        bool


@dataclass
class FNOBrainOutput:
    timestamp:        str
    provider:         str
    model:            str
    iv_regime:        str        # HIGH | NORMAL | LOW
    iv_rank:          float      # 0-100
    market_stance:    str        # BULLISH | NEUTRAL | BEARISH | CAUTIOUS
    risk_score:       int        # 1-10
    narrative:        str
    strategy_recommendations: List[StrategyRecommendation]
    conviction_adjustments: Dict[str, int]  # symbol -> delta
    cautions:         List[str]
    options_summary:  str
    learning_note:    str
    latency_sec:      float = 0.0
    tokens_used:      int   = 0
    raw_response:     str   = ""


# --------------------------------------------------------------------------- #
#  FNO-specific system prompt                                                 #
# --------------------------------------------------------------------------- #

FNO_SYSTEM_PROMPT = """You are ROX Brain F&O, the options-specialist meta-intelligence
for the ROX Proven Edge Engine.

You combine the outputs of 8 specialised trading agents with deep options expertise.
You think like a senior derivatives trader at a prop desk — direct, quantitative,
risk-conscious, and never verbose.

Your F&O-specific expertise:
 - IV regime assessment: HIGH (>60th percentile) -> favour selling strategies
   NORMAL (40th-60th) -> directional spreads; LOW (<40th) -> buying strategies
 - Strategy selection aligned to regime + directional bias
 - Greeks-aware risk assessment (Delta exposure, Gamma risk near expiry)
 - Physical settlement awareness for stock options
 - Earnings/event proximity adjustments

CRITICAL COST STRUCTURE UPDATE — FIX-STT-05 (effective April 1, 2026):
 - F&O STT is increasing significantly from April 1, 2026:
     - Index Futures: 0.02% -> 0.05% of turnover (2.5x increase)
     - Options (buy side): 0.1% -> 0.15% of premium (1.5x increase)
 - Impact: All strategies held past or near April 1 expiry must factor this in.
   Near-expiry short-dated options strategies (weekly expiries) are most affected.
   Adjust breakeven calculations and net P&L projections accordingly.
 - For any strategy recommended within 7 days of April 1, 2026, add a caution
   noting the STT cost increase and its impact on net profitability.
 - Prefer strategies with fewer legs / lower turnover in this period to minimise STT drag.

Always respond with VALID JSON ONLY — no markdown fences, no preamble.

Response schema:
{
  "iv_regime":       "HIGH|NORMAL|LOW",
  "market_stance":   "BULLISH|NEUTRAL|BEARISH|CAUTIOUS",
  "risk_score":      <integer 1-10>,
  "narrative":       "<2-3 paragraph synthesis>",
  "strategy_recommendations": [
    {
      "strategy_name":      "iron_condor|straddle|strangle|bull_call_spread|bear_put_spread|calendar_spread",
      "symbol":             "<SYMBOL>",
      "conviction":         <int 0-100>,
      "rationale":          "<one sentence>",
      "iv_fit":             "HIGH_IV|LOW_IV|ANY",
      "regime_fit":         "BULL|BEAR|CONSOLIDATION|ANY",
      "adjusted_conviction":<int 0-100>,
      "proceed":            true|false
    }
  ],
  "conviction_adjustments": {
    "<SYMBOL>": <signed int delta e.g. -10 or +5>
  },
  "cautions":        ["<string>", ...],
  "options_summary": "<one sentence on overall options market tone>",
  "learning_note":   "<empty string unless trade_results supplied>"
}"""


# --------------------------------------------------------------------------- #
#  Main class                                                                 #
# --------------------------------------------------------------------------- #

class FNOBrainExtension:
    """
    F&O extension for AI Brain. Uses Google Gemini SDK directly
    with FIX-QUOTA-01 fallback to flash model on 429 errors.
    """

    # Strategy selection matrix: (iv_regime, direction) -> preferred strategies
    STRATEGY_MATRIX = {
        ("HIGH",   "BULLISH"):  ["bull_call_spread", "short_put", "iron_condor"],
        ("HIGH",   "BEARISH"):  ["bear_put_spread",  "short_call", "iron_condor"],
        ("HIGH",   "NEUTRAL"):  ["iron_condor", "calendar_spread"],
        ("NORMAL", "BULLISH"):  ["bull_call_spread", "straddle"],
        ("NORMAL", "BEARISH"):  ["bear_put_spread",  "straddle"],
        ("NORMAL", "NEUTRAL"):  ["iron_condor", "calendar_spread"],
        ("LOW",    "BULLISH"):  ["straddle", "strangle", "bull_call_spread"],
        ("LOW",    "BEARISH"):  ["straddle", "strangle", "bear_put_spread"],
        ("LOW",    "NEUTRAL"):  ["straddle", "strangle"],
    }

    # FIX-QUOTA-02: Model cascade for FNOBrain (matches other LLM modules)
    _PRIMARY_MODEL   = "gemini-3.1-pro-preview"
    _FALLBACK_MODEL  = "gemini-3-flash-preview"
    _SECONDARY_FALLBACK = "gemini-2.0-flash"

    def __init__(self):
        self._client = None
        self._provider = "offline"
        self._model    = self._PRIMARY_MODEL
        self._enabled  = os.environ.get("BRAIN_ENABLED", "true").lower() != "false"
        self._setup_backend()

    def _setup_backend(self):
        """Initialize Google Gemini SDK client."""
        if not self._enabled:
            return
        try:
            api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
            if not api_key:
                logger.info("FNOBrain: no GOOGLE_API_KEY — will use rule-based fallback")
                return

            from google import genai
            self._client = genai.Client(api_key=api_key)
            self._provider = "gemini"
            self._model = self._PRIMARY_MODEL
            logger.info(f"FNOBrain ready — gemini/{self._model}")
        except Exception as e:
            logger.warning(f"FNOBrain backend setup failed: {e} — using rule-based fallback")

    # ------------------------------------------------------------------ #
    #  LLM call with FIX-QUOTA-02 fallback                               #
    # ------------------------------------------------------------------ #

    def _call_llm(self, system_prompt: str, user_prompt: str) -> tuple:
        """
        Call Gemini with automatic fallback on 429 quota errors.

        Model cascade:
          1. gemini-3.1-pro-preview (best quality, limited quota)
          2. gemini-3-flash-preview (good quality, higher quota)
          3. gemini-2.0-flash (always available)

        Returns (text, token_count) or raises on total failure.
        """
        models_to_try = [self._PRIMARY_MODEL, self._FALLBACK_MODEL, self._SECONDARY_FALLBACK]
        last_error = None

        for model_name in models_to_try:
            try:
                from google.genai import types as genai_types
                response = self._client.models.generate_content(
                    model=model_name,
                    contents=user_prompt,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.7,
                        max_output_tokens=4096,
                    ),
                )
                text = response.text if response.text else ""
                tokens = getattr(response, 'usage_metadata', None)
                token_count = 0
                if tokens:
                    token_count = getattr(tokens, 'total_token_count', 0) or 0

                if model_name != self._model:
                    logger.info(
                        f"[FIX-QUOTA-02] FNOBrain fallback to {model_name} succeeded "
                        f"(tokens={token_count})"
                    )
                    self._model = model_name  # remember working model

                return text, token_count

            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                    logger.warning(
                        f"[FIX-QUOTA-02] Quota hit on {model_name}: "
                        + err_str[:150]
                        + f" — retrying with fallback model"
                    )
                    last_error = e
                    continue  # try next model in cascade
                else:
                    # Non-quota error — don't retry with different model
                    raise

        # All models exhausted
        if last_error:
            raise last_error
        raise RuntimeError("FNOBrain: all model fallbacks exhausted")

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def fno_synthesize_sync(
        self,
        consensus:      Dict[str, Any],
        equity_setups:  List[Dict[str, Any]],
        market_context: Dict[str, Any],
        fno_context:    Dict[str, Any],
        trade_results:  Optional[List[Dict]] = None,
    ) -> FNOBrainOutput:
        """
        Main entry point — combines agent consensus with F&O context.

        Parameters
        ----------
        consensus       coordinator consensus dict
        equity_setups   list of equity setup dicts
        market_context  {nifty_price, india_vix, pcr, iv_rank, ...}
        fno_context     {greeks_summary, oi_walls, settlement_flags, iv_rank, ...}
        trade_results   closed option trades for post-mortem (optional)
        """
        # Rule-based fallback if no LLM backend
        if not self._client:
            return self._rule_based_output(market_context, fno_context, equity_setups)

        user_msg = self._build_prompt(
            consensus, equity_setups, market_context, fno_context, trade_results
        )

        t0 = time.monotonic()
        try:
            raw, tokens = self._call_llm(FNO_SYSTEM_PROMPT, user_msg)
        except Exception as e:
            logger.error(f"FNOBrain LLM call failed (all fallbacks): {e}")
            return self._rule_based_output(market_context, fno_context, equity_setups)

        latency = time.monotonic() - t0
        logger.info(f"FNOBrain: gemini/{self._model} | {latency:.1f}s | {tokens} tokens")
        return self._parse(raw, market_context, fno_context, equity_setups, latency, tokens)

    @property
    def is_ready(self) -> bool:
        return self._client is not None

    @property
    def info(self) -> str:
        return f"gemini/{self._model}"

    # ------------------------------------------------------------------ #
    #  Prompt builder                                                      #
    # ------------------------------------------------------------------ #

    def _build_prompt(self, consensus, equity_setups, mctx, fno_ctx, trade_results):
        parts = [
            f"=== DATE / TIME ===\n{datetime.now().strftime('%Y-%m-%d %H:%M IST')}",
            "=== AGENT CONSENSUS ===\n" + json.dumps(consensus, indent=2),
            "=== EQUITY SETUPS ===\n"   + json.dumps(equity_setups, indent=2),
            "=== MARKET CONTEXT ===\n"  + json.dumps(mctx, indent=2),
            "=== F&O CONTEXT ===\n"     + json.dumps(fno_ctx, indent=2),
        ]
        if trade_results:
            parts.append("=== CLOSED OPTION TRADES (post-mortem) ===\n"
                         + json.dumps(trade_results, indent=2))
        parts.append("\nSynthesize and return your F&O JSON response.")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------ #
    #  Parse LLM response                                                 #
    # ------------------------------------------------------------------ #

    def _parse(self, raw, mctx, fno_ctx, equity_setups,
               latency, tokens) -> FNOBrainOutput:
        now = datetime.now().isoformat()
        if not raw:
            logger.error("FNOBrain received empty/None response from LLM — using rule-based fallback")
            return self._rule_based_output(mctx, fno_ctx, equity_setups, raw or "", latency, tokens)
        try:
            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0]
            data = json.loads(clean)
        except json.JSONDecodeError as e:
            logger.error(f"FNOBrain JSON parse error: {e}")
            return self._rule_based_output(mctx, fno_ctx, equity_setups, raw, latency, tokens)

        recs = []
        for r in data.get("strategy_recommendations", []):
            recs.append(StrategyRecommendation(
                strategy_name       = r.get("strategy_name", ""),
                symbol              = r.get("symbol", "NIFTY"),
                conviction          = r.get("conviction", 65),
                rationale           = r.get("rationale", ""),
                iv_fit              = r.get("iv_fit", "ANY"),
                regime_fit          = r.get("regime_fit", "ANY"),
                adjusted_conviction = r.get("adjusted_conviction", 65),
                proceed             = r.get("proceed", True),
            ))

        return FNOBrainOutput(
            timestamp      = now,
            provider       = self._provider,
            model          = self._model,
            iv_regime      = data.get("iv_regime", "NORMAL"),
            iv_rank        = float(fno_ctx.get("iv_rank", mctx.get("iv_rank", 50))),
            market_stance  = data.get("market_stance", "NEUTRAL"),
            risk_score     = int(data.get("risk_score", 5)),
            narrative      = data.get("narrative", ""),
            strategy_recommendations = recs,
            conviction_adjustments   = data.get("conviction_adjustments", {}),
            cautions       = data.get("cautions", []),
            options_summary= data.get("options_summary", ""),
            learning_note  = data.get("learning_note", ""),
            latency_sec    = round(latency, 2),
            tokens_used    = tokens,
            raw_response   = raw,
        )

    # ------------------------------------------------------------------ #
    #  Rule-based fallback (works without any LLM)                       #
    # ------------------------------------------------------------------ #

    def _rule_based_output(self, mctx, fno_ctx, equity_setups,
                            raw="", latency=0.0, tokens=0) -> FNOBrainOutput:
        iv_rank  = float(fno_ctx.get("iv_rank", mctx.get("iv_rank", 50)))
        pcr      = float(fno_ctx.get("pcr", mctx.get("pcr", 1.0)))
        vix      = float(mctx.get("india_vix", fno_ctx.get("india_vix", 15.0)))
        regime   = str(mctx.get("market_regime", "BULL"))

        # IV regime
        if iv_rank >= 60:
            iv_regime = "HIGH"
        elif iv_rank >= 40:
            iv_regime = "NORMAL"
        else:
            iv_regime = "LOW"

        # Direction from PCR
        if pcr > 1.2:
            direction = "BULLISH"
        elif pcr < 0.8:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        # Preferred strategies
        preferred = self.STRATEGY_MATRIX.get(
            (iv_regime, direction),
            ["iron_condor"]
        )

        recs = []
        for strat in preferred[:2]:
            conv = 65
            if iv_regime == "HIGH" and strat in ("iron_condor", "short_put", "short_call"):
                conv = 72
            elif iv_regime == "LOW" and strat in ("straddle", "strangle"):
                conv = 68

            # Settlement risk adjustment
            adj = -10 if fno_ctx.get("settlement_risk_present", False) else 0

            recs.append(StrategyRecommendation(
                strategy_name       = strat,
                symbol              = "NIFTY",
                conviction          = conv,
                rationale           = f"Rule-based: {iv_regime} IV regime + {direction} bias",
                iv_fit              = f"{iv_regime}_IV",
                regime_fit          = regime,
                adjusted_conviction = conv + adj,
                proceed             = (conv + adj) >= 60,
            ))

        cautions = []
        if vix > 20:
            cautions.append(f"VIX elevated at {vix:.1f} — prefer defined-risk strategies")
        if fno_ctx.get("settlement_risk_present"):
            cautions.append("Physical settlement risk detected — avoid new short stock positions near expiry")
        if iv_rank > 80:
            cautions.append(f"IV rank {iv_rank:.0f} — very high; premium selling risky if spike continues")

        narrative = (
            f"Rule-based F&O assessment (LLM offline). "
            f"IV regime: {iv_regime} (rank {iv_rank:.0f}). "
            f"Market direction: {direction} (PCR {pcr:.2f}). "
            f"Preferred strategy: {preferred[0].replace('_', ' ').title()}."
        )

        return FNOBrainOutput(
            timestamp      = datetime.now().isoformat(),
            provider       = "rule_based",
            model          = "none",
            iv_regime      = iv_regime,
            iv_rank        = iv_rank,
            market_stance  = direction,
            risk_score     = 5,
            narrative      = narrative,
            strategy_recommendations = recs,
            conviction_adjustments   = {},
            cautions       = cautions,
            options_summary= f"PCR {pcr:.2f} | IV rank {iv_rank:.0f} | VIX {vix:.1f}",
            learning_note  = "",
            latency_sec    = latency,
            tokens_used    = tokens,
            raw_response   = raw,
        )


# --------------------------------------------------------------------------- #
#  Terminal pretty-printer                                                    #
# --------------------------------------------------------------------------- #

def print_fno_brain_output(output: FNOBrainOutput):
    sep  = "=" * 65
    sep2 = "-" * 65
    print(f"\n{sep}")
    ts = output.timestamp[:16] if output.timestamp else "offline"
    print(f"F&O BRAIN  |  {ts}  |  {output.provider.upper()} / {output.model}")
    print(sep)
    print(f"\nIV REGIME : {output.iv_regime} (rank {output.iv_rank:.0f})  |  "
          f"STANCE : {output.market_stance}  |  RISK : {output.risk_score}/10  |  "
          f"{output.latency_sec}s\n")
    print(output.narrative)

    if output.cautions:
        print("\n  F&O CAUTIONS")
        for c in output.cautions:
            print(f"   * {c}")

    if output.strategy_recommendations:
        print(f"\n{sep}")
        print("RECOMMENDED OPTION STRATEGIES")
        print(sep2)
        for r in output.strategy_recommendations:
            flag = "OK" if r.proceed else "NO"
            delta = r.adjusted_conviction - r.conviction
            d_str = f"(+{delta})" if delta > 0 else f"({delta})" if delta < 0 else "(=)"
            print(f"\n  [{flag}]  {r.strategy_name.replace('_',' ').upper()}  on  {r.symbol}")
            print(f"       Conviction: {r.conviction} -> {r.adjusted_conviction} {d_str}  |  IV fit: {r.iv_fit}")
            print(f"       {r.rationale}")

    if output.options_summary:
        print(f"\n  OPTIONS TONE: {output.options_summary}")
    if output.learning_note:
        print(f"  LEARNING: {output.learning_note}")
    print(f"\n{sep}\n")
