"""
ROX Proven Edge Engine v4.0 — FNO Brain Extension (OpenRouter)
===============================================================
Extends the base AI Brain with options-specific reasoning:
  - IV regime assessment  (HIGH / NORMAL / LOW -> strategy selection)
  - Conviction adjustment (IV percentile, earnings proximity, liquidity, settlement)
  - Strategy recommendation  (iron_condor, straddle, bull_call_spread ...)
  - Options post-mortem     (learns from closed option trades)

Uses OpenRouter API. Migrated from Gemini SDK on 2026-04-17.

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

    # OpenRouter model config
    _PRIMARY_MODEL   = os.getenv("OPEN_ROUTER_MODEL", "openrouter/free")
    _FALLBACK_MODEL  = os.getenv("OPEN_ROUTER_MODEL", "openrouter/free")

    def __init__(self):
        self._client = None
        self._provider = "offline"
        self._model    = self._PRIMARY_MODEL
        self._enabled  = os.environ.get("BRAIN_ENABLED", "true").lower() != "false"
        self._api_key  = ""
        self._base_url = os.getenv("OPEN_ROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        self._setup_backend()

    def _setup_backend(self):
        """Initialize OpenRouter API client."""
        if not self._enabled:
            return
        try:
            api_key = os.environ.get("OPEN_ROUTER_API", "").strip()
            if not api_key:
                logger.info("FNOBrain: no OPEN_ROUTER_API — will use rule-based fallback")
                return
            self._api_key = api_key
            self._provider = "openrouter"
            self._model = self._PRIMARY_MODEL
            logger.info(f"FNOBrain ready — openrouter/{self._model}")
        except Exception as e:
            logger.warning(f"FNOBrain backend setup failed: {e} — using rule-based fallback")

    # ------------------------------------------------------------------ #
    #  LLM call via OpenRouter                                            #
    # ------------------------------------------------------------------ #

    def _call_llm(self, system_prompt: str, user_prompt: str) -> tuple:
        """
        Call OpenRouter with automatic fallback on errors.
        Returns (text, token_count) or raises on total failure.
        """
        import httpx

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("OPEN_ROUTER_HTTP_REFERER", "https://rox-engine.local"),
            "X-Title": os.getenv("OPEN_ROUTER_X_TITLE", "ROX Trading Engine"),
        }

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 4096,
            "response_format": {"type": "json_object"},
        }

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"{self._base_url}/chat/completions",
                headers=headers,
                json=payload,
            )

        if resp.status_code != 200:
            raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        choice = data.get("choices", [{}])[0]
        text = choice.get("message", {}).get("content", "")
        usage = data.get("usage", {})
        token_count = usage.get("total_tokens", 0)

        return text, token_count

    # ------------------------------------------------------------------ #
    #  FIX-JSON-01: Strict JSON schema validator                         #
    # ------------------------------------------------------------------ #

    _VALID_IV_REGIMES = {"HIGH", "NORMAL", "LOW"}
    _VALID_STANCES = {"BULLISH", "NEUTRAL", "BEARISH", "CAUTIOUS"}
    _VALID_STRATEGIES = {
        "iron_condor", "straddle", "strangle", "bull_call_spread",
        "bear_put_spread", "calendar_spread", "short_put", "short_call",
    }
    _VALID_IV_FITS = {"HIGH_IV", "LOW_IV", "ANY"}
    _VALID_REGIME_FITS = {"BULL", "BEAR", "CONSOLIDATION", "ANY"}

    @classmethod
    def validate_fno_output(cls, data: dict) -> tuple:
        """
        Validate parsed FNOBrain JSON against schema.
        Returns (is_valid: bool, errors: list[str]).
        """
        errors = []

        # Required top-level fields
        for field in ("iv_regime", "market_stance", "risk_score", "narrative",
                       "strategy_recommendations", "options_summary"):
            if field not in data:
                errors.append(f"missing required field: {field}")

        # iv_regime
        iv_regime = data.get("iv_regime", "")
        if iv_regime not in cls._VALID_IV_REGIMES:
            errors.append(f"iv_regime '{iv_regime}' not in {cls._VALID_IV_REGIMES}")

        # market_stance
        stance = data.get("market_stance", "")
        if stance not in cls._VALID_STANCES:
            errors.append(f"market_stance '{stance}' not in {cls._VALID_STANCES}")

        # risk_score
        risk = data.get("risk_score")
        if not isinstance(risk, int) or not (1 <= risk <= 10):
            errors.append(f"risk_score must be int 1-10, got {risk}")

        # narrative — must be non-empty string
        narrative = data.get("narrative", "")
        if not isinstance(narrative, str) or len(narrative.strip()) < 10:
            errors.append("narrative must be non-empty string >= 10 chars")

        # strategy_recommendations
        recs = data.get("strategy_recommendations", [])
        if not isinstance(recs, list):
            errors.append("strategy_recommendations must be a list")
        else:
            for i, r in enumerate(recs):
                if not isinstance(r, dict):
                    errors.append(f"strategy_recommendations[{i}] must be dict")
                    continue
                sn = r.get("strategy_name", "")
                if sn not in cls._VALID_STRATEGIES:
                    errors.append(f"strategy[{i}].strategy_name '{sn}' invalid")
                for num_field in ("conviction", "adjusted_conviction"):
                    val = r.get(num_field)
                    if not isinstance(val, (int, float)) or not (0 <= val <= 100):
                        errors.append(f"strategy[{i}].{num_field} must be 0-100, got {val}")
                if not isinstance(r.get("proceed"), bool):
                    errors.append(f"strategy[{i}].proceed must be bool")

        # conviction_adjustments
        adj = data.get("conviction_adjustments", {})
        if not isinstance(adj, dict):
            errors.append("conviction_adjustments must be dict")

        # cautions
        cautions = data.get("cautions", [])
        if not isinstance(cautions, list):
            errors.append("cautions must be list")

        return (len(errors) == 0, errors)

    # ------------------------------------------------------------------ #
    #  FIX-JSON-01: LLM call with retry on JSON failure                  #
    # ------------------------------------------------------------------ #

    _MAX_JSON_RETRIES = 2

    def _call_llm_with_retry(self, system_prompt: str, user_prompt: str) -> tuple:
        """
        Call LLM with JSON validation retry.
        Up to _MAX_JSON_RETRIES extra attempts if JSON parse or schema validation fails.
        Returns (parsed_dict, raw_text, token_count).
        Raises on total failure.
        """
        last_raw = ""
        last_errors = []

        for attempt in range(1 + self._MAX_JSON_RETRIES):
            try:
                raw, tokens = self._call_llm(system_prompt, user_prompt)
                last_raw = raw
            except Exception as e:
                logger.error(f"FNOBrain LLM call failed (attempt {attempt+1}): {e}")
                if attempt < self._MAX_JSON_RETRIES:
                    continue
                raise

            # Parse JSON
            if not raw or not raw.strip():
                logger.warning(f"FNOBrain attempt {attempt+1}: empty response, retrying...")
                last_errors = ["empty response"]
                if attempt < self._MAX_JSON_RETRIES:
                    # Add explicit instruction to next attempt
                    user_prompt += "\n\nCRITICAL: Your previous response was empty. You MUST return complete JSON."
                    continue
                raise ValueError("FNOBrain: all attempts returned empty response")

            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0]

            try:
                data = json.loads(clean)
            except json.JSONDecodeError as e:
                logger.warning(
                    f"[FIX-JSON-01] FNOBrain JSON parse error (attempt {attempt+1}): {e}"
                )
                logger.debug(f"[FIX-JSON-01] Raw response (first 500 chars): {raw[:500]}")
                last_errors = [f"JSON parse: {e}"]
                if attempt < self._MAX_JSON_RETRIES:
                    user_prompt += (
                        f"\n\nCRITICAL: Your previous JSON was invalid ({e}). "
                        "Ensure ALL strings are properly closed with quotes. "
                        "Return ONLY complete, valid JSON."
                    )
                    continue
                raise

            # Schema validation
            is_valid, val_errors = self.validate_fno_output(data)
            if not is_valid:
                logger.warning(
                    f"[FIX-JSON-01] FNOBrain schema validation failed (attempt {attempt+1}): "
                    + "; ".join(val_errors)
                )
                logger.debug(f"[FIX-JSON-01] Raw response (first 500 chars): {raw[:500]}")
                last_errors = val_errors
                if attempt < self._MAX_JSON_RETRIES:
                    user_prompt += (
                        f"\n\nCRITICAL: Your JSON had validation errors: {'; '.join(val_errors)}. "
                        "Fix these issues and return valid JSON matching the schema."
                    )
                    continue
                # Last attempt with validation errors — use what we have (partial)
                logger.error(
                    f"[FIX-JSON-01] All {1+self._MAX_JSON_RETRIES} attempts had validation errors. "
                    f"Using last response with errors: {'; '.join(val_errors)}"
                )
                return data, raw, tokens

            # Success
            if attempt > 0:
                logger.info(f"[FIX-JSON-01] FNOBrain JSON succeeded on attempt {attempt+1}")
            return data, raw, tokens

        # Should not reach here
        raise RuntimeError(f"FNOBrain: all {1+self._MAX_JSON_RETRIES} attempts failed: {last_errors}")

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
            data, raw, tokens = self._call_llm_with_retry(FNO_SYSTEM_PROMPT, user_msg)
        except Exception as e:
            logger.error(f"[FIX-JSON-01] FNOBrain LLM call failed (all retries): {e}")
            return self._rule_based_output(market_context, fno_context, equity_setups)

        latency = time.monotonic() - t0
        logger.info(f"FNOBrain: openrouter/{self._model} | {latency:.1f}s | {tokens} tokens")
        return self._parse_validated(data, raw, market_context, fno_context, equity_setups, latency, tokens)

    @property
    def is_ready(self) -> bool:
        return self._api_key != ""

    @property
    def info(self) -> str:
        return f"openrouter/{self._model}"

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
    #  Parse validated LLM response (FIX-JSON-01)                         #
    # ------------------------------------------------------------------ #

    def _parse_validated(self, data: dict, raw: str, mctx, fno_ctx, equity_setups,
                         latency, tokens) -> FNOBrainOutput:
        """Parse pre-validated JSON data into FNOBrainOutput."""
        now = datetime.now().isoformat()

        recs = []
        for r in data.get("strategy_recommendations", []):
            recs.append(StrategyRecommendation(
                strategy_name       = r.get("strategy_name", ""),
                symbol              = r.get("symbol", "NIFTY"),
                conviction          = int(r.get("conviction", 65)),
                rationale           = r.get("rationale", ""),
                iv_fit              = r.get("iv_fit", "ANY"),
                regime_fit          = r.get("regime_fit", "ANY"),
                adjusted_conviction = int(r.get("adjusted_conviction", 65)),
                proceed             = bool(r.get("proceed", True)),
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
    #  Legacy parse (kept for backward compat)                            #
    # ------------------------------------------------------------------ #

    def _parse(self, raw, mctx, fno_ctx, equity_setups,
               latency, tokens) -> FNOBrainOutput:
        """Legacy parser — prefer _parse_validated with retry path."""
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
            logger.error(f"FNOBrain JSON parse error (legacy path): {e}")
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
