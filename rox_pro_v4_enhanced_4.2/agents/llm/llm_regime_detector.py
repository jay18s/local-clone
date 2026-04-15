"""
LLM Regime Detector - LLM-powered market regime detection (Enhancement P1.1)
=============================================================================

Replaces rule-based regime detection with LLM-powered analysis that provides:
- Probability distributions across all regimes
- Human-readable reasoning
- Key factor identification
- Transition warnings

Integrates into LeadCoordinator._tier_11_detect_regime()
Falls back to rule-based if LLM unavailable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

from .base_llm_agent import BaseLLMAgent, LLMConfig, LLMResponse

# Import from parent config
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import MarketRegime, TradeDirection


# Prompt Templates
REGIME_DETECTION_PROMPT = """You are a senior market analyst for Indian equities.

Analyze the following market data and classify the current market regime.

MARKET DATA:
- Nifty Price: {nifty_price}
- Nifty 200DMA: {nifty_200dma}
- Distance from 200DMA: {distance_pct:.1f}%
- India VIX: {vix} (5d ago: {vix_5d_ago}, trend: {vix_trend})
- FII 5-day Flow: ₹{fii_flow:+,} Cr
- DII 5-day Flow: ₹{dii_flow:+,} Cr
- NIFTY PCR: {nifty_pcr}
- BANKNIFTY PCR: {banknifty_pcr}
- Sector Performance (5d): {sector_performance}
- Price Structure: {price_structure}
- ADX: {adx}

GLOBAL & MACRO CONTEXT (FIX-MACRO-01 — do NOT ignore these):
- Brent Crude Oil: ${crude_usd:.1f}/bbl  ← Sharp crude moves directly impact Indian macro (CAD, inflation, OMC margins)
- USD/INR: ₹{usd_inr:.2f}  ← Rupee stress amplifies FII outflows; rupee strength aids them
- Gift Nifty Overnight Gap: {gift_nifty_gap_pct:+.2f}%  ← Leading signal from SGX/GIFT; captures global overnight sentiment
- Key External Catalyst: {external_catalyst}

RECENT NEWS CONTEXT:
{news_summary}

PREVIOUS REGIME: {previous_regime}

Respond ONLY with valid JSON in this exact format:
{{
    "regime": "BULL|BEAR|CONSOLIDATION|MILD_BULL|MILD_BEAR|CORRECTION",
    "confidence": <integer 0-100>,
    "regime_probability_distribution": {{
        "BULL": <float 0-1>,
        "MILD_BULL": <float 0-1>,
        "CONSOLIDATION": <float 0-1>,
        "MILD_BEAR": <float 0-1>,
        "BEAR": <float 0-1>,
        "CORRECTION": <float 0-1>
    }},
    "reasoning": "<2-3 sentences explaining the analysis>",
    "key_factors": [
        "<factor 1>",
        "<factor 2>",
        "<factor 3>"
    ],
    "transition_warning": "<optional warning if regime change is likely, or null>"
}}

IMPORTANT:
- Probabilities must sum to 1.0
- Be conservative: use CONSOLIDATION when signals are mixed
- Consider VIX trend, not just level
- Factor in FII/DII flow direction and magnitude
- External catalysts (crude crash, geopolitical de-escalation, US Fed pivot) can override
  technical signals — a +8% Gift Nifty gap or crude -10% in a single session is a regime event.
- A falling VIX + rising PCR + positive Gift Nifty gap combination is a strong MILD_BULL signal
  even when price is still near 200DMA lows.
"""


@dataclass
class RegimeDetectionResult:
    """Result of LLM-powered regime detection."""
    regime: MarketRegime
    confidence: float  # 0-100
    probability_distribution: Dict[str, float]  # BULL: 0.25, BEAR: 0.05, etc.
    reasoning: str  # Human-readable explanation
    key_factors: List[str]  # Top 3-5 factors that influenced decision
    transition_warning: Optional[str] = None  # Warning if regime change likely
    source: str = "LLM"  # "LLM" or "FALLBACK"
    timestamp: datetime = field(default_factory=datetime.now)
    raw_response: Optional[str] = None


class LLMRegimeDetector(BaseLLMAgent):
    """
    LLM-powered market regime detection.

    Integrates into LeadCoordinator._tier_11_detect_regime()
    Falls back to rule-based if LLM unavailable.
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config, logger_name="LLMRegimeDetector")
        self._last_result: Optional[RegimeDetectionResult] = None
        self._regime_history: List[RegimeDetectionResult] = []
        # 15-minute result cache — regime doesn't change every 60s
        self._cache_result: Optional[RegimeDetectionResult] = None
        self._cache_expires_at: datetime = datetime.min

    def detect_regime(
        self,
        market_data: Dict[str, Any],
        previous_regime: MarketRegime
    ) -> RegimeDetectionResult:
        """
        Main method called by coordinator.

        Args:
            market_data: Dict with nifty_price, vix, fii_flows, etc.
            previous_regime: Previous day's regime for context

        Returns:
            RegimeDetectionResult with regime, confidence, probabilities, reasoning
        """
        # Build prompt
        prompt = self._build_prompt(market_data, previous_regime)

        # ── 15-minute result cache ────────────────────────────────────────────
        # Regime detection with gemini-2.5-pro is expensive. Market regime
        # doesn't change every 60 seconds. Skip the LLM call if we have a
        # fresh cached result from the same regime context.
        now = datetime.now()
        if (
            self._cache_result is not None
            and now < self._cache_expires_at
            and self._cache_result.regime == previous_regime
        ):
            self.logger.debug("[REGIME-CACHE] Returning cached regime (still fresh)")
            return self._cache_result

        # Get LLM response (no fallback_handler - we handle fallback ourselves)
        response = self.generate(
            prompt=prompt,
            expect_json=True,
            fallback_handler=None
        )

        # Parse response - check if we got valid LLM response
        if response.source == "LLM" and response.parsed_json:
            result = self._parse_response(response.parsed_json, response.content)
        else:
            # Use fallback when LLM unavailable
            result = self._fallback_detection(market_data, previous_regime)

        # Store result
        self._last_result = result
        self._regime_history.append(result)
        if len(self._regime_history) > 30:  # Keep last 30 days
            self._regime_history = self._regime_history[-30:]

        # Update 15-minute cache if LLM responded (not just rule-based fallback)
        if result.source == "LLM":
            self._cache_result = result
            self._cache_expires_at = now + timedelta(minutes=15)

        return result

    def _build_prompt(
        self,
        market_data: Dict[str, Any],
        previous_regime: MarketRegime
    ) -> str:
        """Construct LLM prompt with market context."""
        nifty_price = market_data.get("nifty_price", 22500)
        nifty_200dma = market_data.get("nifty_200dma", nifty_price * 0.95)

        # Calculate distance from 200DMA
        distance_pct = 0.0
        if nifty_200dma > 0:
            distance_pct = ((nifty_price - nifty_200dma) / nifty_200dma) * 100

        # VIX data
        vix = market_data.get("india_vix", 15.0)
        vix_history = market_data.get("vix_history", [])
        vix_5d_ago = vix_history[-5] if len(vix_history) >= 5 else vix
        vix_trend = "rising" if vix > vix_5d_ago else "falling" if vix < vix_5d_ago else "stable"

        # Flows
        flow_data = market_data.get("flow_data", {})
        fii_flow = flow_data.get("fii_cash_5day", 0)
        dii_flow = flow_data.get("dii_cash_5day", 0)

        # Derivatives — "pcr" is the key in derivatives_data; "nifty_pcr" key doesn't exist
        deriv_data = market_data.get("derivatives_data", {})
        nifty_pcr = deriv_data.get("pcr", deriv_data.get("nifty_pcr", 1.0))
        # BANKNIFTY PCR comes from index_option_chains, not derivatives_data
        banknifty_chain = market_data.get("index_option_chains", {}).get("BANKNIFTY", {})
        banknifty_pcr = banknifty_chain.get("pcr", 1.0)

        # Sector performance
        sector_perf = market_data.get("sector_performance", {})
        sector_performance = ", ".join(
            f"{k}: {v:+.1f}%" for k, v in list(sector_perf.items())[:5]
        ) if sector_perf else "Neutral"

        # Technical
        price_structure = market_data.get("price_structure", "neutral")
        adx = market_data.get("adx", 25)

        # News — try both keys: "news_items" (direct) and "news_context" (coordinator-injected)
        news_items = market_data.get("news_items") or market_data.get("news_context", [])
        news_summary = "\n".join(
            f"- {item.get('title', item) if isinstance(item, dict) else item}"
            for item in news_items[:8]
        ) if news_items else "No significant news in the last 24 hours"

        # ── FIX-MACRO-01: Global & macro context ─────────────────────────────
        # These were fetched but never injected into the LLM prompt, causing the
        # regime detector to be blind to crude oil moves, rupee stress, Gift Nifty
        # signals, and geopolitical catalysts — the primary drivers on macro-event days.
        # crude_brent_usd and usd_inr now live at TOP LEVEL of market_data (fyers_fetcher).
        macro_data = market_data.get("macro", {})
        crude_usd = (
            market_data.get("crude_brent_usd")       # top-level (fyers_fetcher, primary)
            or market_data.get("crude_price")         # legacy alias
            or macro_data.get("crude_brent_usd", 0.0)
        )
        usd_inr = (
            market_data.get("usd_inr")               # top-level (fyers_fetcher, primary)
            or macro_data.get("usd_inr", 84.0)
        )
        gift_nifty_gap_pct = (
            market_data.get("gift_nifty_gap_pct")    # top-level (fyers_fetcher, primary)
            or macro_data.get("gift_nifty_gap_pct", 0.0)
        )
        # Build external catalyst string from geopolitical summary or overnight_risk
        # FIX-PIPELINE-04: overnight_risk is an OvernightRiskProfile *object*, not a dict.
        # Calling .get() on it silently returns {} so geo_events was always empty.
        # Use .to_agent_context() to convert, then read key_headlines for catalyst text.
        overnight_risk_raw = market_data.get("overnight_risk")
        overnight_risk = {}
        if overnight_risk_raw is not None:
            if hasattr(overnight_risk_raw, "to_agent_context"):
                overnight_risk = overnight_risk_raw.to_agent_context()
            elif isinstance(overnight_risk_raw, dict):
                overnight_risk = overnight_risk_raw

        geo_events = []
        # OvernightRiskProfile.to_agent_context() returns key "restrictions" and narrative,
        # but key_headlines lives directly on the object
        if overnight_risk_raw is not None and hasattr(overnight_risk_raw, "key_headlines"):
            geo_events = list(overnight_risk_raw.key_headlines[:3])
        elif overnight_risk.get("key_headlines"):
            geo_events = list(overnight_risk["key_headlines"][:3])

        # Also surface risk_level and market_stance as part of the catalyst string
        risk_level    = overnight_risk.get("risk_level", "")
        market_stance = overnight_risk.get("market_stance", "")
        geo_prefix = ""
        if risk_level in ("EXTREME", "HIGH", "ELEVATED"):
            geo_prefix = f"[OVERNIGHT: {risk_level} / {market_stance}] "

        external_catalyst = (
            market_data.get("geopolitical_summary")
            or (geo_prefix + "; ".join(geo_events) if geo_events else
                (geo_prefix + "None identified" if geo_prefix else "None identified"))
        )

        return REGIME_DETECTION_PROMPT.format(
            nifty_price=nifty_price,
            nifty_200dma=nifty_200dma,
            distance_pct=distance_pct,
            vix=vix,
            vix_5d_ago=vix_5d_ago,
            vix_trend=vix_trend,
            fii_flow=fii_flow,
            dii_flow=dii_flow,
            nifty_pcr=nifty_pcr,
            banknifty_pcr=banknifty_pcr,
            sector_performance=sector_performance,
            price_structure=price_structure,
            adx=adx,
            crude_usd=crude_usd if crude_usd else 0.0,
            usd_inr=usd_inr if usd_inr else 84.0,
            gift_nifty_gap_pct=gift_nifty_gap_pct if gift_nifty_gap_pct else 0.0,
            external_catalyst=external_catalyst,
            news_summary=news_summary,
            previous_regime=previous_regime.value if hasattr(previous_regime, 'value') else str(previous_regime)
        )

    def _parse_response(
        self,
        parsed: Dict[str, Any],
        raw_response: str
    ) -> RegimeDetectionResult:
        """Parse JSON response from LLM."""
        try:
            # Parse regime
            regime_str = parsed.get("regime", "CONSOLIDATION").upper()
            regime_map = {
                "BULL": MarketRegime.BULL,
                "MILD_BULL": MarketRegime.MILD_BULL,
                "CONSOLIDATION": MarketRegime.CONSOLIDATION,
                "MILD_BEAR": MarketRegime.MILD_BEAR,
                "BEAR": MarketRegime.BEAR,
                "CORRECTION": MarketRegime.CORRECTION,
            }
            regime = regime_map.get(regime_str, MarketRegime.CONSOLIDATION)

            # Parse confidence
            confidence = float(parsed.get("confidence", 50))
            confidence = max(0, min(100, confidence))

            # Parse probability distribution
            prob_dist = parsed.get("regime_probability_distribution", {})
            if prob_dist:
                # Normalize to sum to 1.0
                total = sum(prob_dist.values())
                if total > 0:
                    prob_dist = {k: v / total for k, v in prob_dist.items()}
            else:
                # Default distribution based on detected regime
                prob_dist = {regime_str: 0.5, "CONSOLIDATION": 0.3}
                for r in regime_map:
                    if r not in prob_dist:
                        prob_dist[r] = 0.04

            return RegimeDetectionResult(
                regime=regime,
                confidence=confidence,
                probability_distribution=prob_dist,
                reasoning=parsed.get("reasoning", "LLM analysis completed"),
                key_factors=parsed.get("key_factors", []),
                transition_warning=parsed.get("transition_warning"),
                source="LLM",
                raw_response=raw_response
            )

        except Exception as e:
            self.logger.error(f"Failed to parse regime response: {e}")
            return RegimeDetectionResult(
                regime=MarketRegime.CONSOLIDATION,
                confidence=50,
                probability_distribution={"CONSOLIDATION": 0.5, "MILD_BULL": 0.25, "MILD_BEAR": 0.25},
                reasoning=f"Parse error: {e}",
                key_factors=[],
                source="FALLBACK"
            )

    def _fallback_detection(
        self,
        market_data: Dict[str, Any],
        previous_regime: MarketRegime
    ) -> RegimeDetectionResult:
        """Rule-based fallback if LLM fails."""
        self.logger.info("Using rule-based regime detection fallback")

        score = 0
        fired = 0

        nifty = market_data.get("nifty_price", 0)
        dma200 = market_data.get("nifty_200dma", 0)

        if dma200 > 0:
            gap = (nifty - dma200) / dma200
            if gap > 0.05:
                score += 1
                fired += 1
            elif gap > 0:
                score += 0.5
            elif gap < -0.05:
                score -= 1
                fired += 1
            else:
                score -= 0.5

        ps = market_data.get("price_structure", "neutral")
        if ps == "higher_highs":
            score += 1
            fired += 1
        elif ps == "lower_lows":
            score -= 1
            fired += 1

        adx = market_data.get("adx", 20)
        adx_strong = adx > 25

        fii = market_data.get("flow_data", {}).get("fii_cash_5day", 0)
        if fii > 3000:
            score += 1
            fired += 1
        elif fii < -3000:
            score -= 1
            fired += 1

        vix = market_data.get("india_vix", 15)
        if vix > 22:
            score -= 0.5
        elif vix < 13:
            score += 0.5
            fired += 1

        # Determine regime
        if score >= 2:
            regime = MarketRegime.BULL
        elif score >= 1:
            regime = MarketRegime.MILD_BULL
        elif score <= -2:
            regime = MarketRegime.BEAR
        elif score <= -1:
            regime = MarketRegime.MILD_BEAR
        else:
            regime = MarketRegime.CONSOLIDATION

        # Calculate confidence
        abs_s = abs(score)
        base = 50 + abs_s * 10
        if abs_s >= 2 and adx_strong and fired >= 3:
            conf = min(90, base + fired * 3)
        elif abs_s >= 2:
            conf = min(75, base + fired * 2)
        elif abs_s >= 1:
            conf = min(65, base + fired * 2)
        else:
            conf = min(55, base)

        # Build probability distribution
        prob_dist = {
            "BULL": 0.1,
            "MILD_BULL": 0.15,
            "CONSOLIDATION": 0.3,
            "MILD_BEAR": 0.15,
            "BEAR": 0.1,
            "CORRECTION": 0.2
        }
        if regime == MarketRegime.BULL:
            prob_dist["BULL"] = 0.4
            prob_dist["MILD_BULL"] = 0.25
        elif regime == MarketRegime.BEAR:
            prob_dist["BEAR"] = 0.4
            prob_dist["MILD_BEAR"] = 0.25
        elif regime == MarketRegime.MILD_BULL:
            prob_dist["MILD_BULL"] = 0.35
            prob_dist["BULL"] = 0.2
        elif regime == MarketRegime.MILD_BEAR:
            prob_dist["MILD_BEAR"] = 0.35
            prob_dist["BEAR"] = 0.2

        key_factors = []
        if dma200 > 0:
            gap = (nifty - dma200) / dma200 * 100
            key_factors.append(f"Nifty {gap:+.1f}% from 200DMA")
        if ps != "neutral":
            key_factors.append(f"Price structure: {ps}")
        if abs(fii) > 2000:
            key_factors.append(f"FII 5d flow: ₹{fii:+,}Cr")
        if vix > 18:
            key_factors.append(f"Elevated VIX: {vix:.1f}")

        return RegimeDetectionResult(
            regime=regime,
            confidence=round(conf, 1),
            probability_distribution=prob_dist,
            reasoning=f"Rule-based detection based on {fired} triggered signals, net score: {score:.1f}",
            key_factors=key_factors,
            transition_warning=None,
            source="FALLBACK"
        )

    def get_regime_history(self, days: int = 30) -> List[RegimeDetectionResult]:
        """Get regime detection history."""
        return self._regime_history[-days:]

    def get_last_result(self) -> Optional[RegimeDetectionResult]:
        """Get the most recent regime detection result."""
        return self._last_result
