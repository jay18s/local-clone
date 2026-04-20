"""
ROX Proven Edge Engine v4.1 - KAIRO Agent (Enhanced)
=====================================================
Sentiment Analysis Agent - Market psychology and contrarian signals.

v4.1 Enhancement: Integrates with News Intelligence Layer (news_core.py)
to incorporate real-time geopolitical sentiment, live news sentiment, and
overnight risk data into KAIRO's composite score.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

from .base_agent import BaseAgent, AgentVerdict, AgentReport
import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TradeDirection, MarketRegime, SentimentZone

logger = logging.getLogger("rox.kairo")


@dataclass
class SentimentComponents:
    """Individual sentiment component scores"""
    news_sentiment: float = 0.0  # -100 to +100
    analyst_sentiment: float = 0.0
    social_sentiment: float = 0.0
    corporate_sentiment: float = 0.0
    global_sentiment: float = 0.0


class KairoAgent(BaseAgent):
    """
    KAIRO - Sentiment Analysis Agent (v4.1)

    Aggregates sentiment from multiple channels and identifies contrarian
    opportunities at sentiment extremes.

    v4.1: Now consumes live news_context data from the News Intelligence Layer
    to enrich news_sentiment and global_sentiment when available.

    Baseline weight: 12%
    Weight increases during regime transitions and market extremes.
    """

    # Component weights
    NEWS_WEIGHT = 0.30
    ANALYST_WEIGHT = 0.25
    SOCIAL_WEIGHT = 0.20
    CORPORATE_WEIGHT = 0.15
    GLOBAL_WEIGHT = 0.10

    # Sentiment zone thresholds
    EUPHORIA_THRESHOLD = 70
    BULLISH_UPPER = 70
    BULLISH_LOWER = 40
    BEARISH_UPPER = -40
    BEARISH_LOWER = -70
    PANIC_THRESHOLD = -70

    def __init__(self):
        super().__init__(
            name="KAIRO",
            domain="Sentiment Analysis",
            baseline_weight=0.12
        )

    def analyze(self, data: Dict[str, Any], regime: MarketRegime) -> AgentReport:
        """
        Perform comprehensive sentiment analysis.

        Args:
            data: Should contain:
                - 'news_sentiment': News headlines sentiment (-100 to +100)
                - 'analyst_sentiment': Analyst ratings sentiment
                - 'social_sentiment': Social media sentiment
                - 'corporate_sentiment': Management commentary sentiment
                - 'global_sentiment': Global macro sentiment
                Optional:
                - 'vix': India VIX value
                - 'pcr': Put-call ratio
                - 'narrative': Current market narrative
                - 'news_context': Live news data from news_core (v4.1)
                - 'overnight_risk': Overnight risk profile from NOCTURNAL (v4.1)

        Returns:
            AgentReport with sentiment analysis verdict
        """
        # v4.1: Extract and apply live news context if available
        news_context_data = data.get("news_context", [])
        overnight_risk = data.get("overnight_risk", None)

        # Compute live sentiment overrides from news_context
        live_news_sentiment, live_global_sentiment = self._extract_news_context_sentiment(
            news_context_data, overnight_risk
        )

        # Parse sentiment components — prefer live data when available
        components = SentimentComponents(
            news_sentiment=live_news_sentiment if live_news_sentiment is not None
                           else data.get('news_sentiment', 0),
            analyst_sentiment=data.get('analyst_sentiment', 0),
            social_sentiment=data.get('social_sentiment', 0),
            corporate_sentiment=data.get('corporate_sentiment', 0),
            global_sentiment=live_global_sentiment if live_global_sentiment is not None
                             else data.get('global_sentiment', 0),
        )

        # Calculate composite sentiment score
        composite_score = self._calculate_composite_score(components)

        # Apply overnight risk adjustment (v4.1)
        composite_score = self._apply_overnight_risk_adjustment(composite_score, overnight_risk)

        # Determine sentiment zone
        sentiment_zone = self._determine_sentiment_zone(composite_score)

        # Check for contrarian signals
        contrarian_signal = self._detect_contrarian_signal(
            composite_score, sentiment_zone, data
        )

        # Track narrative
        narrative = data.get('narrative', {})
        narrative_analysis = self._analyze_narrative(narrative, components)

        # Analyze global impact
        global_analysis = self._analyze_global_factors(data)

        # Generate verdict
        verdict = self._generate_verdict(
            composite_score, sentiment_zone, contrarian_signal, regime
        )

        # Build key observations
        key_observations = self._generate_observations(
            components, sentiment_zone, contrarian_signal, global_analysis
        )

        # Add news intelligence observations (v4.1)
        if news_context_data:
            key_observations.append(
                f"News Intelligence: {len(news_context_data)} live articles analysed"
            )
        if overnight_risk and getattr(overnight_risk, "risk_level", None):
            key_observations.append(
                f"Overnight Risk: {overnight_risk.risk_level}"
            )

        return AgentReport(
            agent_name=self.name,
            verdict=verdict,
            analysis_details={
                "composite_score": composite_score,
                "sentiment_zone": sentiment_zone.value,
                "contrarian_signal": contrarian_signal,
                "narrative_strength": narrative_analysis.get("strength", "unknown"),
                "global_impact": global_analysis.get("impact", "neutral"),
                "live_news_used": live_news_sentiment is not None,
                "overnight_risk_level": getattr(overnight_risk, "risk_level", "NONE"),
            },
            key_observations=key_observations,
            metrics={
                "composite_score": composite_score,
                "news_score": components.news_sentiment,
                "analyst_score": components.analyst_sentiment,
                "social_score": components.social_sentiment
            },
            raw_data={
                "components": components.__dict__,
                "vix": data.get('vix', 0),
                "pcr": data.get('pcr', 0)
            }
        )

    # ------------------------------------------------------------------
    # v4.1 — News Intelligence helpers
    # ------------------------------------------------------------------

    def _extract_news_context_sentiment(
        self,
        news_context_data: Any,
        overnight_risk: Any,
    ):
        """
        Convert raw news_context items into news_sentiment and global_sentiment
        scores on the -100..+100 scale that KAIRO expects.

        Returns (news_sentiment, global_sentiment) — either may be None if
        insufficient data is available.
        """
        news_sentiment = None
        global_sentiment = None

        # news_context_data may be a list of NewsItem objects or dicts
        if not news_context_data:
            return news_sentiment, global_sentiment

        try:
            impact_scores = []
            geo_scores = []

            items = news_context_data if isinstance(news_context_data, list) else []
            for item in items:
                # Support both NewsItem objects and plain dicts
                if hasattr(item, "impact_score"):
                    score = item.impact_score  # -1.0 to +1.0
                    category = getattr(item, "category", None)
                    cat_val = category.value if hasattr(category, "value") else str(category)
                    impact_scores.append(score)
                    if "geopolit" in cat_val.lower() or "macro" in cat_val.lower():
                        geo_scores.append(score)
                elif isinstance(item, dict):
                    score = item.get("impact_score", 0)
                    cat_val = str(item.get("category", ""))
                    impact_scores.append(score)
                    if "geopolit" in cat_val.lower() or "macro" in cat_val.lower():
                        geo_scores.append(score)

            if impact_scores:
                avg_impact = sum(impact_scores) / len(impact_scores)
                news_sentiment = avg_impact * 100  # scale to -100..+100

            if geo_scores:
                avg_geo = sum(geo_scores) / len(geo_scores)
                global_sentiment = avg_geo * 100

        except Exception as e:
            logger.debug(f"KAIRO: news context extraction failed: {e}")

        # Also use overnight risk as a sentiment modifier
        if overnight_risk is not None:
            try:
                risk_level = getattr(overnight_risk, "risk_level", "NORMAL")
                gap_size = getattr(overnight_risk, "expected_gap_size", "neutral")
                override = {
                    "EXTREME": -80,
                    "HIGH": -40,
                    "ELEVATED": -15,
                    "NORMAL": 0,
                }.get(str(risk_level).upper(), 0)
                if override != 0:
                    news_sentiment = override if news_sentiment is None else (news_sentiment + override) / 2
            except Exception as e:
                logger.debug(f"KAIRO: overnight risk sentiment override failed: {e}")

        return news_sentiment, global_sentiment

    def _apply_overnight_risk_adjustment(
        self, composite_score: float, overnight_risk: Any
    ) -> float:
        """
        Dampen bullish composite scores when overnight risk is HIGH/EXTREME.
        This prevents KAIRO from emitting strong LONG signals on days when
        NOCTURNAL has flagged elevated geopolitical or macro risk.
        """
        if overnight_risk is None:
            return composite_score

        try:
            risk_level = str(getattr(overnight_risk, "risk_level", "NORMAL")).upper()
            if risk_level == "EXTREME":
                # Force deep bearish territory
                return min(composite_score, -40)
            elif risk_level == "HIGH":
                # Cap bullish sentiment
                return min(composite_score, -10)
            elif risk_level == "ELEVATED":
                # Slightly dampen
                return composite_score * 0.7
        except Exception:
            pass

        return composite_score

    # ------------------------------------------------------------------
    # Core sentiment analysis methods (v3.x — unchanged logic)
    # ------------------------------------------------------------------

    def _calculate_composite_score(self, components: SentimentComponents) -> float:
        """Calculate weighted composite sentiment score"""
        score = (
            components.news_sentiment * self.NEWS_WEIGHT +
            components.analyst_sentiment * self.ANALYST_WEIGHT +
            components.social_sentiment * self.SOCIAL_WEIGHT +
            components.corporate_sentiment * self.CORPORATE_WEIGHT +
            components.global_sentiment * self.GLOBAL_WEIGHT
        )
        return max(-100, min(100, score))

    def _determine_sentiment_zone(self, score: float) -> SentimentZone:
        """Determine the sentiment zone from score"""
        if score > self.EUPHORIA_THRESHOLD:
            return SentimentZone.EUPHORIA
        elif score > self.BULLISH_LOWER:
            return SentimentZone.BULLISH
        elif score < self.PANIC_THRESHOLD:
            return SentimentZone.PANIC
        elif score < self.BEARISH_UPPER:
            return SentimentZone.BEARISH
        else:
            return SentimentZone.NEUTRAL

    def _detect_contrarian_signal(self, score: float, zone: SentimentZone,
                                  data: Dict) -> Dict:
        """Detect contrarian buy/sell signals"""
        contrarian = {
            "detected": False,
            "strength": "none",
            "direction": None,
            "indicators": []
        }

        vix = data.get('vix', 0)
        pcr = data.get('pcr', 0)

        # Euphoria detection (contrarian sell)
        euphoria_indicators = []

        if zone == SentimentZone.EUPHORIA:
            euphoria_indicators.append("Composite score in euphoria zone")

        if score > 80:
            euphoria_indicators.append("Extreme sentiment reading")

        # Check for sustained high sentiment
        sustained_high = data.get('sustained_high_sentiment', False)
        if sustained_high:
            euphoria_indicators.append("Sustained high sentiment for 5+ days")

        # IPO frenzy indicator (would need external data)
        if data.get('ipo_frenzy', False):
            euphoria_indicators.append("IPO frenzy detected")

        if len(euphoria_indicators) >= 2:
            contrarian["detected"] = True
            contrarian["strength"] = "strong" if len(euphoria_indicators) >= 3 else "moderate"
            contrarian["direction"] = TradeDirection.SHORT
            contrarian["indicators"] = euphoria_indicators

        # Capitulation/Panic detection (contrarian buy)
        panic_indicators = []

        if zone == SentimentZone.PANIC:
            panic_indicators.append("Composite score in panic zone")

        if score < -80:
            panic_indicators.append("Extreme negative sentiment")

        if vix > 28:
            panic_indicators.append(f"VIX elevated at {vix}")

        if pcr > 1.5:
            panic_indicators.append(f"Extreme put buying (PCR: {pcr})")

        # Check for FII capitulation
        fii_heavy_selling = data.get('fii_5day_flow', 0) < -10000
        if fii_heavy_selling:
            panic_indicators.append("FII heavy selling (> 10,000 Cr)")

        if len(panic_indicators) >= 2:
            contrarian["detected"] = True
            contrarian["strength"] = "strong" if len(panic_indicators) >= 3 else "moderate"
            contrarian["direction"] = TradeDirection.LONG
            contrarian["indicators"] = panic_indicators

        return contrarian

    def _analyze_narrative(self, narrative: Dict, components: SentimentComponents) -> Dict:
        """Analyze current market narrative"""
        analysis = {
            "current": narrative.get('current', 'Unknown'),
            "strength": narrative.get('strength', 'unknown'),
            "cracks_detected": False
        }

        # Check for narrative cracks
        price_direction = narrative.get('price_direction', 'neutral')
        if (price_direction == 'up' and components.news_sentiment < -20):
            analysis["cracks_detected"] = True
            analysis["crack_type"] = "bullish_narrative_weakening"
        elif (price_direction == 'down' and components.news_sentiment > 20):
            analysis["cracks_detected"] = True
            analysis["crack_type"] = "bearish_narrative_weakening"

        return analysis

    def _analyze_global_factors(self, data: Dict) -> Dict:
        """Analyze global sentiment impact on India"""
        analysis = {
            "impact": "neutral",
            "factors": []
        }

        dxy = data.get('dxy', 0)
        crude = data.get('crude', 0)
        us_market = data.get('us_market_direction', 'neutral')
        vix_global = data.get('vix_global', 0)

        # DXY impact
        if dxy > 105:
            analysis["factors"].append("Strong dollar (DXY > 105) - negative for FII flows")
            analysis["impact"] = "negative"
        elif dxy < 100:
            analysis["factors"].append("Weak dollar (DXY < 100) - positive for FII flows")
            if analysis["impact"] == "neutral":
                analysis["impact"] = "positive"

        # Crude impact
        if crude > 90:
            analysis["factors"].append("High crude ($90+) - negative for India macros")
            analysis["impact"] = "negative"
        elif crude < 75:
            analysis["factors"].append("Low crude ($75-) - positive for India macros")
            if analysis["impact"] != "negative":
                analysis["impact"] = "positive"

        # US market correlation
        if us_market == 'down' and data.get('us_market_change', 0) < -2:
            analysis["factors"].append("US markets down >2% - likely negative open")
            analysis["impact"] = "negative"

        # Global VIX
        if vix_global > 25:
            analysis["factors"].append(f"Global VIX elevated ({vix_global}) - risk-off mode")
            analysis["impact"] = "negative"

        return analysis

    def _generate_verdict(self, composite_score: float, zone: SentimentZone,
                         contrarian_signal: Dict, regime: MarketRegime) -> AgentVerdict:
        """Generate the final verdict"""
        # Calculate LONG threshold based on regime
        _non_bull_regimes = {MarketRegime.BEAR, MarketRegime.CORRECTION, MarketRegime.MILD_BEAR, MarketRegime.CONSOLIDATION}
        _long_threshold = 45 if regime in _non_bull_regimes else 20

        # Start with sentiment direction
        if composite_score > _long_threshold:
            direction = TradeDirection.LONG
            base_conviction = 50 + composite_score * 0.3
        elif composite_score < -20:
            direction = TradeDirection.SHORT
            base_conviction = 50 - composite_score * 0.3
        else:
            direction = TradeDirection.NEUTRAL
            base_conviction = 50

        # Apply contrarian adjustment
        if contrarian_signal["detected"]:
            # Reverse direction on strong contrarian signals
            if contrarian_signal["strength"] == "strong":
                if contrarian_signal["direction"] == TradeDirection.LONG:
                    # Panic = contrarian buy
                    direction = TradeDirection.LONG
                    base_conviction = 75
                else:
                    # Euphoria = contrarian sell/caution
                    if direction == TradeDirection.LONG:
                        base_conviction -= 20
                        direction = TradeDirection.NEUTRAL
            else:
                # Moderate contrarian signal
                base_conviction = max(40, base_conviction - 10)

        # Regime adjustments
        if regime == MarketRegime.BEAR and zone == SentimentZone.PANIC:
            # In bear markets, panic readings are strong buy signals
            base_conviction += 15
            direction = TradeDirection.LONG
        elif regime == MarketRegime.BULL and zone == SentimentZone.EUPHORIA:
            # In bull markets, euphoria is a warning
            base_conviction -= 15

        # Clamp conviction
        conviction = max(0, min(100, base_conviction))

        # Generate reason
        reason = f"Sentiment zone: {zone.value}"
        if contrarian_signal["detected"]:
            reason += f" | Contrarian signal: {contrarian_signal['direction'].value if contrarian_signal['direction'] else 'none'}"

        # Generate risks
        risks = []
        if zone == SentimentZone.EUPHORIA:
            risks.append("Euphoric conditions - reduce position sizes")
        elif zone == SentimentZone.PANIC:
            risks.append("Panic conditions - wait for technical confirmation")
        if abs(composite_score) > 70:
            risks.append("Extreme sentiment readings can persist longer than expected")

        return AgentVerdict(
            direction=direction,
            conviction=conviction,
            weight=self.current_weight,
            reason=reason,
            risks=risks
        )

    def _generate_observations(self, components: SentimentComponents,
                              zone: SentimentZone, contrarian_signal: Dict,
                              global_analysis: Dict) -> List[str]:
        """Generate key observations"""
        observations = []

        # Sentiment zone
        observations.append(f"Sentiment zone: {zone.value}")

        # Component highlights
        if abs(components.news_sentiment) > 50:
            dir_word = "positive" if components.news_sentiment > 0 else "negative"
            observations.append(f"News sentiment strongly {dir_word} ({components.news_sentiment:.0f})")

        if abs(components.social_sentiment) > 60:
            dir_word = "bullish" if components.social_sentiment > 0 else "bearish"
            observations.append(f"Social sentiment {dir_word} ({components.social_sentiment:.0f})")

        # Contrarian signals
        if contrarian_signal["detected"]:
            strength = contrarian_signal["strength"]
            direction = contrarian_signal["direction"]
            observations.append(f"CONTRARIAN ALERT: {strength.upper()} {direction.value if direction else ''} signal detected")

        # Global factors
        for factor in global_analysis.get("factors", [])[:2]:
            observations.append(f"Global: {factor}")

        return observations
