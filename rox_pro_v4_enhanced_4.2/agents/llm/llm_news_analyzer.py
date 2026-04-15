"""
LLM News Impact Analyzer - Trading-focused news intelligence (Enhancement P2.1)
================================================================================

Extends existing GeminiNewsAnalyzer to generate trading-actionable intelligence:
- Sector impact scores
- Index gap predictions
- Actionable signals
- Risk event identification
- Trade restrictions
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Any

from .base_llm_agent import BaseLLMAgent, LLMConfig, LLMResponse

# Import from parent config
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import MarketRegime, TradeDirection


# Prompt Template
NEWS_IMPACT_PROMPT = """You are a senior news analyst for an options trading desk.

Analyze these news items and provide TRADING-FOCUSED intelligence:

NEWS ITEMS (last 24 hours):
{news_items_formatted}

MACRO CATALYSTS — ACTIVE TODAY (FIX-NEWS-04: treat these as primary signals, not secondary):
{macro_catalysts}

CURRENT POSITIONS:
{positions_formatted}

UPCOMING EVENTS (next 7 days):
{events_formatted}

Provide comprehensive trading intelligence in JSON format:
{{
    "overall_market_impact": {{
        "direction": "BULLISH|BEARISH|NEUTRAL",
        "magnitude": "HIGH|MEDIUM|LOW",
        "confidence": <integer 0-100>
    }},
    "sector_impacts": {{
        "IT": {{"impact_score": <float -1 to 1>, "reason": "<explanation>"}},
        "Banking": {{"impact_score": <float -1 to 1>, "reason": "<explanation>"}},
        "Pharma": {{"impact_score": <float -1 to 1>, "reason": "<explanation>"}},
        "Auto": {{"impact_score": <float -1 to 1>, "reason": "<explanation>"}},
        "Metals": {{"impact_score": <float -1 to 1>, "reason": "<explanation>"}},
        "FMCG": {{"impact_score": <float -1 to 1>, "reason": "<explanation>"}},
        "Energy": {{"impact_score": <float -1 to 1>, "reason": "<explanation>"}}
    }},
    "index_impacts": {{
        "NIFTY": {{"expected_gap": "<points range>", "probability": <float 0-1>}},
        "BANKNIFTY": {{"expected_gap": "<points range>", "probability": <float 0-1>}},
        "SENSEX": {{"expected_gap": "<points range>", "probability": <float 0-1>}}
    }},
    "actionable_signals": [
        {{
            "signal_type": "SECTOR_AVOID|SECTOR_OVERWEIGHT|VOLATILITY_BUY|VOLATILITY_SELL|STOCK_SPECIFIC",
            "target": "<sector/index/stock>",
            "reason": "<explanation>"
        }}
    ],
    "risk_events": [
        {{
            "event_name": "<name>",
            "event_date": "<YYYY-MM-DD>",
            "impact_description": "<what to expect>",
            "recommendation": "<action to take>"
        }}
    ],
    "trade_restrictions": [
        "<restriction 1, e.g., HALT_NEW_SHORTS_IN_IT>",
        "<restriction 2, e.g., REDUCE_POSITION_SIZE_25%>"
    ],
    "executive_summary": "<2-3 paragraph actionable summary for the trading desk>"
}}

IMPORTANT:
- Focus on actionable intelligence, not just analysis
- Consider second-order effects (e.g., oil price → inflation → RBI policy)
- Identify both opportunities and risks
- Be specific about which sectors/stocks are affected
"""


@dataclass
class SectorImpact:
    """Sector impact from news."""
    impact_score: float  # -1.0 to +1.0
    reason: str


@dataclass
class IndexImpact:
    """Expected index impact."""
    expected_gap: str  # e.g., "+50 to +100 points"
    probability: float  # 0.0 to 1.0


@dataclass
class ActionableSignal:
    """Trading signal derived from news."""
    signal_type: str  # "SECTOR_AVOID", "VOLATILITY_BUY", etc.
    target: str  # Sector, index, or stock
    reason: str


@dataclass
class RiskEvent:
    """Upcoming risk event."""
    event_name: str
    event_date: str
    impact_description: str
    recommendation: str


@dataclass
class NewsImpactResult:
    """Complete news impact analysis result."""
    overall_market_impact: Dict[str, Any]  # direction, magnitude, confidence
    sector_impacts: Dict[str, SectorImpact]
    index_impacts: Dict[str, IndexImpact]
    actionable_signals: List[ActionableSignal]
    risk_events: List[RiskEvent]
    trade_restrictions: List[str]
    executive_summary: str
    source: str = "LLM"  # "LLM" or "FALLBACK"
    timestamp: datetime = field(default_factory=datetime.now)
    raw_response: Optional[str] = None


class LLMNewsImpactAnalyzer(BaseLLMAgent):
    """
    Trading-focused news impact analysis.

    Extends news_core.GeminiNewsAnalyzer with aggregate analysis
    and actionable signal generation.
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config, logger_name="LLMNewsImpactAnalyzer")
        self._last_analysis: Optional[NewsImpactResult] = None

    def analyze_trading_impact(
        self,
        news_items: List[Dict[str, Any]],
        positions: List[Dict[str, Any]],
        upcoming_events: List[Dict[str, Any]],
        sector_performance: Dict[str, float] = None,
    ) -> NewsImpactResult:
        """
        Generate trading-focused news analysis.

        Args:
            news_items: Analyzed news from base_analyzer (NewsItem objects or dicts)
            positions: Current open positions
            upcoming_events: Calendar events (earnings, policy, etc.)
            sector_performance: {sector: avg_1d_chg%} from FyersFetcher

        Returns:
            NewsImpactResult with sector impacts, signals, restrictions
        """
        # Build prompt
        prompt = self._build_impact_prompt(news_items, positions, upcoming_events, sector_performance)

        # Get LLM response (no fallback_handler - we handle fallback ourselves)
        response = self.generate(
            prompt=prompt,
            expect_json=True,
            fallback_handler=None
        )

        # Parse response - check if we got valid LLM response
        if response.source == "LLM" and response.parsed_json:
            result = self._parse_impact_response(response.parsed_json, response.content)
        else:
            # Use fallback when LLM unavailable
            result = self._fallback_analysis(news_items, positions, upcoming_events)

        # Store result
        self._last_analysis = result
        return result

    def _build_impact_prompt(
        self,
        news_items: List[Dict[str, Any]],
        positions: List[Dict[str, Any]],
        upcoming_events: List[Dict[str, Any]],
        sector_performance: Dict[str, float] = None,
    ) -> str:
        """Construct trading impact prompt."""
        # Format news items — handles both NewsItem dataclasses and plain dicts
        news_lines = []
        for item in news_items[:15]:  # Limit to 15 items
            # NewsItem dataclass from news_core — access attributes directly
            if hasattr(item, 'headline'):
                title    = item.headline
                summary  = getattr(item, 'reasoning', '') or ''
                score    = getattr(item, 'impact_score', 0)
                severity = getattr(item, 'severity', None)
                sentiment = (
                    'BULLISH' if score > 0.1 else
                    'BEARISH' if score < -0.1 else 'NEUTRAL'
                )
                sectors_hit = getattr(item, 'sectors', [])
                news_lines.append(
                    f"- [{sentiment}] {title}"
                    + (f" [{', '.join(sectors_hit)}]" if sectors_hit else "")
                )
                if summary:
                    news_lines.append(f"  {summary[:200]}...")
            # Plain dict fallback (e.g. from regime detector news_items key)
            elif isinstance(item, dict):
                title    = item.get('title', item.get('headline', ''))
                summary  = item.get('summary', item.get('content', item.get('reasoning', '')))[:200]
                sentiment = item.get('sentiment', 'NEUTRAL')
                if title:
                    news_lines.append(f"- [{sentiment}] {title}")
                    if summary:
                        news_lines.append(f"  {summary}...")
        news_items_formatted = "\n".join(news_lines) or "No significant news in the last 24 hours"

        # Sector performance context — helps LLM assess sector-level impact
        sector_ctx = ""
        if sector_performance:
            sp_lines = [f"  {sec}: {chg:+.2f}%" for sec, chg in sector_performance.items()]
            sector_ctx = "\n\nSECTOR PERFORMANCE (today):\n" + "\n".join(sp_lines)

        # Format positions
        pos_lines = []
        for pos in positions[:10]:
            if isinstance(pos, dict):
                symbol = pos.get('symbol', 'N/A')
                qty = pos.get('quantity', 0)
                pnl = pos.get('pnl', 0)
                pos_lines.append(f"- {symbol}: {qty} shares, P&L: ₹{pnl:+,}")
        positions_formatted = "\n".join(pos_lines) or "No open positions"

        # Format events
        event_lines = []
        for event in upcoming_events[:10]:
            if isinstance(event, dict):
                name = event.get('name', event.get('title', ''))
                date_str = event.get('date', '')
                impact = event.get('impact', 'MEDIUM')
                event_lines.append(f"- [{date_str}] {name} (impact: {impact})")
        events_formatted = "\n".join(event_lines) or "No major events in the next 7 days"

        # ── FIX-NEWS-04 + FIX-PIPELINE-03: Build macro catalysts string ──────
        # Geopolitical events, crude moves, and currency stress are primary
        # drivers on macro-event days. They must reach the LLM as first-class
        # inputs — not buried inside unstructured news text — so the magnitude
        # classifier rates them HIGH instead of defaulting to LOW/NEUTRAL.
        # Coordinator injects macro scalars as __-prefixed keys in sector_performance.
        macro_catalyst_lines = []
        if sector_performance:
            crude = (
                sector_performance.get("__crude_brent_usd")
                or sector_performance.get("crude_brent_usd")
                or sector_performance.get("crude_usd")
            )
            if crude and crude > 0:
                macro_catalyst_lines.append(f"- Brent Crude: ${crude:.1f}/bbl")
            usd_inr = (
                sector_performance.get("__usd_inr")
                or sector_performance.get("usd_inr")
            )
            if usd_inr and usd_inr > 0:
                macro_catalyst_lines.append(f"- USD/INR: ₹{usd_inr:.2f}")
            gift_gap = (
                sector_performance.get("__gift_nifty_gap_pct")
                or sector_performance.get("gift_nifty_gap_pct")
            )
            if gift_gap is not None and gift_gap != 0:
                macro_catalyst_lines.append(f"- Gift Nifty Overnight Gap: {gift_gap:+.2f}%")
        # Include any HIGH/CRITICAL geopolitical events from news_items
        # severity is an ImpactSeverity enum — compare by .name not string equality
        for item in news_items[:15]:
            sev = getattr(item, "severity", None) or (item.get("severity") if isinstance(item, dict) else None)
            if sev is not None:
                sev_name = sev.name if hasattr(sev, "name") else str(sev).upper()
                if sev_name in ("HIGH", "CRITICAL", "EXTREME"):
                    title = getattr(item, "headline", None) or (item.get("title", "") if isinstance(item, dict) else str(item))
                    macro_catalyst_lines.append(f"- ⚡ {sev_name} EVENT: {title[:120]}")
        macro_catalysts = "\n".join(macro_catalyst_lines) or "No macro catalysts identified today"

        return NEWS_IMPACT_PROMPT.format(
            news_items_formatted=news_items_formatted + sector_ctx,
            macro_catalysts=macro_catalysts,
            positions_formatted=positions_formatted,
            events_formatted=events_formatted
        )

    def _parse_impact_response(
        self,
        parsed: Dict[str, Any],
        raw_response: str
    ) -> NewsImpactResult:
        """Parse JSON response into structured result."""
        try:
            # Parse overall market impact
            overall = parsed.get("overall_market_impact", {})
            overall_market_impact = {
                "direction": overall.get("direction", "NEUTRAL"),
                "magnitude": overall.get("magnitude", "LOW"),
                "confidence": int(overall.get("confidence", 50))
            }

            # Parse sector impacts
            sector_impacts = {}
            for sector, data in parsed.get("sector_impacts", {}).items():
                sector_impacts[sector] = SectorImpact(
                    impact_score=float(data.get("impact_score", 0)),
                    reason=data.get("reason", "")
                )

            # Parse index impacts
            index_impacts = {}
            for index, data in parsed.get("index_impacts", {}).items():
                index_impacts[index] = IndexImpact(
                    expected_gap=data.get("expected_gap", "0"),
                    probability=float(data.get("probability", 0.5))
                )

            # Parse actionable signals
            actionable_signals = []
            for sig in parsed.get("actionable_signals", []):
                actionable_signals.append(ActionableSignal(
                    signal_type=sig.get("signal_type", ""),
                    target=sig.get("target", ""),
                    reason=sig.get("reason", "")
                ))

            # Parse risk events
            risk_events = []
            for evt in parsed.get("risk_events", []):
                risk_events.append(RiskEvent(
                    event_name=evt.get("event_name", ""),
                    event_date=evt.get("event_date", ""),
                    impact_description=evt.get("impact_description", ""),
                    recommendation=evt.get("recommendation", "")
                ))

            return NewsImpactResult(
                overall_market_impact=overall_market_impact,
                sector_impacts=sector_impacts,
                index_impacts=index_impacts,
                actionable_signals=actionable_signals,
                risk_events=risk_events,
                trade_restrictions=parsed.get("trade_restrictions", []),
                executive_summary=parsed.get("executive_summary", ""),
                source="LLM",
                raw_response=raw_response
            )

        except Exception as e:
            self.logger.error(f"Failed to parse news impact response: {e}")
            return NewsImpactResult(
                overall_market_impact={"direction": "NEUTRAL", "magnitude": "LOW", "confidence": 50},
                sector_impacts={},
                index_impacts={},
                actionable_signals=[],
                risk_events=[],
                trade_restrictions=[],
                executive_summary="Analysis parsing failed",
                source="FALLBACK"
            )

    def _fallback_analysis(
        self,
        news_items: List[Dict[str, Any]],
        positions: List[Dict[str, Any]],
        upcoming_events: List[Dict[str, Any]]
    ) -> NewsImpactResult:
        """Fallback analysis when LLM is unavailable."""
        self.logger.info("Using fallback news impact analysis")

        # Simple keyword-based analysis
        bullish_keywords = ['rally', 'surge', 'gain', 'positive', 'growth', 'upgrade', 'buyback']
        bearish_keywords = ['crash', 'fall', 'decline', 'negative', 'loss', 'downgrade', 'concern']

        bullish_count = 0
        bearish_count = 0

        for item in news_items:
            if isinstance(item, dict):
                text = f"{item.get('title', '')} {item.get('summary', '')}".lower()
                bullish_count += sum(1 for kw in bullish_keywords if kw in text)
                bearish_count += sum(1 for kw in bearish_keywords if kw in text)

        if bullish_count > bearish_count * 1.5:
            direction = "BULLISH"
        elif bearish_count > bullish_count * 1.5:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        magnitude = "HIGH" if abs(bullish_count - bearish_count) > 5 else "MEDIUM" if abs(bullish_count - bearish_count) > 2 else "LOW"

        return NewsImpactResult(
            overall_market_impact={
                "direction": direction,
                "magnitude": magnitude,
                "confidence": 40
            },
            sector_impacts={},
            index_impacts={},
            actionable_signals=[],
            risk_events=[],
            trade_restrictions=[],
            executive_summary=f"Basic keyword analysis: {bullish_count} bullish signals, {bearish_count} bearish signals. LLM analysis unavailable.",
            source="FALLBACK"
        )

    def get_last_analysis(self) -> Optional[NewsImpactResult]:
        """Get the most recent analysis."""
        return self._last_analysis
