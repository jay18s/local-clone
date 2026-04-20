"""
ROX Proven Edge Engine v4.1 - CATALYST Agent (Enhanced)
========================================================
Event Calendar Agent - Upcoming events and their market impact.

v4.1 Enhancement: Integrates with News Intelligence Layer (news_core.py)
to auto-inject live news-derived events into the event calendar, incorporate
overnight risk trading restrictions, and upgrade the event impact scoring
with Gemini-powered analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import logging

from .base_agent import BaseAgent, AgentVerdict, AgentReport
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TradeDirection, MarketRegime

logger = logging.getLogger("rox.catalyst")


class EventCategory(Enum):
    """Event categories"""
    CORPORATE = "CORPORATE"
    SECTOR = "SECTOR"
    MACRO_INDIA = "MACRO_INDIA"
    GLOBAL = "GLOBAL"
    NEWS_DERIVED = "NEWS_DERIVED"   # v4.1: events auto-injected from news_core


class EventImpact(Enum):
    """Event impact levels"""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    MINIMAL = "MINIMAL"


class CatalystVerdict(Enum):
    """CATALYST specific verdicts"""
    CLEAR = "CLEAR"
    CAUTION = "CAUTION"
    HIGH_ALERT = "HIGH_ALERT"


@dataclass
class Event:
    """Market event"""
    name: str
    date: datetime
    category: EventCategory
    impact: EventImpact
    affected_sectors: List[str] = field(default_factory=list)
    affected_stocks: List[str] = field(default_factory=list)
    description: str = ""
    probability_consensus: float = 0.0  # For events with expected outcomes
    # v4.1 additions
    source_headline: str = ""           # Original news headline (if news-derived)
    severity_score: float = 0.0        # Gemini severity score (-1..+1)


@dataclass
class Scenario:
    """Event scenario"""
    name: str
    probability: float
    expected_move: float  # Expected % move
    trade_action: str
    trigger_condition: str


class CatalystAgent(BaseAgent):
    """
    CATALYST - Event Calendar Agent (v4.1)

    Tracks upcoming events, their expected impact, and provides
    pre/post-event positioning guidance.

    v4.1: Enriches event calendar with live news items from the News
    Intelligence Layer and respects NOCTURNAL trading restrictions.

    Baseline weight: 10%
    Weight increases when major events are approaching.
    """

    # High impact macro events
    HIGH_IMPACT_MACRO = [
        "RBI_MONETARY_POLICY",
        "UNION_BUDGET",
        "FED_FOMC",
        "GDP_DATA",
        "CPI_INFLATION"
    ]

    # Pre-event trade management rules
    HIGH_IMPACT_REDUCTION = 0.50  # Reduce positions by 50%
    MEDIUM_IMPACT_REDUCTION = 0.25
    PRE_EVENT_DAYS = {
        EventImpact.HIGH: 2,
        EventImpact.MEDIUM: 1,
        EventImpact.LOW: 0
    }

    def __init__(self):
        super().__init__(
            name="CATALYST",
            domain="Event Calendar",
            baseline_weight=0.10
        )
        self.event_calendar: List[Event] = []

    def analyze(self, data: Dict[str, Any], regime: MarketRegime) -> AgentReport:
        """
        Perform event calendar analysis.

        Args:
            data: Should contain:
                - 'events': List of Event objects or dicts
                - 'current_date': Current date for event proximity
                - 'stock': Stock being analyzed (optional)
                Optional:
                - 'earnings_season': Whether in earnings season
                - 'expiry_week': Whether in F&O expiry week
                - 'news_context': Live news data from news_core (v4.1)
                - 'overnight_risk': Overnight risk profile from NOCTURNAL (v4.1)

        Returns:
            AgentReport with event analysis verdict
        """
        current_date = data.get('current_date', datetime.now())
        if isinstance(current_date, str):
            current_date = datetime.fromisoformat(current_date)

        # Parse events from calendar
        events = self._parse_events(data.get('events', []))

        # v4.1: Inject news-derived events from News Intelligence Layer
        news_context_data = data.get("news_context", [])
        overnight_risk = data.get("overnight_risk", None)

        news_events = self._extract_news_events(news_context_data, current_date)
        events = events + news_events

        self.event_calendar = events

        stock = data.get('stock', '')

        # Filter upcoming events (next 10 trading days ~ 14 calendar days)
        upcoming_events = self._get_upcoming_events(events, current_date, days=14)

        # Categorize by impact and proximity
        event_analysis = self._analyze_events(upcoming_events, current_date, stock)

        # Check for high-alert events
        high_alert_events = self._get_high_alert_events(upcoming_events, current_date)

        # v4.1: Check overnight risk for additional restrictions
        overnight_restrictions = self._get_overnight_restrictions(overnight_risk)

        # Determine overall verdict
        catalyst_verdict = self._determine_catalyst_verdict(
            upcoming_events, high_alert_events, current_date, overnight_risk
        )

        # Generate scenario planning for high-impact events
        scenarios = []
        for event in high_alert_events:
            scenarios.extend(self._create_scenarios(event))

        # Generate trade recommendations
        trade_recommendations = self._generate_trade_recommendations(
            catalyst_verdict, high_alert_events, data.get('expiry_week', False),
            overnight_restrictions
        )

        # Generate verdict
        verdict = self._generate_verdict(
            catalyst_verdict, event_analysis, high_alert_events, regime
        )

        # Build key observations
        key_observations = self._generate_observations(
            upcoming_events, high_alert_events, event_analysis
        )

        # Add v4.1 news-intelligence observations
        if news_events:
            key_observations.append(
                f"News Intelligence: {len(news_events)} live news event(s) auto-injected"
            )
        if overnight_restrictions:
            for r in overnight_restrictions[:2]:
                key_observations.append(f"🚫 Restriction: {r}")

        return AgentReport(
            agent_name=self.name,
            verdict=verdict,
            analysis_details={
                "verdict_type": catalyst_verdict.value,
                "upcoming_events_count": len(upcoming_events),
                "high_alert_count": len(high_alert_events),
                "news_derived_events": len(news_events),
                "events_next_5_days": len([e for e in upcoming_events
                                          if (e.date - current_date).days <= 5]),
                "scenarios": [s.__dict__ for s in scenarios[:3]],
                "trade_recommendations": trade_recommendations,
                "overnight_restrictions": overnight_restrictions,
            },
            key_observations=key_observations,
            metrics={
                "events_next_10_days": len(upcoming_events),
                "high_impact_events": len([e for e in upcoming_events
                                          if e.impact == EventImpact.HIGH]),
                "days_to_next_major": self._days_to_next_major_event(upcoming_events, current_date)
            },
            raw_data={
                "upcoming_events": [
                    {
                        "name": e.name,
                        "date": e.date.isoformat(),
                        "impact": e.impact.value,
                        "category": e.category.value
                    }
                    for e in upcoming_events[:10]
                ]
            }
        )

    # ------------------------------------------------------------------
    # v4.1 — News Intelligence helpers
    # ------------------------------------------------------------------

    def _extract_news_events(
        self, news_context_data: Any, current_date: datetime
    ) -> List[Event]:
        """
        Convert NewsItem objects from news_core into Event objects so they
        appear in the CATALYST event calendar and influence the verdict.

        Only CRITICAL and HIGH severity items are injected — lower severity
        items are too noisy for the event calendar.
        """
        events: List[Event] = []
        if not news_context_data:
            return events

        items = news_context_data if isinstance(news_context_data, list) else []
        for item in items:
            try:
                # Support both NewsItem objects and plain dicts
                if hasattr(item, "severity"):
                    severity_val = item.severity.value if hasattr(item.severity, "value") else 0
                    if severity_val < 3:   # Only CRITICAL(4) and HIGH(3)
                        continue
                    headline = item.headline
                    impact_score = item.impact_score  # -1..+1
                    published = item.published
                    sectors = item.sectors or []
                    symbols = item.symbols or []
                elif isinstance(item, dict):
                    severity_val = item.get("severity", 0)
                    if isinstance(severity_val, str):
                        severity_val = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0}.get(
                            severity_val.upper(), 0
                        )
                    if severity_val < 3:
                        continue
                    headline = item.get("headline", "Unknown")
                    impact_score = item.get("impact_score", 0)
                    published = item.get("published", current_date)
                    sectors = item.get("sectors", [])
                    symbols = item.get("symbols", [])
                else:
                    continue

                # Map Gemini severity → EventImpact
                event_impact = EventImpact.HIGH if severity_val >= 4 else EventImpact.MEDIUM

                events.append(Event(
                    name=f"[NEWS] {headline[:80]}",
                    date=published if isinstance(published, datetime) else current_date,
                    category=EventCategory.NEWS_DERIVED,
                    impact=event_impact,
                    affected_sectors=sectors[:3],
                    affected_stocks=symbols[:5],
                    description=headline,
                    source_headline=headline,
                    severity_score=impact_score,
                ))
            except Exception as e:
                logger.debug(f"CATALYST: failed to parse news item: {e}")
                continue

        return events

    def _get_overnight_restrictions(self, overnight_risk: Any) -> List[str]:
        """Extract trading restrictions from the overnight risk profile."""
        if overnight_risk is None:
            return []
        try:
            restrictions = getattr(overnight_risk, "trading_restrictions", [])
            return list(restrictions) if restrictions else []
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Core event calendar methods
    # ------------------------------------------------------------------

    def _parse_events(self, events_data: List) -> List[Event]:
        """Parse events from list of dicts"""
        events = []
        for e in events_data:
            if isinstance(e, Event):
                events.append(e)
            elif isinstance(e, dict):
                date = e.get('date')
                if isinstance(date, str):
                    date = datetime.fromisoformat(date)
                else:
                    date = date or datetime.now()

                impact_str = e.get('impact', 'LOW')
                impact = EventImpact[impact_str] if impact_str in EventImpact.__members__ else EventImpact.LOW

                category_str = e.get('category', 'CORPORATE')
                category = EventCategory[category_str] if category_str in EventCategory.__members__ else EventCategory.CORPORATE

                events.append(Event(
                    name=e.get('name', 'Unknown Event'),
                    date=date,
                    category=category,
                    impact=impact,
                    affected_sectors=e.get('affected_sectors', []),
                    affected_stocks=e.get('affected_stocks', []),
                    description=e.get('description', ''),
                    probability_consensus=e.get('probability_consensus', 0)
                ))
        return events

    def _get_upcoming_events(self, events: List[Event],
                            current_date: datetime,
                            days: int = 14) -> List[Event]:
        """Get events in the next N days"""
        cutoff = current_date + timedelta(days=days)
        upcoming = [e for e in events if current_date <= e.date <= cutoff]
        return sorted(upcoming, key=lambda x: x.date)

    def _analyze_events(self, events: List[Event], current_date: datetime,
                       stock: str = '') -> Dict:
        """Analyze events for impact"""
        analysis = {
            "by_impact": {impact.value: [] for impact in EventImpact},
            "by_category": {cat.value: [] for cat in EventCategory},
            "stock_specific": [],
            "sector_specific": []
        }

        for event in events:
            analysis["by_impact"][event.impact.value].append(event.name)
            analysis["by_category"][event.category.value].append(event.name)

            if stock and stock in event.affected_stocks:
                analysis["stock_specific"].append(event.name)

            for sector in event.affected_sectors:
                analysis["sector_specific"].append(f"{event.name} ({sector})")

        return analysis

    def _get_high_alert_events(self, events: List[Event],
                               current_date: datetime) -> List[Event]:
        """Get events requiring high alert"""
        high_alert = []

        for event in events:
            days_to_event = (event.date - current_date).days

            # High impact within 2 days
            if event.impact == EventImpact.HIGH and days_to_event <= 2:
                high_alert.append(event)
            # Medium impact within 1 day
            elif event.impact == EventImpact.MEDIUM and days_to_event <= 1:
                high_alert.append(event)
            # Any corporate event for specific stock within 3 days
            elif event.category == EventCategory.CORPORATE and days_to_event <= 3:
                high_alert.append(event)
            # v4.1: News-derived HIGH events are always high alert
            elif event.category == EventCategory.NEWS_DERIVED and event.impact == EventImpact.HIGH:
                high_alert.append(event)

        return high_alert

    def _determine_catalyst_verdict(self, events: List[Event],
                                   high_alert: List[Event],
                                   current_date: datetime,
                                   overnight_risk: Any = None) -> CatalystVerdict:
        """Determine overall CATALYST verdict"""
        # v4.1: Overnight EXTREME/HIGH risk forces HIGH_ALERT
        if overnight_risk is not None:
            risk_level = str(getattr(overnight_risk, "risk_level", "NORMAL")).upper()
            if risk_level in ("EXTREME", "HIGH"):
                return CatalystVerdict.HIGH_ALERT
            elif risk_level == "ELEVATED":
                return CatalystVerdict.CAUTION

        if not events:
            return CatalystVerdict.CLEAR

        # High alert events within 48 hours
        for event in high_alert:
            days_to_event = (event.date - current_date).days
            if event.impact == EventImpact.HIGH and days_to_event <= 2:
                return CatalystVerdict.HIGH_ALERT

        # Medium impact events within 3-5 days
        for event in events:
            days_to_event = (event.date - current_date).days
            if event.impact in [EventImpact.HIGH, EventImpact.MEDIUM] and days_to_event <= 5:
                return CatalystVerdict.CAUTION

        return CatalystVerdict.CLEAR

    def _create_scenarios(self, event: Event) -> List[Scenario]:
        """Create scenario planning for high-impact event"""
        scenarios = []

        if event.category == EventCategory.MACRO_INDIA:
            if "RBI" in event.name:
                scenarios = [
                    Scenario(
                        name="Rate Cut",
                        probability=0.25,
                        expected_move=2.0,
                        trade_action="Banking, Realty longs; reduce IT",
                        trigger_condition="RBI cuts repo rate"
                    ),
                    Scenario(
                        name="Status Quo",
                        probability=0.50,
                        expected_move=0.5,
                        trade_action="Normal operations",
                        trigger_condition="RBI maintains rate"
                    ),
                    Scenario(
                        name="Hawkish Pause",
                        probability=0.25,
                        expected_move=-1.5,
                        trade_action="Reduce rate-sensitive positions",
                        trigger_condition="RBI signals future hikes"
                    )
                ]
            elif "BUDGET" in event.name.upper():
                scenarios = [
                    Scenario(
                        name="Growth-Focused",
                        probability=0.35,
                        expected_move=3.0,
                        trade_action="Infra, Capex, Defence longs",
                        trigger_condition="Strong capex allocation"
                    ),
                    Scenario(
                        name="Populist",
                        probability=0.35,
                        expected_move=-1.0,
                        trade_action="Reduce positions, wait for clarity",
                        trigger_condition="High welfare spending"
                    ),
                    Scenario(
                        name="Neutral",
                        probability=0.30,
                        expected_move=0.5,
                        trade_action="Selective stock picks",
                        trigger_condition="Status quo budget"
                    )
                ]

        elif event.category == EventCategory.GLOBAL:
            if "FED" in event.name.upper():
                scenarios = [
                    Scenario(
                        name="Dovish",
                        probability=0.30,
                        expected_move=1.5,
                        trade_action="FII-favored stocks long",
                        trigger_condition="Fed signals rate cuts"
                    ),
                    Scenario(
                        name="Hawkish",
                        probability=0.30,
                        expected_move=-2.0,
                        trade_action="Reduce positions, hedge",
                        trigger_condition="Fed maintains hawkish stance"
                    ),
                    Scenario(
                        name="Neutral",
                        probability=0.40,
                        expected_move=0.5,
                        trade_action="Normal operations",
                        trigger_condition="Fed delivers as expected"
                    )
                ]

        elif event.category == EventCategory.CORPORATE:
            scenarios = [
                Scenario(
                    name="Beat Estimates",
                    probability=0.35,
                    expected_move=3.0,
                    trade_action="Continue/add to position",
                    trigger_condition="Results beat by >5%"
                ),
                Scenario(
                    name="In-Line",
                    probability=0.40,
                    expected_move=0.5,
                    trade_action="Hold position",
                    trigger_condition="Results as expected"
                ),
                Scenario(
                    name="Miss Estimates",
                    probability=0.25,
                    expected_move=-4.0,
                    trade_action="Exit or reduce position",
                    trigger_condition="Results miss by >5%"
                )
            ]

        elif event.category == EventCategory.NEWS_DERIVED:
            # v4.1: Generic geopolitical/news scenarios
            severity = event.severity_score
            if severity < -0.3:
                scenarios = [
                    Scenario(
                        name="Risk-Off Escalation",
                        probability=0.50,
                        expected_move=-2.0,
                        trade_action="Reduce all positions; hedge with puts",
                        trigger_condition="Negative news escalates"
                    ),
                    Scenario(
                        name="News Priced In",
                        probability=0.50,
                        expected_move=0.5,
                        trade_action="Wait for technicals to stabilise",
                        trigger_condition="Market absorbs headline risk"
                    )
                ]
            else:
                scenarios = [
                    Scenario(
                        name="Positive Catalyst",
                        probability=0.50,
                        expected_move=1.5,
                        trade_action="Look for long entries on sector beneficiaries",
                        trigger_condition="News drives sector rotation"
                    ),
                    Scenario(
                        name="Fades",
                        probability=0.50,
                        expected_move=-0.5,
                        trade_action="Normal stop management",
                        trigger_condition="Initial reaction fades"
                    )
                ]

        return scenarios

    def _generate_trade_recommendations(self, verdict: CatalystVerdict,
                                       high_alert: List[Event],
                                       expiry_week: bool,
                                       overnight_restrictions: List[str] = None) -> List[str]:
        """Generate specific trade recommendations"""
        recommendations = []

        if verdict == CatalystVerdict.HIGH_ALERT:
            recommendations.append("Reduce all affected positions by 50%")
            recommendations.append("Tighten stop losses by 50%")
            recommendations.append("No new entries within 24 hours of event")
            recommendations.append("Prepare both bull and bear case scenarios")
        elif verdict == CatalystVerdict.CAUTION:
            recommendations.append("Reduce affected positions by 25%")
            recommendations.append("No new entries 24 hours before event")
            recommendations.append("Ensure all stops are in place")
        else:
            recommendations.append("Normal trading operations")
            recommendations.append("Standard stop loss management")

        if expiry_week:
            recommendations.append("F&O expiry week - expect volatility")

        # v4.1: Append overnight restrictions
        if overnight_restrictions:
            for r in overnight_restrictions:
                recommendations.append(f"NOCTURNAL: {r}")

        return recommendations

    def _days_to_next_major_event(self, events: List[Event],
                                  current_date: datetime) -> int:
        """Get days until next major event"""
        for event in sorted(events, key=lambda x: x.date):
            if event.impact == EventImpact.HIGH:
                return (event.date - current_date).days
        return 99  # No major events soon

    def _generate_verdict(self, catalyst_verdict: CatalystVerdict,
                         event_analysis: Dict, high_alert: List[Event],
                         regime: MarketRegime) -> AgentVerdict:
        """Generate the final verdict.

        CATALYST is an event-risk gate, not a directional agent.
        Direction is always NEUTRAL — CATALYST only signals how much event risk
        is present, never which direction to trade.

        CLEAR conviction is scaled by how far away the nearest event is:
          - 10+ days clear → conviction 70 (very clear runway)
          - 5–9 days clear → conviction 65
          - <5 days clear  → conviction 60 (close but below caution threshold)
        This avoids the NEUTRAL+80 combination that the cross-examiner was
        correctly flagging as "strongly convinced of nothing" (defective reasoning).
        """
        if catalyst_verdict == CatalystVerdict.CLEAR:
            direction = TradeDirection.NEUTRAL
            days_to_next = event_analysis.get("days_to_next_major", 99)
            if days_to_next >= 10:
                conviction = 70   # Wide clear runway
            elif days_to_next >= 5:
                conviction = 65   # Moderate runway
            else:
                conviction = 60   # Close but still below CAUTION threshold
            reason = f"No major events in next 5 days - clear for trading (nearest={days_to_next}d)"
        elif catalyst_verdict == CatalystVerdict.CAUTION:
            direction = TradeDirection.NEUTRAL
            conviction = 50
            reason = f"Medium impact event(s) approaching - reduce exposure"
        else:  # HIGH_ALERT
            direction = TradeDirection.NEUTRAL  # Block or reduce trades
            conviction = 20
            reason = f"HIGH ALERT: {len(high_alert)} high-impact event(s) within 48 hours"

        # Regime adjustment
        if regime == MarketRegime.BEAR and catalyst_verdict == CatalystVerdict.HIGH_ALERT:
            conviction -= 10  # Extra cautious in bear markets

        # Clamp conviction
        conviction = max(0, min(100, conviction))

        # Generate risks
        risks = []
        if high_alert:
            risks.append(f"{len(high_alert)} high-alert event(s) pending")
        if event_analysis.get("stock_specific"):
            risks.append(f"Stock-specific events: {', '.join(event_analysis['stock_specific'][:2])}")
        news_derived = event_analysis.get("by_category", {}).get("NEWS_DERIVED", [])
        if news_derived:
            risks.append(f"{len(news_derived)} live news event(s) flagged")

        return AgentVerdict(
            direction=direction,
            conviction=conviction,
            weight=self.current_weight,
            reason=reason,
            risks=risks
        )

    def _generate_observations(self, events: List[Event], high_alert: List[Event],
                              event_analysis: Dict) -> List[str]:
        """Generate key observations"""
        observations = []

        # Upcoming events summary
        if events:
            observations.append(f"{len(events)} events in next 10 trading days")
        else:
            observations.append("No significant events in next 10 days - CLEAR")

        # High alert events
        for event in high_alert[:3]:
            observations.append(f"HIGH ALERT: {event.name} ({event.date.strftime('%Y-%m-%d')})")

        # Impact breakdown
        high_count = len(event_analysis["by_impact"]["HIGH"])
        medium_count = len(event_analysis["by_impact"]["MEDIUM"])
        if high_count > 0 or medium_count > 0:
            observations.append(f"High impact: {high_count} | Medium impact: {medium_count}")

        # Category breakdown
        macro_count = len(event_analysis["by_category"]["MACRO_INDIA"])
        global_count = len(event_analysis["by_category"]["GLOBAL"])
        corporate_count = len(event_analysis["by_category"]["CORPORATE"])
        news_count = len(event_analysis["by_category"].get("NEWS_DERIVED", []))

        if macro_count > 0:
            observations.append(f"Macro events: {macro_count}")
        if global_count > 0:
            observations.append(f"Global events: {global_count}")
        if corporate_count > 0:
            observations.append(f"Corporate events: {corporate_count}")
        if news_count > 0:
            observations.append(f"Live news events: {news_count}")

        return observations

    def add_event(self, event: Event):
        """Add event to calendar"""
        self.event_calendar.append(event)

    def get_next_event(self, category: EventCategory = None) -> Optional[Event]:
        """Get next upcoming event, optionally filtered by category"""
        now = datetime.now()
        future_events = [e for e in self.event_calendar if e.date >= now]

        if category:
            future_events = [e for e in future_events if e.category == category]

        if future_events:
            return min(future_events, key=lambda x: x.date)
        return None
