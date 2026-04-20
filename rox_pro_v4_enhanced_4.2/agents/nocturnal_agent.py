# nocturnal_agent.py
"""
ROX Proven Edge Engine v4.1 - NOCTURNAL Agent
==============================================
Overnight Risk Assessment Agent - Pre-market briefing specialist.

Integrates with News Intelligence Layer to provide:
- Overnight gap risk assessment
- Pre-market trading restrictions
- Geopolitical risk monitoring
- Gap probability estimation

Weight: 0.20 (high weight for risk management)
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, time

from agents.base_agent import BaseAgent, AgentVerdict, AgentReport
from agents.news_core import NewsContextProvider, OvernightRiskProfile, ImpactSeverity
from config import TradeDirection, MarketRegime


class NocturnalAgent(BaseAgent):
    """
    NOCTURNAL - Overnight Risk Assessment Agent

    Specialized agent for pre-market risk assessment using news intelligence.
    Runs at 9:00 AM IST to evaluate overnight developments and set trading guardrails.

    Baseline weight: 20%
    """

    def __init__(self):
        super().__init__(
            name="NOCTURNAL",
            domain="Overnight Risk Assessment",
            baseline_weight=0.20
        )
        self.news_context = NewsContextProvider()
        self._last_briefing: Optional[OvernightRiskProfile] = None

    def analyze(self, data: Dict[str, Any], regime: MarketRegime) -> AgentReport:
        """
        Perform overnight risk analysis.

        Args:
            data: Should contain:
                - 'market_context': Current market data (nifty_price, vix, etc.)
                - 'portfolio_status': Current positions
                - 'force_refresh': bool to force news update

        Returns:
            AgentReport with risk assessment and trading restrictions
        """
        market_context = data.get("market_context", {})
        portfolio = data.get("portfolio_status", {})

        # Get overnight risk profile from news context
        risk_profile = self.news_context.get_overnight_risk()

        # If no profile or forced refresh, generate one
        if risk_profile is None or data.get("force_refresh"):
            # This would normally be async, but we use cached version
            risk_profile = self._generate_fallback_profile(market_context)

        self._last_briefing = risk_profile

        # Generate verdict based on risk level
        verdict = self._generate_verdict(risk_profile, regime)

        # Calculate position sizing multiplier
        sizing_multiplier = self._get_sizing_multiplier(risk_profile.risk_level)

        # Identify affected positions
        affected_positions = self._identify_affected_positions(
            portfolio.get("open_positions", []),
            risk_profile.affected_sectors
        )

        return AgentReport(
            agent_name=self.name,
            verdict=verdict,
            analysis_details={
                "risk_profile": risk_profile.to_agent_context(),
                "sizing_multiplier": sizing_multiplier,
                "affected_positions": affected_positions,
                "key_headlines": risk_profile.key_headlines[:5],
                "gap_probability": risk_profile.gap_probability,
                "expected_gap": risk_profile.expected_gap_size
            },
            key_observations=self._generate_observations(risk_profile),
            metrics={
                "risk_score": self._risk_level_to_score(risk_profile.risk_level),
                "confidence": risk_profile.confidence,
                "gap_prob": risk_profile.gap_probability,
                "news_count": len(self.news_context.get_critical_news())
            },
            raw_data={
                "risk_profile": risk_profile.to_agent_context(),
                "all_news": [n.to_dict() for n in self.news_context.get_all_news()[:10]]
            }
        )

    def _generate_verdict(self, profile: OvernightRiskProfile, regime: MarketRegime) -> AgentVerdict:
        """Generate trading verdict from risk profile"""

        # Map risk level to direction and conviction
        risk_direction_map = {
            "EXTREME": (TradeDirection.NEUTRAL, 20),
            "HIGH": (TradeDirection.NEUTRAL, 35),
            "ELEVATED": (TradeDirection.NEUTRAL, 50),
            "NORMAL": (TradeDirection.LONG if profile.market_stance == "LONG" else TradeDirection.NEUTRAL, 65),
            "LOW": (TradeDirection.LONG, 75)
        }

        direction, base_conviction = risk_direction_map.get(
            profile.risk_level, (TradeDirection.NEUTRAL, 50)
        )

        # Override with profile stance if strong
        if profile.market_stance == "LONG" and profile.confidence > 70:
            direction = TradeDirection.LONG
        elif profile.market_stance == "SHORT" and profile.confidence > 70:
            direction = TradeDirection.SHORT
        elif profile.market_stance == "CASH":
            direction = TradeDirection.NEUTRAL
            base_conviction = 10

        # Adjust for regime
        if regime == MarketRegime.BEAR and direction == TradeDirection.LONG:
            base_conviction -= 10
        elif regime == MarketRegime.BULL and direction == TradeDirection.SHORT:
            base_conviction -= 10

        conviction = max(0, min(100, base_conviction))

        # Build reason
        reason_parts = [
            f"Risk Level: {profile.risk_level}",
            f"Gap Prob: {profile.gap_probability:.0%}",
            f"Expected: {profile.expected_gap_size}"
        ]
        if profile.key_headlines:
            reason_parts.append(f"Key: {profile.key_headlines[0][:50]}...")

        # Build risks
        risks = []
        if profile.risk_level in ["EXTREME", "HIGH"]:
            risks.append(f"High overnight risk: {profile.risk_level}")
        if profile.gap_probability > 0.5:
            risks.append(f"High gap probability: {profile.gap_probability:.0%}")
        if "HALT_ALL_NEW_POSITIONS" in profile.trading_restrictions:
            risks.append("Trading halted by risk management")

        return AgentVerdict(
            direction=direction,
            conviction=conviction,
            weight=self.current_weight,
            reason=" | ".join(reason_parts),
            risks=risks
        )

    def _get_sizing_multiplier(self, risk_level: str) -> float:
        """Get position sizing multiplier based on risk"""
        multipliers = {
            "EXTREME": 0.0,  # No new positions
            "HIGH": 0.25,  # 25% of normal size
            "ELEVATED": 0.5,  # 50% of normal size
            "NORMAL": 1.0,  # Normal size
            "LOW": 1.25  # Can increase size
        }
        return multipliers.get(risk_level, 1.0)

    def _identify_affected_positions(self, positions: List[Dict],
                                     sector_impacts: Dict[str, float]) -> List[Dict]:
        """Identify which current positions are affected by news"""
        affected = []

        for pos in positions:
            sector = pos.get("sector", "")
            symbol = pos.get("symbol", "")

            # Check sector impact
            sector_impact = sector_impacts.get(sector, 0)

            # Check symbol-specific news
            symbol_context = self.news_context.get_symbol_context(symbol)

            if abs(sector_impact) > 0.3 or abs(symbol_context.get("score", 0)) > 0.3:
                affected.append({
                    "symbol": symbol,
                    "sector": sector,
                    "sector_impact": sector_impact,
                    "news_impact": symbol_context.get("score", 0),
                    "recommendation": "REDUCE" if sector_impact < -0.5 else "HOLD"
                })

        return affected

    def _risk_level_to_score(self, risk_level: str) -> int:
        """Convert risk level to numeric score"""
        scores = {"EXTREME": 10, "HIGH": 7, "ELEVATED": 5, "NORMAL": 3, "LOW": 1}
        return scores.get(risk_level, 5)

    def _generate_observations(self, profile: OvernightRiskProfile) -> List[str]:
        """Generate key observations"""
        obs = []

        obs.append(f"Overnight Risk: {profile.risk_level} (confidence: {profile.confidence}%)")
        obs.append(f"Gap Assessment: {profile.expected_gap_size} (prob: {profile.gap_probability:.0%})")

        if profile.trading_restrictions:
            obs.append(f"Restrictions: {', '.join(profile.trading_restrictions[:2])}")

        if profile.key_headlines:
            obs.append(f"Top News: {profile.key_headlines[0][:60]}...")

        if profile.affected_sectors:
            top_sector = max(profile.affected_sectors.items(), key=lambda x: abs(x[1]))
            obs.append(f"Key Sector: {top_sector[0]} ({top_sector[1]:+.2f})")

        return obs

    def _generate_fallback_profile(self, market_context: Dict) -> OvernightRiskProfile:
        """Generate fallback profile when news unavailable"""
        return OvernightRiskProfile(
            risk_level="NORMAL",
            market_stance="NEUTRAL",
            confidence=50,
            gap_probability=0.1,
            expected_gap_size="±50 points",
            key_headlines=[],
            affected_sectors={},
            trading_restrictions=[],
            narrative="News intelligence unavailable - using default risk parameters"
        )

    def get_trading_restrictions(self) -> List[str]:
        """Get current trading restrictions"""
        if self._last_briefing:
            return self._last_briefing.trading_restrictions
        return []

    def should_halt_trading(self) -> bool:
        """Check if trading should be halted"""
        if self._last_briefing:
            return self._last_briefing.risk_level == "EXTREME"
        return False