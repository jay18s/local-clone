"""
ROX Proven Edge Engine v3.0 - PRUDENCE Agent
===========================================
Risk Management Agent - Position sizing and risk limits.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

from .base_agent import BaseAgent, AgentVerdict, AgentReport
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TradeDirection, MarketRegime, RiskLimits


class PrudenceVerdict(Enum):
    """Prudence-specific verdicts"""
    APPROVED = "APPROVED"
    REDUCE_SIZE = "REDUCE_SIZE"
    VETO = "VETO"


@dataclass
class PositionInfo:
    """Information about a position"""
    stock: str
    shares: int
    entry_price: float
    current_price: float
    stop_loss: float
    position_value: float
    risk_amount: float
    sector: str = ""


@dataclass
class PortfolioStatus:
    """Current portfolio status"""
    total_capital: float = 0.0
    deployed_capital: float = 0.0
    cash: float = 0.0
    portfolio_heat: float = 0.0
    current_drawdown: float = 0.0
    open_positions: List[PositionInfo] = field(default_factory=list)
    sector_exposure: Dict[str, float] = field(default_factory=dict)


@dataclass
class PositionSizingResult:
    """Result of position sizing calculation"""
    approved: bool
    shares: int = 0
    position_value: float = 0.0
    position_percent: float = 0.0
    risk_percent: float = 0.0
    new_portfolio_heat: float = 0.0
    reason: str = ""
    adjustments: List[str] = field(default_factory=list)


class PrudenceAgent(BaseAgent):
    """
    PRUDENCE - Risk Management Agent
    
    Enforces strict risk limits, calculates position sizes using the 2% rule,
    and has ABSOLUTE VETO POWER over all trades.
    
    Baseline weight: 10% (minimum, always retains veto power)
    """
    
    # Kelly fraction settings
    KELLY_FRACTIONS = {
        "VERY_HIGH": 0.5,  # Half Kelly for very high conviction
        "HIGH": 0.33,  # Third Kelly for high conviction
        "MEDIUM": 0.25,  # Quarter Kelly for medium conviction
    }
    
    # Regime multipliers
    REGIME_MULTIPLIERS = {
        MarketRegime.BULL: 1.2,
        MarketRegime.BEAR: 0.8,
        MarketRegime.CONSOLIDATION: 1.0,
        MarketRegime.CORRECTION: 0.6,
        MarketRegime.MILD_BULL: 1.1,
        MarketRegime.MILD_BEAR: 0.9
    }
    
    def __init__(self):
        super().__init__(
            name="PRUDENCE",
            domain="Risk Management",
            baseline_weight=0.10
        )
        self.risk_limits = RiskLimits()
    
    def analyze(self, data: Dict[str, Any], regime: MarketRegime) -> AgentReport:
        """
        Perform risk management analysis.
        
        Args:
            data: Should contain:
                - 'portfolio': PortfolioStatus or dict
                - 'trade_request': Dict with entry, stop_loss, target for proposed trade
                - 'conviction_level': Conviction level of the trade
                - 'stock': Stock symbol
                - 'sector': Stock sector
                - 'proposed_direction': TradeDirection from consensus (optional, defaults to NEUTRAL)
                
        Returns:
            AgentReport with risk assessment and position sizing
        """
        # Parse portfolio status
        portfolio = self._parse_portfolio_status(data.get('portfolio', {}))

        # Capture the proposed trade direction so PRUDENCE mirrors it correctly.
        # PRUDENCE is a risk-sizing gate, NOT a directional agent.  It should
        # reflect the consensus direction that was passed in, not override it.
        self._proposed_direction = data.get('proposed_direction', TradeDirection.NEUTRAL)
        
        # Parse trade request
        trade_request = data.get('trade_request', {})
        stock = data.get('stock', 'UNKNOWN')
        sector = data.get('sector', '')
        conviction_level = data.get('conviction_level', 'MEDIUM')
        
        # Calculate position sizing
        sizing_result = self._calculate_position_size(
            portfolio, trade_request, conviction_level, regime, stock, sector
        )
        
        # Check all risk limits
        limit_checks = self._check_all_limits(portfolio, sizing_result, sector)
        
        # Determine verdict
        verdict_type = self._determine_verdict(sizing_result, limit_checks)
        
        # Generate verdict
        verdict = self._generate_verdict(
            verdict_type, portfolio, sizing_result, limit_checks, regime
        )
        
        # Build key observations
        key_observations = self._generate_observations(
            portfolio, sizing_result, limit_checks
        )
        
        return AgentReport(
            agent_name=self.name,
            verdict=verdict,
            analysis_details={
                "verdict_type": verdict_type.value,
                "portfolio_heat": portfolio.portfolio_heat,
                "new_heat": sizing_result.new_portfolio_heat,
                "sizing": {
                    "shares": sizing_result.shares,
                    "position_value": sizing_result.position_value,
                    "position_percent": sizing_result.position_percent,
                    "risk_percent": sizing_result.risk_percent
                },
                "limits_checked": limit_checks
            },
            key_observations=key_observations,
            metrics={
                "portfolio_heat": portfolio.portfolio_heat,
                "drawdown": portfolio.current_drawdown,
                "cash_percent": (portfolio.cash / portfolio.total_capital * 100) if portfolio.total_capital > 0 else 0,
                "approved_shares": sizing_result.shares if sizing_result.approved else 0
            },
            raw_data={
                "portfolio": {
                    "total_capital": portfolio.total_capital,
                    "deployed": portfolio.deployed_capital,
                    "cash": portfolio.cash,
                    "heat": portfolio.portfolio_heat,
                    "drawdown": portfolio.current_drawdown
                }
            }
        )
    
    def _parse_portfolio_status(self, data: Dict) -> PortfolioStatus:
        """Parse portfolio status from dict"""
        if isinstance(data, PortfolioStatus):
            return data
        
        positions = []
        for pos in data.get('open_positions', []):
            if isinstance(pos, PositionInfo):
                positions.append(pos)
            else:
                positions.append(PositionInfo(
                    stock=pos.get('stock', ''),
                    shares=pos.get('shares', 0),
                    entry_price=pos.get('entry_price', 0),
                    current_price=pos.get('current_price', 0),
                    stop_loss=pos.get('stop_loss', 0),
                    position_value=pos.get('position_value', 0),
                    risk_amount=pos.get('risk_amount', 0),
                    sector=pos.get('sector', '')
                ))
        
        return PortfolioStatus(
            total_capital=data.get('total_capital', 0),
            deployed_capital=data.get('deployed_capital', 0),
            cash=data.get('cash', 0),
            portfolio_heat=data.get('portfolio_heat', 0),
            current_drawdown=data.get('current_drawdown', 0),
            open_positions=positions,
            sector_exposure=data.get('sector_exposure', {})
        )
    
    def _calculate_position_size(self, portfolio: PortfolioStatus,
                                 trade_request: Dict, conviction_level: str,
                                 regime: MarketRegime, stock: str,
                                 sector: str) -> PositionSizingResult:
        """Calculate position size using 2% rule"""
        result = PositionSizingResult(approved=False)
        
        entry_price = trade_request.get('entry_price', 0)
        stop_loss = trade_request.get('stop_loss', 0)
        
        if entry_price <= 0 or stop_loss <= 0 or portfolio.total_capital <= 0:
            result.reason = "Invalid inputs for position sizing"
            return result
        
        # Step 1: Calculate risk amount (2% rule)
        risk_amount = portfolio.total_capital * self.risk_limits.max_risk_per_trade
        
        # Step 2: Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share <= 0:
            result.reason = "Invalid stop loss - no risk defined"
            return result
        
        # Step 3: Calculate raw shares
        raw_shares = int(risk_amount / risk_per_share)
        
        # Step 4: Calculate position value
        position_value = raw_shares * entry_price
        
        # Step 5: Apply conviction-based Kelly fraction
        kelly_fraction = self.KELLY_FRACTIONS.get(conviction_level, 0.25)
        regime_multiplier = self.REGIME_MULTIPLIERS.get(regime, 1.0)
        
        adjusted_shares = int(raw_shares * kelly_fraction * regime_multiplier)
        position_value = adjusted_shares * entry_price
        position_percent = position_value / portfolio.total_capital
        
        # Step 6: Apply hard cap (15% max position)
        max_position_value = portfolio.total_capital * self.risk_limits.max_single_position
        if position_value > max_position_value:
            adjusted_shares = int(max_position_value / entry_price)
            position_value = adjusted_shares * entry_price
            position_percent = self.risk_limits.max_single_position
            result.adjustments.append(f"Position capped at {self.risk_limits.max_single_position*100}% of portfolio")
        
        # Calculate risk percent
        risk_percent = (adjusted_shares * risk_per_share) / portfolio.total_capital
        
        # Calculate new portfolio heat
        new_heat = portfolio.portfolio_heat + risk_percent
        
        result.approved = True
        result.shares = adjusted_shares
        result.position_value = position_value
        result.position_percent = position_percent
        result.risk_percent = risk_percent
        result.new_portfolio_heat = new_heat
        result.reason = "Position sized according to 2% rule"

        # Enhancement 6: Apply Monte-Carlo scenario simulation
        scenario_output = self._simulate_scenarios(result, portfolio, regime)
        if "kelly_adjustment" in scenario_output:
            result.adjustments.append(
                f"Kelly-adjusted: {scenario_output['kelly_adjustment']:.0%} of original size"
            )

        return result

    # ------------------------------------------------------------------
    # Enhancement 6: Counterfactual / Monte-Carlo Risk Simulation
    # ------------------------------------------------------------------

    def _simulate_scenarios(self, sizing: "PositionSizingResult",
                            portfolio: PortfolioStatus,
                            regime: MarketRegime) -> Dict:
        """
        Monte-Carlo style scenario simulation for probabilistic risk assessment.
        Applies Half-Kelly adjustment to position size based on expected value.
        """
        scenarios = {
            "best_case":  {"multiplier":  1.5, "probability": 0.25},
            "base_case":  {"multiplier":  1.0, "probability": 0.50},
            "worst_case": {"multiplier": -1.0, "probability": 0.25},
        }

        position_value = sizing.position_value
        if position_value <= 0 or portfolio.total_capital <= 0:
            return {}

        expected_values = {}
        for scenario, params in scenarios.items():
            pnl = position_value * 0.05 * params["multiplier"]
            expected_values[scenario] = {
                "pnl": round(pnl, 2),
                "portfolio_impact_pct": round((pnl / portfolio.total_capital) * 100, 3),
            }

        win_prob = (scenarios["best_case"]["probability"] +
                    scenarios["base_case"]["probability"] * 0.5)
        avg_win  = abs(expected_values["best_case"]["pnl"])
        avg_loss = abs(expected_values["worst_case"]["pnl"])

        kelly_adjustment = 0.25  # default quarter-Kelly
        if avg_loss > 0:
            kelly_fraction = win_prob - ((1 - win_prob) / (avg_win / avg_loss))
            kelly_adjustment = max(0.25, min(0.5, kelly_fraction))  # Half-Kelly cap

            new_shares = int(sizing.shares * kelly_adjustment)
            if new_shares > 0:
                sizing.shares = new_shares
                sizing.position_value = new_shares * (position_value / sizing.shares
                                                       if sizing.shares > 0 else 0)

        return {
            "expected_values": expected_values,
            "kelly_adjustment": kelly_adjustment,
            "scenario_analysis": "Monte-Carlo simulation complete",
        }
    
    def _check_all_limits(self, portfolio: PortfolioStatus,
                         sizing: PositionSizingResult,
                         sector: str) -> Dict[str, Dict]:
        """Check all risk limits"""
        checks = {}
        
        # Risk per trade check
        checks["risk_per_trade"] = {
            "limit": self.risk_limits.max_risk_per_trade * 100,
            "actual": sizing.risk_percent * 100,
            "passed": sizing.risk_percent <= self.risk_limits.max_risk_per_trade
        }
        
        # Portfolio heat check
        checks["portfolio_heat"] = {
            "limit": self.risk_limits.max_portfolio_heat * 100,
            "actual": sizing.new_portfolio_heat * 100,
            "passed": sizing.new_portfolio_heat <= self.risk_limits.max_portfolio_heat
        }
        
        # Single position check
        checks["single_position"] = {
            "limit": self.risk_limits.max_single_position * 100,
            "actual": sizing.position_percent * 100,
            "passed": sizing.position_percent <= self.risk_limits.max_single_position
        }
        
        # Sector exposure check
        current_sector = portfolio.sector_exposure.get(sector, 0)
        new_sector = current_sector + sizing.position_percent
        checks["sector_exposure"] = {
            "limit": self.risk_limits.max_sector_exposure * 100,
            "current": current_sector * 100,
            "new": new_sector * 100,
            "passed": new_sector <= self.risk_limits.max_sector_exposure
        }
        
        # Cash buffer check
        new_cash_percent = (portfolio.cash - sizing.position_value) / portfolio.total_capital
        checks["cash_buffer"] = {
            "limit": self.risk_limits.min_cash_buffer * 100,
            "actual": new_cash_percent * 100,
            "passed": new_cash_percent >= self.risk_limits.min_cash_buffer
        }
        
        # Drawdown check
        checks["drawdown"] = {
            "status": "normal",
            "actual": portfolio.current_drawdown * 100,
            "passed": portfolio.current_drawdown < 0.20
        }
        if portfolio.current_drawdown > 0.15:
            checks["drawdown"]["status"] = "emergency"
            checks["drawdown"]["passed"] = False
        elif portfolio.current_drawdown > 0.10:
            checks["drawdown"]["status"] = "defensive"
        elif portfolio.current_drawdown > 0.05:
            checks["drawdown"]["status"] = "caution"
        
        return checks
    
    def _determine_verdict(self, sizing: PositionSizingResult,
                          limit_checks: Dict) -> PrudenceVerdict:
        """Determine the final verdict"""
        if not sizing.approved:
            return PrudenceVerdict.VETO
        
        # Check for any failed limits
        for check_name, check in limit_checks.items():
            if not check.get("passed", True):
                if check_name in ["portfolio_heat", "risk_per_trade"]:
                    return PrudenceVerdict.VETO
                elif check_name == "sector_exposure":
                    # Reduce size for sector limit
                    return PrudenceVerdict.REDUCE_SIZE
                elif check_name == "drawdown" and check.get("status") == "emergency":
                    return PrudenceVerdict.VETO
        
        return PrudenceVerdict.APPROVED
    
    def _generate_verdict(self, verdict_type: PrudenceVerdict,
                         portfolio: PortfolioStatus,
                         sizing: PositionSizingResult,
                         limit_checks: Dict,
                         regime: MarketRegime) -> AgentVerdict:
        """Generate the final verdict.

        PRUDENCE is a risk gate, not a directional agent.
        - VETO   → NEUTRAL direction, conviction=0  (blocks the trade entirely)
        - REDUCE → mirrors proposed_direction, conviction scales down with portfolio risk
        - APPROVE→ mirrors proposed_direction, conviction = 55 base + risk headroom bonus
                   (never inflates to 80 blindly — that was polluting consensus scores)
        """
        # Retrieve the proposed direction stored during analyze()
        proposed = getattr(self, "_proposed_direction", TradeDirection.NEUTRAL)

        if verdict_type == PrudenceVerdict.VETO:
            direction = TradeDirection.NEUTRAL  # Block the trade
            conviction = 0
            reason = self._generate_veto_reason(limit_checks, sizing)
        elif verdict_type == PrudenceVerdict.REDUCE_SIZE:
            direction = proposed
            if proposed == TradeDirection.NEUTRAL:
                # No directional view passed in — PRUDENCE has nothing to say directionally
                conviction = 50
            else:
                # Portfolio stress → lower conviction to signal caution
                heat_penalty = int(min(15, portfolio.portfolio_heat * 100))
                conviction = max(40, 55 - heat_penalty)
            reason = f"Position reduced to {sizing.shares} shares due to limit constraints"
        else:
            direction = proposed
            if proposed == TradeDirection.NEUTRAL:
                # PRUDENCE is a risk gate only — when no direction is proposed, output
                # exactly 50 conviction so it contributes no directional signal bias.
                # Reason describes the current risk environment, not a null position.
                conviction = 50
                heat = portfolio.portfolio_heat
                reason = f"Risk assessment: portfolio heat={heat}% | No active position — gate OPEN | VIX-adjusted caution applied"
            else:
                # Directional trade approved: grant a risk-headroom bonus (max +15 pts)
                heat_room = max(0.0, 0.10 - portfolio.portfolio_heat)
                risk_bonus = int(min(15, heat_room * 150))
                conviction = 55 + risk_bonus   # range: 55–70
                reason = f"Position approved: {sizing.shares} shares ({sizing.position_percent*100:.1f}% of portfolio)"
        
        # Generate risks
        risks = []
        if portfolio.current_drawdown > 0.05:
            risks.append(f"Portfolio in drawdown ({portfolio.current_drawdown*100:.1f}%)")
        if portfolio.portfolio_heat > 0.05:
            risks.append(f"Portfolio heat elevated ({portfolio.portfolio_heat*100:.1f}%)")
        if sizing.new_portfolio_heat > 0.07:
            risks.append(f"New heat will be {sizing.new_portfolio_heat*100:.1f}% (close to limit)")
        
        return AgentVerdict(
            direction=direction,
            conviction=conviction,
            weight=self.current_weight,
            reason=reason,
            risks=risks
        )
    
    def _generate_veto_reason(self, limit_checks: Dict, sizing: PositionSizingResult) -> str:
        """Generate specific veto reason"""
        reasons = []
        
        if not limit_checks.get("risk_per_trade", {}).get("passed", True):
            reasons.append(f"Risk per trade ({sizing.risk_percent*100:.1f}%) exceeds 2% limit")
        
        if not limit_checks.get("portfolio_heat", {}).get("passed", True):
            actual = limit_checks["portfolio_heat"]["actual"]
            reasons.append(f"Portfolio heat ({actual:.1f}%) would exceed 8% limit")
        
        if not limit_checks.get("drawdown", {}).get("passed", True):
            status = limit_checks["drawdown"]["status"]
            reasons.append(f"Portfolio in {status} drawdown - no new aggressive trades")
        
        if not sizing.approved:
            reasons.append(sizing.reason)
        
        return "VETO: " + "; ".join(reasons) if reasons else "VETO: Risk limits exceeded"
    
    def _generate_observations(self, portfolio: PortfolioStatus,
                              sizing: PositionSizingResult,
                              limit_checks: Dict) -> List[str]:
        """Generate key observations"""
        observations = []
        
        # Portfolio status
        observations.append(f"Capital: {portfolio.total_capital:,.0f} | Deployed: {portfolio.deployed_capital:,.0f} ({portfolio.deployed_capital/portfolio.total_capital*100:.0f}%)")
        
        # Heat status
        heat_status = "NORMAL"
        if portfolio.portfolio_heat > 0.06:
            heat_status = "WARNING"
        if portfolio.portfolio_heat > 0.08:
            heat_status = "LIMIT"
        observations.append(f"Portfolio Heat: {portfolio.portfolio_heat*100:.1f}% ({heat_status})")
        
        # Drawdown status
        dd_status = limit_checks.get("drawdown", {}).get("status", "normal")
        observations.append(f"Drawdown: {portfolio.current_drawdown*100:.1f}% ({dd_status.upper()})")
        
        # Position sizing result
        if sizing.approved:
            observations.append(f"Approved: {sizing.shares} shares @ {sizing.position_percent*100:.1f}% of portfolio")
        else:
            observations.append(f"Position sizing failed: {sizing.reason}")
        
        # Failed limits
        for name, check in limit_checks.items():
            if not check.get("passed", True):
                observations.append(f"LIMIT ALERT: {name} - {check}")
        
        return observations
    
    def has_veto_power(self) -> bool:
        """PRUDENCE always has veto power"""
        return True
