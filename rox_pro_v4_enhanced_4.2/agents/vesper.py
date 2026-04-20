"""
ROX Proven Edge Engine v3.0 - VESPER Agent
=========================================
Flow Analysis Agent - Institutional money movement tracking (FII/DII).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum

from .base_agent import BaseAgent, AgentVerdict, AgentReport
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TradeDirection, MarketRegime


class FlowSignal(Enum):
    """Flow signal types"""
    STRONG_ACCUMULATION = "STRONG_ACCUMULATION"
    STEALTH_ACCUMULATION = "STEALTH_ACCUMULATION"
    SMART_MONEY_EXIT = "SMART_MONEY_EXIT"
    SYNCHRONIZED_UPTREND = "SYNCHRONIZED_UPTREND"
    SYNCHRONIZED_DOWNTREND = "SYNCHRONIZED_DOWNTREND"
    STRONG_DISTRIBUTION = "STRONG_DISTRIBUTION"
    CAPITULATION = "CAPITULATION"
    NEUTRAL = "NEUTRAL"


@dataclass
class FlowData:
    """Flow data container"""
    fii_cash_daily: float = 0.0  # In crores
    dii_cash_daily: float = 0.0
    fii_derivative_daily: float = 0.0
    fii_cash_3day: float = 0.0
    dii_cash_3day: float = 0.0
    fii_cash_5day: float = 0.0
    dii_cash_5day: float = 0.0
    flow_momentum: float = 0.0  # 3-day change


@dataclass
class SectorFlow:
    """Sector-level flow data"""
    sector_name: str
    inflow_3day: float = 0.0
    outflow_3day: float = 0.0
    net_flow: float = 0.0


@dataclass
class BulkDeal:
    """Bulk/Block deal information"""
    stock: str
    buyer: str
    seller: str
    quantity: int
    price: float
    premium_discount: float  # % vs CMP
    date: datetime


class VesperAgent(BaseAgent):
    """
    VESPER - Flow Analysis Agent
    
    Tracks institutional money movement (FII, DII, mutual funds) to identify
    where smart money is positioning.
    
    Baseline weight: 18%
    """
    
    # Flow thresholds in crores
    FII_STRONG_BUY_DAILY = 2000
    FII_MODERATE_BUY_DAILY = 500
    FII_STRONG_ACCUMULATION_5DAY = 8000
    FII_MODERATE_ACCUMULATION_5DAY = 3000
    FII_STRONG_DISTRIBUTION_5DAY = -8000
    FII_MODERATE_DISTRIBUTION_5DAY = -3000
    
    def __init__(self):
        super().__init__(
            name="VESPER",
            domain="Flow Analysis",
            baseline_weight=0.18
        )
    
    def analyze(self, data: Dict[str, Any], regime: MarketRegime) -> AgentReport:
        """
        Perform comprehensive flow analysis.
        
        Args:
            data: Must contain:
                - 'flow_data': FlowData or dict with FII/DII flows
                Optional:
                - 'sector_flows': List of SectorFlow
                - 'bulk_deals': List of BulkDeal
                - 'fii_futures_position': FII futures positioning
                - 'derivative_flows': FII derivative flows
                
        Returns:
            AgentReport with flow analysis verdict
        """
        # Parse flow data
        flow_data = self._parse_flow_data(data.get('flow_data', {}))
        
        # Analyze flow patterns
        flow_signal = self._classify_flow_signal(flow_data)
        
        # Calculate flow score
        flow_score = self._calculate_flow_score(flow_data)
        
        # Analyze sector rotation
        sector_flows = self._parse_sector_flows(data.get('sector_flows', []))
        rotation_signal = self._detect_sector_rotation(sector_flows)
        
        # Analyze derivative positioning
        derivative_analysis = self._analyze_derivatives(
            data.get('fii_futures_position', 0),
            data.get('derivative_flows', {})
        )
        
        # Analyze bulk deals
        bulk_deals = data.get('bulk_deals', [])
        bulk_deal_analysis = self._analyze_bulk_deals(bulk_deals)
        
        # Generate verdict
        verdict = self._generate_verdict(
            flow_data, flow_signal, flow_score, 
            derivative_analysis, regime
        )
        
        # Build key observations
        key_observations = self._generate_observations(
            flow_data, flow_signal, rotation_signal, bulk_deal_analysis
        )
        
        return AgentReport(
            agent_name=self.name,
            verdict=verdict,
            analysis_details={
                "flow_signal": flow_signal.value,
                "flow_score": flow_score,
                "fii_5day_flow": flow_data.fii_cash_5day,
                "dii_5day_flow": flow_data.dii_cash_5day,
                "flow_momentum": flow_data.flow_momentum,
                "sector_rotation": rotation_signal,
                "derivative_position": derivative_analysis.get("position", "neutral"),
                "bulk_deal_signal": bulk_deal_analysis.get("signal", "neutral")
            },
            key_observations=key_observations,
            metrics={
                "flow_score": flow_score,
                "fii_5day_cr": flow_data.fii_cash_5day,
                "dii_5day_cr": flow_data.dii_cash_5day,
                "momentum": flow_data.flow_momentum
            },
            raw_data={
                "flow_data": flow_data.__dict__,
                "sector_flows": [s.__dict__ for s in sector_flows]
            }
        )

    # ------------------------------------------------------------------
    # Enhancement 4: Tree-of-Thoughts for ambiguous flow signals
    # ------------------------------------------------------------------

    def analyze_with_tree_of_thoughts(self, data: Dict[str, Any],
                                      regime: MarketRegime) -> AgentReport:
        """
        Tree-of-Thoughts for VESPER: branches over FII flow interpretation.
        Activates only when flow conviction is ambiguous (40-60 range).
        """
        initial_report = self.analyze(data, regime)
        # ToT activates only for ambiguous conviction (40-60).
        # When conviction is already clear (< 40 or > 60), return immediately.
        if not (40 <= initial_report.verdict.conviction <= 60):
            return initial_report

        branches = [
            {
                "name": "sustained_accumulation",
                "description": "FII flows indicate sustained institutional buying",
                "probability": 0.40,
                "fii_multiplier": 1.5,
            },
            {
                "name": "distribution_phase",
                "description": "FII flows signal distribution / exit",
                "probability": 0.35,
                "fii_multiplier": -1.5,
            },
            {
                "name": "neutral_repositioning",
                "description": "Mixed flows; sector rotation rather than directional",
                "probability": 0.25,
                "fii_multiplier": 0,
            },
        ]

        branch_scores = []
        for branch in branches:
            branch_data = dict(data)
            flow_raw = dict(branch_data.get("flow_data", {}))
            for k in ("fii_cash_daily", "fii_cash_3day", "fii_cash_5day"):
                flow_raw[k] = flow_raw.get(k, 0) * branch["fii_multiplier"]
            branch_data["flow_data"] = flow_raw
            branch_report = self.analyze(branch_data, regime)
            branch_scores.append({
                "hypothesis": branch["name"],
                "conviction": branch_report.verdict.conviction,
                "direction": branch_report.verdict.direction,
                "probability": branch["probability"],
            })

        weighted_conviction = sum(b["conviction"] * b["probability"] for b in branch_scores)
        initial_report.verdict.conviction = round(weighted_conviction, 1)
        initial_report.verdict.reason += (
            f" | [ToT: {len(branches)} flow branches, "
            f"weighted conviction={weighted_conviction:.0f}]"
        )
        initial_report.analysis_details["tree_branches"] = branch_scores
        initial_report.verdict.__post_init__()
        return initial_report

    def _parse_flow_data(self, data: Dict) -> FlowData:
        """Parse flow data from dict or return existing FlowData"""
        if isinstance(data, FlowData):
            return data
        
        return FlowData(
            fii_cash_daily=data.get('fii_cash_daily', 0),
            dii_cash_daily=data.get('dii_cash_daily', 0),
            fii_derivative_daily=data.get('fii_derivative_daily', 0),
            fii_cash_3day=data.get('fii_cash_3day', 0),
            dii_cash_3day=data.get('dii_cash_3day', 0),
            fii_cash_5day=data.get('fii_cash_5day', 0),
            dii_cash_5day=data.get('dii_cash_5day', 0),
            flow_momentum=data.get('flow_momentum', 0)
        )
    
    def _parse_sector_flows(self, data: List) -> List[SectorFlow]:
        """Parse sector flow data"""
        flows = []
        for item in data:
            if isinstance(item, SectorFlow):
                flows.append(item)
            elif isinstance(item, dict):
                flows.append(SectorFlow(
                    sector_name=item.get('sector_name', ''),
                    inflow_3day=item.get('inflow_3day', 0),
                    outflow_3day=item.get('outflow_3day', 0),
                    net_flow=item.get('net_flow', 0)
                ))
        return flows
    
    def _classify_flow_signal(self, flow_data: FlowData) -> FlowSignal:
        """Classify the overall flow signal"""
        fii_5d = flow_data.fii_cash_5day
        dii_5d = flow_data.dii_cash_5day
        
        # Synchronized uptrend: Both buying strongly
        if fii_5d > 2000 and dii_5d > 1000:
            return FlowSignal.SYNCHRONIZED_UPTREND
        
        # Synchronized downtrend: Both selling
        if fii_5d < -2000 and dii_5d < -1000:
            return FlowSignal.SYNCHRONIZED_DOWNTREND
        
        # Strong accumulation: FII buying heavily
        if fii_5d > self.FII_STRONG_ACCUMULATION_5DAY:
            return FlowSignal.STRONG_ACCUMULATION
        
        # Stealth accumulation: DII buying while FII sells
        if fii_5d < 0 and dii_5d > 2000:
            return FlowSignal.STEALTH_ACCUMULATION
        
        # Strong distribution: FII selling heavily
        if fii_5d < self.FII_STRONG_DISTRIBUTION_5DAY:
            return FlowSignal.STRONG_DISTRIBUTION
        
        # Capitulation: FII panic selling, DII absorbing
        if fii_5d < -5000 and dii_5d > 3000:
            return FlowSignal.CAPITULATION
        
        # Smart money exit: FII buying but DII selling (unusual)
        if fii_5d > 0 and dii_5d < -1000:
            return FlowSignal.SMART_MONEY_EXIT
        
        return FlowSignal.NEUTRAL
    
    def _calculate_flow_score(self, flow_data: FlowData) -> float:
        """Calculate flow score (0-100)"""
        # Normalize 5-day FII flow to score
        # Assume max meaningful flow is +/- 10,000 Cr
        max_flow = 10000
        
        fii_score = (flow_data.fii_cash_5day / max_flow) * 50 + 50
        fii_score = max(0, min(100, fii_score))
        
        # Add momentum factor
        momentum_adjustment = flow_data.flow_momentum / 1000 * 10
        
        final_score = fii_score + momentum_adjustment
        return max(0, min(100, final_score))
    
    def _detect_sector_rotation(self, sector_flows: List[SectorFlow]) -> Dict:
        """Detect sector rotation patterns"""
        if not sector_flows:
            return {"detected": False, "from": None, "to": None}
        
        # Find sectors with strongest inflows and outflows
        inflows = sorted(
            [s for s in sector_flows if s.net_flow > 0],
            key=lambda x: x.net_flow,
            reverse=True
        )
        outflows = sorted(
            [s for s in sector_flows if s.net_flow < 0],
            key=lambda x: x.net_flow
        )
        
        rotation = {
            "detected": False,
            "from": None,
            "to": None
        }
        
        # Rotation signal: >500 Cr outflow from one sector, >500 Cr inflow to another
        if inflows and outflows:
            if inflows[0].net_flow > 500 and outflows[0].net_flow < -300:
                rotation["detected"] = True
                rotation["from"] = outflows[0].sector_name
                rotation["to"] = inflows[0].sector_name
                rotation["inflow_amount"] = inflows[0].net_flow
                rotation["outflow_amount"] = outflows[0].net_flow
        
        return rotation
    
    def _analyze_derivatives(self, futures_position: float, 
                           derivative_flows: Dict) -> Dict:
        """Analyze FII derivative positioning"""
        analysis = {
            "position": "neutral",
            "signal": "neutral",
            "futures_bias": 0
        }
        
        # Futures position interpretation
        if futures_position > 5000:
            analysis["position"] = "net_long"
            analysis["signal"] = "bullish"
        elif futures_position < -5000:
            analysis["position"] = "net_short"
            analysis["signal"] = "bearish"
        
        analysis["futures_bias"] = futures_position
        
        return analysis
    
    def _analyze_bulk_deals(self, bulk_deals: List[BulkDeal]) -> Dict:
        """Analyze bulk/block deals for signals"""
        if not bulk_deals:
            return {"signal": "neutral", "notable_deals": []}
        
        analysis = {
            "signal": "neutral",
            "notable_deals": [],
            "bullish_count": 0,
            "bearish_count": 0
        }
        
        for deal in bulk_deals[:10]:  # Top 10 recent deals
            # Premium buying is bullish
            if deal.premium_discount > 2:
                analysis["bullish_count"] += 1
                analysis["notable_deals"].append({
                    "stock": deal.stock,
                    "type": "premium_buy",
                    "premium": deal.premium_discount
                })
            # Discount selling is bearish
            elif deal.premium_discount < -2:
                analysis["bearish_count"] += 1
                analysis["notable_deals"].append({
                    "stock": deal.stock,
                    "type": "discount_sell",
                    "discount": deal.premium_discount
                })
        
        # Overall signal
        if analysis["bullish_count"] > analysis["bearish_count"] * 2:
            analysis["signal"] = "bullish"
        elif analysis["bearish_count"] > analysis["bullish_count"] * 2:
            analysis["signal"] = "bearish"
        
        return analysis
    
    def _generate_verdict(self, flow_data: FlowData, flow_signal: FlowSignal,
                         flow_score: float, derivative_analysis: Dict,
                         regime: MarketRegime) -> AgentVerdict:
        """Generate the final verdict"""
        # Determine direction from flow signal
        direction_map = {
            FlowSignal.SYNCHRONIZED_UPTREND: (TradeDirection.LONG, 85),
            FlowSignal.STRONG_ACCUMULATION: (TradeDirection.LONG, 80),
            FlowSignal.STEALTH_ACCUMULATION: (TradeDirection.LONG, 65),
            FlowSignal.CAPITULATION: (TradeDirection.LONG, 70),  # Contrarian
            FlowSignal.SYNCHRONIZED_DOWNTREND: (TradeDirection.SHORT, 85),
            FlowSignal.STRONG_DISTRIBUTION: (TradeDirection.SHORT, 80),
            FlowSignal.SMART_MONEY_EXIT: (TradeDirection.SHORT, 60),
            FlowSignal.NEUTRAL: (TradeDirection.NEUTRAL, 50)
        }
        
        direction, base_conviction = direction_map.get(flow_signal, (TradeDirection.NEUTRAL, 50))

        _non_bull_regimes = {
            MarketRegime.BEAR, MarketRegime.CORRECTION,
            MarketRegime.MILD_BEAR, MarketRegime.CONSOLIDATION
        }
        if flow_signal == FlowSignal.STEALTH_ACCUMULATION and regime in _non_bull_regimes:
            direction = TradeDirection.NEUTRAL
            base_conviction = 50
        
        # Adjust conviction based on flow score
        conviction = base_conviction
        if flow_score > 70:
            conviction += 10
        elif flow_score < 30:
            conviction -= 10
        
        # Derivative confirmation
        if derivative_analysis["signal"] == "bullish" and direction == TradeDirection.LONG:
            conviction += 5
        elif derivative_analysis["signal"] == "bearish" and direction == TradeDirection.SHORT:
            conviction += 5
        elif derivative_analysis["signal"] not in ["neutral", direction.value.lower()]:
            conviction -= 5
        
        # Regime adjustments
        if regime == MarketRegime.BULL and flow_signal == FlowSignal.STRONG_ACCUMULATION:
            conviction += 5
        elif regime == MarketRegime.BEAR and flow_signal == FlowSignal.STRONG_DISTRIBUTION:
            conviction += 5
        
        # Clamp conviction
        conviction = max(0, min(100, conviction))
        
        # Generate reason — describe what the signal means in context
        if flow_signal == FlowSignal.STEALTH_ACCUMULATION and regime in _non_bull_regimes:
            reason = (
                f"DII accumulation signal (monitoring only, not actionable in {regime.value}) | "
                f"FII 5-day: {flow_data.fii_cash_5day:+,.0f} Cr | "
                f"DII offsetting FII selling — watch for regime change"
            )
        else:
            reason = f"Flow signal: {flow_signal.value}"
            if flow_data.fii_cash_5day != 0:
                reason += f" | FII 5-day: {flow_data.fii_cash_5day:+,.0f} Cr"
        
        # Generate risks
        risks = []
        if abs(flow_data.flow_momentum) > 3000:
            risks.append("High flow volatility - momentum may reverse")
        if flow_signal == FlowSignal.CAPITULATION:
            risks.append("Capitulation signal - timing uncertain")
        if direction == TradeDirection.LONG and flow_data.fii_cash_5day < 0:
            risks.append("FII selling despite bullish signal")
        
        return AgentVerdict(
            direction=direction,
            conviction=conviction,
            weight=self.current_weight,
            reason=reason,
            risks=risks
        )
    
    def _generate_observations(self, flow_data: FlowData, flow_signal: FlowSignal,
                              rotation_signal: Dict, bulk_deal_analysis: Dict) -> List[str]:
        """Generate key observations"""
        observations = []
        
        # Flow signal
        observations.append(f"Flow classification: {flow_signal.value}")
        
        # FII/DII summary
        if flow_data.fii_cash_5day != 0:
            fii_dir = "buying" if flow_data.fii_cash_5day > 0 else "selling"
            observations.append(f"FII {fii_dir} {abs(flow_data.fii_cash_5day):,.0f} Cr (5-day)")
        
        if flow_data.dii_cash_5day != 0:
            dii_dir = "buying" if flow_data.dii_cash_5day > 0 else "selling"
            observations.append(f"DII {dii_dir} {abs(flow_data.dii_cash_5day):,.0f} Cr (5-day)")
        
        # Flow momentum
        if flow_data.flow_momentum != 0:
            mom_dir = "accelerating" if flow_data.flow_momentum > 0 else "decelerating"
            observations.append(f"Flow momentum {mom_dir}")
        
        # Sector rotation
        if rotation_signal.get("detected"):
            observations.append(
                f"Rotation detected: {rotation_signal['from']} to {rotation_signal['to']}"
            )
        
        # Bulk deals
        if bulk_deal_analysis.get("notable_deals"):
            deal = bulk_deal_analysis["notable_deals"][0]
            observations.append(f"Notable bulk deal: {deal['stock']} ({deal['type']})")
        
        return observations
