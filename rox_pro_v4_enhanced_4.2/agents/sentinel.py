"""
ROX Proven Edge Engine v3.0 - SENTINEL Agent
===========================================
Derivatives Analysis Agent - Options market and OI analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

from .base_agent import BaseAgent, AgentVerdict, AgentReport
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TradeDirection, MarketRegime


class OISignal(Enum):
    """OI change signals"""
    LONG_BUILDUP = "LONG_BUILDUP"
    SHORT_BUILDUP = "SHORT_BUILDUP"
    SHORT_COVERING = "SHORT_COVERING"
    LONG_UNWINDING = "LONG_UNWINDING"
    NEUTRAL = "NEUTRAL"


@dataclass
class OIWall:
    """Open Interest Wall"""
    strike: float
    oi: int
    wall_type: str  # 'call' or 'put'
    strength: float = 0.0


@dataclass
class DerivativesData:
    """Derivatives market data"""
    pcr: float = 1.0
    pcr_trend: str = "stable"  # 'rising', 'falling', 'stable'
    max_pain: float = 0.0
    current_price: float = 0.0
    india_vix: float = 15.0
    iv_rank: float = 50.0
    call_oi_walls: List[OIWall] = field(default_factory=list)
    put_oi_walls: List[OIWall] = field(default_factory=list)
    oi_signal: OISignal = OISignal.NEUTRAL


class SentinelAgent(BaseAgent):
    """
    SENTINEL - Derivatives Analysis Agent
    
    Analyzes options market structure, OI, PCR, and Greeks to identify
    smart money positioning and support/resistance levels.
    
    Baseline weight: 15%
    """
    
    # PCR interpretation thresholds
    PCR_EXTREME_BEARISH = 1.3  # Contrarian bullish
    PCR_MODERATE_BEARISH = 1.1
    PCR_NEUTRAL_UPPER = 1.1
    PCR_NEUTRAL_LOWER = 0.8
    PCR_MODERATE_BULLISH = 0.8
    PCR_EXTREME_BULLISH = 0.6  # Contrarian bearish
    
    def __init__(self):
        super().__init__(
            name="SENTINEL",
            domain="Derivatives Analysis",
            baseline_weight=0.15
        )
    
    def analyze(self, data: Dict[str, Any], regime: MarketRegime) -> AgentReport:
        """
        Perform comprehensive derivatives analysis.
        
        Args:
            data: Should contain:
                - 'pcr': Put-call ratio
                - 'pcr_trend': PCR trend direction
                - 'max_pain': Max pain level
                - 'current_price': Current index/stock price
                - 'india_vix': India VIX value
                - 'iv_rank': IV rank (0-100)
                - 'oi_change': OI change direction
                - 'price_change': Price change direction
                Optional:
                - 'call_oi_walls': List of call OI walls
                - 'put_oi_walls': List of put OI walls
                
        Returns:
            AgentReport with derivatives analysis verdict
        """
        # Parse derivatives data
        deriv_data = self._parse_derivatives_data(data)
        
        # Analyze PCR
        pcr_analysis = self._analyze_pcr(deriv_data.pcr, deriv_data.pcr_trend)
        
        # Analyze OI structure
        oi_analysis = self._analyze_oi_structure(
            deriv_data.call_oi_walls,
            deriv_data.put_oi_walls,
            deriv_data.current_price
        )
        
        # Determine OI signal
        oi_signal = self._determine_oi_signal(
            data.get('oi_change', 0),
            data.get('price_change', 0)
        )
        
        # Analyze IV environment
        iv_analysis = self._analyze_iv(deriv_data.iv_rank, deriv_data.india_vix)
        
        # Calculate support/resistance from OI
        sr_from_oi = self._derive_sr_from_oi(deriv_data)
        
        # Generate verdict
        verdict = self._generate_verdict(
            deriv_data, pcr_analysis, oi_analysis, oi_signal, iv_analysis, regime
        )
        
        # Build key observations
        key_observations = self._generate_observations(
            deriv_data, pcr_analysis, oi_analysis, iv_analysis
        )
        
        return AgentReport(
            agent_name=self.name,
            verdict=verdict,
            analysis_details={
                "pcr_analysis": pcr_analysis,
                "oi_signal": oi_signal.value,
                "max_pain": deriv_data.max_pain,
                "gap_from_max_pain": deriv_data.current_price - deriv_data.max_pain,
                "vix_level": deriv_data.india_vix,
                "iv_environment": iv_analysis.get("environment", "normal"),
                "support_from_oi": sr_from_oi.get("support"),
                "resistance_from_oi": sr_from_oi.get("resistance")
            },
            key_observations=key_observations,
            metrics={
                "pcr": deriv_data.pcr,
                "vix": deriv_data.india_vix,
                "iv_rank": deriv_data.iv_rank
            },
            raw_data={
                "derivatives_data": {
                    "pcr": deriv_data.pcr,
                    "max_pain": deriv_data.max_pain,
                    "vix": deriv_data.india_vix,
                    "iv_rank": deriv_data.iv_rank
                }
            }
        )
    
    def _parse_derivatives_data(self, data: Dict) -> DerivativesData:
        """Parse derivatives data from dict"""
        # Parse OI walls
        call_walls = []
        for wall in data.get('call_oi_walls', []):
            if isinstance(wall, dict):
                call_walls.append(OIWall(
                    strike=wall.get('strike', 0),
                    oi=wall.get('oi', 0),
                    wall_type='call',
                    strength=wall.get('strength', 0)
                ))
        
        put_walls = []
        for wall in data.get('put_oi_walls', []):
            if isinstance(wall, dict):
                put_walls.append(OIWall(
                    strike=wall.get('strike', 0),
                    oi=wall.get('oi', 0),
                    wall_type='put',
                    strength=wall.get('strength', 0)
                ))
        
        return DerivativesData(
            pcr=data.get('pcr', 1.0),
            pcr_trend=data.get('pcr_trend', 'stable'),
            max_pain=data.get('max_pain', 0),
            current_price=data.get('current_price', 0),
            india_vix=data.get('india_vix', 15),
            iv_rank=data.get('iv_rank', 50),
            call_oi_walls=call_walls,
            put_oi_walls=put_walls
        )
    
    def _analyze_pcr(self, pcr: float, trend: str) -> Dict:
        """Analyze PCR for signals"""
        analysis = {
            "zone": "neutral",
            "contrarian_signal": False,
            "signal_direction": None
        }
        
        if pcr > self.PCR_EXTREME_BEARISH:
            analysis["zone"] = "excessive_bearishness"
            analysis["contrarian_signal"] = True
            analysis["signal_direction"] = "bullish"
        elif pcr > self.PCR_MODERATE_BEARISH:
            analysis["zone"] = "moderately_bearish"
            analysis["contrarian_signal"] = True
            analysis["signal_direction"] = "mild_bullish"
        elif pcr < self.PCR_EXTREME_BULLISH:
            analysis["zone"] = "excessive_bullishness"
            analysis["contrarian_signal"] = True
            analysis["signal_direction"] = "bearish"
        elif pcr < self.PCR_MODERATE_BULLISH:
            analysis["zone"] = "moderately_bullish"
            analysis["contrarian_signal"] = True
            analysis["signal_direction"] = "mild_bearish"
        
        # Trend consideration
        if trend == "rising" and pcr > 1.0:
            analysis["momentum_note"] = "PCR rising in downtrend = potential bottom"
        elif trend == "falling" and pcr < 1.0:
            analysis["momentum_note"] = "PCR falling in uptrend = growing complacency"
        
        return analysis
    
    def _analyze_oi_structure(self, call_walls: List[OIWall], 
                             put_walls: List[OIWall],
                             current_price: float) -> Dict:
        """Analyze OI structure for support/resistance"""
        analysis = {
            "call_resistance": None,
            "put_support": None,
            "call_wall_strength": 0,
            "put_wall_strength": 0,
            "structure_bias": "neutral"
        }
        
        # Find strongest call wall above price (resistance)
        above_calls = [w for w in call_walls if w.strike > current_price]
        if above_calls:
            strongest_call = max(above_calls, key=lambda x: x.oi)
            analysis["call_resistance"] = strongest_call.strike
            analysis["call_wall_strength"] = strongest_call.oi
        
        # Find strongest put wall below price (support)
        below_puts = [w for w in put_walls if w.strike < current_price]
        if below_puts:
            strongest_put = max(below_puts, key=lambda x: x.oi)
            analysis["put_support"] = strongest_put.strike
            analysis["put_wall_strength"] = strongest_put.oi
        
        # Determine structure bias
        if analysis["put_wall_strength"] > analysis["call_wall_strength"] * 1.5:
            analysis["structure_bias"] = "bullish"  # Strong put support
        elif analysis["call_wall_strength"] > analysis["put_wall_strength"] * 1.5:
            analysis["structure_bias"] = "bearish"  # Strong call resistance
        
        return analysis
    
    def _determine_oi_signal(self, oi_change: float, price_change: float) -> OISignal:
        """Determine OI signal from OI and price changes"""
        oi_rising = oi_change > 0
        price_rising = price_change > 0
        
        if oi_rising and price_rising:
            return OISignal.LONG_BUILDUP
        elif oi_rising and not price_rising:
            return OISignal.SHORT_BUILDUP
        elif not oi_rising and price_rising:
            return OISignal.SHORT_COVERING
        elif not oi_rising and not price_rising:
            return OISignal.LONG_UNWINDING
        else:
            return OISignal.NEUTRAL
    
    def _analyze_iv(self, iv_rank: float, vix: float) -> Dict:
        """Analyze implied volatility environment"""
        analysis = {
            "environment": "normal",
            "iv_level": "moderate",
            "recommendation": None
        }
        
        if iv_rank > 60:
            analysis["environment"] = "high_iv"
            analysis["iv_level"] = "high"
            analysis["recommendation"] = "Options expensive - prefer selling"
        elif iv_rank < 30:
            analysis["environment"] = "low_iv"
            analysis["iv_level"] = "low"
            analysis["recommendation"] = "Options cheap - prefer buying"
        
        # VIX overlay
        if vix > 25:
            analysis["environment"] = "high_fear"
            analysis["note"] = "High VIX - expect volatility"
        elif vix < 12:
            analysis["environment"] = "complacent"
            analysis["note"] = "Low VIX - potential for volatility spike"
        
        return analysis
    
    def _derive_sr_from_oi(self, deriv_data: DerivativesData) -> Dict:
        """Derive support/resistance from OI walls"""
        sr = {
            "support": None,
            "resistance": None,
            "confidence": "low"
        }
        
        if deriv_data.put_oi_walls:
            strongest_put = max(deriv_data.put_oi_walls, key=lambda x: x.oi)
            if strongest_put.strike < deriv_data.current_price:
                sr["support"] = strongest_put.strike
        
        if deriv_data.call_oi_walls:
            strongest_call = max(deriv_data.call_oi_walls, key=lambda x: x.oi)
            if strongest_call.strike > deriv_data.current_price:
                sr["resistance"] = strongest_call.strike
        
        if sr["support"] and sr["resistance"]:
            sr["confidence"] = "high"
        elif sr["support"] or sr["resistance"]:
            sr["confidence"] = "medium"
        
        return sr
    
    def _generate_verdict(self, deriv_data: DerivativesData, pcr_analysis: Dict,
                         oi_analysis: Dict, oi_signal: OISignal,
                         iv_analysis: Dict, regime: MarketRegime) -> AgentVerdict:
        """Generate the final verdict"""
        direction = TradeDirection.NEUTRAL
        conviction = 50
        
        # PCR signal
        if pcr_analysis["contrarian_signal"]:
            if pcr_analysis["signal_direction"] == "bullish":
                direction = TradeDirection.LONG
                conviction += 15
            elif pcr_analysis["signal_direction"] == "bearish":
                direction = TradeDirection.SHORT
                conviction += 15
            elif pcr_analysis["signal_direction"] == "mild_bullish":
                if direction != TradeDirection.SHORT:
                    direction = TradeDirection.LONG
                conviction += 5
            elif pcr_analysis["signal_direction"] == "mild_bearish":
                if direction != TradeDirection.LONG:
                    direction = TradeDirection.SHORT
                conviction += 5
        
        # OI signal contribution
        oi_direction_map = {
            OISignal.LONG_BUILDUP: (TradeDirection.LONG, 10),
            OISignal.SHORT_BUILDUP: (TradeDirection.SHORT, 10),
            OISignal.SHORT_COVERING: (TradeDirection.LONG, 5),
            OISignal.LONG_UNWINDING: (TradeDirection.SHORT, 5),
            OISignal.NEUTRAL: (TradeDirection.NEUTRAL, 0)
        }
        
        oi_dir, oi_conv = oi_direction_map.get(oi_signal, (TradeDirection.NEUTRAL, 0))
        if oi_dir == direction:
            conviction += oi_conv
        else:
            conviction -= oi_conv // 2
        
        # OI structure contribution
        if oi_analysis["structure_bias"] == "bullish" and direction == TradeDirection.LONG:
            conviction += 10
        elif oi_analysis["structure_bias"] == "bearish" and direction == TradeDirection.SHORT:
            conviction += 10
        
        # Max Pain gravity
        if deriv_data.max_pain > 0 and deriv_data.current_price > 0:
            gap = (deriv_data.current_price - deriv_data.max_pain) / deriv_data.current_price
            if abs(gap) < 0.02:  # Near max pain
                conviction -= 5  # Reduced directional confidence near max pain
        
        # VIX adjustment
        if deriv_data.india_vix > 22:
            conviction -= 10  # Reduce conviction in high volatility
        elif deriv_data.india_vix < 13:
            conviction += 5  # Add conviction in low volatility
        
        # Clamp conviction
        conviction = max(0, min(100, conviction))
        
        # Generate reason — be explicit about contrarian nature to avoid
        # the cross-examiner misreading zone names as directional statements.
        # e.g. zone="moderately_bearish" + direction=LONG looks contradictory
        # but is correct contrarian logic (fearful market = buying opportunity).
        if pcr_analysis["contrarian_signal"]:
            sentiment = pcr_analysis["zone"].replace("_", " ")
            contrarian_dir = pcr_analysis["signal_direction"].replace("_", " ")
            reason_parts = [
                f"PCR {deriv_data.pcr:.2f} → {sentiment} sentiment "
                f"→ contrarian {contrarian_dir} signal"
            ]
        else:
            reason_parts = [f"PCR: {deriv_data.pcr:.2f} (neutral zone — no contrarian signal)"]
        reason_parts.append(f"OI signal: {oi_signal.value}")
        if deriv_data.india_vix > 20:
            reason_parts.append(f"High VIX: {deriv_data.india_vix}")
        reason = " | ".join(reason_parts)
        
        # Generate risks
        risks = []
        if deriv_data.india_vix > 25:
            risks.append("Very high VIX - volatile environment")
        if iv_analysis["environment"] in ["high_iv", "high_fear"]:
            risks.append("High IV - options expensive, expect IV crush")
        if not oi_analysis["call_resistance"] and not oi_analysis["put_support"]:
            risks.append("No clear OI walls detected")
        
        return AgentVerdict(
            direction=direction,
            conviction=conviction,
            weight=self.current_weight,
            reason=reason,
            risks=risks
        )
    
    def _generate_observations(self, deriv_data: DerivativesData, pcr_analysis: Dict,
                              oi_analysis: Dict, iv_analysis: Dict) -> List[str]:
        """Generate key observations"""
        observations = []
        
        # PCR observation
        observations.append(f"PCR: {deriv_data.pcr:.2f} ({pcr_analysis['zone']})")
        if pcr_analysis.get("momentum_note"):
            observations.append(pcr_analysis["momentum_note"])
        
        # Max Pain observation
        if deriv_data.max_pain > 0:
            gap = deriv_data.current_price - deriv_data.max_pain
            direction = "above" if gap > 0 else "below"
            observations.append(f"Max Pain: {deriv_data.max_pain:.0f} ({direction} current)")
        
        # OI walls
        if oi_analysis["put_support"]:
            observations.append(f"Put OI support: {oi_analysis['put_support']:.0f}")
        if oi_analysis["call_resistance"]:
            observations.append(f"Call OI resistance: {oi_analysis['call_resistance']:.0f}")
        
        # VIX observation
        if deriv_data.india_vix > 20:
            observations.append(f"VIX elevated: {deriv_data.india_vix:.1f}")
        elif deriv_data.india_vix < 13:
            observations.append(f"VIX low: {deriv_data.india_vix:.1f} - complacency warning")
        
        # IV recommendation
        if iv_analysis.get("recommendation"):
            observations.append(iv_analysis["recommendation"])
        
        return observations
