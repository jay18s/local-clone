"""
ROX Proven Edge Engine v3.0 - ORION Agent
========================================
Technical Analysis Agent - Price action, chart patterns, and technical indicators.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import math

from .base_agent import BaseAgent, AgentVerdict, AgentReport
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TradeDirection, MarketRegime


@dataclass
class TechnicalIndicators:
    """Technical indicator values"""
    atr: float = 0.0
    atr_percent: float = 0.0
    rsi: float = 50.0
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    bb_width: float = 0.0
    adx: float = 0.0


@dataclass
class SupportResistanceLevel:
    """Support or resistance level"""
    price: float
    strength: int  # 1-5 stars
    level_type: str  # 'support' or 'resistance'
    touches: int = 1
    volume_confirm: bool = False


@dataclass
class PatternSignal:
    """Detected pattern signal"""
    pattern_name: str
    pattern_type: str  # 'bullish', 'bearish', 'neutral'
    probability: float
    status: str  # 'forming', 'confirmed', 'invalidated'
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target: Optional[float] = None


class OrionAgent(BaseAgent):
    """
    ORION - Technical Analysis Agent
    
    Analyzes multi-timeframe price action, chart patterns, and technical
    indicators to identify high-probability entry and exit points.
    
    Baseline weight: 20%
    """
    
    # Pattern probability mappings
    CANDLESTICK_PATTERNS = {
        "doji_at_support": {"probability": 0.65, "direction": TradeDirection.LONG},
        "bullish_engulfing": {"probability": 0.72, "direction": TradeDirection.LONG},
        "bearish_engulfing": {"probability": 0.72, "direction": TradeDirection.SHORT},
        "morning_star": {"probability": 0.78, "direction": TradeDirection.LONG},
        "evening_star": {"probability": 0.78, "direction": TradeDirection.SHORT},
        "hammer": {"probability": 0.68, "direction": TradeDirection.LONG},
        "hanging_man": {"probability": 0.68, "direction": TradeDirection.SHORT},
        "shooting_star": {"probability": 0.70, "direction": TradeDirection.SHORT},
        "inverted_hammer": {"probability": 0.65, "direction": TradeDirection.LONG},
        "three_white_soldiers": {"probability": 0.80, "direction": TradeDirection.LONG},
        "three_black_crows": {"probability": 0.80, "direction": TradeDirection.SHORT},
    }
    
    CHART_PATTERNS = {
        "head_and_shoulders": {"probability": 0.75, "direction": TradeDirection.SHORT},
        "inverse_head_and_shoulders": {"probability": 0.75, "direction": TradeDirection.LONG},
        "double_top": {"probability": 0.70, "direction": TradeDirection.SHORT},
        "double_bottom": {"probability": 0.73, "direction": TradeDirection.LONG},
        "cup_and_handle": {"probability": 0.74, "direction": TradeDirection.LONG},
        "ascending_triangle": {"probability": 0.72, "direction": TradeDirection.LONG},
        "descending_triangle": {"probability": 0.72, "direction": TradeDirection.SHORT},
        "symmetrical_triangle": {"probability": 0.65, "direction": TradeDirection.NEUTRAL},
        "bull_flag": {"probability": 0.68, "direction": TradeDirection.LONG},
        "bear_flag": {"probability": 0.68, "direction": TradeDirection.SHORT},
    }
    
    def __init__(self):
        super().__init__(
            name="ORION",
            domain="Technical Analysis",
            baseline_weight=0.20
        )
    
    def analyze(self, data: Dict[str, Any], regime: MarketRegime) -> AgentReport:
        """
        Perform comprehensive technical analysis.
        
        Args:
            data: Must contain:
                - 'price_data': Dict with OHLCV data
                - 'indicators': TechnicalIndicators or dict
                - 'stock': Stock symbol
                Optional:
                - 'weekly_trend': Weekly trend direction
                - 'daily_trend': Daily trend direction
                - 'four_hour_trend': 4H trend direction
                - 'one_hour_trend': 1H trend direction
                
        Returns:
            AgentReport with technical analysis verdict
        """
        stock = data.get('stock', 'UNKNOWN')
        price_data = data.get('price_data', {})
        indicators = data.get('indicators', {})
        
        # Parse indicators
        if isinstance(indicators, dict):
            tech_indicators = TechnicalIndicators(
                atr=indicators.get('atr', 0),
                atr_percent=indicators.get('atr_percent', 0),
                rsi=indicators.get('rsi', 50),
                macd=indicators.get('macd', 0),
                macd_signal=indicators.get('macd_signal', 0),
                macd_histogram=indicators.get('macd_histogram', 0),
                bb_upper=indicators.get('bb_upper', 0),
                bb_middle=indicators.get('bb_middle', 0),
                bb_lower=indicators.get('bb_lower', 0),
                bb_width=indicators.get('bb_width', 0),
                adx=indicators.get('adx', 0)
            )
        else:
            tech_indicators = indicators
        
        # Calculate multi-timeframe confluence
        confluence_score = self._calculate_confluence(data, regime)
        
        # Analyze indicators
        indicator_analysis = self._analyze_indicators(tech_indicators, price_data)
        
        # Identify support/resistance levels
        sr_levels = self._identify_support_resistance(price_data)
        
        # Detect patterns
        patterns = self._detect_patterns(price_data, tech_indicators, sr_levels)
        
        # Calculate entry setup
        current_price = price_data.get('close', 0)
        entry_setup = self._calculate_entry_setup(
            current_price, tech_indicators, sr_levels, patterns, regime
        )
        
        # Generate verdict
        verdict = self._generate_verdict(
            confluence_score, indicator_analysis, patterns, entry_setup, regime
        )
        
        # Build report
        key_observations = self._generate_observations(
            tech_indicators, confluence_score, patterns, sr_levels
        )
        
        return AgentReport(
            agent_name=self.name,
            verdict=verdict,
            analysis_details={
                "stock": stock,
                "confluence_score": confluence_score,
                "indicator_analysis": indicator_analysis,
                "patterns_detected": [p.pattern_name for p in patterns],
                "support_levels": [(l.price, l.strength) for l in sr_levels if l.level_type == 'support'],
                "resistance_levels": [(l.price, l.strength) for l in sr_levels if l.level_type == 'resistance'],
            },
            key_observations=key_observations,
            metrics={
                "confluence_score": confluence_score,
                "rsi": tech_indicators.rsi,
                "atr_percent": tech_indicators.atr_percent,
                "adx": tech_indicators.adx
            },
            raw_data={
                "indicators": tech_indicators.__dict__,
                "entry_setup": entry_setup
            }
        )
    
    def _calculate_confluence(self, data: Dict[str, Any], regime: MarketRegime) -> float:
        """Calculate multi-timeframe confluence score (0-100)"""
        weekly = data.get('weekly_trend', 'neutral')
        daily = data.get('daily_trend', 'neutral')
        four_hour = data.get('four_hour_trend', 'neutral')
        one_hour = data.get('one_hour_trend', 'neutral')

        def trend_score(trend):
            return {'bullish': 1, 'bearish': -1, 'neutral': 0}.get(trend, 0)

        # Base multi-timeframe score: max 100, min 0
        # In a BULL market all 4 will be bullish → base = 100
        base = (
            trend_score(weekly) * 40 +
            trend_score(daily) * 30 +
            trend_score(four_hour) * 20 +
            trend_score(one_hour) * 10
        )
        base_norm = (base + 100) / 2   # 0-100

        # --- Stock-specific quality layer: always applied, max ±20 pts ---
        # These differ per stock (RSI, volume, range, extension)
        # and are the primary source of differentiation in a bull market
        rsi_sig       = data.get('rsi_trend',      'neutral')
        volume_sig    = data.get('volume_trend',    'neutral')
        range_sig     = data.get('range_trend',     'neutral')
        extension_sig = data.get('extension_trend', 'neutral')

        quality = (
            trend_score(rsi_sig)       * 8  +   # RSI zone quality   ±8
            trend_score(volume_sig)    * 6  +   # Volume confirm     ±6
            trend_score(range_sig)     * 3  +   # Range position     ±3
            trend_score(extension_sig) * 3      # SMA extension      ±3
        )
        # quality range: -20 to +20

        # Blend: base (0-100) weighted 65%, quality adjusted 35%
        # quality_norm maps -20→30, 0→50, +20→70 → scale to contribute 35%
        quality_norm = (quality + 20) / 40 * 100  # 0-100
        score = base_norm * 0.65 + quality_norm * 0.35
        score = max(0, min(100, score))

        # [FIX-C] Regime adjustment — expanded to cover MILD_BEAR and MILD_BULL.
        # Previously only BULL(+5%) and BEAR(-5%) were handled. This meant
        # MILD_BEAR had zero regime signal, allowing LONG setups through freely.
        # Now we invert the score toward SHORT territory in bearish regimes:
        #   BULL       → +5% boost (bullish bias confirmed)
        #   MILD_BULL  → +3% boost
        #   MILD_BEAR  → score reflected toward 50 then pushed below
        #                (effectively: bearish stocks score higher, bullish lower)
        #   BEAR/CORRECTION → stronger push toward SHORT
        if regime == MarketRegime.BULL and base > 0:
            score = min(100, score * 1.05)
        elif regime == MarketRegime.MILD_BULL and base > 0:
            score = min(100, score * 1.03)
        elif regime in (MarketRegime.MILD_BEAR, MarketRegime.CORRECTION):
            # Invert around 50: a bullish stock (score=70) becomes 30 in MILD_BEAR
            # so that it falls below the SHORT threshold (<35) triggering SHORT verdict.
            # A bearish stock (score=30) becomes 70, rising above LONG threshold (>65).
            score = 100 - score
        elif regime == MarketRegime.BEAR:
            # Stronger version: full inversion + additional push toward SHORT territory
            score = max(0, (100 - score) * 1.05)

        score = max(0, min(100, score))
        return round(score, 1)
    
    def _analyze_indicators(self, indicators: TechnicalIndicators, 
                          price_data: Dict) -> Dict[str, Any]:
        """Analyze technical indicators"""
        analysis = {
            "rsi_signal": "neutral",
            "rsi_zone": "neutral",
            "macd_signal": "neutral",
            "bb_signal": "neutral",
            "volatility_regime": "normal",
            "trend_strength": "weak"
        }
        
        # RSI Analysis
        if indicators.rsi > 70:
            analysis["rsi_signal"] = "overbought"
            analysis["rsi_zone"] = "overbought"
        elif indicators.rsi < 30:
            analysis["rsi_signal"] = "oversold"
            analysis["rsi_zone"] = "oversold"
        elif indicators.rsi > 50:
            analysis["rsi_signal"] = "bullish"
        else:
            analysis["rsi_signal"] = "bearish"
        
        # MACD Analysis
        if indicators.macd > indicators.macd_signal:
            analysis["macd_signal"] = "bullish"
            if indicators.macd_histogram > 0:
                analysis["macd_momentum"] = "strengthening"
        else:
            analysis["macd_signal"] = "bearish"
        
        # Bollinger Bands
        current_price = price_data.get('close', 0)
        if current_price > 0 and indicators.bb_upper > 0:
            if current_price >= indicators.bb_upper:
                analysis["bb_signal"] = "upper_band"
            elif current_price <= indicators.bb_lower:
                analysis["bb_signal"] = "lower_band"
            elif current_price > indicators.bb_middle:
                analysis["bb_signal"] = "upper_half"
            else:
                analysis["bb_signal"] = "lower_half"
        
        # Volatility regime
        if indicators.atr_percent > 3:
            analysis["volatility_regime"] = "high"
        elif indicators.atr_percent < 1:
            analysis["volatility_regime"] = "low"
        
        # Trend strength
        if indicators.adx > 25:
            analysis["trend_strength"] = "strong"
        elif indicators.adx > 20:
            analysis["trend_strength"] = "moderate"
        
        return analysis
    
    def _identify_support_resistance(self, price_data: Dict) -> List[SupportResistanceLevel]:
        """Identify key support and resistance levels"""
        levels = []
        
        # Get price history
        highs = price_data.get('highs', [])
        lows = price_data.get('lows', [])
        closes = price_data.get('closes', [])
        current_price = price_data.get('close', 0)
        
        if not highs or not lows:
            return levels
        
        # Find swing highs (resistance)
        for i in range(1, len(highs) - 1):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                strength = self._calculate_level_strength(highs[i], highs, lows)
                levels.append(SupportResistanceLevel(
                    price=highs[i],
                    strength=strength,
                    level_type='resistance',
                    touches=self._count_touches(highs[i], highs + lows)
                ))
        
        # Find swing lows (support)
        for i in range(1, len(lows) - 1):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                strength = self._calculate_level_strength(lows[i], highs, lows)
                levels.append(SupportResistanceLevel(
                    price=lows[i],
                    strength=strength,
                    level_type='support',
                    touches=self._count_touches(lows[i], highs + lows)
                ))
        
        # Add moving average levels if available
        ma_50 = price_data.get('ma_50')
        ma_200 = price_data.get('ma_200')
        
        if ma_50:
            level_type = 'support' if ma_50 < current_price else 'resistance'
            levels.append(SupportResistanceLevel(
                price=ma_50,
                strength=3,
                level_type=level_type
            ))
        
        if ma_200:
            level_type = 'support' if ma_200 < current_price else 'resistance'
            levels.append(SupportResistanceLevel(
                price=ma_200,
                strength=4,
                level_type=level_type
            ))
        
        # Sort by strength and proximity
        levels.sort(key=lambda x: (-x.strength, abs(x.price - current_price)))
        
        return levels[:10]  # Return top 10 levels
    
    def _calculate_level_strength(self, price: float, highs: List, lows: List) -> int:
        """Calculate strength of a support/resistance level (1-5)"""
        touches = self._count_touches(price, highs + lows, tolerance=0.01)
        return min(5, touches)
    
    def _count_touches(self, price: float, price_list: List, tolerance: float = 0.01) -> int:
        """Count how many times price came near this level"""
        count = 0
        for p in price_list:
            if abs(p - price) / price < tolerance:
                count += 1
        return count
    
    def _detect_patterns(self, price_data: Dict, indicators: TechnicalIndicators,
                        sr_levels: List[SupportResistanceLevel]) -> List[PatternSignal]:
        """Detect chart and candlestick patterns using full PatternRecognitionEngine."""
        patterns = []
        current_price = price_data.get('close', 0)

        # ── FULL PATTERN RECOGNITION ENGINE ───────────────────────────────
        # Use ml_pipeline/pattern_recognition.py instead of the three-pattern
        # simplified inline check.  Falls back gracefully if import fails.
        try:
            import sys, os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from ml_pipeline.pattern_recognition import PatternRecognitionEngine, PatternDirection

            recognizer = PatternRecognitionEngine()
            # Build a minimal OHLCV list from the available price dict.
            # If historical bars are present under 'ohlcv' they're used directly;
            # otherwise we build a single-bar stub (enough for candlestick checks).
            ohlcv_bars = price_data.get("ohlcv", [])
            if not ohlcv_bars:
                ohlcv_bars = [{
                    "open":   price_data.get("open", current_price),
                    "high":   price_data.get("high", current_price),
                    "low":    price_data.get("low", current_price),
                    "close":  current_price,
                    "volume": price_data.get("volume", 0),
                }]

            detected = recognizer.detect_patterns(ohlcv_bars)
            for cp in detected:
                direction_map = {
                    PatternDirection.BULLISH: "bullish",
                    PatternDirection.BEARISH: "bearish",
                    PatternDirection.NEUTRAL: "neutral",
                }
                patterns.append(PatternSignal(
                    pattern_name=cp.pattern_type.value.lower(),
                    pattern_type=direction_map.get(cp.direction, "neutral"),
                    probability=round(cp.confidence, 2),
                    status="confirmed" if cp.confidence >= 0.70 else "forming",
                ))
        except Exception as _pr_err:
            # Graceful fallback to the original simplified candlestick checks
            pass
        # ── END PATTERN RECOGNITION ENGINE ────────────────────────────────

        # ── LEGACY INLINE CHECKS (always run as supplementary signals) ────
        open_price = price_data.get('open', current_price)
        high_price = price_data.get('high', current_price)
        low_price = price_data.get('low', current_price)
        close_price = current_price

        # Doji detection
        body = abs(close_price - open_price)
        total_range = high_price - low_price if high_price != low_price else 1
        if body / total_range < 0.1:
            near_support = any(abs(l.price - current_price) / current_price < 0.02
                             for l in sr_levels if l.level_type == 'support')
            near_resistance = any(abs(l.price - current_price) / current_price < 0.02
                                for l in sr_levels if l.level_type == 'resistance')

            if near_support:
                patterns.append(PatternSignal(
                    pattern_name="doji_at_support",
                    pattern_type="bullish",
                    probability=0.65,
                    status="confirmed"
                ))
            elif near_resistance:
                patterns.append(PatternSignal(
                    pattern_name="doji_at_resistance",
                    pattern_type="bearish",
                    probability=0.65,
                    status="confirmed"
                ))

        # Bollinger Band squeeze
        if indicators.bb_width > 0:
            avg_bb_width = price_data.get('avg_bb_width', indicators.bb_width)
            if indicators.bb_width < avg_bb_width * 0.7:
                patterns.append(PatternSignal(
                    pattern_name="bb_squeeze",
                    pattern_type="neutral",
                    probability=0.70,
                    status="forming"
                ))

        # RSI divergence (simplified check)
        if indicators.rsi > 70 and close_price > open_price:
            patterns.append(PatternSignal(
                pattern_name="potential_bearish_divergence",
                pattern_type="bearish",
                probability=0.60,
                status="watching"
            ))
        elif indicators.rsi < 30 and close_price < open_price:
            patterns.append(PatternSignal(
                pattern_name="potential_bullish_divergence",
                pattern_type="bullish",
                probability=0.60,
                status="watching"
            ))

        return patterns
    
    def _calculate_entry_setup(self, current_price: float, indicators: TechnicalIndicators,
                              sr_levels: List[SupportResistanceLevel],
                              patterns: List[PatternSignal],
                              regime: MarketRegime) -> Dict[str, Any]:
        """Calculate entry, stop loss, and target levels"""
        setup = {
            "entry_zone": (current_price * 0.99, current_price * 1.01),
            "stop_loss": None,
            "target_1": None,
            "target_2": None,
            "risk_reward": 0,
            "direction": TradeDirection.NEUTRAL
        }
        
        if current_price <= 0:
            return setup
        
        # Calculate ATR-based stop
        if indicators.atr > 0:
            atr_stop = indicators.atr * 1.5
        else:
            atr_stop = current_price * 0.03  # Default 3%
        
        # Determine direction based on patterns and levels
        bullish_score = 0
        bearish_score = 0
        
        for pattern in patterns:
            if pattern.pattern_type == "bullish":
                bullish_score += pattern.probability
            elif pattern.pattern_type == "bearish":
                bearish_score += pattern.probability
        
        # Check nearest support/resistance
        nearest_support = min(
            [l for l in sr_levels if l.level_type == 'support' and l.price < current_price],
            key=lambda x: current_price - x.price,
            default=None
        )
        nearest_resistance = min(
            [l for l in sr_levels if l.level_type == 'resistance' and l.price > current_price],
            key=lambda x: x.price - current_price,
            default=None
        )
        
        # Determine setup direction
        if bullish_score > bearish_score and nearest_support:
            setup["direction"] = TradeDirection.LONG
            setup["stop_loss"] = current_price - atr_stop
            if nearest_resistance and nearest_resistance.price > current_price:
                setup["target_1"] = nearest_resistance.price
                setup["target_2"] = nearest_resistance.price * 1.05
            else:
                setup["target_1"] = current_price + atr_stop * 2.0
                setup["target_2"] = current_price + atr_stop * 3.0
        elif bearish_score > bullish_score and nearest_resistance:
            setup["direction"] = TradeDirection.SHORT
            setup["stop_loss"] = current_price + atr_stop
            # FIX-DIRECTION-ROOT: For SHORT, target must be BELOW entry.
            # nearest_support may be ABOVE current_price if the stock has
            # fallen through previous support levels. In that case the
            # structural target is directionally wrong — use ATR instead.
            if nearest_support and nearest_support.price < current_price:
                setup["target_1"] = nearest_support.price
                setup["target_2"] = nearest_support.price * 0.95
            else:
                setup["target_1"] = current_price - atr_stop * 2.0
                setup["target_2"] = current_price - atr_stop * 3.0
        
        # Calculate risk-reward
        if setup["stop_loss"] and setup["target_1"]:
            risk = abs(current_price - setup["stop_loss"])
            reward = abs(setup["target_1"] - current_price)
            if risk > 0:
                setup["risk_reward"] = reward / risk
        
        return setup
    
    def _generate_verdict(self, confluence_score: float, indicator_analysis: Dict,
                         patterns: List[PatternSignal], entry_setup: Dict,
                         regime: MarketRegime) -> AgentVerdict:
        """Generate the final verdict"""
        # Determine direction
        direction = entry_setup.get("direction", TradeDirection.NEUTRAL)
        
        # If no clear setup from entry, use confluence
        if direction == TradeDirection.NEUTRAL:
            if confluence_score > 65:
                direction = TradeDirection.LONG
            elif confluence_score < 35:
                direction = TradeDirection.SHORT
        
        # Calculate conviction — proportional, not bucket-based
        # Base: 50. Confluence maps linearly: score=50→+0, score=100→+25, score=0→-25
        conviction = 50 + (confluence_score - 50) * 0.5

        # RSI contribution (fine-grained, not just overbought/oversold)
        rsi_zone = indicator_analysis.get("rsi_zone", "neutral")
        rsi_value = indicator_analysis.get("rsi_value", 50)
        if direction == TradeDirection.LONG:
            if 45 <= rsi_value <= 60:   conviction += 8   # ideal momentum zone
            elif rsi_value < 45:        conviction += 4   # oversold bounce
            elif rsi_value > 70:        conviction -= 10  # overbought
            elif rsi_value > 65:        conviction -= 4   # getting stretched
        elif direction == TradeDirection.SHORT:
            if 40 <= rsi_value <= 55:   conviction += 8
            elif rsi_value > 55:        conviction += 4
            elif rsi_value < 30:        conviction -= 10
        
        # Pattern contribution
        for pattern in patterns:
            if pattern.pattern_type == direction.value.lower():
                conviction += pattern.probability * 10
        
        # Risk-reward contribution
        rr = entry_setup.get("risk_reward", 0)
        if rr >= 2.0:
            conviction += 10
        elif rr >= 1.5:
            conviction += 5
        elif rr < 1.0:
            conviction -= 10
        
        # [FIX-C] Regime adjustment — expanded penalties for direction/regime mismatch.
        # Old: only BEAR penalised LONG by -5, BULL penalised SHORT by -5.
        # This was far too gentle — MILD_BEAR had zero regime penalty on LONG setups.
        # New: graded penalties ensure wrong-direction setups lose significant conviction.
        if direction == TradeDirection.LONG:
            if regime == MarketRegime.BEAR:
                conviction -= 20    # heavy penalty — LONG in BEAR is almost always wrong
            elif regime in (MarketRegime.MILD_BEAR, MarketRegime.CORRECTION):
                conviction -= 12    # meaningful penalty — most LONG setups will fail regime gate
        elif direction == TradeDirection.SHORT:
            if regime == MarketRegime.BULL:
                conviction -= 20
            elif regime == MarketRegime.MILD_BULL:
                conviction -= 12
        
        # Clamp conviction
        conviction = max(0, min(100, conviction))
        
        # Generate reason
        reasons = []
        if confluence_score > 65:
            reasons.append(f"Strong confluence ({confluence_score:.0f}%)")
        if rsi_zone in ["oversold", "overbought"]:
            reasons.append(f"RSI {rsi_zone}")
        if patterns:
            reasons.append(f"Pattern: {patterns[0].pattern_name}")
        
        reason = "; ".join(reasons) if reasons else "Mixed signals"
        
        # Generate risks
        risks = []
        if confluence_score < 50:
            risks.append("Low timeframe alignment")
        if rsi_zone == "overbought" and direction == TradeDirection.LONG:
            risks.append("Overbought conditions")
        if rr < 1.5:
            risks.append("Poor risk-reward ratio")
        
        return AgentVerdict(
            direction=direction,
            conviction=conviction,
            weight=self.current_weight,
            reason=reason,
            risks=risks
        )
    
    def _generate_observations(self, indicators: TechnicalIndicators,
                              confluence_score: float,
                              patterns: List[PatternSignal],
                              sr_levels: List[SupportResistanceLevel]) -> List[str]:
        """Generate key observations for the report"""
        observations = []
        
        # Confluence observation
        if confluence_score > 75:
            observations.append(f"Strong multi-timeframe alignment ({confluence_score:.0f}%)")
        elif confluence_score < 35:
            observations.append(f"Poor timeframe alignment ({confluence_score:.0f}%)")
        
        # RSI observation
        if indicators.rsi > 70:
            observations.append(f"RSI overbought at {indicators.rsi:.1f}")
        elif indicators.rsi < 30:
            observations.append(f"RSI oversold at {indicators.rsi:.1f}")
        
        # Volatility observation
        if indicators.atr_percent > 3:
            observations.append(f"High volatility: ATR {indicators.atr_percent:.1f}% of price")
        
        # Pattern observations
        for pattern in patterns[:2]:
            observations.append(f"Pattern detected: {pattern.pattern_name} ({pattern.status})")
        
        # S/R observations
        strong_levels = [l for l in sr_levels if l.strength >= 3]
        if strong_levels:
            observations.append(f"Key {strong_levels[0].level_type} at {strong_levels[0].price:.2f} ({strong_levels[0].strength} stars)")
        
        return observations
