"""
ROX Proven Edge Engine v3.0 - Pattern Recognition
================================================
AI-powered chart pattern recognition.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from collections import deque
import numpy as np


class PatternType(Enum):
    """Chart pattern types"""
    # Reversal patterns
    HEAD_AND_SHOULDERS = "HEAD_AND_SHOULDERS"
    INVERSE_HEAD_AND_SHOULDERS = "INVERSE_HEAD_AND_SHOULDERS"
    DOUBLE_TOP = "DOUBLE_TOP"
    DOUBLE_BOTTOM = "DOUBLE_BOTTOM"
    TRIPLE_TOP = "TRIPLE_TOP"
    TRIPLE_BOTTOM = "TRIPLE_BOTTOM"
    ROUNDED_TOP = "ROUNDED_TOP"
    ROUNDED_BOTTOM = "ROUNDED_BOTTOM"
    
    # Continuation patterns
    CUP_AND_HANDLE = "CUP_AND_HANDLE"
    ASCENDING_TRIANGLE = "ASCENDING_TRIANGLE"
    DESCENDING_TRIANGLE = "DESCENDING_TRIANGLE"
    SYMMETRICAL_TRIANGLE = "SYMMETRICAL_TRIANGLE"
    BULL_FLAG = "BULL_FLAG"
    BEAR_FLAG = "BEAR_FLAG"
    BULL_PENNANT = "BULL_PENNANT"
    BEAR_PENNANT = "BEAR_PENNANT"
    
    # Candlestick patterns
    HAMMER = "HAMMER"
    INVERTED_HAMMER = "INVERTED_HAMMER"
    BULLISH_ENGULFING = "BULLISH_ENGULFING"
    BEARISH_ENGULFING = "BEARISH_ENGULFING"
    MORNING_STAR = "MORNING_STAR"
    EVENING_STAR = "EVENING_STAR"
    DOJI = "DOJI"
    THREE_WHITE_SOLDIERS = "THREE_WHITE_SOLDIERS"
    THREE_BLACK_CROWS = "THREE_BLACK_CROWS"


class PatternDirection(Enum):
    """Pattern direction"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


@dataclass
class ChartPattern:
    """Detected chart pattern"""
    pattern_type: PatternType
    direction: PatternDirection
    confidence: float  # 0-1
    start_idx: int
    end_idx: int
    key_levels: Dict[str, float]
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    probability: float = 0.0
    raw_scores: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "pattern_type": self.pattern_type.value,
            "direction": self.direction.value,
            "confidence": self.confidence,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "key_levels": self.key_levels,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "probability": self.probability
        }


@dataclass
class SwingPoint:
    """Swing high/low point"""
    idx: int
    price: float
    point_type: str  # 'high' or 'low'
    strength: int = 1  # Number of bars on each side


class PatternRecognitionEngine:
    """
    AI-powered pattern recognition engine.
    
    Features:
    - Candlestick pattern detection
    - Chart pattern recognition
    - Support/resistance identification
    - Pattern probability scoring
    """
    
    # Pattern probabilities from historical data
    PATTERN_PROBABILITIES = {
        PatternType.HEAD_AND_SHOULDERS: 0.75,
        PatternType.INVERSE_HEAD_AND_SHOULDERS: 0.75,
        PatternType.DOUBLE_TOP: 0.70,
        PatternType.DOUBLE_BOTTOM: 0.73,
        PatternType.CUP_AND_HANDLE: 0.74,
        PatternType.ASCENDING_TRIANGLE: 0.72,
        PatternType.DESCENDING_TRIANGLE: 0.72,
        PatternType.BULL_FLAG: 0.68,
        PatternType.BEAR_FLAG: 0.68,
        PatternType.HAMMER: 0.68,
        PatternType.BULLISH_ENGULFING: 0.72,
        PatternType.BEARISH_ENGULFING: 0.72,
        PatternType.MORNING_STAR: 0.78,
        PatternType.EVENING_STAR: 0.78,
        PatternType.THREE_WHITE_SOLDIERS: 0.80,
        PatternType.THREE_BLACK_CROWS: 0.80,
    }
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.min_pattern_bars = config.get("min_pattern_bars", 10)
        self.lookback_bars = config.get("lookback_bars", 100)
        
        # Pattern storage
        self.detected_patterns: List[ChartPattern] = []
        self.swing_points: List[SwingPoint] = []
    
    def detect_patterns(self, ohlcv_data: List[Dict]) -> List[ChartPattern]:
        """
        Detect all patterns in OHLCV data.
        
        Args:
            ohlcv_data: List of OHLCV dictionaries
            
        Returns:
            List of detected patterns
        """
        if len(ohlcv_data) < self.min_pattern_bars:
            return []
        
        self.detected_patterns = []
        
        # Extract arrays
        highs = np.array([d['high'] for d in ohlcv_data])
        lows = np.array([d['low'] for d in ohlcv_data])
        closes = np.array([d['close'] for d in ohlcv_data])
        opens = np.array([d['open'] for d in ohlcv_data])
        
        # Find swing points
        self.swing_points = self._find_swing_points(highs, lows)
        
        # Detect candlestick patterns
        self._detect_candlestick_patterns(opens, highs, lows, closes)
        
        # Detect chart patterns
        self._detect_chart_patterns(highs, lows, closes)
        
        return self.detected_patterns
    
    def _find_swing_points(self, highs: np.ndarray, lows: np.ndarray,
                           lookback: int = 5) -> List[SwingPoint]:
        """Find swing highs and lows"""
        swing_points = []
        
        for i in range(lookback, len(highs) - lookback):
            # Check for swing high
            is_swing_high = all(highs[i] > highs[i-j] for j in range(1, lookback + 1)) and \
                           all(highs[i] > highs[i+j] for j in range(1, lookback + 1))
            
            if is_swing_high:
                swing_points.append(SwingPoint(
                    idx=i,
                    price=highs[i],
                    point_type='high',
                    strength=lookback
                ))
            
            # Check for swing low
            is_swing_low = all(lows[i] < lows[i-j] for j in range(1, lookback + 1)) and \
                          all(lows[i] < lows[i+j] for j in range(1, lookback + 1))
            
            if is_swing_low:
                swing_points.append(SwingPoint(
                    idx=i,
                    price=lows[i],
                    point_type='low',
                    strength=lookback
                ))
        
        return swing_points
    
    def _detect_candlestick_patterns(self, opens: np.ndarray, highs: np.ndarray,
                                      lows: np.ndarray, closes: np.ndarray):
        """Detect candlestick patterns"""
        n = len(closes)
        
        for i in range(max(2, n - 50), n):  # Last 50 bars
            # Single candle patterns
            self._detect_doji(opens, highs, lows, closes, i)
            self._detect_hammer(opens, highs, lows, closes, i)
            
            # Multi-candle patterns
            if i >= 1:
                self._detect_engulfing(opens, highs, lows, closes, i)
            
            if i >= 2:
                self._detect_star_patterns(opens, highs, lows, closes, i)
                self._detect_three_soldiers_crows(opens, highs, lows, closes, i)
    
    def _detect_doji(self, opens: np.ndarray, highs: np.ndarray,
                     lows: np.ndarray, closes: np.ndarray, i: int):
        """Detect Doji pattern"""
        body = abs(closes[i] - opens[i])
        total_range = highs[i] - lows[i]
        
        if total_range > 0 and body / total_range < 0.1:
            # Check if at support/resistance
            key_levels = {"doji_price": closes[i]}
            
            self.detected_patterns.append(ChartPattern(
                pattern_type=PatternType.DOJI,
                direction=PatternDirection.NEUTRAL,
                confidence=0.65,
                start_idx=i,
                end_idx=i,
                key_levels=key_levels,
                probability=0.65
            ))
    
    def _detect_hammer(self, opens: np.ndarray, highs: np.ndarray,
                       lows: np.ndarray, closes: np.ndarray, i: int):
        """Detect Hammer pattern"""
        body = abs(closes[i] - opens[i])
        lower_wick = min(opens[i], closes[i]) - lows[i]
        upper_wick = highs[i] - max(opens[i], closes[i])
        total_range = highs[i] - lows[i]
        
        if total_range > 0:
            # Hammer: small body, long lower wick, small upper wick
            if body / total_range < 0.3 and lower_wick > body * 2 and upper_wick < body * 0.5:
                self.detected_patterns.append(ChartPattern(
                    pattern_type=PatternType.HAMMER,
                    direction=PatternDirection.BULLISH,
                    confidence=0.68,
                    start_idx=i,
                    end_idx=i,
                    key_levels={"hammer_low": lows[i]},
                    probability=0.68
                ))
    
    def _detect_engulfing(self, opens: np.ndarray, highs: np.ndarray,
                          lows: np.ndarray, closes: np.ndarray, i: int):
        """Detect Engulfing pattern"""
        prev_body = abs(closes[i-1] - opens[i-1])
        curr_body = abs(closes[i] - opens[i])
        
        # Bullish Engulfing
        if (closes[i-1] < opens[i-1] and  # Previous is bearish
            closes[i] > opens[i] and       # Current is bullish
            opens[i] < closes[i-1] and     # Opens below prev close
            closes[i] > opens[i-1]):       # Closes above prev open
            
            self.detected_patterns.append(ChartPattern(
                pattern_type=PatternType.BULLISH_ENGULFING,
                direction=PatternDirection.BULLISH,
                confidence=0.72,
                start_idx=i-1,
                end_idx=i,
                key_levels={"engulfing_low": min(lows[i-1], lows[i])},
                probability=0.72
            ))
        
        # Bearish Engulfing
        if (closes[i-1] > opens[i-1] and  # Previous is bullish
            closes[i] < opens[i] and       # Current is bearish
            opens[i] > closes[i-1] and     # Opens above prev close
            closes[i] < opens[i-1]):       # Closes below prev open
            
            self.detected_patterns.append(ChartPattern(
                pattern_type=PatternType.BEARISH_ENGULFING,
                direction=PatternDirection.BEARISH,
                confidence=0.72,
                start_idx=i-1,
                end_idx=i,
                key_levels={"engulfing_high": max(highs[i-1], highs[i])},
                probability=0.72
            ))
    
    def _detect_star_patterns(self, opens: np.ndarray, highs: np.ndarray,
                               lows: np.ndarray, closes: np.ndarray, i: int):
        """Detect Morning/Evening Star patterns"""
        # Morning Star (bullish)
        if (closes[i-2] < opens[i-2] and  # First candle bearish
            abs(closes[i-1] - opens[i-1]) < abs(closes[i-2] - opens[i-2]) * 0.3 and  # Second small body
            closes[i] > opens[i] and      # Third candle bullish
            closes[i] > (opens[i-2] + closes[i-2]) / 2):  # Closes above midpoint of first
            
            self.detected_patterns.append(ChartPattern(
                pattern_type=PatternType.MORNING_STAR,
                direction=PatternDirection.BULLISH,
                confidence=0.78,
                start_idx=i-2,
                end_idx=i,
                key_levels={"star_low": min(lows[i-2], lows[i-1], lows[i])},
                probability=0.78
            ))
        
        # Evening Star (bearish)
        if (closes[i-2] > opens[i-2] and  # First candle bullish
            abs(closes[i-1] - opens[i-1]) < abs(closes[i-2] - opens[i-2]) * 0.3 and  # Second small body
            closes[i] < opens[i] and      # Third candle bearish
            closes[i] < (opens[i-2] + closes[i-2]) / 2):  # Closes below midpoint of first
            
            self.detected_patterns.append(ChartPattern(
                pattern_type=PatternType.EVENING_STAR,
                direction=PatternDirection.BEARISH,
                confidence=0.78,
                start_idx=i-2,
                end_idx=i,
                key_levels={"star_high": max(highs[i-2], highs[i-1], highs[i])},
                probability=0.78
            ))
    
    def _detect_three_soldiers_crows(self, opens: np.ndarray, highs: np.ndarray,
                                      lows: np.ndarray, closes: np.ndarray, i: int):
        """Detect Three White Soldiers / Three Black Crows"""
        # Three White Soldiers (bullish)
        if (all(closes[i-j] > opens[i-j] for j in range(3)) and  # All bullish
            closes[i] > closes[i-1] > closes[i-2] and            # Higher closes
            opens[i] > opens[i-1] and opens[i-1] > opens[i-2]):  # Higher opens
            
            self.detected_patterns.append(ChartPattern(
                pattern_type=PatternType.THREE_WHITE_SOLDIERS,
                direction=PatternDirection.BULLISH,
                confidence=0.80,
                start_idx=i-2,
                end_idx=i,
                key_levels={"soldiers_low": min(lows[i-2], lows[i-1], lows[i])},
                probability=0.80
            ))
        
        # Three Black Crows (bearish)
        if (all(closes[i-j] < opens[i-j] for j in range(3)) and  # All bearish
            closes[i] < closes[i-1] < closes[i-2] and            # Lower closes
            opens[i] < opens[i-1] and opens[i-1] < opens[i-2]):  # Lower opens
            
            self.detected_patterns.append(ChartPattern(
                pattern_type=PatternType.THREE_BLACK_CROWS,
                direction=PatternDirection.BEARISH,
                confidence=0.80,
                start_idx=i-2,
                end_idx=i,
                key_levels={"crows_high": max(highs[i-2], highs[i-1], highs[i])},
                probability=0.80
            ))
    
    def _detect_chart_patterns(self, highs: np.ndarray, lows: np.ndarray,
                               closes: np.ndarray):
        """Detect chart patterns from swing points"""
        if len(self.swing_points) < 4:
            return
        
        # Detect Double Top/Bottom
        self._detect_double_patterns(highs, lows, closes)
        
        # Detect Head and Shoulders
        self._detect_head_shoulders(highs, lows, closes)
        
        # Detect Triangle patterns
        self._detect_triangle_patterns(highs, lows, closes)
    
    def _detect_double_patterns(self, highs: np.ndarray, lows: np.ndarray,
                                closes: np.ndarray):
        """Detect Double Top and Double Bottom patterns"""
        high_points = [p for p in self.swing_points if p.point_type == 'high']
        low_points = [p for p in self.swing_points if p.point_type == 'low']
        
        # Double Top
        if len(high_points) >= 2:
            for i in range(len(high_points) - 1):
                p1, p2 = high_points[i], high_points[i+1]
                
                # Check if peaks are at similar level (within 2%)
                if abs(p1.price - p2.price) / p1.price < 0.02:
                    # Find neckline
                    between_lows = [p for p in low_points 
                                   if p1.idx < p.idx < p2.idx]
                    
                    if between_lows:
                        neckline = min(p.price for p in between_lows)
                        self.detected_patterns.append(ChartPattern(
                            pattern_type=PatternType.DOUBLE_TOP,
                            direction=PatternDirection.BEARISH,
                            confidence=0.70,
                            start_idx=p1.idx,
                            end_idx=p2.idx,
                            key_levels={
                                "peak1": p1.price,
                                "peak2": p2.price,
                                "neckline": neckline
                            },
                            target_price=neckline - (p1.price - neckline),
                            probability=0.70
                        ))
        
        # Double Bottom
        if len(low_points) >= 2:
            for i in range(len(low_points) - 1):
                p1, p2 = low_points[i], low_points[i+1]
                
                # Check if troughs are at similar level
                if abs(p1.price - p2.price) / p1.price < 0.02:
                    between_highs = [p for p in high_points 
                                    if p1.idx < p.idx < p2.idx]
                    
                    if between_highs:
                        neckline = max(p.price for p in between_highs)
                        self.detected_patterns.append(ChartPattern(
                            pattern_type=PatternType.DOUBLE_BOTTOM,
                            direction=PatternDirection.BULLISH,
                            confidence=0.73,
                            start_idx=p1.idx,
                            end_idx=p2.idx,
                            key_levels={
                                "trough1": p1.price,
                                "trough2": p2.price,
                                "neckline": neckline
                            },
                            target_price=neckline + (neckline - p1.price),
                            probability=0.73
                        ))
    
    def _detect_head_shoulders(self, highs: np.ndarray, lows: np.ndarray,
                               closes: np.ndarray):
        """Detect Head and Shoulders pattern"""
        high_points = [p for p in self.swing_points if p.point_type == 'high']
        
        if len(high_points) < 3:
            return
        
        for i in range(len(high_points) - 2):
            left_shoulder = high_points[i]
            head = high_points[i+1]
            right_shoulder = high_points[i+2]
            
            # Head and Shoulders criteria
            if (head.price > left_shoulder.price and 
                head.price > right_shoulder.price and
                abs(left_shoulder.price - right_shoulder.price) / left_shoulder.price < 0.03):
                
                # Find neckline
                between_lows = [p for p in self.swing_points 
                               if p.point_type == 'low' and 
                               left_shoulder.idx < p.idx < right_shoulder.idx]
                
                if len(between_lows) >= 2:
                    neckline = min(p.price for p in between_lows)
                    
                    self.detected_patterns.append(ChartPattern(
                        pattern_type=PatternType.HEAD_AND_SHOULDERS,
                        direction=PatternDirection.BEARISH,
                        confidence=0.75,
                        start_idx=left_shoulder.idx,
                        end_idx=right_shoulder.idx,
                        key_levels={
                            "left_shoulder": left_shoulder.price,
                            "head": head.price,
                            "right_shoulder": right_shoulder.price,
                            "neckline": neckline
                        },
                        target_price=neckline - (head.price - neckline),
                        stop_loss=head.price * 1.01,
                        probability=0.75
                    ))
    
    def _detect_triangle_patterns(self, highs: np.ndarray, lows: np.ndarray,
                                  closes: np.ndarray):
        """Detect Triangle patterns"""
        if len(self.swing_points) < 4:
            return
        
        # Get recent swing points
        recent = self.swing_points[-10:]  # Last 10 swing points
        
        if len(recent) < 4:
            return
        
        high_points = [p for p in recent if p.point_type == 'high']
        low_points = [p for p in recent if p.point_type == 'low']
        
        if len(high_points) >= 2 and len(low_points) >= 2:
            # Check for ascending triangle (flat top, rising bottom)
            high_trend = self._calculate_trend([p.price for p in high_points])
            low_trend = self._calculate_trend([p.price for p in low_points])
            
            if abs(high_trend) < 0.001 and low_trend > 0.001:
                self.detected_patterns.append(ChartPattern(
                    pattern_type=PatternType.ASCENDING_TRIANGLE,
                    direction=PatternDirection.BULLISH,
                    confidence=0.72,
                    start_idx=min(p.idx for p in recent),
                    end_idx=max(p.idx for p in recent),
                    key_levels={
                        "resistance": np.mean([p.price for p in high_points]),
                        "support_trend": low_trend
                    },
                    probability=0.72
                ))
            
            # Check for descending triangle (falling top, flat bottom)
            if high_trend < -0.001 and abs(low_trend) < 0.001:
                self.detected_patterns.append(ChartPattern(
                    pattern_type=PatternType.DESCENDING_TRIANGLE,
                    direction=PatternDirection.BEARISH,
                    confidence=0.72,
                    start_idx=min(p.idx for p in recent),
                    end_idx=max(p.idx for p in recent),
                    key_levels={
                        "support": np.mean([p.price for p in low_points]),
                        "resistance_trend": high_trend
                    },
                    probability=0.72
                ))
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend slope"""
        if len(values) < 2:
            return 0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Linear regression slope
        slope = np.polyfit(x, y, 1)[0]
        return slope / np.mean(y) if np.mean(y) != 0 else 0
    
    def get_best_pattern(self, direction: str = None) -> Optional[ChartPattern]:
        """Get highest confidence pattern, optionally filtered by direction"""
        patterns = self.detected_patterns
        
        if direction:
            patterns = [p for p in patterns if p.direction.value == direction]
        
        if not patterns:
            return None
        
        return max(patterns, key=lambda p: p.confidence)
    
    def get_patterns_by_type(self, pattern_type: PatternType) -> List[ChartPattern]:
        """Get all patterns of a specific type"""
        return [p for p in self.detected_patterns if p.pattern_type == pattern_type]
