"""
ROX Proven Edge Engine v3.0 - Streaming Indicators
=================================================
Real-time technical indicator calculations.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
from enum import Enum


class IndicatorState(Enum):
    """Indicator signal states"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    OVERBOUGHT = "OVERBOUGHT"
    OVERSOLD = "OVERSOLD"


@dataclass
class IndicatorValue:
    """Single indicator value"""
    value: float
    signal: IndicatorState
    timestamp: float = 0.0
    raw_components: Dict = field(default_factory=dict)


class StreamingEMA:
    """Streaming Exponential Moving Average"""
    
    def __init__(self, period: int):
        self.period = period
        self.alpha = 2 / (period + 1)
        self.ema: Optional[float] = None
        self.initialized = False
        self.count = 0
    
    def update(self, price: float) -> float:
        """Update EMA with new price"""
        self.count += 1
        
        if self.ema is None:
            self.ema = price
        else:
            self.ema = self.alpha * price + (1 - self.alpha) * self.ema
        
        if self.count >= self.period:
            self.initialized = True
        
        return self.ema if self.ema is not None else price
    
    def get(self) -> Optional[float]:
        return self.ema


class StreamingSMA:
    """Streaming Simple Moving Average"""
    
    def __init__(self, period: int):
        self.period = period
        self.buffer: deque = deque(maxlen=period)
        self.sum = 0.0
    
    def update(self, price: float) -> float:
        """Update SMA with new price"""
        if len(self.buffer) == self.period:
            self.sum -= self.buffer[0]
        
        self.buffer.append(price)
        self.sum += price
        
        return self.sum / len(self.buffer)
    
    def get(self) -> Optional[float]:
        if len(self.buffer) < self.period:
            return None
        return self.sum / len(self.buffer)


class StreamingRSI:
    """Streaming Relative Strength Index"""
    
    def __init__(self, period: int = 14):
        self.period = period
        self.prev_price: Optional[float] = None
        self.gains: deque = deque(maxlen=period)
        self.losses: deque = deque(maxlen=period)
        self.avg_gain = 0.0
        self.avg_loss = 0.0
        self.initialized = False
        self.count = 0
    
    def update(self, price: float) -> IndicatorValue:
        """Update RSI with new price"""
        self.count += 1
        
        if self.prev_price is None:
            self.prev_price = price
            return IndicatorValue(
                value=50.0,
                signal=IndicatorState.NEUTRAL
            )
        
        change = price - self.prev_price
        gain = max(0, change)
        loss = max(0, -change)
        
        self.prev_price = price
        
        if not self.initialized:
            self.gains.append(gain)
            self.losses.append(loss)
            
            if self.count >= self.period:
                self.avg_gain = sum(self.gains) / self.period
                self.avg_loss = sum(self.losses) / self.period
                self.initialized = True
        else:
            # Smoothed average
            self.avg_gain = (self.avg_gain * (self.period - 1) + gain) / self.period
            self.avg_loss = (self.avg_loss * (self.period - 1) + loss) / self.period
        
        if not self.initialized or self.avg_loss == 0:
            rsi = 50.0
        else:
            rs = self.avg_gain / self.avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # Determine signal
        if rsi > 70:
            signal = IndicatorState.OVERBOUGHT
        elif rsi < 30:
            signal = IndicatorState.OVERSOLD
        elif rsi > 50:
            signal = IndicatorState.BULLISH
        else:
            signal = IndicatorState.BEARISH
        
        return IndicatorValue(
            value=rsi,
            signal=signal,
            raw_components={
                "avg_gain": self.avg_gain,
                "avg_loss": self.avg_loss
            }
        )


class StreamingMACD:
    """Streaming MACD indicator"""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, 
                 signal_period: int = 9):
        self.fast_ema = StreamingEMA(fast_period)
        self.slow_ema = StreamingEMA(slow_period)
        self.signal_ema = StreamingEMA(signal_period)
        
        self.macd_value: Optional[float] = None
        self.signal_value: Optional[float] = None
        self.histogram: float = 0.0
    
    def update(self, price: float) -> IndicatorValue:
        """Update MACD with new price"""
        fast = self.fast_ema.update(price)
        slow = self.slow_ema.update(price)
        
        if self.fast_ema.initialized and self.slow_ema.initialized:
            self.macd_value = fast - slow
            self.signal_value = self.signal_ema.update(self.macd_value)
            
            if self.signal_value is not None:
                self.histogram = self.macd_value - self.signal_value
        
        # Determine signal
        if self.macd_value is None or self.signal_value is None:
            signal = IndicatorState.NEUTRAL
        elif self.macd_value > self.signal_value and self.histogram > 0:
            signal = IndicatorState.BULLISH
        elif self.macd_value < self.signal_value and self.histogram < 0:
            signal = IndicatorState.BEARISH
        else:
            signal = IndicatorState.NEUTRAL
        
        return IndicatorValue(
            value=self.macd_value or 0,
            signal=signal,
            raw_components={
                "macd": self.macd_value,
                "signal": self.signal_value,
                "histogram": self.histogram
            }
        )


class StreamingATR:
    """Streaming Average True Range"""
    
    def __init__(self, period: int = 14):
        self.period = period
        self.tr_buffer: deque = deque(maxlen=period)
        self.atr: Optional[float] = None
        self.prev_high: Optional[float] = None
        self.prev_low: Optional[float] = None
        self.prev_close: Optional[float] = None
        self.initialized = False
    
    def update(self, high: float, low: float, close: float) -> IndicatorValue:
        """Update ATR with new OHLC"""
        # Calculate True Range
        if self.prev_close is None:
            tr = high - low
        else:
            tr = max(
                high - low,
                abs(high - self.prev_close),
                abs(low - self.prev_close)
            )
        
        self.prev_high = high
        self.prev_low = low
        self.prev_close = close
        
        self.tr_buffer.append(tr)
        
        if len(self.tr_buffer) >= self.period:
            self.atr = sum(self.tr_buffer) / self.period
            self.initialized = True
        
        return IndicatorValue(
            value=self.atr or tr,
            signal=IndicatorState.NEUTRAL,
            raw_components={"tr": tr}
        )
    
    def get_volatility_percent(self, price: float) -> float:
        """Get ATR as percentage of price"""
        if self.atr is None or price == 0:
            return 0
        return (self.atr / price) * 100


class StreamingBollingerBands:
    """Streaming Bollinger Bands"""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev
        self.sma = StreamingSMA(period)
        self.price_buffer: deque = deque(maxlen=period)
        
        self.middle: float = 0.0
        self.upper: float = 0.0
        self.lower: float = 0.0
        self.bandwidth: float = 0.0
    
    def update(self, price: float) -> IndicatorValue:
        """Update Bollinger Bands"""
        self.middle = self.sma.update(price)
        self.price_buffer.append(price)
        
        if len(self.price_buffer) >= self.period:
            # Calculate standard deviation
            mean = self.middle
            variance = sum((p - mean) ** 2 for p in self.price_buffer) / self.period
            std = math.sqrt(variance)
            
            self.upper = self.middle + (self.std_dev * std)
            self.lower = self.middle - (self.std_dev * std)
            
            if self.middle > 0:
                self.bandwidth = (self.upper - self.lower) / self.middle * 100
        
        # Determine signal
        if price >= self.upper:
            signal = IndicatorState.OVERBOUGHT
        elif price <= self.lower:
            signal = IndicatorState.OVERSOLD
        elif price > self.middle:
            signal = IndicatorState.BULLISH
        else:
            signal = IndicatorState.BEARISH
        
        return IndicatorValue(
            value=self.middle,
            signal=signal,
            raw_components={
                "upper": self.upper,
                "middle": self.middle,
                "lower": self.lower,
                "bandwidth": self.bandwidth
            }
        )


class StreamingADX:
    """Streaming Average Directional Index"""
    
    def __init__(self, period: int = 14):
        self.period = period
        self.plus_dm_buffer: deque = deque(maxlen=period)
        self.minus_dm_buffer: deque = deque(maxlen=period)
        self.tr_buffer: deque = deque(maxlen=period)
        
        self.prev_high: Optional[float] = None
        self.prev_low: Optional[float] = None
        self.prev_close: Optional[float] = None
        
        self.adx: float = 0.0
        self.plus_di: float = 0.0
        self.minus_di: float = 0.0
        self.initialized = False
    
    def update(self, high: float, low: float, close: float) -> IndicatorValue:
        """Update ADX with OHLC"""
        if self.prev_high is None:
            self.prev_high = high
            self.prev_low = low
            self.prev_close = close
            return IndicatorValue(value=0, signal=IndicatorState.NEUTRAL)
        
        # Calculate True Range
        tr = max(
            high - low,
            abs(high - self.prev_close),
            abs(low - self.prev_close)
        )
        
        # Calculate Directional Movement
        up_move = high - self.prev_high
        down_move = self.prev_low - low
        
        plus_dm = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm = down_move if down_move > up_move and down_move > 0 else 0
        
        self.tr_buffer.append(tr)
        self.plus_dm_buffer.append(plus_dm)
        self.minus_dm_buffer.append(minus_dm)
        
        self.prev_high = high
        self.prev_low = low
        self.prev_close = close
        
        if len(self.tr_buffer) >= self.period:
            self.initialized = True
            
            atr = sum(self.tr_buffer) / self.period
            avg_plus_dm = sum(self.plus_dm_buffer) / self.period
            avg_minus_dm = sum(self.minus_dm_buffer) / self.period
            
            if atr > 0:
                self.plus_di = (avg_plus_dm / atr) * 100
                self.minus_di = (avg_minus_dm / atr) * 100
                
                di_diff = abs(self.plus_di - self.minus_di)
                di_sum = self.plus_di + self.minus_di
                
                if di_sum > 0:
                    dx = (di_diff / di_sum) * 100
                    self.adx = dx  # Simplified, should be smoothed
        
        # Determine signal
        if self.adx > 25:
            if self.plus_di > self.minus_di:
                signal = IndicatorState.BULLISH
            else:
                signal = IndicatorState.BEARISH
        else:
            signal = IndicatorState.NEUTRAL
        
        return IndicatorValue(
            value=self.adx,
            signal=signal,
            raw_components={
                "adx": self.adx,
                "plus_di": self.plus_di,
                "minus_di": self.minus_di
            }
        )


class StreamingVWAP:
    """Streaming Volume Weighted Average Price"""
    
    def __init__(self):
        self.cum_volume = 0
        self.cum_value = 0.0
        self.vwap = 0.0
        self.session_high = 0.0
        self.session_low = float('inf')
    
    def update(self, price: float, volume: int) -> IndicatorValue:
        """Update VWAP"""
        self.cum_volume += volume
        self.cum_value += price * volume
        
        if self.cum_volume > 0:
            self.vwap = self.cum_value / self.cum_volume
        
        self.session_high = max(self.session_high, price)
        self.session_low = min(self.session_low, price) if price > 0 else self.session_low
        
        # Determine signal
        if price > self.vwap * 1.005:  # 0.5% above VWAP
            signal = IndicatorState.BULLISH
        elif price < self.vwap * 0.995:  # 0.5% below VWAP
            signal = IndicatorState.BEARISH
        else:
            signal = IndicatorState.NEUTRAL
        
        return IndicatorValue(
            value=self.vwap,
            signal=signal,
            raw_components={
                "vwap": self.vwap,
                "cum_volume": self.cum_volume,
                "cum_value": self.cum_value
            }
        )
    
    def reset(self):
        """Reset for new session"""
        self.cum_volume = 0
        self.cum_value = 0.0
        self.vwap = 0.0
        self.session_high = 0.0
        self.session_low = float('inf')


class StreamingIndicators:
    """
    Manager for all streaming indicators.
    
    Features:
    - Real-time indicator updates
    - Multi-timeframe support
    - Indicator state management
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Initialize indicators
        self.rsi = StreamingRSI(period=14)
        self.macd = StreamingMACD()
        self.atr = StreamingATR()
        self.bb = StreamingBollingerBands()
        self.adx = StreamingADX()
        self.vwap = StreamingVWAP()
        
        # Moving averages
        self.ma_20 = StreamingSMA(20)
        self.ma_50 = StreamingSMA(50)
        self.ma_200 = StreamingSMA(200)
        
        # State tracking
        self.last_price: Optional[float] = None
        self.indicator_states: Dict[str, IndicatorValue] = {}
    
    def update(self, price: float, volume: int = 0,
               high: float = None, low: float = None) -> Dict[str, IndicatorValue]:
        """Update all indicators with new tick"""
        high = high or price
        low = low or price
        
        self.last_price = price
        
        # Update each indicator
        self.indicator_states["rsi"] = self.rsi.update(price)
        self.indicator_states["macd"] = self.macd.update(price)
        self.indicator_states["bb"] = self.bb.update(price)
        self.indicator_states["vwap"] = self.vwap.update(price, volume)
        
        # OHLC-dependent indicators
        if high and low:
            self.indicator_states["atr"] = self.atr.update(high, low, price)
            self.indicator_states["adx"] = self.adx.update(high, low, price)
        
        # Moving averages
        self.indicator_states["ma_20"] = IndicatorValue(
            value=self.ma_20.update(price),
            signal=IndicatorState.NEUTRAL
        )
        self.indicator_states["ma_50"] = IndicatorValue(
            value=self.ma_50.update(price),
            signal=IndicatorState.NEUTRAL
        )
        self.indicator_states["ma_200"] = IndicatorValue(
            value=self.ma_200.update(price),
            signal=IndicatorState.NEUTRAL
        )
        
        return self.indicator_states
    
    def get_confluence_score(self) -> float:
        """Calculate overall confluence score (0-100)"""
        if not self.indicator_states:
            return 50.0
        
        bullish_count = 0
        bearish_count = 0
        total = 0
        
        for name, state in self.indicator_states.items():
            if state.signal == IndicatorState.BULLISH:
                bullish_count += 1
            elif state.signal == IndicatorState.BEARISH:
                bearish_count += 1
            elif state.signal == IndicatorState.OVERBOUGHT:
                bearish_count += 0.5
            elif state.signal == IndicatorState.OVERSOLD:
                bullish_count += 0.5
            total += 1
        
        if total == 0:
            return 50.0
        
        # Calculate weighted score
        score = 50 + ((bullish_count - bearish_count) / total) * 50
        return max(0, min(100, score))
    
    def get_trend_direction(self) -> str:
        """Get overall trend direction"""
        score = self.get_confluence_score()
        
        if score > 65:
            return "BULLISH"
        elif score < 35:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def get_volatility_regime(self) -> str:
        """Get volatility regime"""
        if not self.atr.initialized or self.last_price is None:
            return "UNKNOWN"
        
        atr_pct = self.atr.get_volatility_percent(self.last_price)
        
        if atr_pct > 3:
            return "HIGH"
        elif atr_pct > 1.5:
            return "NORMAL"
        else:
            return "LOW"
    
    def get_stop_loss(self, direction: str = "LONG", 
                      multiplier: float = 1.5) -> Optional[float]:
        """Calculate ATR-based stop loss"""
        if not self.atr.initialized or self.last_price is None:
            return None
        
        atr = self.atr.atr
        if direction == "LONG":
            return self.last_price - (atr * multiplier)
        else:
            return self.last_price + (atr * multiplier)
    
    def get_all_states(self) -> Dict:
        """Get all indicator states"""
        return {
            name: {
                "value": state.value,
                "signal": state.signal.value,
                "components": state.raw_components
            }
            for name, state in self.indicator_states.items()
        }
