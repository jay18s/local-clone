"""
ROX Proven Edge Engine v3.0 - Feature Engineering
================================================
Feature generation and management for ML models.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from collections import deque
import numpy as np


@dataclass
class FeatureSet:
    """Set of features for ML model"""
    features: Dict[str, float]
    timestamp: datetime
    symbol: str
    feature_groups: Dict[str, List[str]] = field(default_factory=dict)
    
    def to_vector(self, feature_names: List[str] = None) -> np.ndarray:
        """Convert to feature vector"""
        if feature_names is None:
            return np.array(list(self.features.values()))
        return np.array([self.features.get(f, 0.0) for f in feature_names])
    
    def to_dict(self) -> Dict:
        return {
            "features": self.features,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol
        }


class FeatureEngineer:
    """
    Feature engineering for ML models.
    
    Features groups:
    - Price features
    - Volume features
    - Technical indicators
    - Flow features
    - Sentiment features
    - Derivatives features
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.lookback_periods = config.get("lookback_periods", [5, 10, 20, 50])
        
        # Feature buffers for rolling calculations
        self.price_buffers: Dict[str, deque] = {}
        self.volume_buffers: Dict[str, deque] = {}
        self.return_buffers: Dict[str, deque] = {}
        
        # Feature groups
        self.feature_groups = {
            "price": [],
            "volume": [],
            "technical": [],
            "flow": [],
            "sentiment": [],
            "derivatives": []
        }
    
    def generate_features(self, symbol: str, price_data: Dict,
                         volume_data: Dict = None, indicators: Dict = None,
                         flow_data: Dict = None, sentiment_data: Dict = None,
                         derivatives_data: Dict = None) -> FeatureSet:
        """
        Generate all features for a symbol.
        
        Args:
            symbol: Stock symbol
            price_data: Price information (current, open, high, low, close, prev_close)
            volume_data: Volume information
            indicators: Technical indicators from streaming calculator
            flow_data: FII/DII flow data
            sentiment_data: Sentiment scores
            derivatives_data: Options/futures data
            
        Returns:
            FeatureSet with all generated features
        """
        features = {}
        
        # Initialize buffers if needed
        if symbol not in self.price_buffers:
            self.price_buffers[symbol] = deque(maxlen=max(self.lookback_periods))
            self.volume_buffers[symbol] = deque(maxlen=max(self.lookback_periods))
            self.return_buffers[symbol] = deque(maxlen=max(self.lookback_periods))
        
        # Price features
        price_features = self._generate_price_features(symbol, price_data)
        features.update(price_features)
        self.feature_groups["price"] = list(price_features.keys())
        
        # Volume features
        if volume_data:
            volume_features = self._generate_volume_features(symbol, volume_data)
            features.update(volume_features)
            self.feature_groups["volume"] = list(volume_features.keys())
        
        # Technical indicator features
        if indicators:
            tech_features = self._generate_technical_features(indicators)
            features.update(tech_features)
            self.feature_groups["technical"] = list(tech_features.keys())
        
        # Flow features
        if flow_data:
            flow_features = self._generate_flow_features(flow_data)
            features.update(flow_features)
            self.feature_groups["flow"] = list(flow_features.keys())
        
        # Sentiment features
        if sentiment_data:
            sentiment_features = self._generate_sentiment_features(sentiment_data)
            features.update(sentiment_features)
            self.feature_groups["sentiment"] = list(sentiment_features.keys())
        
        # Derivatives features
        if derivatives_data:
            deriv_features = self._generate_derivatives_features(derivatives_data)
            features.update(deriv_features)
            self.feature_groups["derivatives"] = list(deriv_features.keys())
        
        return FeatureSet(
            features=features,
            timestamp=datetime.now(),
            symbol=symbol,
            feature_groups=self.feature_groups
        )
    
    def _generate_price_features(self, symbol: str, price_data: Dict) -> Dict:
        """Generate price-based features"""
        features = {}
        
        current = price_data.get("current", price_data.get("close", 0))
        prev_close = price_data.get("prev_close", price_data.get("close", 0))
        open_price = price_data.get("open", current)
        high = price_data.get("high", current)
        low = price_data.get("low", current)
        close = price_data.get("close", current)
        
        # Basic returns
        if prev_close > 0:
            features["return_1d"] = (current - prev_close) / prev_close
        else:
            features["return_1d"] = 0
        
        # Update buffers
        self.price_buffers[symbol].append(current)
        
        if len(self.price_buffers[symbol]) > 1:
            prev = self.price_buffers[symbol][-2]
            if prev > 0:
                ret = (current - prev) / prev
                self.return_buffers[symbol].append(ret)
        
        # Rolling returns
        prices = list(self.price_buffers[symbol])
        for period in self.lookback_periods:
            if len(prices) >= period:
                if prices[-period] > 0:
                    features[f"return_{period}d"] = (current - prices[-period]) / prices[-period]
                else:
                    features[f"return_{period}d"] = 0
            else:
                features[f"return_{period}d"] = 0
        
        # Price position in day's range
        day_range = high - low if high > low else 1
        features["price_position"] = (close - low) / day_range if day_range > 0 else 0.5
        
        # Gap features
        if open_price > 0 and prev_close > 0:
            features["gap_pct"] = (open_price - prev_close) / prev_close
        else:
            features["gap_pct"] = 0
        
        # Candle features
        body = abs(close - open_price)
        upper_wick = high - max(close, open_price)
        lower_wick = min(close, open_price) - low
        total_range = high - low
        
        if total_range > 0:
            features["body_ratio"] = body / total_range
            features["upper_wick_ratio"] = upper_wick / total_range
            features["lower_wick_ratio"] = lower_wick / total_range
        else:
            features["body_ratio"] = 0
            features["upper_wick_ratio"] = 0
            features["lower_wick_ratio"] = 0
        
        # Rolling statistics
        returns = list(self.return_buffers[symbol])
        for period in [5, 10, 20]:
            if len(returns) >= period:
                period_returns = returns[-period:]
                features[f"return_mean_{period}"] = np.mean(period_returns)
                features[f"return_std_{period}"] = np.std(period_returns)
                features[f"return_skew_{period}"] = self._skewness(period_returns)
            else:
                features[f"return_mean_{period}"] = 0
                features[f"return_std_{period}"] = 0
                features[f"return_skew_{period}"] = 0
        
        # Price momentum
        features["price_momentum_5"] = self._momentum(prices, 5)
        features["price_momentum_10"] = self._momentum(prices, 10)
        features["price_momentum_20"] = self._momentum(prices, 20)
        
        return features
    
    def _generate_volume_features(self, symbol: str, volume_data: Dict) -> Dict:
        """Generate volume-based features"""
        features = {}
        
        current_volume = volume_data.get("current", volume_data.get("volume", 0))
        
        # Update buffer
        self.volume_buffers[symbol].append(current_volume)
        
        volumes = list(self.volume_buffers[symbol])
        
        # Relative volume
        for period in [5, 10, 20]:
            if len(volumes) >= period:
                avg_volume = np.mean(volumes[-period:])
                features[f"rel_volume_{period}"] = current_volume / avg_volume if avg_volume > 0 else 1
            else:
                features[f"rel_volume_{period}"] = 1
        
        # Volume trend
        if len(volumes) >= 10:
            recent = np.mean(volumes[-5:])
            older = np.mean(volumes[-10:-5])
            features["volume_trend"] = (recent - older) / older if older > 0 else 0
        else:
            features["volume_trend"] = 0
        
        # Volume volatility
        if len(volumes) >= 10:
            features["volume_volatility"] = np.std(volumes[-10:]) / np.mean(volumes[-10:]) if np.mean(volumes[-10:]) > 0 else 0
        else:
            features["volume_volatility"] = 0
        
        return features
    
    def _generate_technical_features(self, indicators: Dict) -> Dict:
        """Generate features from technical indicators"""
        features = {}
        
        # RSI features
        rsi = indicators.get("rsi", {})
        features["rsi_value"] = rsi.get("value", 50)
        features["rsi_overbought"] = 1 if features["rsi_value"] > 70 else 0
        features["rsi_oversold"] = 1 if features["rsi_value"] < 30 else 0
        
        # MACD features
        macd = indicators.get("macd", {})
        components = macd.get("raw_components", {})
        features["macd_value"] = components.get("macd", 0)
        features["macd_signal"] = components.get("signal", 0)
        features["macd_histogram"] = components.get("histogram", 0)
        features["macd_cross"] = 1 if features["macd_histogram"] > 0 else -1
        
        # Bollinger Bands
        bb = indicators.get("bb", {})
        bb_components = bb.get("raw_components", {})
        features["bb_upper"] = bb_components.get("upper", 0)
        features["bb_middle"] = bb_components.get("middle", 0)
        features["bb_lower"] = bb_components.get("lower", 0)
        features["bb_bandwidth"] = bb_components.get("bandwidth", 0)
        features["bb_position"] = self._bb_position(
            indicators.get("last_price", 0),
            features["bb_upper"],
            features["bb_lower"]
        )
        
        # ADX features
        adx = indicators.get("adx", {})
        adx_components = adx.get("raw_components", {})
        features["adx_value"] = adx.get("value", 0)
        features["plus_di"] = adx_components.get("plus_di", 0)
        features["minus_di"] = adx_components.get("minus_di", 0)
        features["adx_trending"] = 1 if features["adx_value"] > 25 else 0
        
        # ATR features
        atr = indicators.get("atr", {})
        features["atr_value"] = atr.get("value", 0)
        
        # Moving averages
        features["ma_20"] = indicators.get("ma_20", {}).get("value", 0)
        features["ma_50"] = indicators.get("ma_50", {}).get("value", 0)
        features["ma_200"] = indicators.get("ma_200", {}).get("value", 0)
        
        # MA crossovers
        last_price = indicators.get("last_price", 0)
        features["above_ma_20"] = 1 if last_price > features["ma_20"] else 0
        features["above_ma_50"] = 1 if last_price > features["ma_50"] else 0
        features["above_ma_200"] = 1 if last_price > features["ma_200"] else 0
        
        # VWAP
        vwap = indicators.get("vwap", {})
        features["vwap_value"] = vwap.get("value", 0)
        features["above_vwap"] = 1 if last_price > features["vwap_value"] else 0
        
        # Confluence score
        features["confluence_score"] = indicators.get("confluence_score", 50)
        
        return features
    
    def _generate_flow_features(self, flow_data: Dict) -> Dict:
        """Generate flow-based features"""
        features = {}
        
        # FII flows
        features["fii_daily"] = flow_data.get("fii_daily", 0)
        features["fii_5day"] = flow_data.get("fii_5day", 0)
        features["fii_momentum"] = flow_data.get("fii_momentum", 0)
        
        # DII flows
        features["dii_daily"] = flow_data.get("dii_daily", 0)
        features["dii_5day"] = flow_data.get("dii_5day", 0)
        
        # Combined flow
        features["total_flow"] = features["fii_5day"] + features["dii_5day"]
        features["flow_alignment"] = 1 if (features["fii_5day"] > 0 and features["dii_5day"] > 0) else \
                                    -1 if (features["fii_5day"] < 0 and features["dii_5day"] < 0) else 0
        
        # Flow score
        features["flow_score"] = flow_data.get("flow_score", 50)
        
        return features
    
    def _generate_sentiment_features(self, sentiment_data: Dict) -> Dict:
        """Generate sentiment-based features"""
        features = {}
        
        # Individual sentiment components
        features["news_sentiment"] = sentiment_data.get("news", 0)
        features["analyst_sentiment"] = sentiment_data.get("analyst", 0)
        features["social_sentiment"] = sentiment_data.get("social", 0)
        features["corporate_sentiment"] = sentiment_data.get("corporate", 0)
        features["global_sentiment"] = sentiment_data.get("global", 0)
        
        # Composite sentiment
        features["composite_sentiment"] = sentiment_data.get("composite", 0)
        
        # Sentiment zone encoding
        zone = sentiment_data.get("zone", "NEUTRAL")
        features["sentiment_zone_euphoria"] = 1 if zone == "EUPHORIA" else 0
        features["sentiment_zone_panic"] = 1 if zone == "PANIC" else 0
        
        # VIX
        features["india_vix"] = sentiment_data.get("vix", 15)
        features["vix_elevated"] = 1 if features["india_vix"] > 20 else 0
        
        return features
    
    def _generate_derivatives_features(self, derivatives_data: Dict) -> Dict:
        """Generate derivatives-based features"""
        features = {}
        
        # PCR
        features["pcr"] = derivatives_data.get("pcr", 1.0)
        features["pcr_extreme_bullish"] = 1 if features["pcr"] > 1.3 else 0
        features["pcr_extreme_bearish"] = 1 if features["pcr"] < 0.6 else 0
        
        # OI
        features["oi_change"] = derivatives_data.get("oi_change", 0)
        features["oi_signal"] = derivatives_data.get("oi_signal", 0)
        
        # Max Pain
        features["max_pain"] = derivatives_data.get("max_pain", 0)
        features["gap_from_max_pain"] = derivatives_data.get("gap_from_max_pain", 0)
        
        # IV
        features["iv_rank"] = derivatives_data.get("iv_rank", 50)
        features["iv_high"] = 1 if features["iv_rank"] > 60 else 0
        
        # Greeks (if available)
        features["net_delta"] = derivatives_data.get("net_delta", 0)
        features["net_gamma"] = derivatives_data.get("net_gamma", 0)
        
        return features
    
    def _skewness(self, data: List[float]) -> float:
        """Calculate skewness"""
        if len(data) < 3:
            return 0
        
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0
        
        return (sum((x - mean) ** 3 for x in data) / n) / (std ** 3)
    
    def _momentum(self, prices: List[float], period: int) -> float:
        """Calculate price momentum"""
        if len(prices) < period + 1:
            return 0
        
        current = prices[-1]
        past = prices[-(period + 1)]
        
        if past == 0:
            return 0
        
        return (current - past) / past
    
    def _bb_position(self, price: float, upper: float, lower: float) -> float:
        """Calculate position within Bollinger Bands"""
        if upper == lower:
            return 0.5
        
        return (price - lower) / (upper - lower)
    
    def get_feature_names(self) -> List[str]:
        """Get all feature names"""
        all_features = []
        for group_features in self.feature_groups.values():
            all_features.extend(group_features)
        return all_features
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Get features organized by group"""
        return self.feature_groups
