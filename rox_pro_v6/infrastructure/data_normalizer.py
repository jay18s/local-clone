"""
ROX Proven Edge Engine v3.0 - Data Normalizer
============================================
Standardize data from different sources into unified format.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

from .data_feed import TickData, OHLCV, QuoteData, MarketDepth, DepthLevel, MarketDataSource


@dataclass
class NormalizedTick:
    """Normalized tick data from any source"""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid: float = 0.0
    ask: float = 0.0
    bid_size: int = 0
    ask_size: int = 0
    trade_type: str = ""
    source: str = ""
    raw_data: Dict = field(default_factory=dict)
    
    def to_tick_data(self) -> TickData:
        """Convert to standard TickData"""
        source_map = {
            "fyers": MarketDataSource.FYERS,
            "zerodha": MarketDataSource.ZERODHA,
            "nse": MarketDataSource.NSE,
            "yahoo": MarketDataSource.YAHOO
        }
        
        return TickData(
            symbol=self.symbol,
            timestamp=self.timestamp,
            price=self.price,
            volume=self.volume,
            bid=self.bid,
            ask=self.ask,
            bid_size=self.bid_size,
            ask_size=self.ask_size,
            trade_type=self.trade_type,
            source=source_map.get(self.source, MarketDataSource.NSE)
        )


@dataclass
class DataQualityMetrics:
    """Data quality assessment"""
    is_valid: bool
    missing_fields: List[str]
    anomalies: List[str]
    confidence_score: float  # 0-1
    
    def to_dict(self) -> Dict:
        return {
            "is_valid": self.is_valid,
            "missing_fields": self.missing_fields,
            "anomalies": self.anomalies,
            "confidence_score": self.confidence_score
        }


class DataNormalizer:
    """
    Normalizes data from various sources into standard format.
    
    Features:
    - Source-specific parsing
    - Data validation
    - Anomaly detection
    - Symbol standardization
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger("DataNormalizer")
        
        # Symbol mappings (exchange -> standard)
        self.symbol_mappings = {
            # NSE symbols
            "NIFTY 50": "NIFTY",
            "NIFTY BANK": "BANKNIFTY",
            "NIFTY IT": "NIFTYIT",
            # Add more as needed
        }
        
        # Anomaly thresholds
        self.price_spike_threshold = config.get("price_spike_threshold", 0.20)  # 20%
        self.volume_spike_threshold = config.get("volume_spike_threshold", 10)  # 10x avg
    
    def normalize_fyers_tick(self, data: Dict) -> Optional[NormalizedTick]:
        """Normalize FYERS tick data"""
        try:
            return NormalizedTick(
                symbol=self._standardize_symbol(data.get("symbol", "")),
                timestamp=self._parse_timestamp(data.get("timestamp")),
                price=float(data.get("ltp", data.get("price", 0))),
                volume=int(data.get("volume", data.get("vol", 0))),
                bid=float(data.get("bid", 0)),
                ask=float(data.get("ask", 0)),
                bid_size=int(data.get("bidSize", data.get("bid_size", 0))),
                ask_size=int(data.get("askSize", data.get("ask_size", 0))),
                trade_type=data.get("type", ""),
                source="fyers",
                raw_data=data
            )
        except Exception as e:
            self.logger.error(f"Fyers normalization error: {e}")
            return None
    
    def normalize_zerodha_tick(self, data: Dict) -> Optional[NormalizedTick]:
        """Normalize Zerodha/Kite tick data"""
        try:
            return NormalizedTick(
                symbol=self._standardize_symbol(data.get("instrument_token", "")),
                timestamp=self._parse_timestamp(data.get("timestamp")),
                price=float(data.get("last_price", 0)),
                volume=int(data.get("volume", 0)),
                bid=float(data.get("depth", {}).get("buy", [{}])[0].get("price", 0)),
                ask=float(data.get("depth", {}).get("sell", [{}])[0].get("price", 0)),
                bid_size=int(data.get("depth", {}).get("buy", [{}])[0].get("quantity", 0)),
                ask_size=int(data.get("depth", {}).get("sell", [{}])[0].get("quantity", 0)),
                trade_type="BUY" if data.get("buy_quantity", 0) > data.get("sell_quantity", 0) else "SELL",
                source="zerodha",
                raw_data=data
            )
        except Exception as e:
            self.logger.error(f"Zerodha normalization error: {e}")
            return None
    
    def normalize_nse_tick(self, data: Dict) -> Optional[NormalizedTick]:
        """Normalize NSE tick data"""
        try:
            return NormalizedTick(
                symbol=self._standardize_symbol(data.get("symbol", "")),
                timestamp=self._parse_timestamp(data.get("timestamp")),
                price=float(data.get("lastPrice", data.get("price", 0))),
                volume=int(data.get("totalTradedVolume", data.get("volume", 0))),
                bid=float(data.get("bidprice", [0])[0] if isinstance(data.get("bidprice"), list) else data.get("bidprice", 0)),
                ask=float(data.get("askprice", [0])[0] if isinstance(data.get("askprice"), list) else data.get("askprice", 0)),
                bid_size=int(data.get("bidQty", 0)),
                ask_size=int(data.get("askQty", 0)),
                trade_type=data.get("tradeType", ""),
                source="nse",
                raw_data=data
            )
        except Exception as e:
            self.logger.error(f"NSE normalization error: {e}")
            return None
    
    def normalize_yahoo_quote(self, data: Dict) -> Optional[NormalizedTick]:
        """Normalize Yahoo Finance quote data"""
        try:
            quote = data.get("quoteResponse", {}).get("result", [{}])[0]
            
            return NormalizedTick(
                symbol=self._standardize_symbol(quote.get("symbol", "").replace(".NS", "")),
                timestamp=datetime.now(),
                price=float(quote.get("regularMarketPrice", 0)),
                volume=int(quote.get("regularMarketVolume", 0)),
                bid=float(quote.get("bid", 0)),
                ask=float(quote.get("ask", 0)),
                bid_size=int(quote.get("bidSize", 0)),
                ask_size=int(quote.get("askSize", 0)),
                trade_type="",
                source="yahoo",
                raw_data=data
            )
        except Exception as e:
            self.logger.error(f"Yahoo normalization error: {e}")
            return None
    
    def normalize(self, source: str, data: Dict) -> Optional[NormalizedTick]:
        """Normalize data from any source"""
        normalizers = {
            "fyers": self.normalize_fyers_tick,
            "zerodha": self.normalize_zerodha_tick,
            "kite": self.normalize_zerodha_tick,
            "nse": self.normalize_nse_tick,
            "yahoo": self.normalize_yahoo_quote
        }
        
        normalizer = normalizers.get(source.lower())
        if normalizer:
            return normalizer(data)
        
        self.logger.warning(f"Unknown source: {source}")
        return None
    
    def validate_tick(self, tick: NormalizedTick, 
                      previous_tick: NormalizedTick = None) -> DataQualityMetrics:
        """Validate tick data quality"""
        missing_fields = []
        anomalies = []
        confidence_score = 1.0
        
        # Check required fields
        if not tick.symbol:
            missing_fields.append("symbol")
        if tick.timestamp is None:
            missing_fields.append("timestamp")
        if tick.price <= 0:
            missing_fields.append("price")
        
        # Check for anomalies
        if previous_tick and previous_tick.price > 0:
            price_change = abs(tick.price - previous_tick.price) / previous_tick.price
            
            if price_change > self.price_spike_threshold:
                anomalies.append(f"Price spike: {price_change*100:.1f}%")
                confidence_score *= 0.7
        
        # Check for zero spreads (might indicate stale data)
        if tick.bid > 0 and tick.ask > 0 and tick.ask - tick.bid == 0:
            anomalies.append("Zero spread - possible stale data")
            confidence_score *= 0.9
        
        # Check for unrealistic volume
        if tick.volume < 0:
            anomalies.append("Negative volume")
            confidence_score *= 0.5
        
        is_valid = len(missing_fields) == 0 and confidence_score > 0.5
        
        return DataQualityMetrics(
            is_valid=is_valid,
            missing_fields=missing_fields,
            anomalies=anomalies,
            confidence_score=confidence_score
        )
    
    def _standardize_symbol(self, symbol: str) -> str:
        """Standardize symbol format"""
        # Remove exchange suffix
        for suffix in [".NS", ".BO", ".BSE"]:
            if symbol.endswith(suffix):
                symbol = symbol[:-len(suffix)]
                break
        
        # Apply mapping
        return self.symbol_mappings.get(symbol, symbol).upper()
    
    def _parse_timestamp(self, timestamp: Any) -> datetime:
        """Parse timestamp from various formats"""
        if timestamp is None:
            return datetime.now()
        
        if isinstance(timestamp, datetime):
            return timestamp
        
        if isinstance(timestamp, (int, float)):
            # Assume Unix timestamp
            return datetime.fromtimestamp(timestamp)
        
        if isinstance(timestamp, str):
            # Try ISO format
            try:
                return datetime.fromisoformat(timestamp)
            except ValueError:
                pass
            
            # Try other formats
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"]:
                try:
                    return datetime.strptime(timestamp, fmt)
                except ValueError:
                    continue
        
        return datetime.now()
    
    def normalize_ohlcv(self, source: str, data: Dict, 
                        timeframe: str = "1min") -> Optional[OHLCV]:
        """Normalize OHLCV data"""
        try:
            symbol = self._standardize_symbol(data.get("symbol", ""))
            timestamp = self._parse_timestamp(data.get("timestamp", data.get("date")))
            
            # Handle different field names
            open_price = float(data.get("open", data.get("o", 0)))
            high_price = float(data.get("high", data.get("h", 0)))
            low_price = float(data.get("low", data.get("l", 0)))
            close_price = float(data.get("close", data.get("c", 0)))
            volume = int(data.get("volume", data.get("v", 0)))
            
            return OHLCV(
                symbol=symbol,
                timestamp=timestamp,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                timeframe=timeframe
            )
        except Exception as e:
            self.logger.error(f"OHLCV normalization error: {e}")
            return None
    
    def normalize_options_chain(self, source: str, 
                                data: Dict) -> List[Dict]:
        """Normalize options chain data"""
        normalized = []
        
        try:
            chain = data.get("optionsChain", data.get("chain", []))
            
            for option in chain:
                normalized.append({
                    "symbol": self._standardize_symbol(option.get("symbol", "")),
                    "strike": float(option.get("strikePrice", option.get("strike", 0))),
                    "expiry": option.get("expiryDate", option.get("expiry")),
                    "option_type": option.get("optionType", option.get("type", "CE")),
                    "ltp": float(option.get("lastPrice", option.get("ltp", 0))),
                    "oi": int(option.get("openInterest", option.get("oi", 0))),
                    "oi_change": int(option.get("changeinOpenInterest", option.get("oi_change", 0))),
                    "volume": int(option.get("totalTradedVolume", option.get("volume", 0))),
                    "iv": float(option.get("impliedVolatility", option.get("iv", 0))),
                    "delta": float(option.get("delta", 0)),
                    "gamma": float(option.get("gamma", 0)),
                    "theta": float(option.get("theta", 0)),
                    "vega": float(option.get("vega", 0))
                })
        except Exception as e:
            self.logger.error(f"Options chain normalization error: {e}")
        
        return normalized
    
    def batch_normalize(self, source: str, 
                        data_list: List[Dict]) -> List[NormalizedTick]:
        """Normalize multiple ticks at once"""
        normalized = []
        
        for data in data_list:
            tick = self.normalize(source, data)
            if tick:
                normalized.append(tick)
        
        return normalized
