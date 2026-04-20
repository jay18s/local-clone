"""
ROX Proven Edge Engine v3.0 - Data Feed Manager
==============================================
Real-time market data integration with multiple sources.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, AsyncIterator
from collections import deque
try:
    import websockets
except ImportError:
    websockets = None  # Optional: install with 'pip install websockets'
try:
    import aiohttp
except ImportError:
    aiohttp = None  # Optional: install with 'pip install aiohttp'


class MarketDataSource(Enum):
    """Available market data sources"""
    FYERS = "fyers"
    YAHOO = "yahoo"
    GOOGLE = "google"
    NSE = "nse"
    ZERODHA = "zerodha"


class DataType(Enum):
    """Types of market data"""
    TICK = "tick"
    OHLCV = "ohlcv"
    QUOTE = "quote"
    DEPTH = "depth"
    OPTIONS_CHAIN = "options_chain"


@dataclass
class TickData:
    """Single tick data structure"""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid: float = 0.0
    ask: float = 0.0
    bid_size: int = 0
    ask_size: int = 0
    trade_type: str = ""  # BUY/SELL
    source: MarketDataSource = MarketDataSource.NSE
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "price": self.price,
            "volume": self.volume,
            "bid": self.bid,
            "ask": self.ask,
            "bid_size": self.bid_size,
            "ask_size": self.ask_size,
            "trade_type": self.trade_type,
            "source": self.source.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "TickData":
        return cls(
            symbol=data["symbol"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            price=data["price"],
            volume=data["volume"],
            bid=data.get("bid", 0.0),
            ask=data.get("ask", 0.0),
            bid_size=data.get("bid_size", 0),
            ask_size=data.get("ask_size", 0),
            trade_type=data.get("trade_type", ""),
            source=MarketDataSource(data.get("source", "nse"))
        )


@dataclass
class OHLCV:
    """OHLCV candlestick data"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    timeframe: str = "1min"
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "timeframe": self.timeframe
        }


@dataclass
class QuoteData:
    """Level 1 quote data"""
    symbol: str
    timestamp: datetime
    last_price: float
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    volume: int
    open_interest: int = 0
    
    @property
    def spread(self) -> float:
        return self.ask - self.bid if self.ask > 0 else 0
    
    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2 if self.bid > 0 and self.ask > 0 else self.last_price


@dataclass
class DepthLevel:
    """Single level in market depth"""
    price: float
    quantity: int
    orders: int


@dataclass
class MarketDepth:
    """Level 3 market depth"""
    symbol: str
    timestamp: datetime
    bids: List[DepthLevel]
    asks: List[DepthLevel]
    
    @property
    def total_bid_quantity(self) -> int:
        return sum(level.quantity for level in self.bids)
    
    @property
    def total_ask_quantity(self) -> int:
        return sum(level.quantity for level in self.asks)
    
    @property
    def bid_ask_imbalance(self) -> float:
        """Calculate order imbalance (-1 to 1)"""
        total = self.total_bid_quantity + self.total_ask_quantity
        if total == 0:
            return 0
        return (self.total_bid_quantity - self.total_ask_quantity) / total


class DataFeedProvider(ABC):
    """Abstract base class for data feed providers"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self._connected = False
        self._callbacks: Dict[str, List[Callable]] = {}
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the data source"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the data source"""
        pass
    
    @abstractmethod
    async def subscribe(self, symbols: List[str], data_type: DataType) -> bool:
        """Subscribe to data for given symbols"""
        pass
    
    @abstractmethod
    async def unsubscribe(self, symbols: List[str]) -> bool:
        """Unsubscribe from data"""
        pass
    
    @abstractmethod
    async def get_historical_data(self, symbol: str, start: datetime, 
                                  end: datetime, interval: str) -> List[OHLCV]:
        """Get historical data"""
        pass
    
    def register_callback(self, event_type: str, callback: Callable):
        """Register callback for data events"""
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []
        self._callbacks[event_type].append(callback)
    
    async def _emit(self, event_type: str, data: Any):
        """Emit data to registered callbacks"""
        if event_type in self._callbacks:
            for callback in self._callbacks[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    self.logger.error(f"Callback error: {e}")
    
    @property
    def is_connected(self) -> bool:
        return self._connected


class FyersProvider(DataFeedProvider):
    """
    FYERS API v3 data provider — real implementation.

    Provides:
    - Historical OHLCV (up to 365 days, any interval)
    - Live quotes (LTP, bid/ask, volume, OI)
    - Market depth (Level 2)
    - WebSocket tick streaming

    Symbol format used internally: NSE:RELIANCE-EQ, NSE:NIFTY50-INDEX
    Pass plain symbols like "RELIANCE", "NIFTY" — conversion is automatic.
    """

    # Fyers interval codes
    _INTERVAL_MAP = {
        "1min": "1",  "2min": "2",  "3min": "3",  "5min": "5",
        "10min": "10", "15min": "15", "20min": "20", "30min": "30",
        "60min": "60", "120min": "120", "240min": "240",
        "1d": "D", "1w": "W", "1M": "M",
    }

    # Index symbols need a different suffix
    _INDEX_SYMBOLS = {"NIFTY", "NIFTY50", "BANKNIFTY", "FINNIFTY",
                      "MIDCPNIFTY", "SENSEX", "INDIAVIX"}

    def __init__(self, config: Dict):
        super().__init__(config)
        self.client_id    = config.get("client_id", config.get("app_id", ""))
        self.access_token = config.get("access_token", "")
        self._fyers       = None   # fyersModel.FyersModel instance
        self._ws          = None   # FyersDataSocket instance
        self._log_path    = config.get("log_path", "logs")

    # ── symbol helpers ────────────────────────────────────────────────────

    # Explicit overrides for tricky symbols
    _SYMBOL_OVERRIDES = {
        "NIFTY":      "NSE:NIFTY50-INDEX",
        "NIFTY50":    "NSE:NIFTY50-INDEX",
        "BANKNIFTY":  "NSE:NIFTYBANK-INDEX",
        "FINNIFTY":   "NSE:FINNIFTY-INDEX",
        "INDIAVIX":   "NSE:INDIAVIX-INDEX",
        "SENSEX":     "BSE:SENSEX-INDEX",
        "BAJAJ-AUTO": "NSE:BAJAJ-AUTO-EQ",
        "M&M":        "NSE:M&M-EQ",
        "MIDCPNIFTY": "NSE:MIDCPNIFTY-INDEX",
    }

    def _to_fyers_symbol(self, symbol: str) -> str:
        """Convert plain symbol → Fyers exchange:symbol-type format."""
        sym_upper = symbol.upper()
        # Check explicit overrides first (handles BAJAJ-AUTO, M&M, NIFTY, etc.)
        if sym_upper in self._SYMBOL_OVERRIDES:
            return self._SYMBOL_OVERRIDES[sym_upper]
        # Generic index detection (startswith checks on clean name)
        sym_clean = sym_upper.replace("&", "").replace("-", "")
        if sym_clean in self._INDEX_SYMBOLS or sym_clean.startswith("NIFTY") or sym_clean.startswith("BANK"):
            return f"NSE:{sym_clean}-INDEX"
        # Regular equity — preserve original symbol name (Fyers is case-sensitive)
        return f"NSE:{sym_upper}-EQ"

    def _to_fyers_symbols(self, symbols: list) -> list:
        return [self._to_fyers_symbol(s) for s in symbols]

    # ── connection ────────────────────────────────────────────────────────

    async def connect(self) -> bool:
        try:
            from fyers_apiv3 import fyersModel as _fm
            self._fyers = _fm.FyersModel(
                client_id=self.client_id,
                token=self.access_token,
                log_path=self._log_path,
                is_async=False,
            )
            # Quick sanity check
            profile = self._fyers.get_profile()
            if profile.get("s") == "ok":
                name = profile.get("data", {}).get("name", "")
                self.logger.info(f"Connected to Fyers API as: {name}")
                self._connected = True
                return True
            else:
                self.logger.error(f"Fyers connect failed: {profile.get('message', profile)}")
                return False
        except ImportError:
            self.logger.error("fyers-apiv3 not installed. Run: pip install fyers-apiv3")
            return False
        except Exception as e:
            self.logger.error(f"Fyers connection error: {e}")
            return False

    async def disconnect(self) -> bool:
        if self._ws:
            try:
                self._ws.close_connection()
            except Exception:
                pass
        self._connected = False
        self._fyers = None
        return True

    # ── subscriptions (WebSocket) ─────────────────────────────────────────

    async def subscribe(self, symbols: List[str], data_type: DataType) -> bool:
        if not self._connected or not self._fyers:
            return False
        try:
            from fyers_apiv3.FyersWebsocket import data_ws

            fyers_symbols = self._to_fyers_symbols(symbols)
            data_type_code = {
                DataType.TICK:  data_ws.SocketDataType.SymbolUpdate,
                DataType.DEPTH: data_ws.SocketDataType.DepthUpdate,
                DataType.QUOTE: data_ws.SocketDataType.SymbolUpdate,
            }.get(data_type, data_ws.SocketDataType.SymbolUpdate)

            def _on_message(msg):
                asyncio.create_task(self._handle_ws_message(msg))

            def _on_error(err):
                self.logger.error(f"Fyers WS error: {err}")

            def _on_close():
                self.logger.warning("Fyers WS connection closed")
                self._connected = False

            self._ws = data_ws.FyersDataSocket(
                access_token=f"{self.client_id}:{self.access_token}",
                log_path=self._log_path,
                litemode=False,
                write_to_file=False,
                reconnect=True,
                on_connect=lambda: self.logger.info("Fyers WS connected"),
                on_close=_on_close,
                on_error=_on_error,
                on_message=_on_message,
            )
            self._ws.subscribe(symbols=fyers_symbols, data_type=data_type_code)
            self._ws.keep_running()
            self.logger.info(f"Subscribed to {len(symbols)} symbols via Fyers WS")
            return True
        except Exception as e:
            self.logger.error(f"Fyers subscribe error: {e}")
            return False

    async def unsubscribe(self, symbols: List[str]) -> bool:
        if self._ws:
            try:
                self._ws.unsubscribe(symbols=self._to_fyers_symbols(symbols))
            except Exception as e:
                self.logger.error(f"Fyers unsubscribe error: {e}")
        return True

    async def _handle_ws_message(self, msg: dict):
        """Parse WebSocket tick and emit to callbacks."""
        try:
            sym_raw = msg.get("symbol", "")
            # Strip exchange prefix and type suffix: NSE:BAJAJ-AUTO-EQ → BAJAJ-AUTO
            if ":" in sym_raw:
                without_exchange = sym_raw.split(":", 1)[1]
                for suffix in ("-EQ", "-INDEX", "-BE", "-SM", "-BL", "-MF"):
                    if without_exchange.endswith(suffix):
                        without_exchange = without_exchange[:-len(suffix)]
                        break
                symbol = without_exchange
            else:
                symbol = sym_raw
            tick = TickData(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(msg.get("timestamp", 0)),
                price=msg.get("ltp", 0.0),
                volume=msg.get("vol_traded_today", 0),
                bid=msg.get("bid_price", 0.0),
                ask=msg.get("ask_price", 0.0),
                bid_size=msg.get("bid_size", 0),
                ask_size=msg.get("ask_size", 0),
                source=MarketDataSource.FYERS,
            )
            await self._emit("tick", tick)
        except Exception as e:
            self.logger.error(f"WS message parse error: {e}")

    # ── quotes (REST) ─────────────────────────────────────────────────────

    def get_quotes(self, symbols: List[str]) -> Dict[str, dict]:
        """
        Fetch live LTP + bid/ask for a list of symbols.
        Returns dict keyed by plain symbol name.
        """
        if not self._connected or not self._fyers:
            return {}
        try:
            fyers_symbols = self._to_fyers_symbols(symbols)
            resp = self._fyers.quotes({"symbols": ",".join(fyers_symbols)})
            if resp.get("s") != "ok":
                self.logger.error(f"Fyers quotes error: {resp.get('message')}")
                return {}
            result = {}
            for item in resp.get("d", []):
                v = item.get("v", {})
                sym_raw = item.get("n", "")
                # Strip exchange prefix (NSE:) and type suffix (-EQ, -INDEX, -BE etc.)
                # e.g. "NSE:BAJAJ-AUTO-EQ" -> "BAJAJ-AUTO"
                #      "NSE:RELIANCE-EQ"   -> "RELIANCE"
                #      "NSE:NIFTY50-INDEX" -> "NIFTY50"
                if ":" in sym_raw:
                    without_exchange = sym_raw.split(":", 1)[1]
                    # Remove known type suffixes from the END only
                    for suffix in ("-EQ", "-INDEX", "-BE", "-SM", "-BL", "-MF"):
                        if without_exchange.endswith(suffix):
                            without_exchange = without_exchange[:-len(suffix)]
                            break
                    plain = without_exchange
                else:
                    plain = sym_raw
                result[plain] = {
                    "ltp":       v.get("lp",  0.0),
                    "open":      v.get("open_price", 0.0),
                    "high":      v.get("high_price",  0.0),
                    "low":       v.get("low_price",   0.0),
                    "close":     v.get("prev_close_price", 0.0),
                    "volume":    v.get("volume", 0),
                    "bid":       v.get("bid_price", 0.0),
                    "ask":       v.get("ask_price", 0.0),
                    "change":    v.get("ch",   0.0),
                    "change_pct":v.get("chp",  0.0),
                }
            return result
        except Exception as e:
            self.logger.error(f"Fyers get_quotes error: {e}")
            return {}

    # ── historical data (REST) ────────────────────────────────────────────

    async def get_historical_data(self, symbol: str, start: datetime,
                                  end: datetime, interval: str = "1d") -> List[OHLCV]:
        """Fetch OHLCV history from Fyers."""
        if not self._connected or not self._fyers:
            return []
        try:
            fyers_sym  = self._to_fyers_symbol(symbol)
            resolution = self._INTERVAL_MAP.get(interval, "D")
            # Fyers v3 API: date_format="0" = epoch timestamps (confusingly named)
            # range_to must be yesterday or earlier — today is "in the future" for Fyers
            import datetime as _dt
            safe_end = min(end, _dt.datetime.now() - _dt.timedelta(days=1))
            payload = {
                "symbol":      fyers_sym,
                "resolution":  resolution,
                "date_format": "0",
                "range_from":  str(int(start.timestamp())),
                "range_to":    str(int(safe_end.timestamp())),
                "cont_flag":   "1",
            }
            resp = self._fyers.history(payload)
            if resp.get("s") != "ok":
                self.logger.error(f"Fyers history error for {symbol}: {resp.get('message')}")
                return []

            candles = resp.get("candles", [])
            ohlcv_list = []
            for c in candles:
                # c = [timestamp, open, high, low, close, volume]
                ohlcv_list.append(OHLCV(
                    symbol    = symbol,
                    timestamp = datetime.fromtimestamp(c[0]),
                    open      = float(c[1]),
                    high      = float(c[2]),
                    low       = float(c[3]),
                    close     = float(c[4]),
                    volume    = int(c[5]),
                    timeframe = interval,
                ))
            return ohlcv_list

        except Exception as e:
            self.logger.error(f"Fyers historical data error for {symbol}: {e}")
            return []

    def get_historical_data_sync(self, symbol: str, start: datetime,
                                 end: datetime, interval: str = "1d") -> List[OHLCV]:
        """Synchronous wrapper — use this from non-async code."""
        import asyncio
        import time
        # Small delay between sequential calls to respect Fyers rate limits
        time.sleep(0.15)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.get_historical_data(symbol, start, end, interval)
                    )
                    return future.result(timeout=30)
            else:
                return loop.run_until_complete(
                    self.get_historical_data(symbol, start, end, interval)
                )
        except Exception as e:
            self.logger.error(f"Sync historical data error: {e}")
            return []




class ZerodhaProvider(DataFeedProvider):
    """Zerodha Kite data provider"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.api_key = config.get("api_key", "")
        self.access_token = config.get("access_token", "")
        self.ws_url = "wss://ws.kite.trade"
    
    async def connect(self) -> bool:
        """Connect to Zerodha WebSocket"""
        try:
            self._connected = True
            self.logger.info("Connected to Zerodha Kite")
            return True
        except Exception as e:
            self.logger.error(f"Zerodha connection error: {e}")
            return False
    
    async def disconnect(self) -> bool:
        self._connected = False
        return True
    
    async def subscribe(self, symbols: List[str], data_type: DataType) -> bool:
        return True
    
    async def unsubscribe(self, symbols: List[str]) -> bool:
        return True
    
    async def get_historical_data(self, symbol: str, start: datetime,
                                  end: datetime, interval: str) -> List[OHLCV]:
        return []


class YahooFinanceProvider(DataFeedProvider):
    """Yahoo Finance data provider (backup)"""
    
    BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self._session = None
    
    async def connect(self) -> bool:
        try:
            self._session = aiohttp.ClientSession()
            self._connected = True
            self.logger.info("Connected to Yahoo Finance")
            return True
        except Exception as e:
            self.logger.error(f"Yahoo connection error: {e}")
            return False
    
    async def disconnect(self) -> bool:
        if self._session:
            await self._session.close()
        self._connected = False
        return True
    
    async def subscribe(self, symbols: List[str], data_type: DataType) -> bool:
        # Yahoo doesn't support real-time WebSocket
        return True
    
    async def unsubscribe(self, symbols: List[str]) -> bool:
        return True
    
    async def get_historical_data(self, symbol: str, start: datetime,
                                  end: datetime, interval: str = "1d") -> List[OHLCV]:
        """Fetch historical data from Yahoo Finance"""
        if not self._session:
            return []
        
        try:
            symbol_yahoo = f"{symbol}.NS"  # NSE suffix
            url = f"{self.BASE_URL}/{symbol_yahoo}"
            params = {
                "period1": int(start.timestamp()),
                "period2": int(end.timestamp()),
                "interval": interval
            }
            
            async with self._session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_yahoo_data(data, symbol)
                return []
        except Exception as e:
            self.logger.error(f"Yahoo data fetch error: {e}")
            return []
    
    def _parse_yahoo_data(self, data: Dict, symbol: str) -> List[OHLCV]:
        """Parse Yahoo Finance response into OHLCV list"""
        try:
            result = data.get("chart", {}).get("result", [])
            if not result:
                return []
            
            quotes = result[0].get("indicators", {}).get("quote", [{}])[0]
            timestamps = result[0].get("timestamp", [])
            
            ohlcv_list = []
            for i, ts in enumerate(timestamps):
                ohlcv_list.append(OHLCV(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(ts),
                    open=quotes.get("open", [])[i],
                    high=quotes.get("high", [])[i],
                    low=quotes.get("low", [])[i],
                    close=quotes.get("close", [])[i],
                    volume=quotes.get("volume", [])[i]
                ))
            
            return ohlcv_list
        except Exception as e:
            self.logger.error(f"Parse error: {e}")
            return []


class DataFeedManager:
    """
    Central manager for all market data feeds.
    
    Handles:
    - Multiple data source connections
    - Automatic failover
    - Data normalization
    - Tick distribution to subscribers
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger("DataFeedManager")
        
        # Initialize providers
        self.providers: Dict[MarketDataSource, DataFeedProvider] = {}
        self.primary_source = MarketDataSource.FYERS
        self.backup_sources = [MarketDataSource.ZERODHA, MarketDataSource.YAHOO]
        
        # Subscription management
        self._subscriptions: Dict[str, set] = {}  # symbol -> set of subscribers
        self._tick_buffers: Dict[str, deque] = {}  # symbol -> circular buffer
        self._buffer_size = config.get("tick_buffer_size", 1000)
        
        # Callbacks
        self._tick_callbacks: List[Callable] = []
        self._quote_callbacks: List[Callable] = []
        
        # State
        self._running = False
        self._tasks: List[asyncio.Task] = []
    
    def add_provider(self, source: MarketDataSource, provider: DataFeedProvider):
        """Add a data provider"""
        self.providers[source] = provider
    
    async def initialize(self):
        """Initialize all providers"""
        # Create providers from config
        providers_config = self.config.get("providers", {})
        
        if "fyers" in providers_config:
            self.providers[MarketDataSource.FYERS] = FyersProvider(providers_config["fyers"])
        
        if "zerodha" in providers_config:
            self.providers[MarketDataSource.ZERODHA] = ZerodhaProvider(providers_config["zerodha"])
        
        if "yahoo" in providers_config:
            self.providers[MarketDataSource.YAHOO] = YahooFinanceProvider(providers_config.get("yahoo", {}))
        
        # Connect to primary source
        if self.primary_source in self.providers:
            await self.providers[self.primary_source].connect()
    
    async def start(self):
        """Start data feed manager"""
        self._running = True
        self.logger.info("Data Feed Manager started")
    
    async def stop(self):
        """Stop data feed manager"""
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        # Disconnect all providers
        for provider in self.providers.values():
            await provider.disconnect()
        
        self.logger.info("Data Feed Manager stopped")
    
    async def subscribe_symbols(self, symbols: List[str], 
                                data_type: DataType = DataType.TICK) -> bool:
        """Subscribe to symbols using primary provider"""
        if self.primary_source in self.providers:
            provider = self.providers[self.primary_source]
            if provider.is_connected:
                success = await provider.subscribe(symbols, data_type)
                if success:
                    for symbol in symbols:
                        if symbol not in self._tick_buffers:
                            self._tick_buffers[symbol] = deque(maxlen=self._buffer_size)
                    return True
        
        # Try backup sources
        for source in self.backup_sources:
            if source in self.providers:
                provider = self.providers[source]
                if await provider.connect():
                    return await provider.subscribe(symbols, data_type)
        
        return False
    
    async def get_tick_buffer(self, symbol: str) -> deque:
        """Get tick buffer for a symbol"""
        if symbol not in self._tick_buffers:
            self._tick_buffers[symbol] = deque(maxlen=self._buffer_size)
        return self._tick_buffers[symbol]
    
    def register_tick_callback(self, callback: Callable):
        """Register callback for tick data"""
        self._tick_callbacks.append(callback)
    
    def register_quote_callback(self, callback: Callable):
        """Register callback for quote data"""
        self._quote_callbacks.append(callback)
    
    async def _on_tick(self, tick: TickData):
        """Handle incoming tick"""
        # Store in buffer
        if tick.symbol in self._tick_buffers:
            self._tick_buffers[tick.symbol].append(tick)
        
        # Call registered callbacks
        for callback in self._tick_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(tick)
                else:
                    callback(tick)
            except Exception as e:
                self.logger.error(f"Tick callback error: {e}")
    
    async def get_historical_data(self, symbol: str, start: datetime,
                                  end: datetime, interval: str = "1d") -> List[OHLCV]:
        """Get historical data from best available source"""
        # Try primary first
        if self.primary_source in self.providers:
            data = await self.providers[self.primary_source].get_historical_data(
                symbol, start, end, interval
            )
            if data:
                return data
        
        # Try backups
        for source in self.backup_sources:
            if source in self.providers:
                data = await self.providers[source].get_historical_data(
                    symbol, start, end, interval
                )
                if data:
                    return data
        
        return []
    
    async def get_current_quote(self, symbol: str) -> Optional[QuoteData]:
        """Get current quote for symbol"""
        # Return from buffer if available
        if symbol in self._tick_buffers and self._tick_buffers[symbol]:
            last_tick = self._tick_buffers[symbol][-1]
            return QuoteData(
                symbol=symbol,
                timestamp=last_tick.timestamp,
                last_price=last_tick.price,
                bid=last_tick.bid,
                ask=last_tick.ask,
                bid_size=last_tick.bid_size,
                ask_size=last_tick.ask_size,
                volume=last_tick.volume
            )
        return None
    
    def get_source_status(self) -> Dict[str, bool]:
        """Get connection status of all sources"""
        return {
            source.value: provider.is_connected 
            for source, provider in self.providers.items()
        }


# ---------------------------------------------------------------------------
# OPTIMUS: Weekly Expiry Options Chain Helper
# ---------------------------------------------------------------------------

from datetime import date as _date, timedelta as _timedelta
from typing import Optional as _Optional


def get_nearest_weekly_expiry(reference: _Optional[_date] = None) -> str:
    """
    Return the ISO-format date of the nearest weekly expiry (Thursday).

    NSE index weekly options expire every Thursday.  If today is Thursday
    and market hours are not over, returns today; otherwise returns the
    next Thursday.

    Args:
        reference: Date to calculate from (defaults to today).

    Returns:
        ISO date string "YYYY-MM-DD".
    """
    today = reference or _date.today()
    # weekday(): Monday=0 … Thursday=3 … Sunday=6
    days_to_thursday = (3 - today.weekday()) % 7
    # If today IS Thursday, keep it (expiry day); zero means same day
    expiry = today + _timedelta(days=days_to_thursday)
    return expiry.isoformat()


def build_mock_options_chain(
    symbol: str = "NIFTY",
    spot_price: float = 22000.0,
    india_vix: float = 15.0,
    pcr: float = 1.0,
    pcr_trend: str = "stable",
    futures_premium: float = 30.0,
) -> dict:
    """
    Build a synthetic / mock options-chain context for OPTIMUS when live data
    is unavailable.

    Useful for testing and simulation mode.  Values are plausible but NOT
    real market data.

    Args:
        symbol          : Index or stock symbol.
        spot_price      : Current spot price.
        india_vix       : India VIX value.
        pcr             : Put-Call Ratio.
        pcr_trend       : "rising" | "falling" | "stable".
        futures_premium : Futures basis in ₹ (positive = contango).

    Returns:
        Dict matching the expected schema for _prepare_optimus_data().
    """
    import math

    strike_gap = 100.0 if "BANK" in symbol.upper() else 50.0
    atm = round(spot_price / strike_gap) * strike_gap

    # Estimate ATM premium (Black-Scholes approximation)
    dte = max(1, (lambda e, t: (
        (_date.fromisoformat(e) - t).days
    ))(get_nearest_weekly_expiry(), _date.today()))
    atm_premium = round(spot_price * (india_vix / 100) * math.sqrt(dte / 252), 1)

    # Generate simplistic OI walls
    call_walls = [
        {"strike": atm + strike_gap * i, "oi": int(1e6 / i)}
        for i in range(1, 4)
    ]
    put_walls = [
        {"strike": atm - strike_gap * i, "oi": int(1e6 / i)}
        for i in range(1, 4)
    ]

    return {
        "symbol": symbol,
        "current_price": spot_price,
        "weekly_expiry": get_nearest_weekly_expiry(),
        "pcr": pcr,
        "pcr_trend": pcr_trend,
        "max_pain": atm,
        "call_oi_walls": call_walls,
        "put_oi_walls": put_walls,
        "ce_oi_change_pct": 0.0,
        "pe_oi_change_pct": 0.0,
        "oi_signal": "NEUTRAL",
        "india_vix": india_vix,
        "iv_rank": min(100, max(0, (india_vix - 10) / 20 * 100)),
        "iv_skew": 0.5,  # slight put skew
        "futures_premium": futures_premium,
        "price_change": 0.0,
        "support_level": atm - 2 * strike_gap,
        "resistance_level": atm + 2 * strike_gap,
    }
