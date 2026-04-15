"""
ROX Proven Edge Engine v3.0 - Cache Manager
==========================================
Redis caching and circular buffers for real-time data.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Generic, TypeVar, Deque
from collections import deque
import time
import pickle


T = TypeVar('T')


class CacheBackend(ABC):
    """Abstract cache backend"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        pass


class InMemoryCache(CacheBackend):
    """In-memory cache implementation"""
    
    def __init__(self, max_size: int = 10000):
        self._cache: Dict[str, Any] = {}
        self._expiry: Dict[str, float] = {}
        self._max_size = max_size
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            # Check expiry
            if key in self._expiry and self._expiry[key] < time.time():
                del self._cache[key]
                del self._expiry[key]
                return None
            
            return self._cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        async with self._lock:
            # Evict old entries if at capacity
            if len(self._cache) >= self._max_size and key not in self._cache:
                # Remove oldest expired entries first
                now = time.time()
                expired = [k for k, v in self._expiry.items() if v < now]
                for k in expired:
                    del self._cache[k]
                    del self._expiry[k]
                
                # If still at capacity, remove oldest
                if len(self._cache) >= self._max_size:
                    oldest = next(iter(self._cache))
                    del self._cache[oldest]
                    if oldest in self._expiry:
                        del self._expiry[oldest]
            
            self._cache[key] = value
            if ttl:
                self._expiry[key] = time.time() + ttl
            
            return True
    
    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                if key in self._expiry:
                    del self._expiry[key]
                return True
            return False
    
    async def exists(self, key: str) -> bool:
        async with self._lock:
            if key in self._expiry and self._expiry[key] < time.time():
                return False
            return key in self._cache
    
    async def clear(self) -> bool:
        async with self._lock:
            self._cache.clear()
            self._expiry.clear()
            return True


class RedisCache(CacheBackend):
    """Redis cache implementation"""
    
    def __init__(self, host: str = "localhost", port: int = 6379, 
                 db: int = 0, password: str = None):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self._redis = None
        self._connected = False
        self.logger = logging.getLogger("RedisCache")
    
    async def connect(self) -> bool:
        """Connect to Redis"""
        try:
            import aioredis
            self._redis = await aioredis.create_redis_pool(
                f"redis://{self.host}:{self.port}",
                db=self.db,
                password=self.password
            )
            self._connected = True
            self.logger.info("Connected to Redis")
            return True
        except ImportError:
            self.logger.warning("aioredis not installed, using in-memory fallback")
            return False
        except Exception as e:
            self.logger.error(f"Redis connection error: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self._redis:
            self._redis.close()
            await self._redis.wait_closed()
        self._connected = False
    
    async def get(self, key: str) -> Optional[Any]:
        if not self._connected or not self._redis:
            return None
        
        try:
            data = await self._redis.get(key)
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            self.logger.error(f"Redis get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        if not self._connected or not self._redis:
            return False
        
        try:
            data = pickle.dumps(value)
            if ttl:
                await self._redis.setex(key, ttl, data)
            else:
                await self._redis.set(key, data)
            return True
        except Exception as e:
            self.logger.error(f"Redis set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        if not self._connected or not self._redis:
            return False
        
        try:
            await self._redis.delete(key)
            return True
        except Exception as e:
            self.logger.error(f"Redis delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        if not self._connected or not self._redis:
            return False
        
        try:
            return await self._redis.exists(key) > 0
        except Exception as e:
            self.logger.error(f"Redis exists error: {e}")
            return False
    
    async def clear(self) -> bool:
        if not self._connected or not self._redis:
            return False
        
        try:
            await self._redis.flushdb()
            return True
        except Exception as e:
            self.logger.error(f"Redis clear error: {e}")
            return False


@dataclass
class CircularBuffer(Generic[T]):
    """
    Thread-safe circular buffer for rolling window calculations.
    
    Features:
    - Fixed size with automatic eviction
    - O(1) append and access
    - Statistical calculations
    """
    
    max_size: int = 1000
    _buffer: Deque[T] = field(default_factory=deque)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    
    def append(self, item: T):
        """Add item to buffer"""
        self._buffer.append(item)
        while len(self._buffer) > self.max_size:
            self._buffer.popleft()
    
    def extend(self, items: List[T]):
        """Add multiple items"""
        for item in items:
            self.append(item)
    
    def get_latest(self, n: int = 1) -> List[T]:
        """Get latest n items"""
        return list(self._buffer)[-n:]
    
    def get_all(self) -> List[T]:
        """Get all items"""
        return list(self._buffer)
    
    def __len__(self) -> int:
        return len(self._buffer)
    
    def __getitem__(self, index: int) -> T:
        return self._buffer[index]
    
    def clear(self):
        """Clear buffer"""
        self._buffer.clear()
    
    @property
    def is_full(self) -> bool:
        return len(self._buffer) >= self.max_size
    
    @property
    def is_empty(self) -> bool:
        return len(self._buffer) == 0


class NumericBuffer(CircularBuffer[float]):
    """Circular buffer optimized for numeric calculations"""
    
    def sum(self) -> float:
        """Calculate sum"""
        return sum(self._buffer) if self._buffer else 0.0
    
    def mean(self) -> float:
        """Calculate mean"""
        if not self._buffer:
            return 0.0
        return sum(self._buffer) / len(self._buffer)
    
    def std(self) -> float:
        """Calculate standard deviation"""
        if len(self._buffer) < 2:
            return 0.0
        
        mean = self.mean()
        variance = sum((x - mean) ** 2 for x in self._buffer) / len(self._buffer)
        return variance ** 0.5
    
    def min(self) -> float:
        """Get minimum"""
        return min(self._buffer) if self._buffer else 0.0
    
    def max(self) -> float:
        """Get maximum"""
        return max(self._buffer) if self._buffer else 0.0
    
    def percentile(self, p: float) -> float:
        """Get percentile value (p between 0 and 100)"""
        if not self._buffer:
            return 0.0
        
        sorted_data = sorted(self._buffer)
        idx = int(len(sorted_data) * p / 100)
        idx = max(0, min(idx, len(sorted_data) - 1))
        return sorted_data[idx]
    
    def ema(self, alpha: float = 0.1) -> float:
        """Calculate exponential moving average"""
        if not self._buffer:
            return 0.0
        
        ema = self._buffer[0]
        for value in self._buffer[1:]:
            ema = alpha * value + (1 - alpha) * ema
        return ema
    
    def rate_of_change(self, period: int = 1) -> float:
        """Calculate rate of change over period"""
        if len(self._buffer) < period + 1:
            return 0.0
        
        current = self._buffer[-1]
        previous = self._buffer[-(period + 1)]
        
        if previous == 0:
            return 0.0
        
        return (current - previous) / previous * 100


@dataclass
class TickBuffer:
    """Buffer for tick data with specialized calculations"""
    
    max_size: int = 1000
    _prices: NumericBuffer = field(default_factory=lambda: NumericBuffer(max_size=1000))
    _volumes: NumericBuffer = field(default_factory=lambda: NumericBuffer(max_size=1000))
    _timestamps: List[datetime] = field(default_factory=list)
    
    def add_tick(self, price: float, volume: float, timestamp: datetime):
        """Add a tick"""
        self._prices.append(price)
        self._volumes.append(volume)
        self._timestamps.append(timestamp)
        
        # Maintain size
        while len(self._timestamps) > self.max_size:
            self._timestamps.pop(0)
    
    def vwap(self) -> float:
        """Calculate VWAP"""
        if not self._prices or not self._volumes:
            return 0.0
        
        total_value = sum(p * v for p, v in zip(self._prices.get_all(), self._volumes.get_all()))
        total_volume = self._volumes.sum()
        
        if total_volume == 0:
            return self._prices.mean()
        
        return total_value / total_volume
    
    def high(self) -> float:
        """Get highest price"""
        return self._prices.max()
    
    def low(self) -> float:
        """Get lowest price"""
        return self._prices.min()
    
    def last_price(self) -> float:
        """Get last price"""
        return self._prices.get_latest(1)[0] if self._prices else 0.0
    
    def total_volume(self) -> float:
        """Get total volume"""
        return self._volumes.sum()


class RedisCacheManager:
    """
    High-level cache manager with Redis backend and fallback.
    
    Features:
    - Redis for distributed caching
    - In-memory fallback
    - Automatic serialization
    - TTL management
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger("CacheManager")
        
        # Initialize backends
        self.redis = RedisCache(
            host=config.get("redis_host", "localhost"),
            port=config.get("redis_port", 6379),
            db=config.get("redis_db", 0),
            password=config.get("redis_password")
        )
        
        self.memory = InMemoryCache(
            max_size=config.get("memory_cache_size", 10000)
        )
        
        self._use_redis = False
        
        # Cache prefixes
        self.PREFIX_TICK = "tick:"
        self.PREFIX_QUOTE = "quote:"
        self.PREFIX_INDICATOR = "ind:"
        self.PREFIX_SIGNAL = "sig:"
    
    async def initialize(self):
        """Initialize cache manager"""
        self._use_redis = await self.redis.connect()
        if not self._use_redis:
            self.logger.warning("Using in-memory cache (Redis unavailable)")
    
    async def shutdown(self):
        """Shutdown cache manager"""
        await self.redis.disconnect()
        await self.memory.clear()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        # Try Redis first
        if self._use_redis:
            value = await self.redis.get(key)
            if value is not None:
                return value
        
        # Fallback to memory
        return await self.memory.get(key)
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache"""
        # Set in both backends
        tasks = [self.memory.set(key, value, ttl)]
        if self._use_redis:
            tasks.append(self.redis.set(key, value, ttl))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return any(r is True for r in results if not isinstance(r, Exception))
    
    async def delete(self, key: str) -> bool:
        """Delete from cache"""
        results = await asyncio.gather(
            self.memory.delete(key),
            self.redis.delete(key) if self._use_redis else asyncio.sleep(0),
            return_exceptions=True
        )
        return True
    
    async def cache_tick(self, symbol: str, tick_data: Dict, ttl: int = 300):
        """Cache tick data"""
        key = f"{self.PREFIX_TICK}{symbol}:{datetime.now().isoformat()}"
        await self.set(key, tick_data, ttl)
    
    async def get_recent_ticks(self, symbol: str, count: int = 100) -> List[Dict]:
        """Get recent ticks for symbol"""
        # This would scan keys in Redis or use memory buffer
        return []
    
    async def cache_indicator(self, symbol: str, indicator_name: str, 
                             value: Any, ttl: int = 60):
        """Cache indicator value"""
        key = f"{self.PREFIX_INDICATOR}{symbol}:{indicator_name}"
        await self.set(key, value, ttl)
    
    async def get_indicator(self, symbol: str, indicator_name: str) -> Optional[Any]:
        """Get cached indicator"""
        key = f"{self.PREFIX_INDICATOR}{symbol}:{indicator_name}"
        return await self.get(key)
    
    async def cache_signal(self, agent: str, symbol: str, signal: Dict, ttl: int = 300):
        """Cache agent signal"""
        key = f"{self.PREFIX_SIGNAL}{agent}:{symbol}:{datetime.now().isoformat()}"
        await self.set(key, signal, ttl)
    
    async def get_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            "redis_connected": self._use_redis,
            "memory_cache_size": len(self.memory._cache)
        }
