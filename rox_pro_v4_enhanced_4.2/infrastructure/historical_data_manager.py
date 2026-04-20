"""
ROX Proven Edge Engine — Smart Historical Data Manager
======================================================
Solves the Fyers API rate-limit problem with three layers:
  1. DuckDB local cache  — only fetch what you don't have
  2. Async throttle      — honour Fyers' 10 req/sec limit
  3. Smart chunking      — Fyers caps intraday history at 100 days per call

Usage (drop-in replacement inside main_production.py):
    from infrastructure.historical_data_manager import HistoricalDataManager
    hdm = HistoricalDataManager(fyers_provider)
    await hdm.warm_cache(NIFTY_50_STOCKS[:20])   # pre-market, once
    candles = await hdm.get("SBIN", "1d", lookback_days=200)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import duckdb
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# --------------------------------------------------------------------------- #
#  Fyers hard limits (v3 API, as of 2025)                                     #
#  - Historical REST:  10 requests / second per token                         #
#  - Intraday data:    max 100 calendar days per request                      #
#  - Daily data:       max 365 calendar days per request                      #
#  - Calls per day:    ~1 000 (unofficial, be conservative)                   #
# --------------------------------------------------------------------------- #

FYERS_RPS          = 8          # target: 8/s to have 20 % headroom
INTRADAY_CHUNK     = 90         # days per intraday call  (< 100 limit)
DAILY_CHUNK        = 350        # days per daily call     (< 365 limit)
CACHE_DIR          = Path(__file__).parent.parent / "data" / "cache"
DB_PATH            = CACHE_DIR / "historical_ohlcv.duckdb"

logger = logging.getLogger("rox.hdm")


# --------------------------------------------------------------------------- #
#  Tiny ORM – DuckDB-backed                                                #
# --------------------------------------------------------------------------- #

@contextmanager
def _db(path: Path = DB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    try:
        yield con
    finally:
        con.close()


def _init_db():
    with _db() as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            symbol    TEXT    NOT NULL,
            timeframe TEXT    NOT NULL,
            ts        INTEGER NOT NULL,   -- unix epoch seconds
            open      REAL    NOT NULL,
            high      REAL    NOT NULL,
            low       REAL    NOT NULL,
            close     REAL    NOT NULL,
            volume    INTEGER NOT NULL,
            PRIMARY KEY (symbol, timeframe, ts)
        )""")
        con.execute("CREATE INDEX IF NOT EXISTS ix_sym_tf ON ohlcv(symbol, timeframe)")
        con.execute("""
        CREATE TABLE IF NOT EXISTS fetch_log (
            symbol    TEXT    NOT NULL,
            timeframe TEXT    NOT NULL,
            from_ts   INTEGER NOT NULL,
            to_ts     INTEGER NOT NULL,
            fetched_at INTEGER NOT NULL,
            rows      INTEGER NOT NULL,
            PRIMARY KEY (symbol, timeframe, from_ts, to_ts)
        )""")


# --------------------------------------------------------------------------- #
#  Token-bucket rate limiter                                                  #
# --------------------------------------------------------------------------- #

class _TokenBucket:
    """Thread-safe async token bucket for smooth rate-limiting."""

    def __init__(self, rate: float, capacity: float):
        self._rate     = rate
        self._capacity = capacity
        self._tokens   = capacity
        self._last     = time.monotonic()
        self._lock     = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now    = time.monotonic()
            elapsed = now - self._last
            self._tokens = min(self._capacity,
                               self._tokens + elapsed * self._rate)
            self._last = now
            if self._tokens < 1:
                wait = (1 - self._tokens) / self._rate
                await asyncio.sleep(wait)
                self._tokens = 0
            else:
                self._tokens -= 1


# --------------------------------------------------------------------------- #
#  Main class                                                                 #
# --------------------------------------------------------------------------- #

class HistoricalDataManager:
    """
    Wraps the FyersProvider and adds:
      - persistent DuckDB cache (survives restarts)
      - token-bucket throttle so you never breach 10 req/s
      - smart date chunking for long lookback windows
      - delta fetch: only downloads the missing date ranges
    """

    # Map ROX timeframe codes → (Fyers resolution, chunk_days, stale_after_secs)
    _TF_META: Dict[str, tuple] = {
        "1m":  ("1",   INTRADAY_CHUNK, 120),
        "5m":  ("5",   INTRADAY_CHUNK, 300),
        "15m": ("15",  INTRADAY_CHUNK, 900),
        "1h":  ("60",  INTRADAY_CHUNK, 3600),
        "1d":  ("D",   DAILY_CHUNK,    86400),
        "1w":  ("W",   DAILY_CHUNK,    604800),
    }

    def __init__(self, fyers_provider, rps: float = FYERS_RPS):
        self._fyers   = fyers_provider
        self._bucket  = _TokenBucket(rps, rps)   # burst = 1 second
        _init_db()
        logger.info(f"HistoricalDataManager ready (cache: {DB_PATH})")

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    async def get(self,
                  symbol: str,
                  timeframe: str = "1d",
                  lookback_days: int = 365,
                  end: Optional[datetime] = None) -> List[dict]:
        """
        Return OHLCV list for `symbol` covering last `lookback_days` days.
        Fetches only the missing segments from Fyers; rest comes from cache.
        """
        end   = end or datetime.now()
        start = end - timedelta(days=lookback_days)
        await self._ensure_coverage(symbol, timeframe, start, end)
        return self._load_from_cache(symbol, timeframe, start, end)

    async def warm_cache(self,
                         symbols: List[str],
                         timeframe: str = "1d",
                         lookback_days: int = 365,
                         concurrency: int = 4) -> Dict[str, int]:
        """
        Pre-market warm-up: download history for a list of symbols
        concurrently but rate-limited.  Returns {symbol: row_count}.
        """
        sem     = asyncio.Semaphore(concurrency)
        results = {}

        async def _fetch_one(sym):
            async with sem:
                rows = await self.get(sym, timeframe, lookback_days)
                results[sym] = len(rows)
                logger.info(f"  warm_cache: {sym} → {len(rows)} bars")

        logger.info(f"Warming cache for {len(symbols)} symbols "
                    f"({timeframe}, {lookback_days}d) ...")
        await asyncio.gather(*[_fetch_one(s) for s in symbols])
        total = sum(results.values())
        logger.info(f"Cache warm complete. {total} total bars stored.")
        return results

    async def refresh_today(self,
                            symbols: List[str],
                            timeframe: str = "1d") -> Dict[str, int]:
        """
        Lightweight intra-day refresh — only fetches today's bar / the
        last few hours.  Call this each cycle inside the main loop.
        """
        end   = datetime.now()
        start = end - timedelta(days=2)    # safe overlap
        results = {}
        for sym in symbols:
            rows = await self._fetch_and_store(sym, timeframe, start, end)
            results[sym] = rows
        return results

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    async def _ensure_coverage(self,
                                symbol: str,
                                timeframe: str,
                                start: datetime,
                                end: datetime):
        """Identify missing date ranges and fetch them."""
        missing = self._find_missing_ranges(symbol, timeframe, start, end)
        if not missing:
            return
        for seg_start, seg_end in missing:
            await self._fetch_and_store(symbol, timeframe, seg_start, seg_end)

    def _find_missing_ranges(self,
                              symbol: str,
                              timeframe: str,
                              start: datetime,
                              end: datetime) -> List[tuple]:
        """
        Returns list of (from, to) datetime pairs that are not yet in the
        cache, by querying the fetch_log table.
        """
        start_ts = int(start.timestamp())
        end_ts   = int(end.timestamp())

        with _db() as con:
            rows = con.execute("""
                SELECT from_ts, to_ts FROM fetch_log
                WHERE symbol=? AND timeframe=?
                  AND to_ts >= ? AND from_ts <= ?
                ORDER BY from_ts
            """, (symbol, timeframe, start_ts, end_ts)).fetchall()

        if not rows:
            return [(start, end)]

        # Minimum gap size per timeframe — gaps smaller than this are skipped.
        # Prevents sub-day datetime.now() drift from firing unnecessary API calls.
        _MIN_GAP: Dict[str, int] = {
            "1m": 120, "5m": 300, "15m": 900, "1h": 3600,
            "1d": 82800,   # 23 h  (handles end-of-day drift)
            "1w": 518400,  # 6 days
        }
        min_gap_secs = _MIN_GAP.get(timeframe, 3600)

        gaps = []
        cursor = start_ts

        for r in rows:
            r_from, r_to = r[0], r[1]
            if cursor < r_from and (r_from - cursor) >= min_gap_secs:
                gaps.append((
                    datetime.fromtimestamp(cursor),
                    datetime.fromtimestamp(r_from)
                ))
            cursor = max(cursor, r_to)

        if cursor < end_ts and (end_ts - cursor) >= min_gap_secs:
            gaps.append((
                datetime.fromtimestamp(cursor),
                datetime.fromtimestamp(end_ts)
            ))

        return gaps

    async def _fetch_and_store(self,
                                symbol: str,
                                timeframe: str,
                                start: datetime,
                                end: datetime) -> int:
        """Chunk the date range, throttle, call Fyers, persist."""
        _, chunk_days, _ = self._TF_META.get(timeframe, ("D", DAILY_CHUNK, 86400))
        chunks = self._date_chunks(start, end, chunk_days)
        total_stored = 0

        for ch_start, ch_end in chunks:
            await self._bucket.acquire()          # respect rate limit
            try:
                candles = await self._fyers.get_historical_data(
                    symbol, ch_start, ch_end, timeframe
                )
            except Exception as e:
                logger.error(f"Fyers fetch failed {symbol} {ch_start}→{ch_end}: {e}")
                await asyncio.sleep(2)            # back-off on error
                continue

            if candles:
                stored = self._upsert_candles(symbol, timeframe, candles)
                total_stored += stored
                self._log_fetch(symbol, timeframe, ch_start, ch_end, stored)
                logger.debug(f"  {symbol} {timeframe}: +{stored} bars "
                             f"({ch_start.date()}→{ch_end.date()})")

        return total_stored

    @staticmethod
    def _date_chunks(start: datetime,
                     end: datetime,
                     chunk_days: int) -> List[tuple]:
        """Split a date range into chunks of at most chunk_days days."""
        chunks = []
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + timedelta(days=chunk_days), end)
            chunks.append((cursor, chunk_end))
            cursor = chunk_end
        return chunks

    def _upsert_candles(self, symbol: str, timeframe: str, candles) -> int:
        """Insert-or-ignore OHLCV rows into DuckDB.  Returns count stored."""
        rows = []
        for c in candles:
            # Support both OHLCV dataclass objects and plain dicts
            if hasattr(c, "timestamp"):
                ts  = int(c.timestamp.timestamp())
                row = (symbol, timeframe, ts,
                       c.open, c.high, c.low, c.close, c.volume)
            else:
                ts  = int(datetime.fromisoformat(c["timestamp"]).timestamp())
                row = (symbol, timeframe, ts,
                       c["open"], c["high"], c["low"], c["close"], c["volume"])
            rows.append(row)

        with _db() as con:
            con.executemany("""
                INSERT INTO ohlcv
                    (symbol, timeframe, ts, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (symbol, timeframe, ts) DO NOTHING
            """, rows)

        return len(rows)

    def _log_fetch(self, symbol, timeframe, start, end, rows):
        with _db() as con:
            con.execute("""
                INSERT INTO fetch_log
                    (symbol, timeframe, from_ts, to_ts, fetched_at, rows)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (symbol, timeframe, from_ts, to_ts)
                DO UPDATE SET fetched_at=excluded.fetched_at, rows=excluded.rows
            """, (symbol, timeframe,
                  int(start.timestamp()), int(end.timestamp()),
                  int(time.time()), rows))

    def _load_from_cache(self,
                          symbol: str,
                          timeframe: str,
                          start: datetime,
                          end: datetime) -> List[dict]:
        start_ts = int(start.timestamp())
        end_ts   = int(end.timestamp())
        with _db() as con:
            rows = con.execute("""
                SELECT ts, open, high, low, close, volume
                FROM ohlcv
                WHERE symbol=? AND timeframe=?
                  AND ts >= ? AND ts <= ?
                ORDER BY ts
            """, (symbol, timeframe, start_ts, end_ts)).fetchall()

        return [
            {
                "timestamp": datetime.fromtimestamp(r[0]).isoformat(),
                "open":   r[1],
                "high":   r[2],
                "low":    r[3],
                "close":  r[4],
                "volume": r[5],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------ #
    #  Sync wrappers — for use inside main_production.py's              #
    #  ThreadPoolExecutor / non-async context                            #
    # ------------------------------------------------------------------ #

    def _run_sync(self, coro):
        """Run an async coroutine from sync code safely."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside an existing loop (e.g. Jupyter / some frameworks)
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(asyncio.run, coro).result(timeout=120)
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)

    def get_sync(self,
                 symbol: str,
                 timeframe: str = "1d",
                 lookback_days: int = 365) -> List[dict]:
        """Synchronous version of get(). Returns list of OHLCV dicts."""
        return self._run_sync(self.get(symbol, timeframe, lookback_days))

    def warm_cache_sync(self,
                        symbols: List[str],
                        timeframe: str = "1d",
                        lookback_days: int = 365) -> Dict[str, int]:
        """Synchronous version of warm_cache()."""
        return self._run_sync(
            self.warm_cache(symbols, timeframe, lookback_days, concurrency=4)
        )

    def refresh_today_sync(self,
                           symbols: List[str],
                           timeframe: str = "1d") -> Dict[str, int]:
        """Synchronous version of refresh_today()."""
        return self._run_sync(self.refresh_today(symbols, timeframe))

    def cache_stats(self) -> dict:
        """Quick diagnostic: how many rows are cached per symbol/tf."""
        with _db() as con:
            rows = con.execute("""
                SELECT symbol, timeframe, COUNT(*) as bars,
                       MIN(ts) as oldest, MAX(ts) as newest
                FROM ohlcv GROUP BY symbol, timeframe
                ORDER BY symbol, timeframe
            """).fetchall()
        return {
            f"{r[0]}|{r[1]}": {
                "bars":   r[2],
                "oldest": datetime.fromtimestamp(r[3]).date().isoformat(),
                "newest": datetime.fromtimestamp(r[4]).date().isoformat(),
            }
            for r in rows
        }
