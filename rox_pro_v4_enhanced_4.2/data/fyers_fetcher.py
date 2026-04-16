"""
ROX Proven Edge Engine v4.0 — Fyers Live Data Fetcher
======================================================
Fetches real-time and historical market data from the Fyers API
and returns it in the exact schema expected by UnifiedCoordinator
/ LeadCoordinator.generate_trading_plan().

Required .env keys (populated by fyers_login.py):
    FYERS_APP_ID        — e.g.  XYZ123-100
    FYERS_ACCESS_TOKEN  — refreshed daily via fyers_login.py
    FYERS_ENABLED       — must be "true"

Performance design
------------------
- The Fyers client is created ONCE and reused across all cycles.
- OHLCV history (300+ days of daily bars, used for SMA-200/ATR/RSI) is
  fetched once per calendar day and cached in memory. This reduces
  each 60-second cycle from ~50 API calls to just 1 quotes batch
  call + 1 option-chain call (~2-3 seconds total).
- Quotes (LTP, OHLC, volume) are fetched fresh every cycle.

Usage (internal — called by ROXUnifiedEngine._fetch_live_data):
    from data.fyers_fetcher import FyersFetcher
    fetcher = FyersFetcher(api_config)          # create once
    data = fetcher.fetch_market_data()          # call every cycle
"""

import logging
import time
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional

logger = logging.getLogger("FyersFetcher")

# IMPROVEMENT 3: Sector average PE estimates for NEXUS fundamental scoring.
# These are approximate long-term sector median PEs for the Indian market.
# NEXUS uses industry_pe to determine if a stock is over/undervalued vs its sector.
_SECTOR_PE_ESTIMATES: Dict[str, float] = {
    "Banking":        14.0,
    "IT":             28.0,
    "Energy":         12.0,
    "Auto":           20.0,
    "Metals":         10.0,
    "Pharma":         30.0,
    "FMCG":           45.0,
    "Infrastructure": 22.0,
    "Telecom":        25.0,
    "Others":         22.0,
}

def _sector_pe_estimate(symbol: str) -> float:
    """Return the estimated sector median PE for a given stock symbol."""
    try:
        from config import SECTOR_MAPPING
        for sector, stocks in SECTOR_MAPPING.items():
            if symbol in stocks:
                return _SECTOR_PE_ESTIMATES.get(sector, 22.0)
    except Exception:
        pass
    return 22.0

# Macro data (FII/DII, PE, yield) — fetched via separate fetcher, cached daily
_macro_fetcher = None
def _get_macro_fetcher():
    global _macro_fetcher
    if _macro_fetcher is None:
        from data.macro_fetcher import MacroFetcher
        _macro_fetcher = MacroFetcher()
    return _macro_fetcher

# ── Fyers symbol format helpers ──────────────────────────────────────────────

NSE_PREFIX  = "NSE:"
BSE_PREFIX  = "BSE:"
INDEX_NIFTY      = "NSE:NIFTY50-INDEX"
INDEX_VIX        = "NSE:INDIAVIX-INDEX"
INDEX_BANKNIFTY  = "NSE:NIFTYBANK-INDEX"
INDEX_SENSEX     = "BSE:SENSEX-INDEX"
INDEX_FINNIFTY   = "NSE:FINNIFTY-INDEX"
INDEX_BANKEX     = "BSE:BANKEX-INDEX"

# All five liquid index option markets
_ALL_INDICES = {
    "NIFTY":     INDEX_NIFTY,
    "BANKNIFTY": INDEX_BANKNIFTY,
    "SENSEX":    INDEX_SENSEX,
    "FINNIFTY":  INDEX_FINNIFTY,
    "BANKEX":    INDEX_BANKEX,
}

def _eq(symbol: str) -> str:
    """Convert bare ticker to Fyers NSE equity symbol."""
    return f"{NSE_PREFIX}{symbol}-EQ"

# ── Symbol alias overrides ────────────────────────────────────────────────────
# Some symbols have known issues with cont_flag=1 (corporate actions, demergers).
# Map them here to their correct Fyers-API symbol string.
_SYMBOL_OVERRIDES: Dict[str, str] = {
    # ── TATAMOTORS post-demerger (Oct 2025) ──────────────────────────────────
    # Tata Motors demerged its CV and PV businesses into two separately listed
    # entities effective October 2025.  The old NSE:TATAMOTORS-EQ no longer
    # trades.  The two new tickers are:
    #   TATAMOTORS_CV → NSE:TMCV-EQ   (Commercial Vehicles — legacy entity)
    #   TATAMOTORS_PV → NSE:TMPV-EQ   (Passenger Vehicles — new entity)
    # We use internal aliases (TATAMOTORS_CV / TATAMOTORS_PV) throughout the
    # codebase and resolve them to Fyers symbols here.
    "TATAMOTORS_CV": "NSE:TMCV-EQ",
    "TATAMOTORS_PV": "NSE:TMPV-EQ",

    # BAJAJ-AUTO: hyphen in ticker is passed verbatim to Fyers — no override needed.
    # "BAJAJ-AUTO": "NSE:BAJAJ-AUTO-EQ",  # Fyers handles this natively

    # M&M: ampersand is handled by Fyers natively.
    # "M&M": "NSE:M&M-EQ",  # Fyers handles this natively
}

def _eq_safe(symbol: str) -> str:
    """Return the Fyers equity symbol, using override if defined."""
    return _SYMBOL_OVERRIDES.get(symbol, f"{NSE_PREFIX}{symbol}-EQ")

def _index(symbol: str) -> str:
    return f"{NSE_PREFIX}{symbol}-INDEX"


# ── Simple SMA / ATR helpers (computed from OHLCV history) ──────────────────

def _sma(closes: List[float], period: int) -> float:
    """Simple Moving Average over the last `period` closes.

    FIX-SMA-200: When insufficient data for the full period, compute from
    all available data instead of returning the last close (which made
    200DMA ≈ current price, showing 0.0000% distance).  A partial-period
    average is a far better approximation than a single data point.
    Requires at least `period // 3` data points to return a value;
    otherwise returns 0.0 to signal "insufficient data".
    """
    if not closes:
        return 0.0
    if len(closes) >= period:
        return sum(closes[-period:]) / period
    # Insufficient data for full period — use all available data
    # but require at least period//3 points for a meaningful average.
    min_required = max(period // 3, 20)   # at least 20 data points
    if len(closes) >= min_required:
        avg = sum(closes) / len(closes)
        logger.debug(
            f"_sma({period}): only {len(closes)} data points "
            f"(need {period}), using all-available avg={avg:.2f}"
        )
        return avg
    # Truly insufficient data — return 0 so callers can handle N/A
    return 0.0

def _atr(candles: List[Dict], period: int = 14) -> float:
    """Average True Range over last `period` bars."""
    trs = []
    for i in range(1, len(candles)):
        h = candles[i]["high"]; l = candles[i]["low"]; pc = candles[i-1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if not trs:
        return candles[-1]["close"] * 0.015
    return sum(trs[-period:]) / min(len(trs), period)

def _rsi(closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains  = [d for d in deltas[-period:] if d > 0]
    losses = [-d for d in deltas[-period:] if d < 0]
    avg_gain = sum(gains) / period if gains else 0
    avg_loss = sum(losses) / period if losses else 1e-9
    rs = avg_gain / avg_loss
    return round(100 - 100 / (1 + rs), 2)

def _trend(close: float, sma20: float, sma50: float, sma200: float) -> str:
    if close > sma20 > sma50 > sma200:
        return "UPTREND"
    if close < sma20 < sma50 < sma200:
        return "DOWNTREND"
    return "SIDEWAYS"

def _ema(values: List[float], period: int) -> List[float]:
    """Exponential Moving Average — returns full series, same length as input."""
    if not values:
        return []
    k = 2.0 / (period + 1)
    ema_vals = [values[0]]
    for v in values[1:]:
        ema_vals.append(v * k + ema_vals[-1] * (1 - k))
    return ema_vals

def _macd(closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9
          ) -> tuple:
    """Returns (macd_line, signal_line, histogram). Falls back to (0,0,0)."""
    if len(closes) < slow + signal:
        return 0.0, 0.0, 0.0
    fast_ema  = _ema(closes, fast)
    slow_ema  = _ema(closes, slow)
    macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]
    sig_line  = _ema(macd_line, signal)
    histogram = macd_line[-1] - sig_line[-1]
    return round(macd_line[-1], 4), round(sig_line[-1], 4), round(histogram, 4)

def _bollinger_bands(closes: List[float], period: int = 20, num_std: float = 2.0
                     ) -> tuple:
    """Returns (upper, middle, lower). Falls back using last close."""
    if not closes:
        return 0.0, 0.0, 0.0
    window = closes[-period:] if len(closes) >= period else closes
    mid    = sum(window) / len(window)
    var    = sum((x - mid) ** 2 for x in window) / len(window)
    std    = var ** 0.5
    return round(mid + num_std * std, 2), round(mid, 2), round(mid - num_std * std, 2)

def _adx_approx(candles: List[Dict], period: int = 14) -> float:
    """
    Wilder-smoothed ADX — the industry-standard 14-period Average Directional Index.
    Uses Wilder's running smoothing (equivalent to EMA alpha=1/period) on TR, DM+, DM−,
    then computes DX per bar and smooths into ADX.  Requires at least 2*period candles.
    Falls back to 20 if insufficient data.
    """
    n = len(candles)
    if n < period * 2 + 1:
        return 20.0

    # Step 1: raw TR, DM+, DM− series
    tr_raw, dmp_raw, dmn_raw = [], [], []
    for i in range(1, n):
        h  = candles[i]["high"];   l  = candles[i]["low"]
        ph = candles[i-1]["high"]; pl = candles[i-1]["low"]
        pc = candles[i-1]["close"]
        up   = h - ph;  down = pl - l
        tr_raw.append(max(h - l, abs(h - pc), abs(l - pc)))
        dmp_raw.append(up   if up   > down and up   > 0 else 0.0)
        dmn_raw.append(down if down > up   and down > 0 else 0.0)

    # Step 2: seed Wilder sums with first `period` bars
    tr14  = sum(tr_raw[:period])
    dmp14 = sum(dmp_raw[:period])
    dmn14 = sum(dmn_raw[:period])

    # Step 3: walk the rest, accumulating smoothed DX values
    dx_list: List[float] = []
    for i in range(period, len(tr_raw)):
        # Wilder smoothing: new = prev - prev/period + current
        tr14  = tr14  - tr14  / period + tr_raw[i]
        dmp14 = dmp14 - dmp14 / period + dmp_raw[i]
        dmn14 = dmn14 - dmn14 / period + dmn_raw[i]
        if tr14 == 0:
            continue
        di_plus  = dmp14 / tr14 * 100
        di_minus = dmn14 / tr14 * 100
        denom = di_plus + di_minus
        if denom == 0:
            continue
        dx_list.append(abs(di_plus - di_minus) / denom * 100)

    if not dx_list:
        return 20.0

    # Step 4: Wilder-smooth the DX series into ADX
    adx = sum(dx_list[:period]) / period   # seed
    for dx in dx_list[period:]:
        adx = (adx * (period - 1) + dx) / period

    return round(adx, 1)


# ── Main fetcher class ───────────────────────────────────────────────────────

class FyersFetcher:
    """
    Fetches live market data from Fyers API and returns it in the
    shape expected by coordinator.generate_trading_plan().

    Parameters
    ----------
    api_config : APIConfig
        Must have fyers_app_id and fyers_access_token populated.
    watchlist : list[str], optional
        NSE tickers to fetch (bare symbols, e.g. "RELIANCE").
        Defaults to Nifty 50.
    history_days : int
        Number of calendar days of OHLCV history to fetch for
        indicator calculation (default 300 → enough for SMA-200).
    """

    def __init__(self, api_config, watchlist: Optional[List[str]] = None,
                 history_days: int = 300):
        self.api_config   = api_config
        self.history_days = history_days
        # FIX-SMA-200: Increased from 60 → 300 calendar days so we have
        # enough daily bars for a true 200-day SMA.  300 calendar days ≈
        # 210 trading days, giving a real 200DMA instead of the all-available
        # fallback in _sma().  Extra API time is negligible (one call per
        # symbol with a wider date range, not more calls).

        from config import NIFTY_50_STOCKS
        self.watchlist = watchlist or NIFTY_50_STOCKS

        # History cache: rebuilt once per calendar day, reused every 60s cycle
        self._history_cache: Dict[str, List[Dict]] = {}
        # Rolling VIX history (last 20 readings, one per 60-second cycle)
        self._vix_history: List[float] = []
        self._history_cache_date: Optional[date]   = None

        # IMPROVEMENT 4: OI change cache for SENTINEL
        # Stores previous session's total OI per index so we can compute
        # real OI delta (long buildup vs short covering) across live cycles.
        self._prev_ce_oi_cache: Dict[str, int] = {}
        self._prev_pe_oi_cache: Dict[str, int] = {}

        self._fyers = self._init_client()

        # ── Startup: sync open F&O positions from Fyers ───────────────────────
        # Done once at object creation so the coordinator always knows the
        # current open book before the first market-data cycle fires.
        self.fno_positions: List[Dict] = self.sync_fno_positions()

    # ── Fyers client ─────────────────────────────────────────────────────────

    def _init_client(self):
        try:
            from fyers_apiv3 import fyersModel
        except ImportError:
            raise RuntimeError(
                "fyers-apiv3 not installed. Run: pip install fyers-apiv3"
            )
        client_id    = self.api_config.fyers_app_id or self.api_config.fyers_api_key
        access_token = self.api_config.fyers_access_token
        if not client_id or not access_token:
            raise RuntimeError(
                "FYERS_APP_ID and FYERS_ACCESS_TOKEN must be set in .env. "
                "Run fyers_login.py first."
            )
        fyers = fyersModel.FyersModel(
            client_id=client_id,
            token=access_token,
            log_path="logs",
            is_async=False,
        )
        logger.info(f"Fyers client initialised | app_id={client_id[:8]}...")
        return fyers

    # ── Public API ───────────────────────────────────────────────────────────

    def fetch_market_data(self) -> Dict[str, Any]:
        """
        Fetch all required data and return in coordinator schema.

        Fast path (every 60s cycle):
          - 1 quotes batch call  — LTP, OHLC, volume for all stocks + indices
          - 1 option-chain call  — PCR, OI walls, max pain

        Slow path (once per calendar day, cached):
          - 51 history calls     — 60+ days of daily OHLCV bars for
                                   SMA/ATR/RSI/ADX calculation
        """
        today = date.today()

        # ── Slow path: rebuild history cache once per day ─────────────────
        if self._history_cache_date != today or not self._history_cache:
            logger.info("Fetching OHLCV history (daily cache refresh)...")
            self._history_cache      = self._fetch_history_all()
            self._history_cache_date = today
            logger.info(f"History cached for {len(self._history_cache)} symbols")
        else:
            logger.debug("Using cached OHLCV history")

        ohlcv_history = self._history_cache

        # ── Fast path: fresh quotes every cycle ───────────────────────────
        logger.info("Fetching live quotes from Fyers...")
        quotes      = self._fetch_quotes()
        nifty_quote = quotes.get("NIFTY50", {})
        vix_quote   = quotes.get("INDIAVIX", {})

        nifty_price = nifty_quote.get("ltp", 22500.0)
        india_vix   = vix_quote.get("ltp", 15.0)

        # Intraday change data (used by agents for short-term momentum)
        nifty_open       = nifty_quote.get("open_price", nifty_price)
        nifty_prev_close = nifty_quote.get("prev_close", nifty_price)
        nifty_change_pct = nifty_quote.get("change_pct", 0.0)
        nifty_intraday_range = {
            "open":       nifty_open,
            "high":       nifty_quote.get("high_price", nifty_price),
            "low":        nifty_quote.get("low_price",  nifty_price),
            "prev_close": nifty_prev_close,
            "change_pct": nifty_change_pct,
        }

        # IMPROVEMENT 7: Fetch real 15-min NIFTY bars for true multi-timeframe analysis.
        # ORION currently derives all timeframes from the same daily OHLCV. These 15-min
        # bars let it compute short-term RSI/EMA so 1H/4H confluence becomes real.
        # Wrapped in try/except — if Fyers 15-min API fails, pipeline continues normally.
        nifty_15min_bars: List[Dict] = []
        try:
            nifty_15min_bars = self._fetch_intraday_history(INDEX_NIFTY, days=5)
        except Exception as _e15:
            logger.debug(f"15-min bars unavailable: {_e15}")

        # 3. Build per-stock price_data and indicators
        price_data = {}
        indicators = {}
        for sym in self.watchlist:
            q       = quotes.get(sym, {})
            candles = ohlcv_history.get(sym, [])
            closes  = [c["close"] for c in candles]

            close  = q.get("ltp", closes[-1] if closes else 0.0)
            open_  = q.get("open_price", close)
            high   = q.get("high_price",  close * 1.01)
            low    = q.get("low_price",   close * 0.99)
            volume = q.get("volume", 0)
            atr    = _atr(candles) if len(candles) > 2 else close * 0.015

            # Use last 30d avg volume as avg_volume
            avg_vol = (
                int(sum(c["volume"] for c in candles[-30:]) / 30)
                if len(candles) >= 30 else max(volume, 1)
            )

            price_data[sym] = {
                "close":     close,           # live LTP — for P&L display
                "ref_close": closes[-1] if closes else close,  # prev session EOD close
                                              # ↑ used by coordinator for setup anchoring
                                              #   so entry/SL/target don't drift with
                                              #   intra-minute price noise
                "open": open_,
                "high": high,   "low": low,
                "volume": volume, "avg_volume": avg_vol, "atr": atr,
            }

            sma20  = _sma(closes, 20)
            sma50  = _sma(closes, 50)
            sma200 = _sma(closes, 200)
            rsi    = _rsi(closes)
            adx    = _adx_approx(candles)
            trend  = _trend(close, sma20, sma50, sma200)
            vol_ratio = volume / avg_vol if avg_vol > 0 else 1.0

            macd_line, macd_sig, macd_hist = _macd(closes)
            bb_upper, bb_mid, bb_lower     = _bollinger_bands(closes)
            bb_width = round((bb_upper - bb_lower) / bb_mid, 4) if bb_mid else 0.0

            indicators[sym] = {
                "rsi": rsi, "sma20": sma20, "sma50": sma50, "sma200": sma200,
                "atr": atr, "adx": adx, "trend": trend,
                "volume_ratio": round(vol_ratio, 2),
                # MACD
                "macd": macd_line, "macd_signal": macd_sig, "macd_histogram": macd_hist,
                # Bollinger Bands
                "bb_upper": bb_upper, "bb_middle": bb_mid, "bb_lower": bb_lower,
                "bb_width": bb_width,
            }

        # 4. Nifty 200-DMA from history
        nifty_candles = ohlcv_history.get("NIFTY50", [])
        nifty_closes  = [c["close"] for c in nifty_candles]
        nifty_200dma  = _sma(nifty_closes, 200)
        nifty_adx     = _adx_approx(nifty_candles)

        # Real NIFTY indicators — feed directly to _prepare_orion_data
        n_sma20  = _sma(nifty_closes, 20)
        n_sma50  = _sma(nifty_closes, 50)
        n_rsi    = _rsi(nifty_closes)
        n_macd, n_macd_sig, n_macd_hist = _macd(nifty_closes)
        n_bb_upper, n_bb_mid, n_bb_lower = _bollinger_bands(nifty_closes)
        n_bb_width = round((n_bb_upper - n_bb_lower) / n_bb_mid, 4) if n_bb_mid else 0.0
        n_atr    = _atr(nifty_candles) if len(nifty_candles) > 2 else nifty_price * 0.01
        n_trend  = _trend(nifty_price, n_sma20, n_sma50, nifty_200dma)
        nifty_indicators = {
            "rsi": n_rsi, "sma20": n_sma20, "sma50": n_sma50, "sma200": nifty_200dma,
            "atr": n_atr, "adx": nifty_adx, "trend": n_trend,
            "macd": n_macd, "macd_signal": n_macd_sig, "macd_histogram": n_macd_hist,
            "bb_upper": n_bb_upper, "bb_middle": n_bb_mid, "bb_lower": n_bb_lower,
            "bb_width": n_bb_width,
        }

        price_structure = (
            "higher_highs"  if nifty_price > nifty_200dma * 1.02 else
            "lower_lows"    if nifty_price < nifty_200dma * 0.98 else
            "neutral"
        )

        # 5. Derivatives data — fetch option chain for all 5 indices
        # Primary index (NIFTY) feeds backwards-compatible derivatives_data.
        # All 5 chains go into index_option_chains for the OptionAdvisor.
        derivatives_data, index_option_chains = self._fetch_all_derivatives(
            nifty_price, india_vix, quotes
        )

        # 6. Real FII/DII flow, Nifty PE, G-sec yield from MacroFetcher
        try:
            macro_fetcher = _get_macro_fetcher()
            # Inject the live Fyers client so MacroFetcher can use it as a
            # G-sec source — the most reliable path when already authenticated.
            macro_fetcher.set_fyers_client(self._fyers)
            macro     = macro_fetcher.fetch()
            flow_data = macro["flow_data"]
            nifty_pe  = macro["nifty_pe"]
            gsec_yield = macro["gsec_yield"]
            # FIX 4.2: GIFT Nifty pre-market gap cues
            gift_nifty_price    = macro.get("gift_nifty_price", 0.0)
            gift_nifty_gap_pct  = macro.get("gift_nifty_gap_pct", 0.0)
            dow_futures_chg_pct = macro.get("dow_futures_chg_pct", 0.0)
            usd_inr             = macro.get("usd_inr", 0.0)
        except Exception as e:
            logger.warning(f"MacroFetcher failed ({e}), using defaults")
            flow_data  = self._fetch_flow_data()
            nifty_pe   = 22.5
            gsec_yield = 7.0
            gift_nifty_price    = 0.0
            gift_nifty_gap_pct  = 0.0
            dow_futures_chg_pct = 0.0
            usd_inr             = 0.0

        # FIX-MACRO-01 (USD/INR): if macro_fetcher returned 0, try Fyers FX quote.
        # Fyers provides currency futures as live quotes during market hours.
        # Use rolling contract tickers: try current month then next month.
        # Format: NSE:USDINR{YY}{MON}FUT where MON = APR, MAY, etc.
        if usd_inr == 0.0:
            try:
                from datetime import date as _date_cls
                _today     = _date_cls.today()
                _months    = ["JAN","FEB","MAR","APR","MAY","JUN",
                              "JUL","AUG","SEP","OCT","NOV","DEC"]
                _yr2       = str(_today.year)[-2:]   # "26"
                _cur_mon   = _months[_today.month - 1]
                _nxt_mon   = _months[_today.month % 12]  # wraps Dec→Jan
                _fx_syms   = ",".join([
                    f"NSE:USDINR{_yr2}{_cur_mon}FUT",
                    f"NSE:USDINR{_yr2}{_nxt_mon}FUT",
                    "NSE:USDINR",                      # spot rate (if available)
                ])
                _fx_resp = self._fyers.quotes({"symbols": _fx_syms})
                if _fx_resp.get("s") == "ok":
                    for _item in _fx_resp.get("d", []):
                        _ltp = float(_item.get("v", {}).get("lp", 0) or 0)
                        if 60 < _ltp < 120:
                            _sym = _item.get("n", "?")
                            # Futures price ≈ spot + forward points (~0.2-0.5 INR typically)
                            # We use it as-is since the forward premium is small and
                            # better than zero. Spot rate will differ by ~0.3-1.5 INR.
                            usd_inr = _ltp
                            logger.info(f"USD/INR (Fyers FX {_sym}): ₹{usd_inr:.2f} [futures price]")
                            break
            except Exception as _fx_e:
                logger.debug(f"Fyers FX USD/INR fetch skipped: {_fx_e}")

        logger.info(
            f"Live data fetched | Nifty={nifty_price:.0f} | "
            f"VIX={india_vix:.1f} | stocks={len(price_data)} | "
            f"FII 5d={flow_data.get('fii_cash_5day', 0):+,.0f}Cr | "
            f"PE={nifty_pe:.1f} | yield={gsec_yield:.2f}% | "
        )

        # ── Sector performance: 1-day % change averaged across sector constituents ──
        # Fed to LLMRegimeDetector (sector momentum signal) and LLMPatternValidator.
        # Also used to build approximate sector_flows for VESPER rotation analysis.
        from config import SECTOR_MAPPING as _SECTOR_MAP
        sector_performance: Dict[str, float] = {}
        sector_flows_list: list = []
        for _sector, _syms in _SECTOR_MAP.items():
            _changes = []
            for _sym in _syms:
                _pd = price_data.get(_sym, {})
                _close   = _pd.get("close", 0)
                _ref     = _pd.get("ref_close", 0) or _close
                if _close and _ref and _ref > 0:
                    _changes.append((_close - _ref) / _ref * 100)
            if _changes:
                _avg_chg = sum(_changes) / len(_changes)
                sector_performance[_sector] = round(_avg_chg, 2)
                # Approximate sector net_flow from price momentum:
                # +1% average = positive inflow signal (~₹500Cr proxy per sector)
                _net = round(_avg_chg * 500, 0)
                sector_flows_list.append({
                    "sector_name":  _sector,
                    "inflow_3day":  max(0, _net),
                    "outflow_3day": max(0, -_net),
                    "net_flow":     _net,
                })

        # ── VIX rolling history ──────────────────────────────────────────────────
        self._vix_history.append(india_vix)
        if len(self._vix_history) > 20:
            self._vix_history = self._vix_history[-20:]

        # IMPROVEMENT 3: Extract per-stock fundamentals from Fyers quote data.
        # Fyers returns pe_ratio, eps, book_value in the quote 'v' dict for equities.
        # This activates NEXUS stock-level analysis (previously always got empty dict).
        fundamental_data: Dict[str, Dict] = {}
        for sym in self.watchlist:
            q = quotes.get(sym, {})
            # Fyers quote fields: pe_ratio / pe, eps, book_value, 52w_high, 52w_low
            pe = q.get("pe_ratio", 0) or q.get("pe", 0)
            eps = q.get("eps", 0)
            book_value = q.get("book_value", 0)
            week_52_high = q.get("52w_high", 0)
            week_52_low  = q.get("52w_low",  0)
            if pe and pe > 0:
                fundamental_data[sym] = {
                    "pe_ratio":    round(float(pe), 2),
                    "eps":         round(float(eps), 2),
                    "book_value":  round(float(book_value), 2),
                    "52w_high":    round(float(week_52_high), 2),
                    "52w_low":     round(float(week_52_low),  2),
                    # Sector PE benchmark for NEXUS over/under-valuation scoring
                    "industry_pe": _sector_pe_estimate(sym),
                    # Approximate ROE from EPS / book_value (NEXUS can use this)
                    "roe": round((float(eps) / float(book_value) * 100), 2)
                              if book_value and float(book_value) > 0 else 0.0,
                }
        if fundamental_data:
            logger.debug(f"Fundamental data extracted for {len(fundamental_data)} stocks")

        return {
            "nifty_price":      nifty_price,
            "nifty_200dma":     nifty_200dma if nifty_200dma > 0 else nifty_price * 0.95,
            "nifty_indicators": nifty_indicators,          # ← real computed NIFTY indicators
            "nifty_intraday":   nifty_intraday_range,     # open/high/low/prev_close/change_pct
            "nifty_change_pct": nifty_change_pct,         # quick access for agents
            "nifty_15min":      nifty_15min_bars,         # IMPROVEMENT 7: real 15-min bars
            "india_vix":        india_vix,
            "adx":              nifty_adx,
            "price_structure":  price_structure,
            "nifty_pe":         nifty_pe,
            "gsec_yield":       gsec_yield,
            "price_data":       price_data,
            "indicators":       indicators,
            "flow_data":        flow_data,
            "sentiment_data":   {"news": 50, "analyst": 50, "social": 50, "global": 50},
            "derivatives_data": derivatives_data,
            "fundamental_data": fundamental_data,   # IMPROVEMENT 3: populated from quotes
            "event_data":       {"events": []},
            "ohlcv_history":    ohlcv_history,
            # Computed analytics — fed directly to LLM modules
            "sector_performance": sector_performance,   # {sector: avg_1d_chg%}
            "sector_flows":       sector_flows_list,    # VESPER rotation signal
            "vix_history":        list(self._vix_history),  # LLMRegimeDetector VIX trend
            # Startup-synced F&O positions (refreshed once per object lifetime)
            "fno_positions":    self.fno_positions,
            # Per-index option chain data for DirectionalOptionAdvisor
            "index_option_chains": index_option_chains,
            # Individual index spot prices (DirectionalOptionAdvisor reads these)
            "banknifty_price": quotes.get("NIFTYBANK", {}).get("ltp", 0.0),
            "sensex_price":    quotes.get("SENSEX",    {}).get("ltp", 0.0),
            "finnifty_price":  quotes.get("FINNIFTY",  {}).get("ltp", 0.0),
            "bankex_price":    quotes.get("BANKEX",    {}).get("ltp", 0.0),
            # FIX 4.2: GIFT Nifty / global pre-market cues
            "gift_nifty_price":    gift_nifty_price,
            "gift_nifty_gap_pct":  gift_nifty_gap_pct,
            "dow_futures_chg_pct": dow_futures_chg_pct,
            "usd_inr":             usd_inr,
        }

    # ── Fyers API calls ───────────────────────────────────────────────────────

    def _fetch_quotes(self) -> Dict[str, Dict]:
        """
        Fetch LTP + OHLC for all watchlist stocks plus Nifty/VIX indices.
        Returns dict keyed by bare symbol.
        """
        # Build symbol list: stocks + indices
        stock_symbols = [_eq_safe(s) for s in self.watchlist]
        index_symbols = [INDEX_NIFTY, INDEX_VIX, INDEX_BANKNIFTY,
                              INDEX_SENSEX, INDEX_FINNIFTY, INDEX_BANKEX]
        all_symbols   = stock_symbols + index_symbols

        # Fyers quotes API accepts comma-separated symbols (max 50 per call)
        result = {}
        batch_size = 50
        for i in range(0, len(all_symbols), batch_size):
            batch = all_symbols[i:i + batch_size]
            try:
                resp = self._fyers.quotes({"symbols": ",".join(batch)})
                if resp.get("s") != "ok":
                    logger.warning(f"Quotes API error: {resp.get('message', resp)}")
                    continue
                for item in resp.get("d", []):
                    v   = item.get("v", {})
                    sym = item.get("n", "")
                    # strip exchange prefix and -EQ / -INDEX suffix
                    bare = sym.replace("NSE:", "").replace("BSE:", "")
                    bare = bare.replace("-EQ", "").replace("-INDEX", "")
                    result[bare] = {
                        "ltp":         v.get("lp",  v.get("close_price", 0)),
                        "open_price":  v.get("open_price",  0),
                        "high_price":  v.get("high_price",  0),
                        "low_price":   v.get("low_price",   0),
                        "volume":      v.get("volume", 0),
                        "prev_close":  v.get("prev_close_price", 0),
                        "change_pct":  v.get("ch", 0),
                    }
            except Exception as e:
                logger.error(f"Quote fetch error (batch {i//batch_size}): {e}")
            time.sleep(0.3)      # rate limit

        # Remap Fyers bare names → internal alias names for post-demerger symbols.
        # e.g. result["TMCV"] → result["TATAMOTORS_CV"] so the rest of the
        # pipeline (which uses our internal watchlist name) finds the right key.
        _FYERS_TO_ALIAS = {v.replace("NSE:", "").replace("BSE:", "").replace("-EQ", ""): k
                           for k, v in _SYMBOL_OVERRIDES.items()}
        for fyers_bare, alias in _FYERS_TO_ALIAS.items():
            if fyers_bare in result and alias not in result:
                result[alias] = result[fyers_bare]

        return result

    def _fetch_history_all(self) -> Dict[str, List[Dict]]:
        """
        Fetch daily OHLCV history for all stocks + Nifty index.
        Called once per calendar day; result is cached in memory.
        Returns {symbol: [{date,open,high,low,close,volume}, ...]}
        """
        end_date   = date.today()
        start_date = end_date - timedelta(days=self.history_days + 30)  # buffer for weekends/holidays

        ohlcv = {}
        symbols_to_fetch = self.watchlist + ["NIFTY50"]
        total = len(symbols_to_fetch)
        for i, sym in enumerate(symbols_to_fetch, 1):
            fyers_sym = INDEX_NIFTY if sym == "NIFTY50" else _eq_safe(sym)
            candles   = self._fetch_candles(fyers_sym, start_date, end_date)
            if candles:
                ohlcv[sym] = candles
            if i % 10 == 0 or i == total:
                logger.info(f"  History progress: {i}/{total} symbols fetched")
            time.sleep(0.4)   # increased from 0.2 to avoid 429 rate-limit errors

        # Report any symbols that returned no candles so they are easy to diagnose
        missing = [s for s in symbols_to_fetch if s not in ohlcv]
        if missing:
            logger.warning(
                f"History missing for {len(missing)} symbol(s) "
                f"(bad symbol / delisted / API error): {missing}"
            )
        return ohlcv

    def _fetch_intraday_history(self, fyers_symbol: str, days: int = 5) -> List[Dict]:
        """
        IMPROVEMENT 7: Fetch 15-minute OHLCV bars for the last N trading days.

        Used to compute short-term RSI/EMA on intraday data, enabling real
        multi-timeframe confluence analysis in ORION (daily + 15-min alignment).

        Returns list of {date, open, high, low, close, volume} dicts with
        'resolution' key set to '15m' so callers can differentiate from daily bars.
        """
        try:
            end_ts   = int(datetime.now().timestamp())
            start_ts = int((datetime.now() - timedelta(days=days + 2)).timestamp())  # +2 for weekends
            payload  = {
                "symbol":      fyers_symbol,
                "resolution":  "15",
                "date_format": "0",        # epoch timestamps
                "range_from":  str(start_ts),
                "range_to":    str(end_ts),
                "cont_flag":   "1",
            }
            resp = self._fyers.history(payload)
            if resp.get("s") != "ok":
                return []
            candles = []
            for bar in resp.get("candles", []):
                ts, o, h, l, c, v = bar
                candles.append({
                    "date":       datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M"),
                    "open":  o, "high": h, "low": l, "close": c, "volume": int(v),
                    "resolution": "15m",
                })
            return candles
        except Exception as e:
            logger.debug(f"Intraday history fetch failed for {fyers_symbol}: {e}")
            return []

    def _fetch_candles(self, fyers_symbol: str,
                       start: date, end: date) -> List[Dict]:
        """
        Fetch daily OHLCV candles for one symbol.

        Attempt 1: cont_flag=1  — adjusted continuous data (preferred).
        Attempt 2: cont_flag=0  — raw unadjusted data.

        Some equity symbols with recent corporate actions (bonus issue, split,
        merger) fail with cont_flag=1 on the Fyers v3 API even though the
        symbol is perfectly valid (e.g. TATAMOTORS after a demerger event).
        The retry without cont_flag almost always succeeds in those cases.
        """
        base_payload = {
            "symbol":      fyers_symbol,
            "resolution":  "D",
            "date_format": "1",              # epoch timestamps
            "range_from":  start.strftime("%Y-%m-%d"),
            "range_to":    end.strftime("%Y-%m-%d"),
        }

        for attempt, cont_flag in enumerate(["1", "0"], start=1):
            try:
                payload = {**base_payload, "cont_flag": cont_flag}
                resp    = self._fyers.history(payload)

                # Handle rate limiting — back off and retry once
                if isinstance(resp, dict) and resp.get("code") == 429:
                    logger.debug(f"  {fyers_symbol}: rate-limited (429), backing off 3s")
                    time.sleep(3)
                    resp = self._fyers.history(payload)

                if resp.get("s") == "ok":
                    candles = []
                    for bar in resp.get("candles", []):
                        ts, o, h, l, c, v = bar
                        candles.append({
                            "date":   datetime.fromtimestamp(ts).strftime("%Y-%m-%d"),
                            "open":   float(o), "high": float(h),
                            "low":    float(l), "close": float(c),
                            "volume": int(v),
                        })
                    if candles:
                        if attempt == 2:
                            logger.debug(
                                f"  {fyers_symbol}: fetched with cont_flag=0 "
                                f"(cont_flag=1 failed — likely recent corp action)"
                            )
                        return candles
                    # ok response but empty candles — try next flag
                    logger.debug(
                        f"  {fyers_symbol}: cont_flag={cont_flag} returned 0 candles"
                    )

                else:
                    msg = resp.get("message", "unknown error")
                    logger.debug(
                        f"  {fyers_symbol}: cont_flag={cont_flag} error — {msg}"
                    )

            except Exception as exc:
                logger.debug(f"  {fyers_symbol}: cont_flag={cont_flag} exception — {exc}")

        logger.warning(f"Candle fetch failed for {fyers_symbol} with both cont_flag values")
        # BSE fallback: some symbols fail on NSE (demerger, corp actions, etc.).
        # For NSE:TMCV-EQ / NSE:TMPV-EQ (TATAMOTORS post-demerger Oct 2025),
        # BSE equivalents are BSE:TMCV-EQ and BSE:TMPV-EQ respectively.
        # For all other NSE:XXX-EQ failures try generic BSE suffix formats.
        if fyers_symbol.startswith("NSE:") and fyers_symbol.endswith("-EQ"):
            bare = fyers_symbol.replace("NSE:", "").replace("-EQ", "")
            # Specific known BSE mappings for post-demerger Tata Motors entities
            _KNOWN_BSE = {
                "TMCV": ["BSE:TMCV-EQ"],
                "TMPV": ["BSE:TMPV-EQ"],
                "LTIM": ["BSE:LTIM-EQ", "BSE:LTIM-A", "BSE:LTIM-BE"],   # LTIMindtree BSE fallback
            }
            bse_candidates = _KNOWN_BSE.get(bare, [f"BSE:{bare}-A", f"BSE:{bare}-EQ", f"BSE:{bare}-BE"])
            for bse_sym in bse_candidates:
                logger.info(f"  {bare}: NSE history failed, trying {bse_sym}")
                for cont_flag in ["1", "0"]:
                    try:
                        payload = {**base_payload, "symbol": bse_sym, "cont_flag": cont_flag}
                        resp = self._fyers.history(payload)
                        if resp.get("s") == "ok":
                            candles = []
                            for bar in resp.get("candles", []):
                                ts, o, h, l, c, v = bar
                                candles.append({
                                    "date":   datetime.fromtimestamp(ts).strftime("%Y-%m-%d"),
                                    "open":   float(o), "high": float(h),
                                    "low":    float(l), "close": float(c),
                                    "volume": int(v),
                                })
                            if candles:
                                logger.info(
                                    f"  {bare}: {len(candles)} bars via {bse_sym}"
                                )
                                return candles
                    except Exception as exc:
                        logger.debug(f"  {bse_sym} failed: {exc}")
        return []

    def _fetch_all_derivatives(
        self,
        nifty_price: float,
        india_vix:   float,
        quotes:      Dict[str, Dict],
    ):
        """
        Fetch option chains for all 5 liquid index option markets.

        Returns
        -------
        derivatives_data : dict  — backwards-compatible NIFTY-only data (for existing agents)
        index_option_chains : dict  — per-index chain data keyed by index name
                                      consumed by DirectionalOptionAdvisor
        """
        index_option_chains: Dict[str, Dict] = {}

        # Strike intervals per index (matches IndexSpec in directional_option_advisor)
        _INTERVALS = {
            "NIFTY":     50,
            "BANKNIFTY": 100,
            "SENSEX":    100,
            "FINNIFTY":  50,
            "BANKEX":    100,
        }

        # Get spot for each index from the already-fetched quotes dict
        _BARE_MAP = {
            "NIFTY":     "NIFTY50",
            "BANKNIFTY": "NIFTYBANK",
            "SENSEX":    "SENSEX",
            "FINNIFTY":  "FINNIFTY",
            "BANKEX":    "BANKEX",
        }
        _FYERS_SYMBOLS = {
            "NIFTY":     INDEX_NIFTY,
            "BANKNIFTY": INDEX_BANKNIFTY,
            "SENSEX":    INDEX_SENSEX,
            "FINNIFTY":  INDEX_FINNIFTY,
            "BANKEX":    INDEX_BANKEX,
        }

        # Global IV rank from VIX (used as fallback for indices with no chain)
        global_iv_rank = min(100, max(0, int((india_vix - 10) / (30 - 10) * 100)))

        for idx_name, fyers_sym in _FYERS_SYMBOLS.items():
            bare = _BARE_MAP[idx_name]
            q    = quotes.get(bare, {})
            spot = float(q.get("ltp", 0))
            if spot <= 0:
                # Store a defaults-only entry so OptionAdvisor still runs
                index_option_chains[idx_name] = {
                    "spot": 0, "pcr": 1.0, "pcr_trend": "stable",
                    "max_pain": 0, "iv_rank": global_iv_rank,
                    "call_oi_walls": [], "put_oi_walls": [],
                }
                continue

            interval = _INTERVALS.get(idx_name, 50)
            atm      = round(spot / interval) * interval

            try:
                resp = self._fyers.optionchain({
                    "symbol":      fyers_sym,
                    "strikecount": 10,
                    "timestamp":   "",
                })
                if resp.get("s") != "ok":
                    raise ValueError(resp.get("message", "option chain error"))

                # ── Fyers API v3 option chain structure (confirmed from live logs) ──
                # Response: data = {
                #   "callOi":      <int>   — aggregate call OI across all expiries
                #   "putOi":       <int>   — aggregate put OI across all expiries
                #   "optionsChain": []     — EMPTY at top level (ignore)
                #   "expiryData":  [...]   — 3-18 per-expiry dicts
                #     Each entry: { "expiryDate": "...",
                #                   "optionsChain": [
                #                     { "strikePrice": 25600,
                #                       "CE": {"oi": 123456, "ltp": 45.5, ...},
                #                       "PE": {"oi": 89012,  "ltp": 23.1, ...} }
                #                   ]}
                # }
                data_block  = resp.get("data", {})
                expiry_data = data_block.get("expiryData", [])

                # ── Aggregate OI for PCR (fast, reliable from top-level fields) ─
                agg_call_oi = int(data_block.get("callOi", 0) or
                                  data_block.get("call_oi", 0) or 0)
                agg_put_oi  = int(data_block.get("putOi",  0) or
                                  data_block.get("put_oi",  0) or 0)

                # ── Per-strike chain from nearest expiry with data ────────────
                chain: list = []
                for expiry_entry in expiry_data:
                    candidate = expiry_entry.get("optionsChain", [])
                    if candidate:
                        chain = candidate
                        logger.debug(
                            f"  {idx_name}: expiry "
                            f"'{expiry_entry.get('expiryDate', '?')}' "
                            f"({len(chain)} strikes)"
                        )
                        break

                if not chain and not agg_call_oi and not agg_put_oi:
                    logger.warning(
                        f"  {idx_name}: no OI data at all — "
                        f"data keys: {list(data_block.keys())}, "
                        f"expiryData count: {len(expiry_data)}"
                    )

                # ── Build per-strike maps ─────────────────────────────────────
                total_call_oi: int = agg_call_oi   # start from aggregate
                total_put_oi:  int = agg_put_oi
                call_oi_by_strike: Dict[float, int] = {}
                put_oi_by_strike:  Dict[float, int] = {}
                max_pain_map:      Dict[float, float] = {}
                # FIX 1: Real option LTPs per strike (replaces BS estimate)
                ltp_by_strike:     Dict[float, Dict[str, float]] = {}

                for row in chain:
                    # v3 uses nested CE/PE sub-dicts; v2 used flat fields
                    strike = float(
                        row.get("strikePrice") or
                        row.get("strike_price") or 0
                    )
                    if strike == 0:
                        continue
                    ce_data = row.get("CE") or row.get("ce") or {}
                    pe_data = row.get("PE") or row.get("pe") or {}
                    c_oi = int(
                        ce_data.get("oi") or ce_data.get("openInterest") or
                        row.get("call_oi") or row.get("callOI") or
                        row.get("ce_oi") or 0
                    )
                    p_oi = int(
                        pe_data.get("oi") or pe_data.get("openInterest") or
                        row.get("put_oi") or row.get("putOI") or
                        row.get("pe_oi") or 0
                    )
                    # FIX 1: Extract real LTPs from Fyers expiryData rows
                    ce_ltp = float(
                        ce_data.get("ltp") or ce_data.get("lastPrice") or
                        ce_data.get("last_price") or 0.0
                    )
                    pe_ltp = float(
                        pe_data.get("ltp") or pe_data.get("lastPrice") or
                        pe_data.get("last_price") or 0.0
                    )
                    if ce_ltp > 0 or pe_ltp > 0:
                        ltp_by_strike[strike] = {"ce": ce_ltp, "pe": pe_ltp}

                    call_oi_by_strike[strike] = c_oi
                    put_oi_by_strike[strike]  = p_oi
                    # Only accumulate strike OI if aggregate was unavailable
                    if not agg_call_oi:
                        total_call_oi += c_oi
                    if not agg_put_oi:
                        total_put_oi  += p_oi
                    call_loss = sum(max(0, s - strike) * oi
                                    for s, oi in call_oi_by_strike.items())
                    put_loss  = sum(max(0, strike - s) * oi
                                    for s, oi in put_oi_by_strike.items())
                    max_pain_map[strike] = call_loss + put_loss

                # FIX 1: Derive ATM CE/PE LTPs for the strategy builders
                atm_ce_ltp = 0.0
                atm_pe_ltp = 0.0
                if ltp_by_strike:
                    # Find the strike closest to ATM
                    closest_strike = min(ltp_by_strike.keys(), key=lambda s: abs(s - atm))
                    atm_ce_ltp = ltp_by_strike[closest_strike].get("ce", 0.0)
                    atm_pe_ltp = ltp_by_strike[closest_strike].get("pe", 0.0)
                    logger.debug(
                        f"  {idx_name}: ATM={closest_strike:.0f} "
                        f"CE_LTP={atm_ce_ltp:.1f} PE_LTP={atm_pe_ltp:.1f} "
                        f"(real market prices)"
                    )

                # ── PCR ───────────────────────────────────────────────────────
                if total_call_oi == 0 and total_put_oi == 0:
                    pcr = 1.0
                    logger.warning(f"  {idx_name}: OI=0 → PCR neutral 1.0 (pre-open)")
                else:
                    pcr = round(total_put_oi / max(total_call_oi, 1), 3)
                    logger.debug(
                        f"  {idx_name}: PCR={pcr:.3f} "
                        f"(call={total_call_oi:,} put={total_put_oi:,})"
                    )
                call_walls = sorted(
                    [{"strike": s, "oi": o}
                     for s, o in call_oi_by_strike.items() if s >= atm],
                    key=lambda x: x["oi"], reverse=True,
                )[:2]
                put_walls = sorted(
                    [{"strike": s, "oi": o}
                     for s, o in put_oi_by_strike.items() if s <= atm],
                    key=lambda x: x["oi"], reverse=True,
                )[:2]
                max_pain_strike = (min(max_pain_map, key=max_pain_map.get)
                                   if max_pain_map else atm)
                pcr_trend  = "rising" if pcr > 1.2 else ("falling" if pcr < 0.8 else "stable")
                # Per-index IV rank: use VIX proxy (no per-index historical vol series)
                iv_rank    = global_iv_rank

                # IMPROVEMENT 4: Real OI change delta (long buildup vs short covering)
                # Compare current session OI to previous cached value.
                # On first run, cache is empty so delta = 0 (safe fallback).
                prev_ce = self._prev_ce_oi_cache.get(idx_name, total_call_oi)
                prev_pe = self._prev_pe_oi_cache.get(idx_name, total_put_oi)
                ce_oi_delta_pct = round((total_call_oi - prev_ce) / max(prev_ce, 1) * 100, 2)
                pe_oi_delta_pct = round((total_put_oi  - prev_pe) / max(prev_pe, 1) * 100, 2)
                # Update cache for next cycle
                self._prev_ce_oi_cache[idx_name] = total_call_oi
                self._prev_pe_oi_cache[idx_name] = total_put_oi

                index_option_chains[idx_name] = {
                    "spot":          spot,
                    "pcr":           pcr,
                    "pcr_trend":     pcr_trend,
                    "max_pain":      max_pain_strike,
                    "iv_rank":       iv_rank,
                    "call_oi_walls": call_walls,
                    "put_oi_walls":  put_walls,
                    "total_call_oi": total_call_oi,
                    "total_put_oi":  total_put_oi,
                    # Real OI change deltas for SENTINEL
                    "ce_oi_delta_pct": ce_oi_delta_pct,
                    "pe_oi_delta_pct": pe_oi_delta_pct,
                    # FIX 1: Real ATM option LTPs from Fyers chain API
                    "atm_ce_ltp":    atm_ce_ltp,
                    "atm_pe_ltp":    atm_pe_ltp,
                    "ltp_by_strike": ltp_by_strike,
                }
                logger.debug(
                    f"  Chain {idx_name}: spot={spot:.0f} PCR={pcr:.2f} "
                    f"max_pain={max_pain_strike:.0f} iv_rank={iv_rank}"
                )

            except Exception as exc:
                logger.warning(f"Option chain failed for {idx_name}: {exc}")
                index_option_chains[idx_name] = {
                    "spot": spot, "pcr": 1.0, "pcr_trend": "stable",
                    "max_pain": atm, "iv_rank": global_iv_rank,
                    "call_oi_walls": [{"strike": atm + interval, "oi": 0},
                                      {"strike": atm + interval * 2, "oi": 0}],
                    "put_oi_walls":  [{"strike": atm - interval, "oi": 0},
                                      {"strike": atm - interval * 2, "oi": 0}],
                }
            time.sleep(0.5)   # rate limit between option chain calls (was 0.2)

        # Backwards-compatible derivatives_data (NIFTY only, existing agents)
        nifty_chain = index_option_chains.get("NIFTY", {})
        tc_oi = nifty_chain.get("total_call_oi", 0)
        tp_oi = nifty_chain.get("total_put_oi", 0)
        nifty_pcr = nifty_chain.get("pcr", 1.0)

        # Derive oi_signal from PCR + OI wall proximity
        if nifty_pcr < 0.65:
            oi_signal = "SHORT_BUILDUP"     # heavy call writing, bears in control
        elif nifty_pcr > 1.35:
            oi_signal = "LONG_BUILDUP"      # heavy put writing, bulls supported
        elif nifty_pcr < 0.80:
            oi_signal = "SHORT_BUILDUP"
        elif nifty_pcr > 1.20:
            oi_signal = "LONG_BUILDUP"
        else:
            oi_signal = "NEUTRAL"

        # CE/PE OI imbalance as proxy for change direction
        # Positive = more call OI than balanced → bearish pressure
        total_oi = tc_oi + tp_oi or 1
        ce_oi_change_pct = round((tc_oi / total_oi - 0.5) * 100, 1)   # >0 = call-heavy
        pe_oi_change_pct = round((tp_oi / total_oi - 0.5) * 100, 1)   # >0 = put-heavy

        # IMPROVEMENT 4: Prefer real OI delta from cache over static imbalance proxy.
        # ce_oi_delta_pct = % change in call OI vs previous cycle (positive = buildup)
        # This allows SENTINEL to correctly classify LONG_BUILDUP vs SHORT_COVERING.
        nifty_ce_delta = nifty_chain.get("ce_oi_delta_pct", ce_oi_change_pct)
        nifty_pe_delta = nifty_chain.get("pe_oi_delta_pct", pe_oi_change_pct)

        # IV skew proxy: higher PCR → elevated put IV relative to call IV
        iv_skew = round((1.0 - nifty_pcr) * 5.0, 2)   # positive = call skew (bearish)

        derivatives_data = {
            "pcr":               nifty_pcr,
            "pcr_trend":         nifty_chain.get("pcr_trend", "stable"),
            "max_pain":          nifty_chain.get("max_pain", round(nifty_price / 50) * 50),
            "india_vix":         india_vix,
            "iv_rank":           nifty_chain.get("iv_rank", global_iv_rank),
            "call_oi_walls":     nifty_chain.get("call_oi_walls", []),
            "put_oi_walls":      nifty_chain.get("put_oi_walls",  []),
            "total_call_oi":     tc_oi,
            "total_put_oi":      tp_oi,
            "oi_signal":         oi_signal,
            "ce_oi_change_pct":  nifty_ce_delta,   # IMPROVEMENT 4: real cycle delta
            "pe_oi_change_pct":  nifty_pe_delta,   # IMPROVEMENT 4: real cycle delta
            "iv_skew":           iv_skew,
            "futures_premium":   0.0,   # placeholder — no futures quote fetched
            "price_change":      0.0,   # filled by nifty_change_pct below
        }

        # FIX 2: Real futures basis — fetch nearest NIFTY futures LTP
        # Symbol: NSE:NIFTY{YY}{MON}FUT  e.g. NSE:NIFTY26MARFUT
        # Roll over after the 25th of the month to the next expiry.
        try:
            import calendar as _cal
            _now = datetime.now()
            _yr2 = str(_now.year)[2:]
            _mon = _now.strftime("%b").upper()
            if _now.day >= 25:  # roll to next month near expiry
                _next = _now.replace(day=1)
                _next = _next.replace(month=(_now.month % 12) + 1) if _now.month < 12 \
                        else _next.replace(year=_now.year + 1, month=1)
                _yr2  = str(_next.year)[2:]
                _mon  = _next.strftime("%b").upper()
            _fut_sym = f"NSE:NIFTY{_yr2}{_mon}FUT"
            _fut_resp = self._fyers.quotes({"symbols": _fut_sym})
            if _fut_resp and _fut_resp.get("code") == 200:
                _fd = (_fut_resp.get("d") or [{}])[0].get("v", {})
                _fut_ltp = float(_fd.get("lp") or _fd.get("close_price") or 0.0)
                if _fut_ltp > 0 and nifty_price > 0:
                    _futures_premium = round(_fut_ltp - nifty_price, 2)
                    derivatives_data["futures_premium"] = _futures_premium
                    logger.debug(
                        f"  NIFTY futures ({_fut_sym}): LTP={_fut_ltp:.2f} "
                        f"spot={nifty_price:.2f} basis={_futures_premium:+.2f}"
                    )
        except Exception as _fe:
            logger.debug(f"Futures basis fetch skipped: {_fe}")

        # FIX-PCR-01: Show PCR=N/A for indices where option chain data is
        # unavailable (e.g. BANKEX on BSE segment where Fyers may not support
        # full option chain fetch). PCR=0.00 is misleading — it implies
        # zero put OI, but really means "no data available".
        _pcr_parts = []
        for k, v in index_option_chains.items():
            if v.get("spot", 0) > 0:
                _pcr = v.get("pcr", 0)
                if _pcr > 0:
                    _pcr_parts.append(f"{k} PCR={_pcr:.2f}")
                else:
                    _pcr_parts.append(f"{k} PCR=N/A")
        logger.info(
            f"Option chains fetched: " + " | ".join(_pcr_parts)
        )
        return derivatives_data, index_option_chains

    def sync_fno_positions(self) -> List[Dict]:
        """
        Fetch all open F&O (and equity) positions from the Fyers broker.

        Called ONCE at engine startup so the coordinator's risk layer
        starts with a complete picture of the current open book instead
        of assuming a flat book.

        Returns
        -------
        list of dicts, each with keys:
            symbol        — Fyers symbol string  e.g. "NSE:NIFTY25JANFUT"
            bare_symbol   — exchange-stripped name e.g. "NIFTY25JANFUT"
            underlying    — best-guess underlying  e.g. "NIFTY"
            qty           — net quantity (negative = short)
            side          — "BUY" | "SELL"
            avg_price     — average fill price
            ltp           — last traded price
            pnl           — unrealised P&L (₹)
            product_type  — "MARGIN" | "INTRADAY" | "CO" | …
            exchange      — "NSE" | "BSE" | "MCX"
        """
        try:
            resp = self._fyers.positions()
            if resp.get("s") != "ok":
                logger.warning(
                    f"Fyers positions() error: {resp.get('message', resp)}"
                )
                return []

            raw_positions = resp.get("netPositions", [])
            positions: List[Dict] = []

            for pos in raw_positions:
                qty = int(pos.get("netQty", 0))
                if qty == 0:
                    continue        # fully squared off — skip

                sym  = pos.get("symbol", "")
                bare = sym.replace("NSE:", "").replace("BSE:", "").replace("MCX:", "")

                # Best-guess underlying: strip year/month/expiry suffix
                # e.g. "NIFTY25JANFUT" → "NIFTY"
                import re as _re
                underlying = _re.split(r"\d", bare)[0].upper() if bare else bare

                positions.append({
                    "symbol":       sym,
                    "bare_symbol":  bare,
                    "underlying":   underlying,
                    "qty":          qty,
                    "side":         "BUY" if qty > 0 else "SELL",
                    "avg_price":    float(pos.get("netAvg",       0)),
                    "ltp":          float(pos.get("ltp",          0)),
                    "pnl":          float(pos.get("pl",           0)),
                    "product_type": pos.get("productType",       ""),
                    "exchange":     pos.get("exchange",          "NSE"),
                })

            total_pnl = sum(p["pnl"] for p in positions)
            logger.info(
                f"F&O startup sync: {len(positions)} open position(s) | "
                f"unrealised P&L ₹{total_pnl:+,.0f}"
            )
            if positions:
                for p in positions:
                    logger.info(
                        f"  {p['side']:4s} {p['bare_symbol']:30s} "
                        f"qty={p['qty']:+5d}  avg=₹{p['avg_price']:,.2f}  "
                        f"pnl=₹{p['pnl']:+,.0f}"
                    )
            return positions

        except Exception as exc:
            logger.warning(f"F&O startup position sync failed: {exc}")
            return []

    def _fetch_flow_data(self) -> Dict[str, float]:
        """
        FII/DII flow data. Fyers doesn't expose a direct FII/DII endpoint,
        so we return zero-defaults; the coordinator handles missing flow data
        gracefully (VESPER uses it but won't crash without it).

        If you have an NSE or third-party flow API, plug it in here.
        """
        return {
            "fii_cash_5day": 0.0,
            "dii_cash_5day": 0.0,
        }
