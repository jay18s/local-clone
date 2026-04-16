"""
ROX Proven Edge Engine v4.0 — Macro Data Fetcher
=================================================
Fetches real FII/DII institutional flow data, Nifty P/E ratio,
and India 10-year G-sec yield to feed VESPER and NEXUS agents.

Data sources (all free, no API key required):
    FII/DII flows  — NSE India public JSON API
    Nifty P/E      — yfinance (^NSEI trailingPE) with NSE fallback
    G-sec yield    — yfinance (^IRX proxy) + hardcoded fallback

Cache strategy (same as fyers_fetcher):
    Macro data changes at end-of-day; cached once per calendar day.
    Each 60s live cycle reuses the in-memory cache.

Usage:
    from data.macro_fetcher import MacroFetcher
    fetcher = MacroFetcher()                # create once
    macro   = fetcher.fetch()              # call each cycle (cached intraday)

    # Returns dict with keys:
    #   flow_data   : {fii_cash_5day, dii_cash_5day, fii_cash_daily,
    #                  dii_cash_daily, fii_cash_3day, dii_cash_3day,
    #                  flow_momentum, fii_derivative_daily}
    #   nifty_pe    : float
    #   gsec_yield  : float   (10Y, percent)
"""

import logging
import time
from datetime import date, datetime, timedelta
from typing import Dict, Any, List, Optional

logger = logging.getLogger("MacroFetcher")

# ── NSE public endpoints (no auth, but require browser-like headers) ─────────
_NSE_BASE    = "https://www.nseindia.com"
_NSE_FII_DII = f"{_NSE_BASE}/api/fiidiiTradeReact"
_NSE_INDICES = f"{_NSE_BASE}/api/allIndices"

_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36",
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.nseindia.com/",
}


class MacroFetcher:
    """
    Fetches macro market data (FII/DII flows, Nifty PE, G-sec yield).
    Designed to be instantiated once and reused across 60s cycles.
    All data is cached for the calendar day — only one real fetch per day.
    """

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self._session = self._make_session()

        # Daily cache
        self._cache: Optional[Dict[str, Any]] = None
        self._cache_date: Optional[date]      = None

        # Optional Fyers client injected by FyersFetcher at runtime
        self._fyers_client = None

    def set_fyers_client(self, client) -> None:
        """
        Inject a live Fyers API client so MacroFetcher can use it as a
        primary G-sec source.  Called by FyersFetcher.fetch_market_data()
        after the client is initialised and verified.

        Setting this does NOT invalidate the daily cache — the client is
        only used on the next cache-miss (once per calendar day).
        """
        if self._fyers_client is None and client is not None:
            self._fyers_client = client
            logger.debug("MacroFetcher: Fyers client registered for G-sec fetch")

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch(self) -> Dict[str, Any]:
        """
        Return macro data dict. Uses in-memory cache during the day.
        Forces a fresh fetch once per calendar day.
        """
        today = date.today()
        if self._cache_date == today and self._cache is not None:
            logger.debug("MacroFetcher: using daily cache")
            return self._cache

        logger.info("MacroFetcher: fetching fresh macro data...")
        data = self._fetch_all()
        self._cache      = data
        self._cache_date = today
        logger.info(\
            f"MacroFetcher: FII 5d={data['flow_data']['fii_cash_5day']:+,.0f} Cr | "\
            f"DII 5d={data['flow_data']['dii_cash_5day']:+,.0f} Cr | "\
            f"PE={data['nifty_pe']:.1f} | yield={data['gsec_yield']:.2f}% | "\
            f"USD/INR=₹{data.get('usd_inr', 0):.2f}"\
        )
        return data

    # ── Internal orchestration ────────────────────────────────────────────────

    def _fetch_all(self) -> Dict[str, Any]:
        flow_data  = self._fetch_fii_dii()
        nifty_pe   = self._fetch_nifty_pe()
        gsec_yield = self._fetch_gsec_yield()
        # FIX 4.2: GIFT Nifty / Global pre-market cues
        gift_data  = self._fetch_gift_nifty()
        return {
            "flow_data":           flow_data,
            "nifty_pe":            nifty_pe,
            "gsec_yield":          gsec_yield,
            "gift_nifty_price":    gift_data.get("gift_nifty_price", 0.0),
            "gift_nifty_gap_pct":  gift_data.get("gift_nifty_gap_pct", 0.0),
            "dow_futures_chg_pct": gift_data.get("dow_futures_chg_pct", 0.0),
            "usd_inr":             gift_data.get("usd_inr", 0.0),
        }

    # ── FII / DII flows ───────────────────────────────────────────────────────

    def _fetch_fii_dii(self) -> Dict[str, float]:
        """
        Fetch last 5 days of FII/DII cash segment data from NSE.
        NSE returns rows like:
            {date, buyValue, sellValue, netValue, category}
        where category is "FII/FPI" or "DII".
        """
        try:
            resp = self._session.get(_NSE_FII_DII, timeout=self.timeout)
            resp.raise_for_status()
            rows = resp.json()             # list of dicts

            fii_daily: List[float] = []
            dii_daily: List[float] = []

            for row in rows:
                cat = row.get("category", "").upper()
                net = self._parse_cr(row.get("netValue", 0))
                if "FII" in cat or "FPI" in cat:
                    fii_daily.append(net)
                elif "DII" in cat:
                    dii_daily.append(net)

            # NSE returns ~10 rows (5 FII + 5 DII), most-recent first
            fii_daily = fii_daily[:5]
            dii_daily = dii_daily[:5]

            return self._build_flow_dict(fii_daily, dii_daily)

        except Exception as e:
            logger.warning(f"FII/DII fetch failed ({e}), using zeros")
            return self._zero_flow()

    def _build_flow_dict(self, fii_daily: List[float],
                         dii_daily: List[float]) -> Dict[str, float]:
        """Aggregate daily FII/DII series into the fields VESPER expects."""
        def safe_sum(lst, n):
            return round(sum(lst[:n]), 2) if lst else 0.0

        fii_1 = fii_daily[0] if fii_daily else 0.0
        dii_1 = dii_daily[0] if dii_daily else 0.0
        fii_3 = safe_sum(fii_daily, 3)
        dii_3 = safe_sum(dii_daily, 3)
        fii_5 = safe_sum(fii_daily, 5)
        dii_5 = safe_sum(dii_daily, 5)

        # Momentum: change from day-3 sum to day-1 sum (annualised direction)
        fii_3_prev = safe_sum(fii_daily[1:], 3) if len(fii_daily) > 1 else fii_3
        momentum   = round(fii_3 - fii_3_prev, 2)

        return {
            "fii_cash_daily":      fii_1,
            "dii_cash_daily":      dii_1,
            "fii_cash_3day":       fii_3,
            "dii_cash_3day":       dii_3,
            "fii_cash_5day":       fii_5,
            "dii_cash_5day":       dii_5,
            "flow_momentum":       momentum,
            "fii_derivative_daily": 0.0,   # NSE API doesn't expose F&O flows
        }

    @staticmethod
    def _zero_flow() -> Dict[str, float]:
        return {
            "fii_cash_daily": 0.0, "dii_cash_daily": 0.0,
            "fii_cash_3day":  0.0, "dii_cash_3day":  0.0,
            "fii_cash_5day":  0.0, "dii_cash_5day":  0.0,
            "flow_momentum":  0.0, "fii_derivative_daily": 0.0,
        }

    @staticmethod
    def _parse_cr(value) -> float:
        """Parse NSE crore values that may come as strings with commas."""
        try:
            return float(str(value).replace(",", ""))
        except (ValueError, TypeError):
            return 0.0

    # ── Nifty P/E ─────────────────────────────────────────────────────────────

    def _fetch_nifty_pe(self) -> float:
        """
        Fetch current Nifty 50 trailing P/E.
        Primary  : yfinance  ^NSEI trailingPE
        Fallback : NSE allIndices endpoint (pe field)
        Default  : 22.5
        """
        # Try yfinance first
        pe = self._nifty_pe_yfinance()
        if pe and 10 < pe < 60:
            return round(pe, 2)

        # Try NSE indices endpoint
        pe = self._nifty_pe_nse()
        if pe and 10 < pe < 60:
            return round(pe, 2)

        logger.warning("Nifty PE unavailable, using default 22.5")
        return 22.5

    def _nifty_pe_yfinance(self) -> Optional[float]:
        try:
            import yfinance as yf
            ticker = yf.Ticker("^NSEI")
            pe = ticker.info.get("trailingPE")
            if pe:
                logger.debug(f"Nifty PE from yfinance: {pe}")
            return float(pe) if pe else None
        except Exception as e:
            logger.debug(f"yfinance PE fetch failed: {e}")
            return None

    def _nifty_pe_nse(self) -> Optional[float]:
        """Fallback: scrape PE from NSE allIndices response."""
        try:
            resp = self._session.get(_NSE_INDICES, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            for idx in data.get("data", []):
                if "NIFTY 50" in idx.get("indexSymbol", "").upper():
                    pe = idx.get("pe")
                    if pe:
                        logger.debug(f"Nifty PE from NSE: {pe}")
                        return float(pe)
        except Exception as e:
            logger.debug(f"NSE PE fetch failed: {e}")
        return None

    # ── G-sec yield ───────────────────────────────────────────────────────────
    #
    # Five-source cascade (tried in order, first valid result wins):
    #
    #   SOURCE 1 — Fyers API (NSE bond quotes)
    #     Uses the already-authenticated Fyers client injected via
    #     set_fyers_client().  Fetches live NSE bond market data for
    #     all G-secs and picks the paper with maturity closest to 10Y.
    #     Most reliable when live trading — same session, zero extra auth.
    #
    #   SOURCE 2 — NSE India liveBonds JSON  (multiple URL variants)
    #     NSE public JSON endpoints for government/corporate bond prices.
    #     The session is already warm from FII/DII fetch so cookies are set.
    #     Tries: liveBonds, liveBonds?type=gsec, bonds (fallback URLs).
    #
    #   SOURCE 3 — yfinance  (history + download, multiple Reuters tickers)
    #     yf.download() is used first (more robust for rate series), then
    #     yf.Ticker().history() as a secondary path.
    #
    #   SOURCE 4 — Stooq.com CSV download
    #     Free EOD global bond data, no auth. Symbol: "10ind.b"
    #
    #   SOURCE 5 — CCIL ZCYC HTML scrape
    #     CCIL Zero Coupon Yield Curve page; authoritative but slow.
    #
    #   DEFAULT — 7.0%
    # ─────────────────────────────────────────────────────────────────────────

    _GSEC_YF_TICKERS = [
        "IN10Y=RR",
        "INBMK10Y=RR",
        "IGB10YT=RR",
        "INGVT10YT=RR",
        "^INBMK10Y",
    ]

    _NSE_BOND_URLS = [
        "https://www.nseindia.com/api/liveBonds",
        "https://www.nseindia.com/api/liveBonds?type=gsec",
        "https://www.nseindia.com/api/liveBonds?type=gsec_bonds",
        "https://www.nseindia.com/api/bonds",
    ]

    _STOOQ_10Y_URL = "https://stooq.com/q/d/l/?s=10ind.b&i=d"
    _CCIL_ZCYC_URL = "https://www.ccilindia.com/Research/MarketData.aspx"

    # RBI DBIE database — official, reliable, no auth required
    # Returns JSON with GSec yield series (ID: FBIL_ZCYC_10Y or similar)
    _RBI_DBIE_URL  = (
        "https://dbie.rbi.org.in/DBIE/dbie.rbi?site=publications"
        "&publicationsViewBy=N&publicationsType=9"
    )
    # RBI Yield Curve API (FBIL-published, updated daily)
    _FBIL_URL = "https://www.fbil.org.in/uploads/FBIL_GOI_YC.json"

    # Valid range guard — India 10Y G-sec yield is almost always 5.5 – 10%
    _YIELD_MIN, _YIELD_MAX = 4.5, 12.0

    def _fetch_gsec_yield(self) -> float:
        """
        Fetch India 10-year G-sec yield via 5-source cascade.
        Returns the first value passing the sanity range [4.5, 12.0]%.
        """
        sources = [
            ("fyers-bond-quotes",  self._gsec_fyers),
            ("fbil-json",          self._gsec_fbil),
            ("rbi-benchmark",      self._gsec_rbi),
            ("nse-liveBonds",      self._gsec_nse_livebonds),
            ("yfinance",           self._gsec_yfinance),
            ("stooq-csv",          self._gsec_stooq),
            ("ccil-zcyc",          self._gsec_ccil),
        ]
        for name, method in sources:
            try:
                val = method()
                if val and self._YIELD_MIN < val < self._YIELD_MAX:
                    logger.info(f"G-sec yield [{name}]: {val:.2f}%")
                    return round(val, 2)
                if val is not None:
                    logger.debug(
                        f"G-sec [{name}] returned {val} — outside valid range, skipping"
                    )
            except Exception as exc:
                logger.debug(f"G-sec [{name}] error: {exc}")

        logger.warning(
            "G-sec yield unavailable from all 7 sources — using default 7.0%"
        )
        return 7.0

    # ── Source 1: Fyers bond market quotes ───────────────────────────────────

    def _gsec_fyers(self) -> Optional[float]:
        """
        Fetch G-sec yield via the live Fyers API client (NSE bond segment).

        NSE lists government securities in the debt market segment.
        The Fyers quotes API accepts bond symbols in the format:
          NSE:<coupon_bps>GS<maturity_year>  e.g. "NSE:730GS2032-GB"

        Strategy: fetch the NSE liveBonds list (already tried in source 2
        but here we have auth) to get the current benchmark 10Y symbol,
        then query Fyers quotes for its YTM (yield to maturity).

        Falls back to a hardcoded list of recent benchmark symbols if the
        bond-list fetch fails, since the benchmark changes infrequently.
        """
        if self._fyers_client is None:
            return None

        # Hardcoded recent 10Y benchmark G-sec symbols on NSE
        # Updated set covers likely benchmarks for 2024-2026
        # Format: NSE:<coupon*100>GS<year>-GB
        # Current 10Y benchmark (Feb 2026): 6.79% GS 2034
        # Also include adjacent maturities (7-12Y window) as fallback.
        # These are updated for 2025-2026 RBI issuance calendar.
        benchmark_candidates = [
            "NSE:679GS2034-GB",   # Current benchmark as of Q4 FY26
            "NSE:692GS2034-GB",
            "NSE:715GS2034-GB",
            "NSE:730GS2035-GB",
            "NSE:726GS2033-GB",
            "NSE:719GS2034-GB",
            "NSE:685GS2034-GB",
            "NSE:730GS2032-GB",
            "NSE:720GS2033-GB",
            "NSE:685GS2035-GB",
            "NSE:693GS2034-GB",
            "NSE:710GS2034-GB",
        ]

        try:
            resp = self._fyers_client.quotes(
                {"symbols": ",".join(benchmark_candidates)}
            )
            logger.debug(f"  Fyers G-sec quotes response: {resp}")
            if resp.get("s") != "ok":
                logger.debug(f"  Fyers G-sec API error: {resp.get('message', resp)}")
                return None
            if not resp.get("d"):
                logger.debug("  Fyers G-sec: no data items in response")
                return None

            today       = date.today()
            best_ytm    = None
            best_diff   = float("inf")

            for item in resp.get("d", []):
                v   = item.get("v", {})
                sym = item.get("n", "")

                # Extract maturity year from symbol, e.g. "NSE:730GS2035-GB" → 2035
                import re as _re
                m = _re.search(r"GS(\d{4})", sym)
                if not m:
                    continue
                mat_year  = int(m.group(1))
                years_out = mat_year - today.year

                # ytm field — Fyers exposes this for debt instruments
                ytm = v.get("ytm") or v.get("yield") or v.get("yld")
                if ytm is None:
                    # Fallback: derive approximate yield from last price
                    # Bond price ≈ 100 → yield ≈ coupon rate
                    ltp = v.get("lp") or v.get("close_price")
                    coupon_m = _re.search(r"^NSE:(\d+)GS", sym)
                    if ltp and coupon_m:
                        coupon = int(coupon_m.group(1)) / 100.0
                        # Very rough YTM approximation from clean price
                        if 85 < float(ltp) < 115:
                            face  = 100.0
                            price = float(ltp)
                            ytm   = (coupon + (face - price) / max(years_out, 1)) / \
                                    ((face + price) / 2) * 100
                        else:
                            continue
                    else:
                        continue

                ytm_f = float(ytm)
                diff  = abs(years_out - 10.0)

                if 7 <= years_out <= 12 and diff < best_diff:
                    if self._YIELD_MIN < ytm_f < self._YIELD_MAX:
                        best_diff = diff
                        best_ytm  = ytm_f

            if best_ytm:
                logger.debug(f"  Fyers G-sec 10Y proxy: {best_ytm:.2f}%")
            return best_ytm

        except Exception as exc:
            logger.debug(f"Fyers G-sec fetch error: {exc}")
            return None

    # ── Source 2: FBIL GOI Yield Curve JSON ─────────────────────────────────
    #
    #   FBIL (Financial Benchmarks India Pvt. Ltd.) publishes the official
    #   GOI yield curve daily.  The JSON endpoint returns tenor→yield pairs
    #   including the 10Y point.  No auth, small payload, very reliable.
    # ─────────────────────────────────────────────────────────────────────────

    def _gsec_fbil(self) -> Optional[float]:
        """
        Fetch India 10Y G-sec yield from FBIL's official yield curve JSON.

        FBIL publishes the RBI-mandated benchmark yield curve daily at:
          https://www.fbil.org.in/uploads/FBIL_GOI_YC.json

        Response format (typical):
          [{"tenor": "1", "rate": "6.89"}, ..., {"tenor": "10", "rate": "6.79"}, ...]
          OR
          {"data": [{"maturity": 10.0, "yield": 6.79}, ...]}

        Also tries the FBIL term money / MIBOR page as a secondary path.
        """
        urls_to_try = [
            "https://www.fbil.org.in/uploads/FBIL_GOI_YC.json",
            "https://www.fbil.org.in/uploads/FBIL_GOVTsecBenchmark.json",
            "https://www.fbil.org.in/#!#GSEC",   # HTML fallback parsed below
        ]

        for url in urls_to_try:
            try:
                resp = self._session.get(url, timeout=self.timeout,
                                         headers={**_HEADERS,
                                                   "Referer": "https://www.fbil.org.in/"})
                if resp.status_code != 200:
                    logger.debug(f"  FBIL {url}: HTTP {resp.status_code}")
                    continue

                content_type = resp.headers.get("content-type", "")
                text = resp.text.strip()

                # ── JSON path ──────────────────────────────────────────────
                if "json" in content_type or text.startswith(("[", "{")):
                    try:
                        data = resp.json()
                        # Normalise to a list of records
                        records = data if isinstance(data, list) else data.get("data", [])

                        best_yield = None
                        best_diff  = float("inf")

                        for rec in records:
                            # Handle various field name conventions
                            tenor_raw = (rec.get("tenor") or rec.get("maturity")
                                         or rec.get("term") or rec.get("year"))
                            yield_raw = (rec.get("rate")  or rec.get("yield")
                                         or rec.get("yld")  or rec.get("value"))
                            if tenor_raw is None or yield_raw is None:
                                continue
                            try:
                                tenor = float(str(tenor_raw))
                                yld   = float(str(yield_raw).replace(",", ""))
                            except (ValueError, TypeError):
                                continue

                            diff = abs(tenor - 10.0)
                            if (7.0 <= tenor <= 12.0
                                    and diff < best_diff
                                    and self._YIELD_MIN < yld < self._YIELD_MAX):
                                best_diff  = diff
                                best_yield = yld

                        if best_yield is not None:
                            logger.debug(f"  FBIL JSON ({url}): {best_yield}%")
                            return best_yield

                    except Exception as exc:
                        logger.debug(f"  FBIL JSON parse error ({url}): {exc}")

                # ── HTML scrape path ───────────────────────────────────────
                import re as _re
                # Look for pattern like "10 Year: 6.79" or "10Y ... 6.79%"
                m = _re.search(
                    r"10\s*[Yy](?:ear)?[^<]{0,40}?([5-9]\.[0-9]{2,4})",
                    text, _re.IGNORECASE,
                )
                if m:
                    val = float(m.group(1))
                    if self._YIELD_MIN < val < self._YIELD_MAX:
                        logger.debug(f"  FBIL HTML scrape ({url}): {val}%")
                        return val

            except Exception as exc:
                logger.debug(f"  FBIL {url}: {exc}")

        return None

    # ── Source 3: RBI / FBIL Benchmark Rate (web scrape) ─────────────────────

    def _gsec_rbi(self) -> Optional[float]:
        """
        Fetch India 10Y G-sec yield from the RBI / FBIL benchmark pages.

        Tries multiple authoritative Indian govt finance pages that publish
        the benchmark 10Y GSec yield as plaintext/JSON — no auth required.

        Sources tried:
          A) FBIL Treasury Bills / GSec benchmark rate page
          B) RBI's published reference rate page (rbi.org.in)
          C) NSE India bond data page (html scrape)
        """
        import re as _re

        sources = [
            # A: NSE debt market summary — often has benchmark yield in page
            ("https://www.nseindia.com/market-data/bonds-traded-in-capital-market",
             r"(?:10[- ]?[Yy](?:ear)?|10Y)[^\d]{0,30}([6-8]\.[0-9]{2})"),
            # B: MoneyControl India 10Y bond page (reliable scrape)
            ("https://www.moneycontrol.com/bonds/india-10-year-bond-yield/",
             r"([6-8]\.[0-9]{2,4})"),
            # C: Investing.com India 10Y (very widely available)
            ("https://in.investing.com/rates-bonds/india-10-year-bond-yield",
             r'"price"\s*:\s*"?([6-8]\.[0-9]{2})"?'),
        ]

        for url, pattern in sources:
            try:
                resp = self._session.get(
                    url, timeout=self.timeout,
                    headers={**_HEADERS, "Referer": url.rsplit("/", 1)[0]},
                    allow_redirects=True
                )
                if resp.status_code != 200:
                    logger.debug(f"  RBI/bench {url}: HTTP {resp.status_code}")
                    continue

                html = resp.text
                m = _re.search(pattern, html, _re.IGNORECASE)
                if m:
                    val = float(m.group(1))
                    if self._YIELD_MIN < val < self._YIELD_MAX:
                        logger.debug(f"  RBI/bench ({url}): {val}%")
                        return val
                else:
                    logger.debug(f"  RBI/bench ({url}): pattern not matched")

            except Exception as exc:
                logger.debug(f"  RBI/bench {url}: {exc}")

        return None

    # ── Source 2: NSE liveBonds JSON ─────────────────────────────────────────

    def _gsec_nse_livebonds(self) -> Optional[float]:
        """
        Fetch India 10Y yield from NSE's live bonds endpoint.

        Tries multiple URL variants since the exact path differs across
        NSE API versions.  The session is already warm (FII/DII cookies set).

        Response structure varies by endpoint:
          /api/liveBonds → list of {symbol, series, yieldRate, maturityDate, ...}
          /api/bonds     → may be wrapped: {"data": [...]}
        """
        today = date.today()

        for url in self._NSE_BOND_URLS:
            try:
                resp = self._session.get(url, timeout=self.timeout)
                if resp.status_code != 200:
                    logger.debug(f"  NSE bonds {url}: HTTP {resp.status_code}")
                    continue

                payload = resp.json()

                # Unwrap {"data": [...]} envelope if present
                bonds = payload if isinstance(payload, list) else \
                        payload.get("data", payload.get("bonds", []))

                if not bonds:
                    logger.debug(f"  NSE bonds {url}: empty response")
                    continue

                best_yield = None
                best_diff  = float("inf")

                for bond in bonds:
                    # Try multiple field name variants
                    yld_raw = (bond.get("yieldRate")
                               or bond.get("yield")
                               or bond.get("ytm")
                               or bond.get("yld")
                               or bond.get("yield_rate"))
                    mat_str = (bond.get("maturityDate")
                               or bond.get("matDate")
                               or bond.get("redemptionDate")
                               or bond.get("maturtyDate"))  # NSE typo seen in wild

                    # Only process central government G-secs
                    series = str(bond.get("series", "")
                                 or bond.get("securityType", "")
                                 or bond.get("instrumentType", "")).upper()
                    if series and not any(k in series for k in
                                          ("GSEC", "GOVT", "GOV", "CENTRAL", "SGB", "")):
                        continue

                    if not mat_str or yld_raw is None:
                        continue

                    for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d/%m/%Y", "%b %d, %Y"):
                        try:
                            mat_date  = datetime.strptime(mat_str.strip(), fmt).date()
                            years_out = (mat_date - today).days / 365.25
                            diff      = abs(years_out - 10.0)
                            yld_f     = float(str(yld_raw).replace(",", ""))

                            if (7.0 <= years_out <= 12.0
                                    and diff < best_diff
                                    and self._YIELD_MIN < yld_f < self._YIELD_MAX):
                                best_diff  = diff
                                best_yield = yld_f
                            break
                        except ValueError:
                            continue

                if best_yield is not None:
                    logger.debug(f"  NSE liveBonds ({url}): {best_yield}%")
                    return best_yield

            except Exception as exc:
                logger.debug(f"  NSE bonds {url}: {exc}")
                continue

        return None

    # ── Source 3: yfinance (download + history) ───────────────────────────────

    def _gsec_yfinance(self) -> Optional[float]:
        """
        Fetch India 10Y yield via yfinance.

        Uses yf.download() first — it bypasses some of the caching/session
        issues that affect yf.Ticker().history() for Reuters bond tickers.
        Falls back to .history() for each ticker if download returns nothing.
        """
        try:
            import yfinance as yf
            import warnings
            warnings.filterwarnings("ignore", category=FutureWarning)
        except ImportError:
            return None

        # Path A: yf.download()
        for sym in self._GSEC_YF_TICKERS:
            try:
                df = yf.download(sym, period="5d", progress=False,
                                 auto_adjust=True, actions=False)
                if df is not None and not df.empty:
                    close_col = "Close" if "Close" in df.columns else df.columns[-1]
                    val = float(df[close_col].dropna().iloc[-1])
                    if val > 0:
                        logger.debug(f"  yf.download ({sym}): {val}")
                        return val
            except Exception as exc:
                logger.debug(f"  yf.download ({sym}) failed: {exc}")

        # Path B: yf.Ticker().history()
        for sym in self._GSEC_YF_TICKERS:
            try:
                hist = yf.Ticker(sym).history(period="5d", auto_adjust=True)
                if hist is not None and not hist.empty:
                    closes = hist["Close"].dropna()
                    if not closes.empty:
                        val = float(closes.iloc[-1])
                        if val > 0:
                            logger.debug(f"  yf.history ({sym}): {val}")
                            return val
            except Exception as exc:
                logger.debug(f"  yf.history ({sym}) failed: {exc}")

        return None

    # ── Source 4: Stooq CSV download ─────────────────────────────────────────

    def _gsec_stooq(self) -> Optional[float]:
        """
        Download India 10Y bond yield CSV from Stooq (no auth, free EOD).
        URL: https://stooq.com/q/d/l/?s=10ind.b&i=d
        Format: Date,Open,High,Low,Close,Volume  (Close = yield %)
        """
        try:
            resp = self._session.get(
                self._STOOQ_10Y_URL,
                timeout=self.timeout,
                headers={**self._session.headers, "Referer": "https://stooq.com/"},
            )
            resp.raise_for_status()
            text = resp.text.strip()

            if not text or "No data" in text or len(text) < 30:
                return None

            lines = [ln.strip() for ln in text.splitlines()
                     if ln.strip() and not ln.lower().startswith("date")]
            if not lines:
                return None

            last_row = lines[-1].split(",")
            if len(last_row) < 5:
                return None

            val = float(last_row[4])   # Close = yield
            logger.debug(f"  Stooq ({last_row[0]}): {val}%")
            return val

        except Exception as exc:
            logger.debug(f"Stooq G-sec error: {exc}")
            return None

    # ── Source 5: CCIL ZCYC HTML scrape ──────────────────────────────────────

    def _gsec_ccil(self) -> Optional[float]:
        """
        Scrape 10Y yield from CCIL's Zero Coupon Yield Curve page.
        Tries two regex patterns to handle minor HTML layout variations.
        """
        try:
            import re
            resp = self._session.get(self._CCIL_ZCYC_URL, timeout=self.timeout)
            resp.raise_for_status()
            html = resp.text

            # Pattern A: <td>10</td> followed by yield cell
            m = re.search(
                r"<td[^>]*>\s*10\s*</td>\s*<td[^>]*>\s*([\d.]+)\s*</td>",
                html, re.IGNORECASE,
            )
            if m:
                val = float(m.group(1))
                logger.debug(f"  CCIL ZCYC (A): {val}%")
                return val

            # Pattern B: "10 Year" or "10Y" text near a decimal
            m = re.search(
                r"10\s*[Yy](?:ear)?[s]?[^<]{0,30}([\d]{1,2}\.[0-9]{1,4})",
                html, re.IGNORECASE,
            )
            if m:
                val = float(m.group(1))
                logger.debug(f"  CCIL ZCYC (B): {val}%")
                return val

            logger.debug("CCIL ZCYC: 10Y row not found in HTML")
            return None

        except Exception as exc:
            logger.debug(f"CCIL ZCYC error: {exc}")
            return None

    # ── HTTP session ──────────────────────────────────────────────────────────

    def _fetch_gift_nifty(self) -> Dict[str, float]:
        """
        FIX 4.2: Fetch GIFT Nifty futures and Dow Jones futures for pre-market cues.
        which are the two most critical macro inputs for Indian market regime detection.
        USD/INR stress amplifies FII outflows and tightens financial conditions.

        All tickers are free via yfinance — no authentication required.
        Returns: dict with gift_nifty_price, gift_nifty_gap_pct, dow_futures_chg_pct,
        """
        result = {
            "gift_nifty_price":    0.0,
            "gift_nifty_gap_pct":  0.0,
            "dow_futures_chg_pct": 0.0,
            "usd_inr":             0.0,   # FIX-MACRO-01
        }
        try:
            import yfinance as yf

            # GIFT Nifty
            gift_ticker = yf.Ticker("^NSGIF")
            gift_info   = gift_ticker.fast_info
            gift_price  = float(getattr(gift_info, "last_price", 0) or 0)
            gift_prev   = float(getattr(gift_info, "previous_close", 0) or 0)

            if gift_price > 0 and gift_prev > 0:
                gap_pct = round((gift_price - gift_prev) / gift_prev * 100, 2)
                result["gift_nifty_price"]   = round(gift_price, 2)
                result["gift_nifty_gap_pct"] = gap_pct
                logger.info(
                    f"GIFT Nifty: {gift_price:.0f} | gap={gap_pct:+.2f}% "
                    f"({'gap-up' if gap_pct > 0 else 'gap-down' if gap_pct < 0 else 'flat'})"
                )

            # Dow Jones Futures (YM=F)
            dow_ticker = yf.Ticker("YM=F")
            dow_info   = dow_ticker.fast_info
            dow_price  = float(getattr(dow_info, "last_price", 0) or 0)
            dow_prev   = float(getattr(dow_info, "previous_close", 0) or 0)

            if dow_price > 0 and dow_prev > 0:
                dow_chg = round((dow_price - dow_prev) / dow_prev * 100, 2)
                result["dow_futures_chg_pct"] = dow_chg
                logger.debug(f"Dow Futures: {dow_price:.0f} | chg={dow_chg:+.2f}%")

            # BZ=F (ICE Brent) is often stale or empty; CL=F (NYMEX WTI) is more reliable.
            # history() is more robust than fast_info for weekends and after-hours.
            # WTI/Brent spread ~$3-5 is acceptable for macro regime detection.
            try:
                crude_tickers = ["CL=F", "BZ=F"]  # WTI primary, Brent backup
                for _ticker in crude_tickers:
                    try:
                        _ct = yf.Ticker(_ticker)
                        # Try fast_info first (fastest)
                        _fi = _ct.fast_info
                        _p  = float(getattr(_fi, "last_price", 0) or 0)
                        if _p > 70:
                            result["crude_brent_usd"] = round(_p, 2)
                            logger.info(f"Crude ({_ticker}): ${_p:.2f}")
                            break
                        # Fall back to history (works on weekends — returns last close)
                        _hist = _ct.history(period="5d", auto_adjust=True)
                        if not _hist.empty:
                            _p = float(_hist["Close"].dropna().iloc[-1])
                            if _p > 70:
                                result["crude_brent_usd"] = round(_p, 2)
                                logger.info(f"Crude ({_ticker}, history): ${_p:.2f}")
                                break
                    except Exception:
                        continue
            except Exception as _ce:
                logger.debug(f"Crude oil fetch skipped: {_ce}")

            # FIX-MACRO-01: USD/INR spot rate (INR=X on Yahoo Finance)
            # Rupee depreciation > 0.5% in a session signals FII stress / risk-off.
            try:
                inr_info  = yf.Ticker("INR=X").fast_info
                inr_price = float(getattr(inr_info, "last_price", 0) or 0)
                if inr_price > 0:
                    result["usd_inr"] = round(inr_price, 4)
                    logger.info(f"USD/INR: ₹{inr_price:.2f}")
            except Exception as _ie:
                logger.debug(f"USD/INR fetch skipped: {_ie}")

        except Exception as exc:
            logger.debug(f"GIFT Nifty/Dow futures fetch skipped: {exc}")

        # Fallback for USD/INR if primary source failed
        # (common post-market, weekend, or when INR=X is stale), try alternative sources.
        # Same cascading pattern as _fetch_gsec_yield's 7-source fallback chain.
        if result.get("usd_inr", 0.0) == 0.0:
            result["usd_inr"] = self._usd_inr_stooq() or self._usd_inr_rbi() or 0.0

        return result

    # ── USD/INR fallback sources ──────────────────────────────────────────────

    def _usd_inr_stooq(self) -> Optional[float]:
        """
        FIX-MACRO-01: USD/INR spot rate from Stooq (free, no auth).
        Ticker: usdinr (USD per INR, need to invert) or inrusd.
        Returns INR per USD (e.g. 83.5) or None on failure.
        """
        urls = [
            ("https://stooq.com/q/d/l/?s=usdinr&i=d", False),   # USD/INR direct
            ("https://stooq.com/q/d/l/?s=inrusd&i=d", True),    # INR/USD → invert
        ]
        for url, invert in urls:
            try:
                resp = self._session.get(
                    url, timeout=self.timeout,
                    headers={**self._session.headers, "Referer": "https://stooq.com/"},
                )
                resp.raise_for_status()
                text = resp.text.strip()
                if not text or "No data" in text or len(text) < 30:
                    continue
                lines = [ln for ln in text.splitlines()
                         if ln.strip() and not ln.lower().startswith("date")]
                if not lines:
                    continue
                parts = lines[-1].split(",")
                if len(parts) < 5:
                    continue
                val = float(parts[4])
                if invert and val > 0:
                    val = round(1.0 / val, 4)
                if 60 < val < 120:      # sanity: INR/USD between 60-120
                    logger.info(f"USD/INR (stooq): ₹{val:.2f}")
                    return round(val, 4)
            except Exception as exc:
                logger.debug(f"Stooq USD/INR error ({url}): {exc}")
        return None

    def _usd_inr_rbi(self) -> Optional[float]:
        """
        FIX-MACRO-01: USD/INR reference rate from RBI's FBIL JSON feed.
        RBI publishes the daily reference rate at ~1:30 PM IST.
        Returns INR per USD or None on failure.
        """
        try:
            import re
            urls = [
                "https://www.fbil.org.in/uploads/FBIL_USD_INR_Reference_Rate.json",
                "https://www.fbil.org.in/#!#USDINR",
            ]
            for url in urls:
                resp = self._session.get(
                    url, timeout=self.timeout,
                    headers={**self._session.headers, "Referer": "https://www.fbil.org.in/"},
                )
                resp.raise_for_status()
                text = resp.text
                # Try JSON path first
                try:
                    import json as _json
                    data = _json.loads(text)
                    # Common FBIL JSON shapes
                    for key in ("rate", "referenceRate", "USD", "usdinr"):
                        if key in data:
                            val = float(str(data[key]).replace(",", ""))
                            if 60 < val < 120:
                                logger.info(f"USD/INR (RBI/FBIL): ₹{val:.2f}")
                                return round(val, 4)
                except Exception:
                    pass
                # HTML fallback
                m = re.search(r'USD[/\s]*INR[^0-9]*([\d]{2,3}\.[\d]{2,4})', text, re.I)
                if m:
                    val = float(m.group(1))
                    if 60 < val < 120:
                        logger.info(f"USD/INR (RBI HTML): ₹{val:.2f}")
                        return round(val, 4)
        except Exception as exc:
            logger.debug(f"RBI/FBIL USD/INR error: {exc}")
        return None

    def _make_session(self):
        """
        Build a requests.Session with NSE-compatible headers and a
        session cookie (NSE requires a cookie obtained from the homepage).
        """
        import requests
        session = requests.Session()
        session.headers.update(_HEADERS)

        # Warm up NSE session cookie — NSE blocks requests without a prior
        # homepage visit. Ignore failures; FII/DII endpoint may still work.
        try:
            session.get(_NSE_BASE, timeout=8)
        except Exception:
            pass

        return session
