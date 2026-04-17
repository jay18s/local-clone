# news_core.py
"""
ROX Proven Edge Engine v4.1 - News Intelligence Core
=====================================================
Real-time news analysis using OpenRouter API for geopolitical risk assessment,
market sentiment, and overnight gap prediction.

Integrates with: KAIRO (sentiment), CATALYST (events), NOCTURNAL (risk)
Migrated from Gemini to OpenRouter on 2026-04-17.
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Literal, Any
from dataclasses import dataclass, field
from enum import Enum
import feedparser
import aiohttp
import hashlib
import time

import httpx

logger = logging.getLogger("rox.news")


class ImpactSeverity(Enum):
    CRITICAL = 4  # Market-wide crash/rally potential (war, major policy)
    HIGH = 3  # Sector-wide impact (tariffs on specific industry)
    MEDIUM = 2  # Single stock/limited impact
    LOW = 1  # Noise/minor impact
    NONE = 0


class NewsCategory(Enum):
    GEOPOLITICAL = "geopolitical"  # War, conflicts, diplomacy
    POLICY = "policy"  # RBI, Fed, government policy
    TRADE = "trade"  # Tariffs, trade wars
    EARNINGS = "earnings"  # Corporate results
    MACRO = "macro"  # GDP, inflation, employment
    SENTIMENT = "sentiment"  # Market psychology


@dataclass
class NewsItem:
    """Structured news item with Gemini-generated analysis"""
    headline: str
    source: str
    published: datetime
    url: str
    content: str
    symbols: List[str]  # Related Indian stocks/indices
    sectors: List[str]  # Affected sectors
    indices: List[str]  # Affected indices (NIFTY50, BANKNIFTY, etc.)
    impact_score: float  # -1.0 to +1.0
    severity: ImpactSeverity
    category: NewsCategory
    time_horizon: str  # immediate, overnight, sustained
    reasoning: str  # Gemini explanation
    raw_analysis: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "headline": self.headline,
            "source": self.source,
            "published": self.published.isoformat(),
            "impact_score": self.impact_score,
            "severity": self.severity.name,
            "category": self.category.value,
            "symbols": self.symbols,
            "sectors": self.sectors,
            "indices": self.indices,
            "time_horizon": self.time_horizon,
            "reasoning": self.reasoning
        }


@dataclass
class OvernightRiskProfile:
    """Pre-market risk assessment output"""
    risk_level: str  # EXTREME, HIGH, ELEVATED, NORMAL, LOW
    market_stance: str  # LONG, SHORT, NEUTRAL, CASH
    confidence: int  # 0-100
    gap_probability: float  # 0.0-1.0
    expected_gap_size: str  # e.g., "+200/-150 points NIFTY"
    key_headlines: List[str]
    affected_sectors: Dict[str, float]  # sector -> impact score
    trading_restrictions: List[str]
    narrative: str  # Executive summary
    timestamp: datetime = field(default_factory=datetime.now)

    def to_agent_context(self) -> Dict:
        """Convert to format consumable by other agents"""
        return {
            "risk_level": self.risk_level,
            "market_stance": self.market_stance,
            "confidence": self.confidence,
            "gap_probability": self.gap_probability,
            "expected_gap": self.expected_gap_size,
            "sector_impacts": self.affected_sectors,
            "restrictions": self.trading_restrictions,
            "narrative": self.narrative,
            "timestamp": self.timestamp.isoformat()
        }


class GeminiNewsAnalyzer:
    """
    OpenRouter-powered news analysis engine.
    Uses configured model for both filtering and deep analysis.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPEN_ROUTER_API", "")
        self.flash_model = None
        self.pro_model = None
        self._base_url = os.getenv("OPEN_ROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        self._init_models()

        # Caching for duplicate detection
        self._seen_hashes: set = set()
        self._cache_ttl = 3600  # 1 hour

    def _init_models(self):
        """Initialize OpenRouter model config."""
        if not self.api_key:
            logger.warning("OpenRouter API not configured. News analysis will use fallback.")
            return

        model = os.getenv("OPEN_ROUTER_MODEL", "openrouter/free")
        self.flash_model = model
        self.pro_model = model

        logger.info(f"News Analyzer initialized | model={model} (via OpenRouter)")

    def _is_configured(self) -> bool:
        return self.api_key != "" and self.flash_model is not None

    def _openrouter_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("OPEN_ROUTER_HTTP_REFERER", "https://rox-engine.local"),
            "X-Title": os.getenv("OPEN_ROUTER_X_TITLE", "ROX Trading Engine"),
        }

    async def _call_openrouter(self, model: str, prompt: str) -> str:
        """Make an async call to OpenRouter."""
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a senior macro analyst for Indian equity markets."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 4096,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                headers=self._openrouter_headers(),
                json=payload,
            )

        if resp.status_code != 200:
            raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")

    # Maximum headlines per call
    _BATCH_SIZE = 20

    async def analyze_batch(self, headlines: List[Dict], urgency: str = "normal") -> List[NewsItem]:
        """
        Batch analyze news headlines.

        Args:
            headlines: List of dicts with 'headline', 'source', 'url', 'published'
            urgency: 'immediate' (use Flash) or 'deep' (use Pro)

        Batches are capped at _BATCH_SIZE to prevent JSON truncation when
        Gemini hits the output token limit mid-array (41 headlines was too many).
        """
        if not self._is_configured():
            return self._fallback_analysis(headlines)

        # Deduplicate
        unique_headlines = []
        for h in headlines:
            content_hash = hashlib.md5(h['headline'].encode()).hexdigest()[:12]
            if content_hash not in self._seen_hashes:
                self._seen_hashes.add(content_hash)
                unique_headlines.append(h)

        if not unique_headlines:
            return []

        # Chunk into batches to avoid output token truncation
        all_items: List[NewsItem] = []
        for i in range(0, len(unique_headlines), self._BATCH_SIZE):
            chunk = unique_headlines[i : i + self._BATCH_SIZE]
            prompt = self._build_analysis_prompt(chunk)
            try:
                if urgency == "immediate":
                    response = await self._call_flash(prompt)
                else:
                    response = await self._call_pro(prompt)
                items = self._parse_response(chunk, response)
                all_items.extend(items)
            except Exception as e:
                logger.error(f"Gemini analysis failed (chunk {i//self._BATCH_SIZE + 1}): {e}")
                all_items.extend(self._fallback_analysis(chunk))

        return all_items

    def _build_analysis_prompt(self, headlines: List[Dict]) -> str:
        """Build analysis prompt for Gemini"""
        headlines_json = json.dumps(headlines, indent=2)

        return f"""You are a senior macro analyst for Indian equity markets (NSE/BSE).
Analyze these news headlines for impact on Indian stocks, sectors, and indices.

HEADLINES TO ANALYZE:
{headlines_json}

For EACH headline, provide analysis in this JSON format:
{{
    "headline": "exact headline text",
    "impact_score": float between -1.0 (extremely bearish) and +1.0 (extremely bullish),
    "severity": "CRITICAL|HIGH|MEDIUM|LOW|NONE",
    "category": "GEOPOLITICAL|POLICY|TRADE|EARNINGS|MACRO|SENTIMENT",
    "affected_sectors": ["IT", "Pharma", "Banking", "Oil & Gas", "Auto", "Metals", "FMCG", "Realty"],
    "affected_indices": ["NIFTY50", "BANKNIFTY", "NIFTYIT", "NIFTYBANK", "NIFTYMETAL"],
    "time_horizon": "immediate|overnight|sustained",
    "reasoning": "2-sentence explanation of market impact mechanism"
}}

SEVERITY GUIDELINES:
- CRITICAL: Wars, major trade wars, systemic banking crises, nuclear events, major terrorist attacks
- HIGH: Sector-wide tariffs, RBI/Fed policy shocks, major corporate fraud, border conflicts
- MEDIUM: Company-specific earnings, minor policy changes, regional tensions
- LOW: Routine announcements, minor executive changes, speculation

INDIAN MARKET CONTEXT:
- NIFTY 50 is the benchmark index (~22,500 level)
- Key sectors: IT (TCS, Infosys), Banking (HDFC Bank, ICICI), Oil (Reliance), Pharma
- Sensitive to: FII flows, USD/INR, crude oil prices, US Fed policy, China relations

Return as a JSON array. Be precise and quantitative."""

    async def _call_flash(self, prompt: str) -> str:
        """Call OpenRouter (fast model)"""
        return await self._call_openrouter(self.flash_model, prompt)

    async def _call_pro(self, prompt: str) -> str:
        """Call OpenRouter (smart model)"""
        return await self._call_openrouter(self.pro_model, prompt)

    def _parse_response(self, original_headlines: List[Dict], response_text: str) -> List[NewsItem]:
        """Parse Gemini response into NewsItem objects"""
        try:
            # Robust JSON cleaning — strip markdown fences, BOM, trailing commas
            text = response_text.strip().lstrip("\ufeff")
            # Strip ```json ... ``` or ``` ... ``` fences
            import re as _re
            text = _re.sub(r"^```(?:json)?\s*", "", text, flags=_re.MULTILINE)
            text = _re.sub(r"\s*```$", "", text, flags=_re.MULTILINE)
            text = text.strip()
            # Remove trailing commas before ] or } (common Gemini quirk)
            text = _re.sub(r",\s*([\]\}])", r"\1", text)

            analyses = json.loads(text)
            if not isinstance(analyses, list):
                analyses = [analyses]

            items = []
            for i, analysis in enumerate(analyses):
                if i >= len(original_headlines):
                    break

                orig = original_headlines[i]

                # Map severity
                sev_str = analysis.get("severity", "NONE")
                try:
                    severity = ImpactSeverity[sev_str]
                except KeyError:
                    severity = ImpactSeverity.NONE

                # Map category
                cat_str = analysis.get("category", "SENTIMENT")
                try:
                    category = NewsCategory[cat_str]
                except KeyError:
                    category = NewsCategory.SENTIMENT

                item = NewsItem(
                    headline=orig.get("headline", analysis.get("headline", "")),
                    source=orig.get("source", "Unknown"),
                    published=datetime.fromisoformat(orig.get("published", datetime.now().isoformat())),
                    url=orig.get("url", ""),
                    content=orig.get("content", ""),
                    symbols=analysis.get("affected_symbols", []),
                    sectors=analysis.get("affected_sectors", []),
                    indices=analysis.get("affected_indices", []),
                    impact_score=float(analysis.get("impact_score", 0)),
                    severity=severity,
                    category=category,
                    time_horizon=analysis.get("time_horizon", "overnight"),
                    reasoning=analysis.get("reasoning", ""),
                    raw_analysis=analysis
                )
                items.append(item)

            return items

        except Exception as e:
            logger.error(f"Failed to parse Gemini response: {e}")
            return self._fallback_analysis(original_headlines)

    def _fallback_analysis(self, headlines: List[Dict]) -> List[NewsItem]:
        """Rule-based fallback when Gemini unavailable"""
        items = []
        for h in headlines:
            headline = h.get("headline", "").upper()

            # Simple keyword matching
            impact_score = 0.0
            severity = ImpactSeverity.NONE
            category = NewsCategory.SENTIMENT
            sectors = []
            indices = ["NIFTY50"]

            # Geopolitical
            if any(k in headline for k in ["WAR", "CONFLICT", "ATTACK", "MISSILE", "BOMB"]):
                impact_score = -0.8
                severity = ImpactSeverity.CRITICAL
                category = NewsCategory.GEOPOLITICAL
                sectors = ["Oil & Gas", "Defense"]

            # Trade/Tariffs
            elif any(k in headline for k in ["TARIFF", "TRADE WAR", "TRUMP", "SANCTIONS"]):
                impact_score = -0.6
                severity = ImpactSeverity.HIGH
                category = NewsCategory.TRADE
                sectors = ["IT", "Pharma", "Metals"]

            # Policy
            elif any(k in headline for k in ["RBI", "FED", "RATE", "POLICY"]):
                impact_score = -0.3 if "HIKE" in headline else 0.2
                severity = ImpactSeverity.HIGH
                category = NewsCategory.POLICY
                sectors = ["Banking", "Realty"]

            # Earnings
            elif any(k in headline for k in ["EARNINGS", "RESULTS", "PROFIT", "LOSS"]):
                impact_score = 0.1
                severity = ImpactSeverity.MEDIUM
                category = NewsCategory.EARNINGS

            item = NewsItem(
                headline=h.get("headline", ""),
                source=h.get("source", "Unknown"),
                published=datetime.now(),
                url=h.get("url", ""),
                content="",
                symbols=[],
                sectors=sectors,
                indices=indices,
                impact_score=impact_score,
                severity=severity,
                category=category,
                time_horizon="overnight",
                reasoning=f"Fallback analysis based on keywords: {headline[:50]}...",
                raw_analysis={}
            )
            items.append(item)

        return items

    async def generate_overnight_assessment(self, news_items: List[NewsItem],
                                            market_context: Dict) -> OvernightRiskProfile:
        """
        Generate comprehensive overnight risk assessment using Gemini Pro.
        """
        if not self._is_configured() or not news_items:
            return self._fallback_risk_profile(news_items)

        # Prepare context
        critical_news = [n for n in news_items if n.severity.value >= ImpactSeverity.HIGH.value]

        context = {
            "critical_headlines": [
                {
                    "headline": n.headline,
                    "impact": n.impact_score,
                    "severity": n.severity.name,
                    "sectors": n.sectors,
                    "reasoning": n.reasoning
                }
                for n in critical_news[:10]
            ],
            "market_context": market_context,
            "timestamp": datetime.now().isoformat()
        }

        prompt = f"""As a pre-market risk analyst for Indian equities, analyze these overnight developments:

{json.dumps(context, indent=2)}

Provide a comprehensive risk assessment in this JSON format:
{{
    "risk_level": "EXTREME|HIGH|ELEVATED|NORMAL|LOW",
    "market_stance": "LONG|SHORT|NEUTRAL|CASH",
    "confidence": 0-100,
    "gap_probability": 0.0-1.0,
    "expected_gap_size": "e.g., +150/-100 points NIFTY",
    "affected_sectors": {{"IT": -0.5, "Banking": +0.3, ...}},
    "trading_restrictions": [
        "HALT_ALL_NEW_POSITIONS",
        "REDUCE_POSITION_SIZE_50%",
        "MANDATORY_HEDGE_ALL_EXPOSURE"
    ],
    "narrative": "2-paragraph executive summary of key risks and opportunities"
}}

RISK LEVEL DEFINITIONS:
- EXTREME: War, major systemic crisis, expected >3% gap
- HIGH: Major policy shock, expected 1.5-3% gap
- ELEVATED: Significant uncertainty, expected 0.5-1.5% gap
- NORMAL: Minor news, expected <0.5% gap
- LOW: No significant overnight developments

TRADING RESTRICTION OPTIONS:
- HALT_ALL_NEW_POSITIONS
- REDUCE_EXISTING_BY_75%
- REDUCE_POSITION_SIZE_50%
- REDUCE_POSITION_SIZE_25%
- MANDATORY_HEDGE_ALL_EXPOSURE
- NO_OVERNIGHT_FUTURES
- AVOID_UNHEDGED_OPTIONS
- INCREASE_SL_WIDTH_50%
- MAX_POSITION_SIZE_25K"""

        try:
            response = await self._call_pro(prompt)
            data = json.loads(response.strip().replace("```json", "").replace("```", ""))

            return OvernightRiskProfile(
                risk_level=data.get("risk_level", "NORMAL"),
                market_stance=data.get("market_stance", "NEUTRAL"),
                confidence=data.get("confidence", 50),
                gap_probability=data.get("gap_probability", 0.0),
                expected_gap_size=data.get("expected_gap_size", "0"),
                key_headlines=[n.headline for n in critical_news[:5]],
                affected_sectors=data.get("affected_sectors", {}),
                trading_restrictions=data.get("trading_restrictions", []),
                narrative=data.get("narrative", "No significant developments")
            )

        except Exception as e:
            logger.error(f"Overnight assessment failed: {e}")
            return self._fallback_risk_profile(news_items)

    def _fallback_risk_profile(self, news_items: List[NewsItem]) -> OvernightRiskProfile:
        """Rule-based risk profile when Gemini unavailable"""
        if not news_items:
            return OvernightRiskProfile(
                risk_level="NORMAL",
                market_stance="NEUTRAL",
                confidence=50,
                gap_probability=0.1,
                expected_gap_size="±50 points",
                key_headlines=[],
                affected_sectors={},
                trading_restrictions=[],
                narrative="No overnight news data available"
            )

        # Calculate aggregate metrics
        avg_impact = sum(n.impact_score for n in news_items) / len(news_items)
        max_severity = max((n.severity for n in news_items), key=lambda x: x.value)

        # Determine risk level
        if max_severity == ImpactSeverity.CRITICAL:
            risk_level = "EXTREME"
        elif max_severity == ImpactSeverity.HIGH:
            risk_level = "HIGH"
        elif max_severity == ImpactSeverity.MEDIUM:
            risk_level = "ELEVATED"
        elif max_severity == ImpactSeverity.LOW:
            risk_level = "NORMAL"
        else:
            risk_level = "LOW"

        # Determine stance
        if avg_impact < -0.5:
            stance = "SHORT"
        elif avg_impact > 0.5:
            stance = "LONG"
        else:
            stance = "NEUTRAL"

        # Build restrictions
        restrictions = []
        if risk_level == "EXTREME":
            restrictions = ["HALT_ALL_NEW_POSITIONS", "MANDATORY_HEDGE_ALL_EXPOSURE"]
        elif risk_level == "HIGH":
            restrictions = ["REDUCE_POSITION_SIZE_50%", "NO_OVERNIGHT_FUTURES"]

        return OvernightRiskProfile(
            risk_level=risk_level,
            market_stance=stance,
            confidence=60,
            gap_probability=0.3 if max_severity.value >= 3 else 0.1,
            expected_gap_size="±100-200 points" if max_severity.value >= 3 else "±50 points",
            key_headlines=[n.headline for n in news_items[:5] if n.severity.value >= 2],
            affected_sectors={},
            trading_restrictions=restrictions,
            narrative=f"Fallback analysis: {len(news_items)} items, avg impact {avg_impact:+.2f}"
        )


class NewsFetcher:
    """
    Multi-source news fetcher with caching and deduplication.
    """

    def __init__(self, analyzer: GeminiNewsAnalyzer):
        self.analyzer = analyzer
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_fetch: Dict[str, datetime] = {}
        self._cache: Dict[str, List[NewsItem]] = {}

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        return self._session

    async def fetch_all_sources(self) -> List[NewsItem]:
        """Fetch from all configured sources"""
        all_news = []

        # Fetch from multiple sources concurrently
        tasks = [
            self._fetch_google_news(),
            self._fetch_rss_feeds(),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, list):
                all_news.extend(result)
            elif isinstance(result, Exception):
                logger.warning(f"News source failed: {result}")

        # Deduplicate by headline similarity
        unique_news = self._deduplicate(all_news)

        # Analyze through Gemini
        if unique_news:
            headlines = [
                {
                    "headline": n.headline,
                    "source": n.source,
                    "url": n.url,
                    "published": n.published.isoformat(),
                    "content": n.content[:500] if n.content else ""
                }
                for n in unique_news
            ]
            analyzed = await self.analyzer.analyze_batch(headlines, urgency="immediate")
            return analyzed

        return []

    async def _fetch_google_news(self) -> List[NewsItem]:
        """Fetch from Google News RSS"""
        queries = [
            "India stock market NSE BSE",
            "NIFTY BANKNIFTY trading",
            "RBI policy India",
            "Trump tariffs trade war",
            "Middle East war oil India",
            "US Fed interest rates emerging markets",
            "China India border tension economic"
        ]

        all_items = []
        for query in queries:
            try:
                rss_url = f"https://news.google.com/rss/search?q={query.replace(' ', '%20')}&hl=en-IN"
                feed = feedparser.parse(rss_url)

                for entry in feed.entries[:5]:
                    item = NewsItem(
                        headline=entry.title,
                        source=entry.get("source", {}).get("title", "Google News"),
                        published=datetime.now(),  # Parse properly in production
                        url=entry.link,
                        content=entry.get("summary", ""),
                        symbols=[],
                        sectors=[],
                        indices=[],
                        impact_score=0.0,
                        severity=ImpactSeverity.NONE,
                        category=NewsCategory.SENTIMENT,
                        time_horizon="overnight",
                        reasoning=""
                    )
                    all_items.append(item)

            except Exception as e:
                logger.debug(f"Google News fetch failed for {query}: {e}")

        return all_items

    async def _fetch_rss_feeds(self) -> List[NewsItem]:
        """Fetch from direct RSS feeds"""
        feeds = [
            "https://www.moneycontrol.com/rss/MCtopnews.xml",
            "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
            "https://www.livemint.com/rss/markets",
        ]

        all_items = []
        for feed_url in feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:3]:
                    item = NewsItem(
                        headline=entry.title,
                        source=feed.feed.get("title", "RSS"),
                        published=datetime.now(),
                        url=entry.link,
                        content=entry.get("summary", ""),
                        symbols=[],
                        sectors=[],
                        indices=[],
                        impact_score=0.0,
                        severity=ImpactSeverity.NONE,
                        category=NewsCategory.SENTIMENT,
                        time_horizon="overnight",
                        reasoning=""
                    )
                    all_items.append(item)
            except Exception as e:
                logger.debug(f"RSS fetch failed for {feed_url}: {e}")

        return all_items

    def _deduplicate(self, items: List[NewsItem]) -> List[NewsItem]:
        """Remove duplicate/similar headlines"""
        seen = set()
        unique = []

        for item in items:
            # Simple dedup: first 10 words hash
            key = " ".join(item.headline.split()[:10]).lower()
            if key not in seen:
                seen.add(key)
                unique.append(item)

        return unique

    async def fetch_geopolitical_only(self) -> List[NewsItem]:
        """Fetch only high-severity geopolitical news"""
        all_news = await self.fetch_all_sources()
        return [n for n in all_news if n.severity.value >= ImpactSeverity.HIGH.value]

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


class NewsContextProvider:
    """
    Singleton provider for news context to agents.
    Maintains current news state and serves queries from agents.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.analyzer = GeminiNewsAnalyzer()
        self.fetcher = NewsFetcher(self.analyzer)
        self._current_news: List[NewsItem] = []
        self._overnight_profile: Optional[OvernightRiskProfile] = None
        self._last_update: Optional[datetime] = None
        self._initialized = True

    async def update(self):
        """Fetch and analyze latest news"""
        self._current_news = await self.fetcher.fetch_all_sources()
        self._last_update = datetime.now()
        logger.info(f"News context updated: {len(self._current_news)} items")

    async def update_overnight_profile(self, market_context: Dict):
        """Generate overnight risk assessment"""
        self._overnight_profile = await self.analyzer.generate_overnight_assessment(
            self._current_news, market_context
        )
        logger.info(f"Overnight profile updated: {self._overnight_profile.risk_level}")

    def get_symbol_context(self, symbol: str) -> Dict:
        """Get news context for specific symbol"""
        relevant = [
            n for n in self._current_news
            if symbol in n.symbols or any(symbol in s for s in n.sectors)
        ]

        if not relevant:
            return {"score": 0, "headline": None, "severity": "NONE"}

        avg_impact = sum(n.impact_score for n in relevant) / len(relevant)
        worst = max(relevant, key=lambda x: abs(x.impact_score))

        return {
            "score": avg_impact,
            "headline": worst.headline,
            "severity": worst.severity.name,
            "reasoning": worst.reasoning,
            "count": len(relevant)
        }

    def get_sector_context(self, sector: str) -> Dict:
        """Get news context for sector"""
        relevant = [n for n in self._current_news if sector in n.sectors]
        if not relevant:
            return {"score": 0, "count": 0}

        return {
            "score": sum(n.impact_score for n in relevant) / len(relevant),
            "count": len(relevant),
            "headlines": [n.headline for n in relevant[:3]]
        }

    def get_overnight_risk(self) -> Optional[OvernightRiskProfile]:
        return self._overnight_profile

    def get_all_news(self) -> List[NewsItem]:
        return self._current_news

    def get_critical_news(self) -> List[NewsItem]:
        return [n for n in self._current_news if n.severity.value >= ImpactSeverity.HIGH.value]


# Convenience function for initialization
def get_news_context() -> NewsContextProvider:
    return NewsContextProvider()