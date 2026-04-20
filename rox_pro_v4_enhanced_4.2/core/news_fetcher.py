"""
ROX Proven Edge Engine v3.0 - News Fetcher Module
================================================
Handles fetching of geopolitical and stock price change news.
"""

import os
import sys
import json
import hashlib
import logging
import asyncio
import aiohttp
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import re

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core.config import NewsConfig, BASE_DIR
    from core.logger import get_logger, log_async_execution
except ImportError:
    from config import NewsConfig, BASE_DIR
    from logger import get_logger, log_async_execution


logger = get_logger("NewsFetcher")


@dataclass
class NewsArticle:
    """Represents a news article."""
    title: str
    summary: str
    source: str
    url: str
    published_date: datetime
    category: str = "general"
    sentiment_score: float = 0.0
    relevance_score: float = 0.0
    keywords_matched: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            "published_date": self.published_date.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "NewsArticle":
        data["published_date"] = datetime.fromisoformat(data["published_date"])
        return cls(**data)


@dataclass
class GeopoliticalEvent:
    """Represents a significant geopolitical event."""
    title: str
    description: str
    event_type: str  # conflict, trade, election, diplomatic, crisis
    severity: str  # low, medium, high, critical
    countries_involved: List[str]
    market_impact: str  # positive, negative, neutral, uncertain
    affected_sectors: List[str]
    affected_stocks: List[str]
    source: str
    url: str
    published_date: datetime
    sentiment_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            "published_date": self.published_date.isoformat(),
        }


@dataclass
class StockPriceNews:
    """Represents stock price-related news."""
    stock: str
    company_name: str
    title: str
    summary: str
    news_type: str  # earnings, corporate_action, price_movement, analyst_rating
    price_impact: str  # positive, negative, neutral
    source: str
    url: str
    published_date: datetime
    price_change_pct: float = 0.0
    volume_change_pct: float = 0.0
    sentiment_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            "published_date": self.published_date.isoformat(),
        }


class NewsCache:
    """Simple file-based cache for news articles."""
    
    def __init__(self, cache_dir: Path = None, duration_minutes: int = 30):
        self.cache_dir = cache_dir or (BASE_DIR / "data" / "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.duration_minutes = duration_minutes
    
    def _get_cache_key(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()
    
    def _get_cache_file(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"
    
    def get(self, url: str) -> Optional[List[Dict]]:
        """Get cached data if valid."""
        cache_file = self._get_cache_file(self._get_cache_key(url))
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            cached_time = datetime.fromisoformat(data['cached_at'])
            if datetime.now() - cached_time < timedelta(minutes=self.duration_minutes):
                return data['articles']
            
            return None
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None
    
    def set(self, url: str, articles: List[Dict]):
        """Cache articles."""
        cache_file = self._get_cache_file(self._get_cache_key(url))
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'cached_at': datetime.now().isoformat(),
                    'url': url,
                    'articles': articles
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")


class NewsFetcher:
    """
    Main news fetching service.
    
    Supports:
    - RSS feed parsing
    - NewsAPI integration (optional)
    - Geopolitical news
    - Stock price change news
    """
    
    # Severity keywords for geopolitical events
    SEVERITY_KEYWORDS = {
        "critical": ["war", "invasion", "nuclear", "military strike", "terror"],
        "high": [
            # Escalation events
            "sanctions", "conflict", "crisis", "embargo", "attack",
            # FIX-NEWS-03: De-escalation / resolution events are equally HIGH magnitude
            # because they cause sharp macro reversals (crude crash, VIX collapse, risk-on rally).
            # These were previously falling through to LOW/MEDIUM, causing the news layer
            # to rate NEUTRAL on days with dominant bullish macro catalysts.
            "ceasefire", "de-escalation", "peace talks", "truce", "resolution",
            "postpone strike", "halt attack", "Iran deal", "diplomacy breakthrough",
            "crude plunge", "crude crash", "oil plunge", "oil crash", "crude drop",
            "fed pivot", "rate cut surprise", "inflation falls", "cpi miss",
        ],
        "medium": ["tension", "dispute", "tariff", "election", "treaty"],
        "low": ["summit", "meeting", "diplomatic", "negotiation"]
    }
    
    # Country to market impact mapping
    COUNTRY_IMPACT = {
        "china": {"negative": ["telecom", "pharma", "auto"], "positive": ["it", "chemicals"]},
        "usa": {"negative": ["it", "pharma"], "positive": ["banking"]},
        "russia": {"negative": ["energy", "metals"], "positive": []},
        "middle east": {"negative": ["energy", "aviation"], "positive": []},
    }
    
    # Stock name variations
    STOCK_NAMES = {
        "reliance": "RELIANCE",
        "tcs": "TCS",
        "tata consultancy": "TCS",
        "infosys": "INFY",
        "hdfc bank": "HDFCBANK",
        "icici bank": "ICICIBANK",
        "sb i": "SBIN",
        "state bank": "SBIN",
        "tata motors": "TATAMOTORS",
        "tata steel": "TATASTEEL",
        "bharti airtel": "BHARTIARTL",
        "airtel": "BHARTIARTL",
        "wipro": "WIPRO",
        "hcl tech": "HCLTECH",
        "sun pharma": "SUNPHARMA",
        "maruti": "MARUTI",
        "bajaj finance": "BAJFINANCE",
        "adani": "ADANIENT",
        "l&t": "LT",
        "larsen": "LT",
        "ongc": "ONGC",
        "ntpc": "NTPC",
        "power grid": "POWERGRID",
    }
    
    def __init__(self, config: NewsConfig = None):
        self.config = config or NewsConfig()
        self.cache = NewsCache(duration_minutes=self.config.cache_duration_minutes)
        self.session = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def close(self):
        """Close HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    @log_async_execution
    async def fetch_rss_feed(self, url: str) -> List[NewsArticle]:
        """Fetch and parse an RSS feed."""
        # Check cache first
        cached = self.cache.get(url)
        if cached:
            logger.debug(f"Using cached data for {url}")
            return [NewsArticle.from_dict(a) for a in cached]
        
        articles = []
        
        try:
            session = await self._get_session()
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    content = await response.text()
                    articles = self._parse_rss(content, url)
                    
                    # Cache the results
                    self.cache.set(url, [a.to_dict() for a in articles])
                else:
                    logger.warning(f"RSS fetch failed for {url}: Status {response.status}")
        
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching RSS feed: {url}")
        except Exception as e:
            logger.error(f"Error fetching RSS feed {url}: {e}")
        
        return articles
    
    def _parse_rss(self, content: str, source_url: str) -> List[NewsArticle]:
        """Parse RSS XML content."""
        articles = []
        
        try:
            root = ET.fromstring(content)
            
            # Find channel
            channel = root.find('channel')
            if channel is None:
                return articles
            
            source_name = channel.findtext('title', 'Unknown Source')
            
            # Find items
            for item in channel.findall('item'):
                title = item.findtext('title', '')
                description = item.findtext('description', '')
                link = item.findtext('link', '')
                pub_date_str = item.findtext('pubDate', '')
                
                # Parse date
                try:
                    # Try various date formats
                    for fmt in [
                        '%a, %d %b %Y %H:%M:%S %z',
                        '%a, %d %b %Y %H:%M:%S GMT',
                        '%Y-%m-%dT%H:%M:%SZ',
                        '%Y-%m-%dT%H:%M:%S%z',
                    ]:
                        try:
                            pub_date = datetime.strptime(pub_date_str.strip(), fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        pub_date = datetime.now()
                except:
                    pub_date = datetime.now()
                
                if title and link:
                    # Extract source from URL
                    source = source_name
                    if 'economictimes' in source_url.lower():
                        source = 'Economic Times'
                    elif 'moneycontrol' in source_url.lower():
                        source = 'Moneycontrol'
                    elif 'livemint' in source_url.lower():
                        source = 'LiveMint'
                    elif 'bbc' in source_url.lower():
                        source = 'BBC News'
                    elif 'nytimes' in source_url.lower():
                        source = 'NY Times'
                    elif 'reuters' in source_url.lower():
                        source = 'Reuters'
                    elif 'timesofindia' in source_url.lower():
                        source = 'Times of India'
                    
                    articles.append(NewsArticle(
                        title=self._clean_text(title),
                        summary=self._clean_text(description),
                        source=source,
                        url=link,
                        published_date=pub_date
                    ))
        
        except ET.ParseError as e:
            logger.error(f"RSS parse error: {e}")
        
        return articles
    
    def _clean_text(self, text: str) -> str:
        """Clean HTML and special characters from text."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove CDATA
        text = re.sub(r'<!\[CDATA\[|\]\]>', '', text)
        # Decode HTML entities
        import html
        text = html.unescape(text)
        # Clean whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    def _calculate_relevance(self, article: NewsArticle, keywords: List[str]) -> Tuple[float, List[str]]:
        """Calculate relevance score and find matching keywords."""
        text = f"{article.title} {article.summary}".lower()
        matched = []
        score = 0.0
        
        for keyword in keywords:
            if keyword.lower() in text:
                matched.append(keyword)
                # Higher weight for title matches
                if keyword.lower() in article.title.lower():
                    score += 2.0
                else:
                    score += 1.0
        
        return score, matched
    
    def _determine_severity(self, text: str) -> str:
        """Determine severity level of geopolitical event."""
        text = text.lower()
        
        for severity, keywords in self.SEVERITY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    return severity
        
        return "low"
    
    def _extract_countries(self, text: str) -> List[str]:
        """Extract mentioned countries from text."""
        countries = []
        country_keywords = [
            "china", "usa", "united states", "russia", "india", "pakistan",
            "iran", "israel", "ukraine", "japan", "uk", "britain",
            "germany", "france", "saudi", "uae", "middle east", "europe",
            "asia", "africa", "north korea", "south korea", "taiwan"
        ]
        
        text = text.lower()
        for country in country_keywords:
            if country in text:
                countries.append(country)
        
        return list(set(countries))
    
    def _determine_market_impact(self, countries: List[str], severity: str) -> str:
        """Determine potential market impact."""
        if severity in ["critical", "high"]:
            return "negative"
        
        for country in countries:
            if country in self.COUNTRY_IMPACT:
                return "uncertain"
        
        return "neutral"
    
    def _extract_stocks(self, text: str) -> List[str]:
        """Extract mentioned stocks from text."""
        stocks = []
        text_lower = text.lower()
        
        for name, symbol in self.STOCK_NAMES.items():
            if name in text_lower:
                stocks.append(symbol)
        
        return list(set(stocks))
    
    @log_async_execution
    async def fetch_geopolitical_news(self) -> List[GeopoliticalEvent]:
        """Fetch and process geopolitical news."""
        logger.info("Fetching geopolitical news...")
        
        all_articles = []
        tasks = [
            self.fetch_rss_feed(url)
            for url in self.config.geopolitical_sources
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Error in fetch task: {result}")
        
        # Filter and convert to geopolitical events
        events = []
        for article in all_articles:
            score, keywords = self._calculate_relevance(
                article, self.config.geopolitical_keywords
            )
            
            if score > 0:
                full_text = f"{article.title} {article.summary}"
                
                event = GeopoliticalEvent(
                    title=article.title,
                    description=article.summary,
                    event_type=self._classify_event_type(full_text),
                    severity=self._determine_severity(full_text),
                    countries_involved=self._extract_countries(full_text),
                    market_impact="uncertain",
                    affected_sectors=[],
                    affected_stocks=[],
                    source=article.source,
                    url=article.url,
                    published_date=article.published_date,
                    sentiment_score=self._calculate_sentiment(full_text)
                )
                
                # Determine market impact
                event.market_impact = self._determine_market_impact(
                    event.countries_involved, event.severity
                )
                
                # Determine affected sectors
                event.affected_sectors = self._get_affected_sectors(
                    event.countries_involved, event.market_impact
                )
                
                # Extract mentioned stocks
                event.affected_stocks = self._extract_stocks(full_text)
                
                events.append(event)
        
        # Sort by severity and date
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        events.sort(key=lambda x: (
            severity_order.get(x.severity, 4),
            -x.published_date.timestamp()
        ))
        
        logger.info(f"Fetched {len(events)} geopolitical events")
        return events[:self.config.max_articles_per_fetch]
    
    def _classify_event_type(self, text: str) -> str:
        """Classify the type of geopolitical event."""
        text = text.lower()
        
        if any(w in text for w in ["war", "conflict", "military", "attack"]):
            return "conflict"
        elif any(w in text for w in ["trade", "tariff", "sanction", "embargo"]):
            return "trade"
        elif any(w in text for w in ["election", "vote", "poll"]):
            return "election"
        elif any(w in text for w in ["treaty", "summit", "meeting", "diplomatic"]):
            return "diplomatic"
        else:
            return "crisis"
    
    def _get_affected_sectors(self, countries: List[str], impact: str) -> List[str]:
        """Get sectors affected by event."""
        sectors = set()
        
        for country in countries:
            if country in self.COUNTRY_IMPACT:
                if impact == "negative":
                    sectors.update(self.COUNTRY_IMPACT[country].get("negative", []))
        
        return list(sectors)
    
    def _calculate_sentiment(self, text: str) -> float:
        """Calculate simple sentiment score."""
        positive_words = [
            "growth", "surge", "rally", "gain", "rise", "positive",
            "boost", "strong", "success", "recovery", "optimistic"
        ]
        negative_words = [
            "crash", "fall", "drop", "decline", "loss", "negative",
            "crisis", "threat", "risk", "fear", "concern", "weak"
        ]
        
        text = text.lower()
        positive = sum(1 for w in positive_words if w in text)
        negative = sum(1 for w in negative_words if w in text)
        
        total = positive + negative
        if total == 0:
            return 0.0
        
        return (positive - negative) / total * 100  # -100 to +100
    
    @log_async_execution
    async def fetch_stock_price_news(self, stocks: List[str] = None) -> List[StockPriceNews]:
        """Fetch stock price change related news."""
        logger.info("Fetching stock price news...")
        
        all_articles = []
        tasks = [
            self.fetch_rss_feed(url)
            for url in self.config.stock_news_sources
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Error in fetch task: {result}")
        
        # Filter and convert to stock price news
        news_items = []
        for article in all_articles:
            score, keywords = self._calculate_relevance(
                article, self.config.stock_keywords
            )
            
            if score > 0:
                full_text = f"{article.title} {article.summary}"
                
                # Try to extract stock symbol
                mentioned_stocks = self._extract_stocks(full_text)
                primary_stock = mentioned_stocks[0] if mentioned_stocks else "MARKET"
                
                news = StockPriceNews(
                    stock=primary_stock,
                    company_name=self._get_company_name(primary_stock),
                    title=article.title,
                    summary=article.summary,
                    news_type=self._classify_stock_news(full_text),
                    price_impact=self._determine_price_impact(full_text),
                    source=article.source,
                    url=article.url,
                    published_date=article.published_date,
                    sentiment_score=self._calculate_sentiment(full_text)
                )
                
                news_items.append(news)
        
        # Filter by specific stocks if provided
        if stocks:
            news_items = [
                n for n in news_items 
                if n.stock in stocks or n.stock == "MARKET"
            ]
        
        # Sort by date
        news_items.sort(key=lambda x: x.published_date, reverse=True)
        
        logger.info(f"Fetched {len(news_items)} stock news items")
        return news_items[:self.config.max_articles_per_fetch]
    
    def _get_company_name(self, symbol: str) -> str:
        """Get company name from symbol."""
        names = {
            "RELIANCE": "Reliance Industries",
            "TCS": "Tata Consultancy Services",
            "INFY": "Infosys",
            "HDFCBANK": "HDFC Bank",
            "ICICIBANK": "ICICI Bank",
            "SBIN": "State Bank of India",
            "TATAMOTORS": "Tata Motors",
            "TATASTEEL": "Tata Steel",
            "BHARTIARTL": "Bharti Airtel",
            "WIPRO": "Wipro",
            "HCLTECH": "HCL Technologies",
            "SUNPHARMA": "Sun Pharma",
            "MARUTI": "Maruti Suzuki",
            "BAJFINANCE": "Bajaj Finance",
            "ADANIENT": "Adani Enterprises",
            "LT": "Larsen & Toubro",
            "MARKET": "General Market"
        }
        return names.get(symbol, symbol)
    
    def _classify_stock_news(self, text: str) -> str:
        """Classify the type of stock news."""
        text = text.lower()
        
        if any(w in text for w in ["result", "earning", "profit", "revenue"]):
            return "earnings"
        elif any(w in text for w in ["dividend", "bonus", "split", "buyback"]):
            return "corporate_action"
        elif any(w in text for w in ["rating", "upgrade", "downgrade", "analyst"]):
            return "analyst_rating"
        else:
            return "price_movement"
    
    def _determine_price_impact(self, text: str) -> str:
        """Determine likely price impact."""
        text = text.lower()
        
        positive_signals = [
            "surge", "rally", "gain", "rise", "jump", "soar", "climb",
            "up", "higher", "boost", "positive", "buy", "upgrade"
        ]
        negative_signals = [
            "fall", "drop", "decline", "crash", "slump", "sink", "tumble",
            "down", "lower", "negative", "sell", "downgrade", "loss"
        ]
        
        positive_count = sum(1 for s in positive_signals if s in text)
        negative_count = sum(1 for s in negative_signals if s in text)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    async def get_market_news_summary(self) -> Dict[str, Any]:
        """Get comprehensive market news summary."""
        geopolitical_events = await self.fetch_geopolitical_news()
        stock_news = await self.fetch_stock_price_news()
        
        # Calculate overall sentiment
        geo_sentiment = sum(e.sentiment_score for e in geopolitical_events) / max(1, len(geopolitical_events))
        stock_sentiment = sum(n.sentiment_score for n in stock_news) / max(1, len(stock_news))
        
        return {
            "timestamp": datetime.now().isoformat(),
            "geopolitical": {
                "count": len(geopolitical_events),
                "critical_events": [e for e in geopolitical_events if e.severity in ["critical", "high"]],
                "overall_sentiment": geo_sentiment,
                "events": [e.to_dict() for e in geopolitical_events[:10]]
            },
            "stock_news": {
                "count": len(stock_news),
                "overall_sentiment": stock_sentiment,
                "news_items": [n.to_dict() for n in stock_news[:10]]
            },
            "combined_sentiment": (geo_sentiment + stock_sentiment) / 2,
            "market_impact_assessment": self._assess_overall_impact(geopolitical_events, stock_news)
        }
    
    def _assess_overall_impact(self, geopolitical: List[GeopoliticalEvent], 
                                stock_news: List[StockPriceNews]) -> Dict:
        """Assess overall market impact."""
        impact = {
            "direction": "neutral",
            "confidence": 50,
            "key_factors": [],
            "warnings": []
        }
        
        # Check geopolitical risks
        critical_events = [e for e in geopolitical if e.severity in ["critical", "high"]]
        if critical_events:
            impact["warnings"].extend([
                f"Geopolitical risk: {e.title[:50]}..."
                for e in critical_events[:3]
            ])
            impact["direction"] = "cautious"
        
        # Check stock news sentiment
        positive_news = len([n for n in stock_news if n.price_impact == "positive"])
        negative_news = len([n for n in stock_news if n.price_impact == "negative"])
        
        if positive_news > negative_news * 1.5:
            impact["direction"] = "positive"
            impact["confidence"] = 60 + min(20, (positive_news - negative_news) * 5)
        elif negative_news > positive_news * 1.5:
            impact["direction"] = "negative"
            impact["confidence"] = 60 + min(20, (negative_news - positive_news) * 5)
        
        # Key factors
        if geopolitical:
            for event in geopolitical[:3]:
                if event.countries_involved:
                    impact["key_factors"].append(
                        f"{event.event_type.title()} - {', '.join(event.countries_involved[:2])}"
                    )
        
        return impact


# Synchronous wrapper for convenience
def fetch_news_sync(news_type: str = "all") -> Dict:
    """
    Synchronous wrapper for news fetching.
    
    Args:
        news_type: "geopolitical", "stock", or "all"
        
    Returns:
        Dictionary with news results
    """
    async def _fetch():
        fetcher = NewsFetcher()
        try:
            if news_type == "geopolitical":
                events = await fetcher.fetch_geopolitical_news()
                return {"geopolitical_events": [e.to_dict() for e in events]}
            elif news_type == "stock":
                news = await fetcher.fetch_stock_price_news()
                return {"stock_news": [n.to_dict() for n in news]}
            else:
                return await fetcher.get_market_news_summary()
        finally:
            await fetcher.close()
    
    return asyncio.run(_fetch())


if __name__ == "__main__":
    # Test the news fetcher
    print("Testing News Fetcher...")
    print("=" * 50)
    
    result = fetch_news_sync("all")
    
    print(f"\nGeopolitical Events: {result.get('geopolitical', {}).get('count', 0)}")
    print(f"Stock News: {result.get('stock_news', {}).get('count', 0)}")
    print(f"Combined Sentiment: {result.get('combined_sentiment', 0):.2f}")
    print(f"Market Impact: {result.get('market_impact_assessment', {})}")
