"""News Fetcher — fetches market news for stocks and indices.
Uses DuckDuckGo API (no API key needed) + optional Playwright fallback."""

import logging, asyncio
from datetime import datetime, timedelta

logger = logging.getLogger("news_fetcher")

class NewsFetcher:
    def __init__(self):
        self._cache = {}
        self._cache_ttl = timedelta(hours=2)

    async def search_news(self, query, max_results=5):
        """Search news for a stock or market query."""
        cache_key = f"{query}:{max_results}"
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if datetime.now() - entry["time"] < self._cache_ttl:
                return entry["results"]

        results = await self._search_ddg(f"{query} stock market news", max_results)
        self._cache[cache_key] = {"results": results, "time": datetime.now()}
        return results

    async def search_headlines(self, max_results=8):
        """Fetch general market headlines."""
        return await self._search_ddg("Indian stock market today Nifty Sensex", max_results)

    async def stock_news(self, ticker, max_results=4):
        """Fetch news specific to a stock."""
        news = await self._search_ddg(f"{ticker} NSE stock news today", max_results)
        if not news:
            news = await self._search_ddg(f"{ticker} share price latest", max_results)
        return news

    async def sector_news(self, sector, max_results=4):
        """Fetch news for a sector."""
        return await self._search_ddg(f"{sector} sector India stock market", max_results)

    async def _search_ddg(self, query, max_results):
        try:
            from duckduckgo_search import DDGS
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                    })
            return results
        except Exception as e:
            logger.warning(f"DDG search failed for '{query[:30]}': {e}")
            return []


# Singleton
_instance = None

def get_news_fetcher():
    global _instance
    if _instance is None:
        _instance = NewsFetcher()
    return _instance
