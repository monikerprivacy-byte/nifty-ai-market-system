import json, logging
from urllib.parse import quote_plus

logger = logging.getLogger("browser_research")

class BrowserResearch:
    def __init__(self):
        self._browser_available = False
        self._check_available()

    def _check_available(self):
        try:
            from playwright.async_api import async_playwright
            self._browser_available = True
        except ImportError:
            self._browser_available = False

    async def search(self, query):
        """Search the web. Uses DuckDuckGo API directly, Playwright as fallback."""
        results = await self._search_ddg_api(query)
        if results:
            return results

        if self._browser_available:
            return await self._search_playwright(query)
        return [{"title": "Search unavailable", "url": "", "snippet": "No search backend available"}]

    async def _search_ddg_api(self, query):
        try:
            from duckduckgo_search import DDGS
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=5):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                    })
            return results
        except Exception as e:
            logger.warning(f"DDGS search failed: {e}")
            return []

    async def _search_playwright(self, query):
        try:
            from playwright.async_api import async_playwright
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True,
                    args=["--disable-blink-features=AutomationControlled"])
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
                )
                page = await context.new_page()
                await page.add_init_script("""
                    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                """)
                await page.goto(f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}",
                    wait_until="domcontentloaded", timeout=15000)
                await page.wait_for_timeout(2000)
                results = []
                links = await page.query_selector_all("a[rel='nofollow']")
                for link in links[:5]:
                    title = await link.inner_text()
                    href = await link.get_attribute("href") or ""
                    results.append({"title": title.strip(), "url": href, "snippet": ""})
                await browser.close()
                return results
        except Exception as e:
            logger.warning(f"Playwright search failed: {e}")
            return []

    async def fetch_page_text(self, url):
        """Get text content from a URL using Playwright"""
        if not self._browser_available:
            return {"error": "Browser not available"}
        try:
            from playwright.async_api import async_playwright
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=20000)
                await page.wait_for_timeout(2000)
                text = await page.inner_text("body")
                await browser.close()
                return {"url": url, "text": text[:5000], "title": await page.title()}
        except Exception as e:
            return {"error": str(e)}
