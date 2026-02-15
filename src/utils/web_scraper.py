"""
Web Scraper - Extract content from URLs
Supports both requests (fast) and Selenium (full JS rendering)
Includes multilingual support with dynamic Accept-Language headers
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
from loguru import logger
import requests
from bs4 import BeautifulSoup
import time
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.multilingual_config import get_multilingual_config


@dataclass
class ScrapedContent:
    """Scraped web content"""
    url: str
    title: Optional[str]
    text: str
    html: str
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    detected_language: Optional[str] = None


class WebScraper:
    """
    Web content scraper with multiple backends:
    - requests + BeautifulSoup (fast, free)
    - Selenium (slower, full JS support)

    Supports multilingual scraping with dynamic Accept-Language headers.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, language: str = 'en'):
        """
        Initialize web scraper.

        Args:
            config: Scraping configuration from api_config.yaml
            language: Language code for Accept-Language header (default: 'en')
        """
        self.config = config or {}
        self.method = self.config.get('method', 'requests')
        self.language = language
        self.ml_config = get_multilingual_config()

        # Base headers
        self.base_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        # Set headers with language-specific Accept-Language
        self.headers = self._get_headers_for_language(language)

        self.timeout = self.config.get('requests', {}).get('timeout', 10)

    def _get_headers_for_language(self, language: str) -> Dict[str, str]:
        """Get headers with appropriate Accept-Language for the language"""
        headers = self.base_headers.copy()
        accept_lang = self.ml_config.get_accept_language(language)
        headers['Accept-Language'] = accept_lang
        return headers

    def set_language(self, language: str):
        """
        Set the language for scraping.

        Args:
            language: ISO 639-1 language code
        """
        if language != self.language:
            self.language = language
            self.headers = self._get_headers_for_language(language)
            logger.debug(f"WebScraper language set to: {language}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def scrape(self, url: str, extract_main_content: bool = True) -> ScrapedContent:
        """
        Scrape content from URL.

        Args:
            url: URL to scrape
            extract_main_content: If True, extract only main content (remove ads, nav, etc.)

        Returns:
            Scraped content with text and metadata
        """
        logger.info(f"Scraping: {url}")

        try:
            if self.method == 'selenium':
                return self._scrape_selenium(url, extract_main_content)
            else:
                return self._scrape_requests(url, extract_main_content)

        except Exception as e:
            logger.error(f"Scraping failed for {url}: {e}")
            return ScrapedContent(
                url=url,
                title=None,
                text='',
                html='',
                success=False,
                error=str(e)
            )

    def _scrape_requests(self, url: str, extract_main: bool) -> ScrapedContent:
        """Scrape using requests + BeautifulSoup (fast)"""
        response = requests.get(url, headers=self.headers, timeout=self.timeout)
        response.raise_for_status()

        html = response.text
        soup = BeautifulSoup(html, 'lxml')

        # Extract title
        title = None
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()

        # Extract main content
        if extract_main:
            text = self._extract_main_content(soup)
        else:
            text = soup.get_text(separator=' ', strip=True)

        metadata = {
            'content_type': response.headers.get('Content-Type'),
            'content_length': len(text),
            'status_code': response.status_code
        }

        return ScrapedContent(
            url=url,
            title=title,
            text=text,
            html=html,
            success=True,
            metadata=metadata
        )

    def _scrape_selenium(self, url: str, extract_main: bool) -> ScrapedContent:
        """Scrape using Selenium (slower, full JS support)"""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC

            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument(f'user-agent={self.headers["User-Agent"]}')

            driver = webdriver.Chrome(options=options)

            try:
                driver.get(url)

                # Wait for page load
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, 'body'))
                )

                # Give time for JS to execute
                time.sleep(2)

                title = driver.title
                html = driver.page_source
                soup = BeautifulSoup(html, 'lxml')

                if extract_main:
                    text = self._extract_main_content(soup)
                else:
                    text = soup.get_text(separator=' ', strip=True)

                return ScrapedContent(
                    url=url,
                    title=title,
                    text=text,
                    html=html,
                    success=True,
                    metadata={'method': 'selenium'}
                )

            finally:
                driver.quit()

        except ImportError:
            logger.warning("Selenium not installed, falling back to requests")
            return self._scrape_requests(url, extract_main)

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """
        Extract main content from HTML, removing navigation, ads, etc.

        Uses common content selectors and heuristics.
        """
        # Remove unwanted elements
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement']):
            tag.decompose()

        # Try common main content selectors
        main_selectors = [
            'article',
            'main',
            '[role="main"]',
            '.article-content',
            '.post-content',
            '.entry-content',
            '#content',
            '.content'
        ]

        for selector in main_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                text = main_content.get_text(separator=' ', strip=True)
                if len(text) > 200:  # Reasonable content length
                    return text

        # Fallback: Extract all paragraphs
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text(strip=True) for p in paragraphs])

        if len(text) > 100:
            return text

        # Last resort: All text
        return soup.get_text(separator=' ', strip=True)

    def extract_passages(
        self,
        content: ScrapedContent,
        query: str,
        max_passages: int = 3,
        min_length: int = 50
    ) -> list[str]:
        """
        Extract relevant passages related to a query.

        Args:
            content: Scraped content
            query: Search query/subclaim
            max_passages: Maximum passages to return
            min_length: Minimum passage length (characters)

        Returns:
            List of relevant text passages
        """
        if not content.success or not content.text:
            return []

        # Split into sentences/paragraphs
        soup = BeautifulSoup(content.html, 'lxml')
        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]
        paragraphs = [p for p in paragraphs if len(p) >= min_length]

        # Simple relevance scoring based on keyword overlap
        query_words = set(query.lower().split())

        scored_paragraphs = []
        for para in paragraphs:
            para_words = set(para.lower().split())
            overlap = len(query_words & para_words)
            if overlap > 0:
                scored_paragraphs.append((overlap, para))

        # Sort by relevance and return top passages
        scored_paragraphs.sort(reverse=True, key=lambda x: x[0])
        passages = [para for score, para in scored_paragraphs[:max_passages]]

        logger.info(f"Extracted {len(passages)} relevant passages from {content.url}")
        return passages
