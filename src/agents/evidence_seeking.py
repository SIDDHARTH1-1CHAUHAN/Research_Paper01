"""
Evidence Seeking Agent - 3-stage evidence retrieval pipeline
Stage 1: Internet Search
Stage 2: Credibility Check
Stage 3: Content Extraction

Supports multilingual search with language-specific regions and credibility sources.
Uses configurable prompts from prompts.yaml for LLM-based content extraction.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger
from datetime import datetime
import sys
sys.path.append('..')
from src.utils.web_scraper import WebScraper
from src.utils.credibility_checker import CredibilityChecker
from src.utils.multilingual_config import get_multilingual_config
from src.utils.prompts_loader import get_prompts_loader


@dataclass
class Evidence:
    """Single piece of evidence"""
    id: str
    subclaim_id: str
    source_url: str
    source_name: str
    passage: str
    credibility_score: float
    credibility_level: str
    retrieved_at: str
    metadata: Dict[str, Any]
    language: str = 'en'


@dataclass
class EvidenceResult:
    """Result from Evidence Seeking Agent"""
    subclaim_id: str
    evidence: List[Evidence]
    total_sources: int
    high_credibility_count: int
    language: str = 'en'


class EvidenceSeekingAgent:
    """
    Agent 3: Evidence Seeking

    Three-stage pipeline:
    1. Internet Search (DuckDuckGo/SerperAPI)
    2. Credibility Assessment (Heuristic/MBFC)
    3. Content Extraction (BeautifulSoup)

    Supports multilingual search with language-specific regions.
    """

    def __init__(self, config: Dict[str, Any] = None, language: str = 'en', llm_interface=None):
        """
        Initialize Evidence Seeking Agent.

        Args:
            config: Configuration dictionary
            language: Language code for search region and credibility (default: 'en')
            llm_interface: Optional LLM interface for intelligent content extraction
        """
        self.config = config or {}
        self.language = language
        self.llm = llm_interface
        self.ml_config = get_multilingual_config()

        # Initialize components
        self.scraper = WebScraper(self.config.get('scraping', {}), language=language)
        self.credibility = CredibilityChecker(self.config.get('credibility', {}), language=language)

        # Configuration
        self.max_results = self.config.get('max_search_results', 10)
        self.credibility_threshold = self.config.get('credibility_threshold', 'medium')
        self.max_passages = self.config.get('max_passages_per_source', 3)

        # Get search region for language
        self.search_region = self.ml_config.get_search_region(language)

        logger.info(f"Evidence Seeking Agent initialized (lang={language}, region={self.search_region})")

    def set_language(self, language: str):
        """
        Set the language for evidence seeking.

        Args:
            language: ISO 639-1 language code
        """
        if language != self.language:
            self.language = language
            self.search_region = self.ml_config.get_search_region(language)
            self.scraper.set_language(language)
            self.credibility.set_language(language)
            logger.info(f"Evidence Seeking Agent language set to: {language}")

    def process(self, query_results: List[Any], language: Optional[str] = None) -> List[EvidenceResult]:
        """
        Retrieve evidence for all queries.

        Args:
            query_results: List of QueryResult objects
            language: Optional language override

        Returns:
            List of EvidenceResult objects
        """
        # Update language if provided
        if language:
            self.set_language(language)

        logger.info(f"Seeking evidence for {len(query_results)} subclaims (lang={self.language})")

        results = []
        for qr in query_results:
            evidence_list = []
            evidence_id = 1

            # Get query language if available
            query_lang = getattr(qr, 'language', self.language)

            for query in qr.queries:
                # Stage 1: Search (with language-specific region)
                search_results = self._search_web(query, language=query_lang)

                # Stage 2 & 3: Check credibility and extract content
                for url in search_results[:self.max_results]:
                    try:
                        # Credibility check
                        cred_score = self.credibility.check(url)

                        if not self.credibility.meets_threshold(cred_score, self.credibility_threshold):
                            logger.debug(f"Filtered low credibility source: {url}")
                            continue

                        # Content extraction
                        content = self.scraper.scrape(url)

                        if not content.success:
                            continue

                        # Extract relevant passages
                        passages = self.scraper.extract_passages(
                            content,
                            qr.subclaim_text,
                            max_passages=self.max_passages
                        )

                        for passage in passages:
                            evidence = Evidence(
                                id=f"EV{evidence_id}",
                                subclaim_id=qr.subclaim_id,
                                source_url=url,
                                source_name=content.title or url,
                                passage=passage,
                                credibility_score=cred_score.score,
                                credibility_level=cred_score.level.value,
                                retrieved_at=datetime.utcnow().isoformat(),
                                metadata=cred_score.details,
                                language=query_lang
                            )
                            evidence_list.append(evidence)
                            evidence_id += 1

                    except Exception as e:
                        logger.error(f"Error processing {url}: {e}")
                        continue

            high_cred_count = len([e for e in evidence_list if e.credibility_level == 'high'])

            results.append(EvidenceResult(
                subclaim_id=qr.subclaim_id,
                evidence=evidence_list,
                total_sources=len(evidence_list),
                high_credibility_count=high_cred_count,
                language=query_lang
            ))

            logger.info(f"Found {len(evidence_list)} evidence items for {qr.subclaim_id} ({high_cred_count} high credibility)")

        return results

    def _search_web(self, query: str, language: str = 'en') -> List[str]:
        """
        Search the web for a query.

        Args:
            query: Search query
            language: Language code for region-specific search

        Returns:
            List of URLs
        """
        # Get search region for the language
        region = self.ml_config.get_search_region(language)

        # For demo, return mock URLs based on query keywords
        # In production, use DuckDuckGo or SerperAPI with region parameter
        logger.info(f"Searching for: {query} (region={region})")

        # Mock search results with language-appropriate Wikipedia
        wiki_lang = language if language != 'zh' else 'zh'
        mock_results = [
            f"https://{wiki_lang}.wikipedia.org/wiki/" + query.replace(' ', '_'),
            "https://www.britannica.com/topic/" + query.replace(' ', '-'),
            "https://www.example.edu/research/" + query.replace(' ', '-'),
        ]

        return mock_results

    def extract_relevant_content_with_llm(self, query: str, content: str) -> Optional[str]:
        """
        Use LLM to extract only relevant information from content.

        Uses the content_retrieval_agent prompt from prompts.yaml.

        Args:
            query: The search query or subclaim
            content: Raw content from web page

        Returns:
            Extracted relevant information or None
        """
        if not self.llm:
            return None

        try:
            prompts_loader = get_prompts_loader()
            prompt = prompts_loader.get_prompt(
                'content_retrieval_agent',
                query=query,
                content=content[:5000]  # Limit content to avoid token limits
            )

            if not prompt:
                return None

            response = self.llm.generate(prompt)

            # Check if response indicates no relevant information
            if response and 'none' not in response.lower()[:50]:
                return response.strip()

            return None

        except Exception as e:
            logger.error(f"LLM content extraction failed: {e}")
            return None

    def process_with_llm(self, query_results: List[Any], language: Optional[str] = None) -> List[EvidenceResult]:
        """
        Retrieve evidence using LLM for intelligent content extraction.

        Args:
            query_results: List of QueryResult objects
            language: Optional language override

        Returns:
            List of EvidenceResult objects with LLM-extracted content
        """
        if language:
            self.set_language(language)

        logger.info(f"Seeking evidence with LLM for {len(query_results)} subclaims")

        results = []
        for qr in query_results:
            evidence_list = []
            evidence_id = 1
            query_lang = getattr(qr, 'language', self.language)

            for query in qr.queries:
                # Stage 1: Search
                search_results = self._search_web(query, language=query_lang)

                # Stage 2 & 3: Check credibility and extract content with LLM
                for url in search_results[:self.max_results]:
                    try:
                        # Credibility check
                        cred_score = self.credibility.check(url)

                        if not self.credibility.meets_threshold(cred_score, self.credibility_threshold):
                            continue

                        # Content extraction
                        content = self.scraper.scrape(url)

                        if not content.success:
                            continue

                        # Use LLM to extract relevant content if available
                        if self.llm and content.text:
                            relevant_content = self.extract_relevant_content_with_llm(
                                qr.subclaim_text,
                                content.text
                            )
                            if relevant_content:
                                evidence = Evidence(
                                    id=f"EV{evidence_id}",
                                    subclaim_id=qr.subclaim_id,
                                    source_url=url,
                                    source_name=content.title or url,
                                    passage=relevant_content,
                                    credibility_score=cred_score.score,
                                    credibility_level=cred_score.level.value,
                                    retrieved_at=datetime.utcnow().isoformat(),
                                    metadata={**cred_score.details, 'llm_extracted': True},
                                    language=query_lang
                                )
                                evidence_list.append(evidence)
                                evidence_id += 1
                        else:
                            # Fallback to regular passage extraction
                            passages = self.scraper.extract_passages(
                                content,
                                qr.subclaim_text,
                                max_passages=self.max_passages
                            )

                            for passage in passages:
                                evidence = Evidence(
                                    id=f"EV{evidence_id}",
                                    subclaim_id=qr.subclaim_id,
                                    source_url=url,
                                    source_name=content.title or url,
                                    passage=passage,
                                    credibility_score=cred_score.score,
                                    credibility_level=cred_score.level.value,
                                    retrieved_at=datetime.utcnow().isoformat(),
                                    metadata=cred_score.details,
                                    language=query_lang
                                )
                                evidence_list.append(evidence)
                                evidence_id += 1

                    except Exception as e:
                        logger.error(f"Error processing {url}: {e}")
                        continue

            high_cred_count = len([e for e in evidence_list if e.credibility_level == 'high'])

            results.append(EvidenceResult(
                subclaim_id=qr.subclaim_id,
                evidence=evidence_list,
                total_sources=len(evidence_list),
                high_credibility_count=high_cred_count,
                language=query_lang
            ))

            logger.info(f"Found {len(evidence_list)} evidence items for {qr.subclaim_id}")

        return results
