"""
Evidence Seeking Agent - 3-stage evidence retrieval pipeline
Stage 1: Internet Search
Stage 2: Credibility Check
Stage 3: Content Extraction
"""

from typing import List, Dict, Any
from dataclasses import dataclass
from loguru import logger
from datetime import datetime
import sys
sys.path.append('..')
from src.utils.web_scraper import WebScraper
from src.utils.credibility_checker import CredibilityChecker


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


@dataclass
class EvidenceResult:
    """Result from Evidence Seeking Agent"""
    subclaim_id: str
    evidence: List[Evidence]
    total_sources: int
    high_credibility_count: int


class EvidenceSeekingAgent:
    """
    Agent 3: Evidence Seeking

    Three-stage pipeline:
    1. Internet Search (DuckDuckGo/SerperAPI)
    2. Credibility Assessment (Heuristic/MBFC)
    3. Content Extraction (BeautifulSoup)
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Evidence Seeking Agent.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Initialize components
        self.scraper = WebScraper(self.config.get('scraping', {}))
        self.credibility = CredibilityChecker(self.config.get('credibility', {}))

        # Configuration
        self.max_results = self.config.get('max_search_results', 10)
        self.credibility_threshold = self.config.get('credibility_threshold', 'medium')
        self.max_passages = self.config.get('max_passages_per_source', 3)

        logger.info("Evidence Seeking Agent initialized")

    def process(self, query_results: List[Any]) -> List[EvidenceResult]:
        """
        Retrieve evidence for all queries.

        Args:
            query_results: List of QueryResult objects

        Returns:
            List of EvidenceResult objects
        """
        logger.info(f"Seeking evidence for {len(query_results)} subclaims")

        results = []
        for qr in query_results:
            evidence_list = []
            evidence_id = 1

            for query in qr.queries:
                # Stage 1: Search
                search_results = self._search_web(query)

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
                                metadata=cred_score.details
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
                high_credibility_count=high_cred_count
            ))

            logger.info(f"Found {len(evidence_list)} evidence items for {qr.subclaim_id} ({high_cred_count} high credibility)")

        return results

    def _search_web(self, query: str) -> List[str]:
        """
        Search the web for a query.

        Args:
            query: Search query

        Returns:
            List of URLs
        """
        # For demo, return mock URLs based on query keywords
        # In production, use DuckDuckGo or SerperAPI
        logger.info(f"Searching for: {query}")

        # Mock search results
        mock_results = [
            "https://en.wikipedia.org/wiki/" + query.replace(' ', '_'),
            "https://www.britannica.com/topic/" + query.replace(' ', '-'),
            "https://www.example.edu/research/" + query.replace(' ', '-'),
        ]

        return mock_results
