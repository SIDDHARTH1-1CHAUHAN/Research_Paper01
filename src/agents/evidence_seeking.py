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
import yaml
import os
import sys
import requests
sys.path.append('..')
from src.utils.web_scraper import WebScraper
from src.utils.credibility_checker import CredibilityChecker

@dataclass
class Evidence:
    """Single piece of evidence"""
    source_url: str
    source_name: str
    passage: str
    retrieved_at: str


@dataclass
class EvidenceResult:
    """Result from Evidence Seeking Agent"""
    subclaim_text: str
    queries: List[str]
    evidence: List[Evidence]
    total_sources: int

class EvidenceSeekingAgent:
    """
    Agent 3: Evidence Seeking

    Three-stage pipeline:
    1. Internet Search (DuckDuckGo/SerperAPI)
    2. Credibility Assessment (Heuristic/MBFC)
    3. Content Extraction (BeautifulSoup)
    """

    def __init__(self, config: Dict[str, Any] = None,llm_interface=None):
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
        self.llm = llm_interface
        logger.info("Evidence Seeking Agent initialized")
        
        # Load prompts
        self.prompt_template = None
        try:
            prompt_path = os.path.join(os.path.dirname(__file__), '../../config/agent_prompt.yaml')
            with open(prompt_path, 'r') as f:
                self.prompt_template = yaml.safe_load(f).get('evidence_seeking', '')
        except Exception as e:
            logger.warning(f"Could not load agent_prompt.yaml: {e}")

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
        evidence_id_counter = 0
        for qr in query_results:
            evidence_list = []
            for query in qr.queries:
                # Stage 1: Search
                search_results = self._search_web(query, "HoVER")

                # Stage 2 & 3: Check credibility and extract content
                for url in search_results[:self.max_results]:
                    try:

                        # Content extraction
                        content = self.scraper.scrape(url)
                        passages = []
                        # Use LLM for extraction if available
                        content_text = getattr(content, 'text', str(content))[:5000]

                        prompt = self.prompt_template.format(
                            query=query,
                            content=content_text
                        )
                        extracted = self.llm._generate_ollama(prompt)

                        if extracted and "None" not in extracted:
                            passages = [extracted]

                        for passage in passages:
                            evidence = Evidence(
                                source_url=url,
                                source_name=content.title or url,
                                passage=passage,
                                retrieved_at=datetime.utcnow().isoformat(),
                            )
                            evidence_list.append(evidence)
                            evidence_id_counter += 1

                    except Exception as e:
                        logger.error(f"Error processing {url}: {e}")
                        continue

            results.append(EvidenceResult(
                subclaim_text=qr.subclaim_text,
                queries=qr.queries,
                evidence=evidence_list,
                total_sources=len(evidence_list),
            ))


        return results

    def _search_web(self, query: str,dataset:str) -> List[str]:
        """
        Search the web for a query.

        Args:
            query: Search query

        Returns:
            List of URLs
        """
        if dataset in ["HoVER", "FEVEROUS",""]:
            # Use Wikipedia API to search specifically on Wikipedia
            results = self._search_wikipedia(query)
            logger.debug(f"wikipedia search results: {results}")
            return results
        
        elif dataset == "SciFact-Open":
            # Use Serper for Google Scholar and PubMed as requested
            results = []
            results.extend(self._search_serper(query, search_type="scholar"))
            # For PubMed, we can use general search with site:pubmed.ncbi.nlm.nih.gov
            results.extend(self._search_serper(f"{query} site:pubmed.ncbi.nlm.nih.gov"))
            return results
        
        else:
            # General search using Wikipedia
            return self._search_wikipedia(query)

    def _search_wikipedia(self, query: str) -> List[str]:
        """Search using Wikipedia Free API"""
        try:
            # Clean up the query if it contains site:wikipedia.org from earlier versions
            clean_query = query.replace(" site:wikipedia.org", "").strip()
            
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "list": "search",
                "srsearch": clean_query,
                "format": "json",
                "srlimit": 1
            }
            headers = {
                'User-Agent': 'ResearchTool/1.0'
            }
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            results = []
            if "query" in data and "search" in data["query"]:
                for item in data["query"]["search"]:
                    title = item["title"].replace(" ", "_")
                    results.append(f"https://en.wikipedia.org/wiki/{title}")
                    
            return results
        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
            return []

    
