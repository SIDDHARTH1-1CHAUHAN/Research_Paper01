"""
Query Generation Agent - Creates diverse search queries for evidence retrieval
"""

from typing import List, Dict, Any
from dataclasses import dataclass
from loguru import logger
import re


@dataclass
class QueryResult:
    """Result from Query Generation Agent"""
    subclaim_id: str
    subclaim_text: str
    queries: List[str]


class QueryGenerationAgent:
    """
    Agent 2: Query Generation

    Generates diverse search queries for each subclaim using:
    - Entity-aware keywords
    - Paraphrasing
    - Temporal variants
    - SEO optimization
    """

    def __init__(self, config: Dict[str, Any] = None, llm_interface=None):
        """
        Initialize Query Generation Agent.

        Args:
            config: Configuration dictionary
            llm_interface: Optional LLM interface for intelligent query generation
        """
        self.config = config or {}
        self.llm = llm_interface

        # Configuration (k=3-4 is optimal per research)
        self.queries_per_subclaim = self.config.get('queries_per_subclaim', 3)

        logger.info(f"Query Generation Agent initialized (k={self.queries_per_subclaim})")

    def process(self, subclaims: List[Dict[str, Any]]) -> List[QueryResult]:
        """
        Generate search queries for subclaims.

        Args:
            subclaims: List of verifiable subclaims

        Returns:
            List of QueryResults with generated queries
        """
        logger.info(f"Generating queries for {len(subclaims)} subclaims")

        results = []
        for subclaim in subclaims:
            queries = self._generate_queries(
                subclaim['text'],
                subclaim.get('entities', []),
                k=self.queries_per_subclaim
            )

            results.append(QueryResult(
                subclaim_id=subclaim['id'],
                subclaim_text=subclaim['text'],
                queries=queries
            ))

        logger.info(f"Generated {sum(len(r.queries) for r in results)} total queries")
        return results

    def _generate_queries(self, text: str, entities: List[str], k: int) -> List[str]:
        """
        Generate k diverse queries for a subclaim.

        Args:
            text: Subclaim text
            entities: Extracted entities
            k: Number of queries to generate

        Returns:
            List of query strings
        """
        queries = []

        # Query 1: Direct quote search
        if entities:
            queries.append(f"{' '.join(entities)} {text}")
        else:
            queries.append(text)

        # Query 2: Question form
        question = self._convert_to_question(text)
        queries.append(question)

        # Query 3: Keyword extraction
        keywords = self._extract_keywords(text, entities)
        queries.append(' '.join(keywords))

        # Query 4: Add temporal context if present
        if k > 3:
            temporal = self._add_temporal_context(text)
            if temporal != text:
                queries.append(temporal)

        # Return exactly k queries
        return queries[:k]

    def _convert_to_question(self, text: str) -> str:
        """Convert statement to question form"""
        # Simple heuristics
        if 'is' in text.lower():
            return text.replace('.', '?')
        elif 'was' in text.lower():
            return f"When {text.lower().replace('.', '?')}"
        else:
            return f"What about {text.lower().replace('.', '?')}"

    def _extract_keywords(self, text: str, entities: List[str]) -> List[str]:
        """Extract important keywords"""
        # Remove common words
        stopwords = {'the', 'is', 'in', 'at', 'on', 'and', 'or', 'a', 'an'}
        words = text.lower().split()
        keywords = [w for w in words if w not in stopwords and len(w) > 3]

        # Add entities
        keywords.extend(entities)

        return list(set(keywords))[:8]  # Top 8 keywords

    def _add_temporal_context(self, text: str) -> str:
        """Add temporal keywords if dates found"""
        # Find years (4 digits)
        years = re.findall(r'\b\d{4}\b', text)
        if years:
            return f"{text} history timeline {years[0]}"
        return text
