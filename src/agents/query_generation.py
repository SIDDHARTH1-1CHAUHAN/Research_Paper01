"""
Query Generation Agent - Creates diverse search queries for evidence retrieval
Supports multilingual query generation with language-specific stopwords
Uses configurable prompts from prompts.yaml for LLM-based query generation
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from loguru import logger
import re
import json

from src.utils.multilingual_config import get_multilingual_config
from src.utils.prompts_loader import get_prompts_loader


@dataclass
class QueryResult:
    """Result from Query Generation Agent"""
    subclaim_id: str
    subclaim_text: str
    queries: List[str]
    language: str = 'en'


class QueryGenerationAgent:
    """
    Agent 2: Query Generation

    Generates diverse search queries for each subclaim using:
    - Entity-aware keywords
    - Paraphrasing
    - Temporal variants
    - SEO optimization

    Supports multilingual claims through language-specific stopwords.
    """

    def __init__(self, config: Dict[str, Any] = None, llm_interface=None, language: str = 'en'):
        """
        Initialize Query Generation Agent.

        Args:
            config: Configuration dictionary
            llm_interface: Optional LLM interface for intelligent query generation
            language: Language code for stopwords and query patterns (default: 'en')
        """
        self.config = config or {}
        self.llm = llm_interface
        self.language = language
        self.ml_config = get_multilingual_config()

        # Configuration (k=3-4 is optimal per research)
        self.queries_per_subclaim = self.config.get('queries_per_subclaim', 3)
        self.generate_english_queries = self.config.get('generate_english_queries', True)

        # Load language-specific stopwords
        self._load_stopwords()

        logger.info(f"Query Generation Agent initialized (k={self.queries_per_subclaim}, lang={self.language})")

    def _load_stopwords(self):
        """Load stopwords for the current language"""
        self.stopwords = self.ml_config.get_stopwords(self.language)
        # Add common English stopwords as fallback
        if not self.stopwords:
            self.stopwords = {'the', 'is', 'in', 'at', 'on', 'and', 'or', 'a', 'an'}

    def set_language(self, language: str):
        """
        Set the language for query generation.

        Args:
            language: ISO 639-1 language code
        """
        if language != self.language:
            self.language = language
            self._load_stopwords()
            logger.info(f"Query Generation Agent language set to: {language}")

    def process(self, subclaims: List[Dict[str, Any]], language: Optional[str] = None) -> List[QueryResult]:
        """
        Generate search queries for subclaims.

        Args:
            subclaims: List of verifiable subclaims
            language: Optional language override

        Returns:
            List of QueryResults with generated queries
        """
        # Update language if provided
        if language:
            self.set_language(language)

        logger.info(f"Generating queries for {len(subclaims)} subclaims (lang={self.language})")

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
                queries=queries,
                language=self.language
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
        """Extract important keywords using language-specific stopwords"""
        # Use language-specific stopwords
        words = text.lower().split()
        keywords = [w for w in words if w not in self.stopwords and len(w) > 2]

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

    def process_with_llm(self, subclaims: List[Dict[str, Any]], language: Optional[str] = None) -> List[QueryResult]:
        """
        Generate search queries using LLM with configurable prompts.

        Args:
            subclaims: List of verifiable subclaims
            language: Optional language override

        Returns:
            List of QueryResults with LLM-generated queries
        """
        if language:
            self.set_language(language)

        if not self.llm:
            logger.warning("LLM not available, falling back to heuristic query generation")
            return self.process(subclaims, language)

        logger.info(f"Generating queries with LLM for {len(subclaims)} subclaims")

        try:
            # Load prompt from configuration
            prompts_loader = get_prompts_loader()

            # Prepare subclaims data for prompt
            subclaims_text = json.dumps([
                {"claim": sc['text'], "id": sc['id']}
                for sc in subclaims
            ], indent=2)

            prompt = prompts_loader.get_prompt(
                'query_generation_agent',
                subclaims=subclaims_text
            )

            if not prompt:
                return self.process(subclaims, language)

            # Generate queries using LLM
            response = self.llm.generate_structured(
                f"{prompt}\n\nGenerate {self.queries_per_subclaim} questions per subclaim.",
                output_schema=[{
                    "claim": "string",
                    "questions": ["string"]
                }]
            )

            # Process LLM response
            results = []
            llm_results = response if isinstance(response, list) else response.get('results', [])

            for subclaim in subclaims:
                # Find matching LLM result
                llm_queries = None
                for lr in llm_results:
                    if lr.get('claim') == subclaim['text'] or subclaim['id'] in str(lr):
                        llm_queries = lr.get('questions', [])
                        break

                if llm_queries:
                    queries = llm_queries[:self.queries_per_subclaim]
                else:
                    # Fallback to heuristic for this subclaim
                    queries = self._generate_queries(
                        subclaim['text'],
                        subclaim.get('entities', []),
                        k=self.queries_per_subclaim
                    )

                results.append(QueryResult(
                    subclaim_id=subclaim['id'],
                    subclaim_text=subclaim['text'],
                    queries=queries,
                    language=self.language
                ))

            logger.info(f"LLM generated {sum(len(r.queries) for r in results)} total queries")
            return results

        except Exception as e:
            logger.error(f"LLM query generation failed: {e}, falling back to heuristic")
            return self.process(subclaims, language)
