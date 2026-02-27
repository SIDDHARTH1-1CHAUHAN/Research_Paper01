"""
Query Generation Agent - Creates diverse search queries for evidence retrieval
"""

from typing import List, Dict, Any
from dataclasses import dataclass
from loguru import logger
import re
import json
import sys
sys.path.append('..')
import yaml
from pathlib import Path


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
        self.queries_per_subclaim = self.config.get('queries_per_subclaim', 1)

        self.prompts = self._load_prompts()

        logger.info(f"Query Generation Agent initialized (k={self.queries_per_subclaim})")

    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompts from YAML"""
        possible_paths = [
            Path("config/agent_prompt.yaml"),
            Path("../config/agent_prompt.yaml"),
            Path("../../config/agent_prompt.yaml"),
        ]

        for path in possible_paths:
            if path.exists():
                with open(path, 'r') as f:
                    return yaml.safe_load(f)
        return {}

    def process(self, subclaims: List[Dict[str, Any]]) -> List[QueryResult]:
        """
        Generate search queries for subclaims.

        Args:
            subclaims: List of verifiable subclaims

        Returns:
            List of QueryResults with generated queries
        """

        results = []
        for subclaim in subclaims:
            logger.debug(f"Processing subclaim: {subclaim}")
            queries = self._generate_queries(
                subclaim.get('text', ''),
                subclaim.get('predicate', ''),
                k=self.queries_per_subclaim
            )

            results.append(QueryResult(
                subclaim_id=subclaim.get('predicate', ''),
                subclaim_text=subclaim.get('description'),  
                queries=queries
            ))

        logger.info(f"Generated {sum(len(r.queries) for r in results)} total queries")
        return results

    def _generate_queries(self, text: str, subclaim: str, k: int) -> List[str]:
        """
        Generate k diverse queries for a subclaim using LLM.

        Args:
            text: Subclaim text
            subclaim: Subclaim predicate
            k: Number of queries to generate

        Returns:
            List of query strings
        """
        prompt_template = self.prompts.get('query_generation', '')
        system_prompt = prompt_template.format(k=k, subclaim=subclaim)
        response = self.llm._generate_ollama(
            role="user",
            prompt=system_prompt,
        )
        logger.info(f"LLM response for query generation: {response}")

        # Robust JSON extraction with fallback
        parsed = None
        if response and response.strip():
            # 1. Try direct parsing
            try:
                parsed = json.loads(response)
            except json.JSONDecodeError:
                pass

            # 2. Try extracting JSON from markdown code blocks
            if parsed is None:
                json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                        pass

            # 3. Try extracting a JSON array
            if parsed is None:
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        pass

            # 4. Try extracting a JSON object
            if parsed is None:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        pass

        if parsed is None:
            logger.warning(f"Could not parse JSON from LLM response: {response}")
            return []

        if isinstance(parsed, list) and len(parsed) > 0:
            queries = parsed[0].get('questions', [])
        elif isinstance(parsed, dict):
            queries = parsed.get('questions', [])
        else:
            queries = []
        logger.debug(f"Generated queries: {queries}: subclaim")
        return queries[:k]

