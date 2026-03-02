"""
Input Ingestion Agent - FOL-based claim decomposition and verifiability filtering
"""

from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from loguru import logger
import sys
sys.path.append('..')
from src.utils.fol_parser import FOLParser


@dataclass
class IngestionResult:
    """Result from Input Ingestion Agent"""
    original_claim: str
    verifiable_subclaims: List[Dict[str, Any]]



class InputIngestionAgent:
    """
    Agent 1: Input Ingestion

    Decomposes complex claims into atomic subclaims using First-Order Logic
    and filters non-verifiable claims (opinions, vague statements, future predictions).
    """

    def __init__(self, config: Dict[str, Any] = None, llm_interface=None):
        """
        Initialize Input Ingestion Agent.
        Args:
            config: Configuration dictionary
            llm_interface: Optional LLM interface for intelligent decomposition
        """
        self.config = config or {}
        self.llm = llm_interface
        self.parser = FOLParser(llm_interface=llm_interface)

        # Configuration
        self.filter_non_verifiable = self.config.get('filter_non_verifiable', True)

        logger.info("Input Ingestion Agent initialized")

    def process(self, claim: str) -> IngestionResult:
        """
        Process a claim through decomposition and filtering.

        Args:
            claim: Natural language claim to process

        Returns:
            IngestionResult with verifiable and filtered subclaims
        """
        logger.info(f"Processing claim: {claim}")

        # Step 1: Decompose claim into subclaims
        subclaims = self.parser._decompose_with_llm(claim)

        verifiable_subclaims = []
        filtered_subclaims = []

        for idx, sc in enumerate(subclaims):
            subclaim_text = sc.get('description', sc.get('predicate', ''))
            classification = self.parser.facts(subclaim_text)
            verifiability = classification.get('verifiability', 'VERIFIABLE')

            normalized = {
                'text': subclaim_text,
                'predicate': sc.get('predicate', ''),
                'verifiability': verifiability,
                'description': sc.get('description', []),
            }

            if normalized['verifiability'] == "VERIFIABLE":
                verifiable_subclaims.append(normalized)
            else:
                filtered_subclaims.append(normalized)

        result = IngestionResult(
            original_claim=claim,
            verifiable_subclaims=verifiable_subclaims,
        )

        logger.info(f"Decomposed into {len(verifiable_subclaims)} verifiable subclaims, filtered {len(filtered_subclaims)}")

        return result

    def to_dict(self, result: IngestionResult) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return asdict(result)
