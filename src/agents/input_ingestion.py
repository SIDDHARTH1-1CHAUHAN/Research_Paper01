"""
Input Ingestion Agent - FOL-based claim decomposition and verifiability filtering
"""

from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from loguru import logger
import sys
sys.path.append('..')
from src.utils.fol_parser import FOLParser, SubClaim, VerifiabilityType


@dataclass
class IngestionResult:
    """Result from Input Ingestion Agent"""
    original_claim: str
    verifiable_subclaims: List[Dict[str, Any]]
    filtered_subclaims: List[Dict[str, Any]]
    filtered_count: int


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
        self.max_subclaims = self.config.get('max_subclaims', 10)
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
        subclaims = self.parser.decompose(claim, max_subclaims=self.max_subclaims)

        # Step 2: Separate verifiable from non-verifiable
        verifiable = []
        filtered = []

        for sc in subclaims:
            sc_dict = {
                'id': sc.id,
                'text': sc.text,
                'logical_form': sc.logical_form,
                'verifiability': sc.verifiability.value,
                'entities': sc.entities,
                'properties': sc.properties
            }

            if sc.verifiability == VerifiabilityType.VERIFIABLE:
                verifiable.append(sc_dict)
            else:
                filtered.append(sc_dict)

        result = IngestionResult(
            original_claim=claim,
            verifiable_subclaims=verifiable,
            filtered_subclaims=filtered,
            filtered_count=len(filtered)
        )

        logger.info(f"Decomposed into {len(verifiable)} verifiable subclaims, filtered {len(filtered)}")

        return result

    def to_dict(self, result: IngestionResult) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return asdict(result)
