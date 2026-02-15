"""
Input Ingestion Agent - FOL-based claim decomposition and verifiability filtering
Includes LLM-based subclaim classification using configurable prompts
"""

from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from loguru import logger
import sys
sys.path.append('..')
from src.utils.fol_parser import FOLParser, SubClaim, VerifiabilityType
from src.utils.prompts_loader import get_prompts_loader


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

    def classify_subclaim(self, claim: str) -> Dict[str, Any]:
        """
        Classify a subclaim as verifiable or non-verifiable using LLM.

        Uses the subclaim_classification_agent prompt from prompts.yaml.

        Args:
            claim: The subclaim text to classify

        Returns:
            Dictionary with classification and explanation
        """
        if not self.llm:
            # Fallback to heuristic classification
            return self._classify_heuristic(claim)

        try:
            prompts_loader = get_prompts_loader()
            prompt = prompts_loader.get_prompt('subclaim_classification_agent', claim=claim)

            if not prompt:
                return self._classify_heuristic(claim)

            response = self.llm.generate_structured(
                prompt,
                output_schema={
                    "classification": "string",
                    "explanation": "string"
                }
            )

            return {
                'classification': response.get('classification', 'VERIFIABLE'),
                'explanation': response.get('explanation', '')
            }

        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return self._classify_heuristic(claim)

    def _classify_heuristic(self, claim: str) -> Dict[str, Any]:
        """
        Heuristic-based subclaim classification (fallback).

        Args:
            claim: The subclaim text to classify

        Returns:
            Dictionary with classification and explanation
        """
        claim_lower = claim.lower()

        # Opinion indicators
        opinion_words = ['believe', 'think', 'should', 'ought', 'better', 'worse', 'best', 'worst', 'opinion']
        if any(word in claim_lower for word in opinion_words):
            return {
                'classification': 'NON-VERIFIABLE',
                'explanation': 'Contains subjective opinion indicators'
            }

        # Future/hypothetical indicators
        future_words = ['will', 'would', 'might', 'may', 'could', 'going to', 'if']
        if any(word in claim_lower for word in future_words):
            return {
                'classification': 'NON-VERIFIABLE',
                'explanation': 'Refers to future or hypothetical events'
            }

        # Vague indicators
        vague_words = ['some', 'many', 'few', 'several', 'various', 'often', 'sometimes']
        if any(word in claim_lower for word in vague_words) and len(claim.split()) < 10:
            return {
                'classification': 'NON-VERIFIABLE',
                'explanation': 'Too vague or ambiguous to verify'
            }

        return {
            'classification': 'VERIFIABLE',
            'explanation': 'Makes factual, objective assertions that can be checked'
        }

    def process_with_classification(self, claim: str) -> IngestionResult:
        """
        Process a claim with additional LLM-based classification.

        Args:
            claim: Natural language claim to process

        Returns:
            IngestionResult with verifiable and filtered subclaims
        """
        logger.info(f"Processing claim with classification: {claim}")

        # Step 1: Decompose claim into subclaims
        subclaims = self.parser.decompose(claim, max_subclaims=self.max_subclaims)

        # Step 2: Classify each subclaim using LLM
        verifiable = []
        filtered = []

        for sc in subclaims:
            classification_result = self.classify_subclaim(sc.text)

            sc_dict = {
                'id': sc.id,
                'text': sc.text,
                'logical_form': sc.logical_form,
                'verifiability': sc.verifiability.value,
                'entities': sc.entities,
                'properties': sc.properties,
                'llm_classification': classification_result
            }

            if classification_result['classification'] == 'VERIFIABLE':
                verifiable.append(sc_dict)
            else:
                filtered.append(sc_dict)

        result = IngestionResult(
            original_claim=claim,
            verifiable_subclaims=verifiable,
            filtered_subclaims=filtered,
            filtered_count=len(filtered)
        )

        logger.info(f"Classified into {len(verifiable)} verifiable subclaims, filtered {len(filtered)}")

        return result
