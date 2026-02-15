"""
First-Order Logic (FOL) Parser
Decomposes complex claims into atomic predicates using FOL principles
Supports multilingual claims through language-specific keyword configuration
Uses configurable prompts from prompts.yaml for LLM-based decomposition
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import re

from src.utils.multilingual_config import get_multilingual_config
from src.utils.prompts_loader import get_prompts_loader


class VerifiabilityType(Enum):
    """Types of claim verifiability"""
    VERIFIABLE = "verifiable"
    NON_VERIFIABLE_OPINION = "opinion"
    NON_VERIFIABLE_VAGUE = "vague"
    NON_VERIFIABLE_FUTURE = "future_prediction"


@dataclass
class SubClaim:
    """Represents an atomic subclaim"""
    id: str
    text: str
    logical_form: str
    verifiability: VerifiabilityType
    entities: List[str]
    properties: List[str]


class FOLParser:
    """
    First-Order Logic parser for claim decomposition.

    Decomposes complex claims into atomic predicates:
    - P(entity, property, value)
    - Relation(entity1, entity2)

    Supports multilingual claims through language-specific keywords.
    """

    def __init__(self, llm_interface=None, language: str = 'en'):
        """
        Initialize FOL parser.

        Args:
            llm_interface: Optional LLM interface for intelligent decomposition
            language: Language code for keyword configuration (default: 'en')
        """
        self.llm = llm_interface
        self.language = language
        self.ml_config = get_multilingual_config()
        self._load_language_keywords()

    def _load_language_keywords(self):
        """Load language-specific keywords for verifiability detection"""
        self.opinion_words = self.ml_config.get_opinion_words(self.language)
        self.future_words = self.ml_config.get_future_words(self.language)
        self.vague_words = self.ml_config.get_vague_words(self.language)

        # Fallback to English defaults if empty
        if not self.opinion_words:
            self.opinion_words = ['believe', 'think', 'should', 'ought', 'better', 'worse', 'best', 'worst']
        if not self.future_words:
            self.future_words = ['will', 'would', 'might', 'may', 'could', 'going to']
        if not self.vague_words:
            self.vague_words = ['some', 'many', 'few', 'several', 'various', 'often', 'sometimes']

    def set_language(self, language: str):
        """
        Change the language for keyword loading.

        Args:
            language: ISO 639-1 language code
        """
        self.language = language
        self._load_language_keywords()

    def decompose(self, claim: str, max_subclaims: int = 10, language: Optional[str] = None) -> List[SubClaim]:
        """
        Decompose a complex claim into atomic subclaims.

        Args:
            claim: Original claim to decompose
            max_subclaims: Maximum number of subclaims to generate
            language: Optional language override for this decomposition

        Returns:
            List of atomic subclaims
        """
        # Update language if provided
        if language and language != self.language:
            self.set_language(language)

        logger.info(f"Decomposing claim (lang={self.language}): {claim}")

        if self.llm:
            return self._decompose_with_llm(claim, max_subclaims)
        else:
            return self._decompose_heuristic(claim, max_subclaims)

    def _decompose_with_llm(self, claim: str, max_subclaims: int) -> List[SubClaim]:
        """Use LLM for intelligent claim decomposition with configurable prompts"""
        # Load prompt from configuration
        prompts_loader = get_prompts_loader()
        config_prompt = prompts_loader.get_prompt('claim_decomposition_agent', claim=claim)

        # Use configured prompt if available, otherwise fallback to default
        if config_prompt:
            system_prompt = """You are a First-Order Logic expert specializing in claim decomposition.
Your task is to define all atomic predicates in the claim.

Each predicate must:
- Represent a single factual assertion
- Use a clear predicate structure (e.g., Location, Born, Won, Held)
- Include a short verification instruction"""

            user_prompt = f"""{config_prompt}

Additionally, provide JSON output with up to {max_subclaims} atomic subclaims:
{{
  "subclaims": [
    {{
      "id": "SC1",
      "text": "atomic claim text",
      "logical_form": "Predicate(entity, property, value)",
      "verifiability": "verifiable" | "opinion" | "vague" | "future_prediction",
      "entities": ["entity1", "entity2"],
      "properties": ["property1"]
    }}
  ]
}}"""
        else:
            # Fallback to original prompt
            system_prompt = """You are a First-Order Logic expert specializing in claim decomposition.

Decompose complex claims into atomic subclaims using FOL predicates:
- Predicate(entity, property, value)
- Relation(entity1, entity2)

Rules:
1. Each subclaim should be independently verifiable
2. Use proper predicate notation
3. Extract all entities and properties
4. Classify verifiability"""

            user_prompt = f"""Decompose this claim into atomic subclaims:

CLAIM: {claim}

Provide up to {max_subclaims} atomic subclaims in JSON format:
{{
  "subclaims": [
    {{
      "id": "SC1",
      "text": "atomic claim text",
      "logical_form": "Predicate(entity, property, value)",
      "verifiability": "verifiable" | "opinion" | "vague" | "future_prediction",
      "entities": ["entity1", "entity2"],
      "properties": ["property1"]
    }}
  ]
}}"""

        try:
            response = self.llm.generate_structured(
                user_prompt,
                output_schema={
                    "subclaims": [{
                        "id": "string",
                        "text": "string",
                        "logical_form": "string",
                        "verifiability": "string",
                        "entities": ["string"],
                        "properties": ["string"]
                    }]
                },
                system_prompt=system_prompt
            )

            subclaims = []
            for i, sc in enumerate(response.get('subclaims', [])[:max_subclaims]):
                subclaims.append(SubClaim(
                    id=sc.get('id', f'SC{i+1}'),
                    text=sc['text'],
                    logical_form=sc['logical_form'],
                    verifiability=VerifiabilityType(sc.get('verifiability', 'verifiable')),
                    entities=sc.get('entities', []),
                    properties=sc.get('properties', [])
                ))

            logger.info(f"Decomposed into {len(subclaims)} subclaims")
            return subclaims

        except Exception as e:
            logger.error(f"LLM decomposition failed: {e}, falling back to heuristic")
            return self._decompose_heuristic(claim, max_subclaims)

    def _decompose_heuristic(self, claim: str, max_subclaims: int) -> List[SubClaim]:
        """Heuristic-based decomposition (fallback)"""
        subclaims = []

        # Split by conjunctions (and, comma)
        parts = re.split(r'\s+and\s+|\s*,\s*|\s+that\s+', claim, flags=re.IGNORECASE)

        for i, part in enumerate(parts[:max_subclaims]):
            part = part.strip()
            if not part:
                continue

            # Extract entities (capitalized words, proper nouns)
            entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', part)

            # Simple predicate form
            logical_form = f"Claim('{part}')"

            # Basic verifiability check
            verifiability = self._check_verifiability_heuristic(part)

            subclaims.append(SubClaim(
                id=f'SC{i+1}',
                text=part,
                logical_form=logical_form,
                verifiability=verifiability,
                entities=entities,
                properties=[]
            ))

        logger.info(f"Heuristic decomposition produced {len(subclaims)} subclaims")
        return subclaims

    def _check_verifiability_heuristic(self, text: str) -> VerifiabilityType:
        """
        Simple heuristic to check if a claim is verifiable.
        Uses language-specific keywords loaded from configuration.
        """
        text_lower = text.lower()

        # Opinion indicators (using language-specific keywords)
        if any(word in text_lower for word in self.opinion_words):
            return VerifiabilityType.NON_VERIFIABLE_OPINION

        # Future predictions (using language-specific keywords)
        if any(word in text_lower for word in self.future_words):
            return VerifiabilityType.NON_VERIFIABLE_FUTURE

        # Vague claims (using language-specific keywords)
        if any(word in text_lower for word in self.vague_words) and len(text.split()) < 10:
            return VerifiabilityType.NON_VERIFIABLE_VAGUE

        return VerifiabilityType.VERIFIABLE

    def filter_verifiable(self, subclaims: List[SubClaim]) -> List[SubClaim]:
        """
        Filter out non-verifiable subclaims.

        Args:
            subclaims: List of subclaims

        Returns:
            Only verifiable subclaims
        """
        verifiable = [sc for sc in subclaims if sc.verifiability == VerifiabilityType.VERIFIABLE]
        logger.info(f"Filtered to {len(verifiable)} verifiable subclaims from {len(subclaims)} total")
        return verifiable
