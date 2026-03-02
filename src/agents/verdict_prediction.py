"""
Verdict Prediction Agent - Evidence aggregation and verdict synthesis
"""

from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from loguru import logger
import yaml
import os
import json
import sys
sys.path.append('..')
from src.utils.llm_interface import LLMInterface


@dataclass
class SubclaimVerdict:
    """Verdict for a single subclaim"""
    supports:str
    claim:str



@dataclass
class VerdictResult:
    """Final verdict result"""
    subclaim_verdicts: List[SubclaimVerdict]
    final_verdict: str
    explanation: str


class VerdictPredictionAgent:
    """
    Agent 4: Verdict Prediction

    Aggregates evidence using weighted voting:
    - HIGH credibility: 1.0 weight
    - MEDIUM credibility: 0.6 weight
    - LOW credibility: 0.3 weight

    Decision thresholds:
    - Score > 0.7 → SUPPORTED
    - Score < 0.3 → NOT_SUPPORTED
    """

    def __init__(self, config: Dict[str, Any] = None, llm_interface=None):
        """
        Initialize Verdict Prediction Agent.

        Args:
            config: Configuration dictionary
            llm_interface: Optional LLM for explanation generation
        """
        self.config = config or {}
        self.llm = llm_interface
        self.prompt_template = None

        if self.llm and isinstance(self.llm, LLMInterface):
            try:
                prompt_path = os.path.join(os.path.dirname(__file__), '../../config/agent_prompt.yaml')
                with open(prompt_path, 'r') as f:
                    prompts = yaml.safe_load(f)
                    self.prompt_template = prompts.get('verdict_prediction', {})
            except Exception as e:
                logger.warning(f"Could not load agent_prompt.yaml for verdict prediction: {e}")


        logger.info("Verdict Prediction Agent initialized")

    def process(
        self,
        evideneces: List[Any]
    ) -> VerdictResult:
        """
        Predict verdict based on evidence.

        Args:
            original_claim: Original claim text
            evidence_results: List of EvidenceResult objects

        Returns:
            VerdictResult with final verdict and explanation
        """

        # Process each subclaim
        subclaim_verdicts = []
        for evidence_result in evideneces:
            logger.debug(f"Evidence result: {evidence_result}")
            verdict = self._predict_subclaim_verdict(evidence_result.subclaim_text, evidence_result.evidence)
            subclaim_verdicts.append(verdict)

        # Aggregate to final verdict
        final_verdict_bool = self._aggregate_verdicts(subclaim_verdicts)
        final_verdict = "SUPPORTED" if final_verdict_bool else "NOT_SUPPORTED"
        logger.debug(f"subclaim verdicts: {subclaim_verdicts}")
        logger.debug(f"Final verdict: {final_verdict}")
        # Generate explanation
        explanation = "Explanation generation not implemented yet."

        result = VerdictResult(
            subclaim_verdicts=subclaim_verdicts,
            final_verdict=final_verdict,
            explanation=explanation,
        )

        logger.info(f"Final result is : {result}")
        return result

    def _predict_subclaim_verdict(self, original_claim: str, evidence_result: List[any]) -> SubclaimVerdict:
        """Predict verdict for a single subclaim using LLM"""

        # Format evidence for the prompt
        cell_parts = []
        # cell_parts.append(f"Subclaim: {evidence_result.subclaim_text}")
        # cell_parts.append(f"Queries: {evidence_result.queries}")
        # cell_parts.append("Evidence:")
        
        for ev in evidence_result:
            cell_parts.append(f"- Source: {ev.source_name}")
            cell_parts.append(f"  Passage: {ev.passage}")
        
        cell = "\n".join(cell_parts)

        # Generate verdict
        prompt = self.prompt_template.format(claim=original_claim, cell=cell)
        response = self.llm._generate_ollama(prompt)
        logger.debug(f"verdiction agent prompt is this : {response}")

            # Parse JSON
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1]
            
        data = json.loads(response.strip())
        return SubclaimVerdict(
            supports=data['label'],
            claim=data['explanation'],
        )

    def _aggregate_verdicts(
        self,
        subclaim_verdicts: List[SubclaimVerdict]
    ) -> bool:
        """Aggregate verdicts to final verdict"""
        for claim in subclaim_verdicts:
            if (claim.supports=="not_supported"):
                return False
        return True

    def to_dict(self, result: VerdictResult) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return asdict(result)
    