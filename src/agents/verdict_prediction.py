"""
Verdict Prediction Agent - Evidence aggregation and verdict synthesis
"""

from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from loguru import logger
import time


@dataclass
class SubclaimVerdict:
    """Verdict for a single subclaim"""
    subclaim_id: str
    verdict: str  # "SUPPORTED" | "NOT_SUPPORTED"
    confidence: float
    evidence_count: int
    high_credibility_count: int


@dataclass
class VerdictResult:
    """Final verdict result"""
    original_claim: str
    subclaim_verdicts: List[SubclaimVerdict]
    final_verdict: str
    overall_confidence: float
    explanation: str
    metadata: Dict[str, Any]


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

        # Credibility weights
        self.weights = self.config.get('weights', {
            'high_credibility': 1.0,
            'medium_credibility': 0.6,
            'low_credibility': 0.3
        })

        # Decision thresholds
        self.support_threshold = self.config.get('support_threshold', 0.7)
        self.refute_threshold = self.config.get('refute_threshold', 0.3)

        logger.info("Verdict Prediction Agent initialized")

    def process(
        self,
        original_claim: str,
        evidence_results: List[Any]
    ) -> VerdictResult:
        """
        Predict verdict based on evidence.

        Args:
            original_claim: Original claim text
            evidence_results: List of EvidenceResult objects

        Returns:
            VerdictResult with final verdict and explanation
        """
        start_time = time.time()
        logger.info(f"Predicting verdict for: {original_claim}")

        # Process each subclaim
        subclaim_verdicts = []
        for er in evidence_results:
            verdict = self._predict_subclaim_verdict(er)
            subclaim_verdicts.append(verdict)

        # Aggregate to final verdict
        final_verdict, overall_confidence = self._aggregate_verdicts(subclaim_verdicts)

        # Generate explanation
        explanation = self._generate_explanation(
            original_claim,
            subclaim_verdicts,
            evidence_results,
            final_verdict
        )

        processing_time = time.time() - start_time

        # Calculate metadata
        total_sources = sum(sv.evidence_count for sv in subclaim_verdicts)
        high_cred_sources = sum(sv.high_credibility_count for sv in subclaim_verdicts)

        result = VerdictResult(
            original_claim=original_claim,
            subclaim_verdicts=subclaim_verdicts,
            final_verdict=final_verdict,
            overall_confidence=overall_confidence,
            explanation=explanation,
            metadata={
                'total_sources': total_sources,
                'high_credibility_sources': high_cred_sources,
                'processing_time': processing_time,
                'subclaims_evaluated': len(subclaim_verdicts)
            }
        )

        logger.info(f"Final verdict: {final_verdict} (confidence: {overall_confidence:.2f})")
        return result

    def _predict_subclaim_verdict(self, evidence_result: Any) -> SubclaimVerdict:
        """Predict verdict for a single subclaim"""
        if not evidence_result.evidence:
            return SubclaimVerdict(
                subclaim_id=evidence_result.subclaim_id,
                verdict="NOT_SUPPORTED",
                confidence=0.0,
                evidence_count=0,
                high_credibility_count=0
            )

        # Weighted voting
        total_weight = 0
        support_weight = 0

        for evidence in evidence_result.evidence:
            # Get weight based on credibility
            weight = self.weights.get(f'{evidence.credibility_level}_credibility', 0.3)

            # Simple keyword matching for support (in real system, use LLM)
            is_supporting = len(evidence.passage) > 50  # Mock: assume longer passages are supporting

            if is_supporting:
                support_weight += weight

            total_weight += weight

        # Calculate score
        score = support_weight / total_weight if total_weight > 0 else 0

        # Determine verdict
        if score >= self.support_threshold:
            verdict = "SUPPORTED"
        elif score <= self.refute_threshold:
            verdict = "NOT_SUPPORTED"
        else:
            verdict = "INSUFFICIENT_EVIDENCE"

        return SubclaimVerdict(
            subclaim_id=evidence_result.subclaim_id,
            verdict=verdict,
            confidence=score,
            evidence_count=len(evidence_result.evidence),
            high_credibility_count=evidence_result.high_credibility_count
        )

    def _aggregate_verdicts(
        self,
        subclaim_verdicts: List[SubclaimVerdict]
    ) -> tuple[str, float]:
        """Aggregate subclaim verdicts to final verdict"""
        if not subclaim_verdicts:
            return "NOT_SUPPORTED", 0.0

        # All subclaims must be SUPPORTED for final SUPPORTED
        all_supported = all(sv.verdict == "SUPPORTED" for sv in subclaim_verdicts)

        if all_supported:
            avg_confidence = sum(sv.confidence for sv in subclaim_verdicts) / len(subclaim_verdicts)
            return "SUPPORTED", avg_confidence
        else:
            # If any subclaim is NOT_SUPPORTED, final is NOT_SUPPORTED
            return "NOT_SUPPORTED", 0.3

    def _generate_explanation(
        self,
        claim: str,
        verdicts: List[SubclaimVerdict],
        evidence_results: List[Any],
        final_verdict: str
    ) -> str:
        """Generate human-readable explanation"""
        lines = []
        lines.append(f"Claim: {claim}")
        lines.append(f"\nFinal Verdict: {final_verdict}\n")

        # Explain each subclaim
        for verdict, evidence_result in zip(verdicts, evidence_results):
            lines.append(f"Subclaim {verdict.subclaim_id}: {verdict.verdict}")
            lines.append(f"  Confidence: {verdict.confidence:.2f}")
            lines.append(f"  Evidence: {verdict.evidence_count} sources ({verdict.high_credibility_count} high credibility)")

            # Show top evidence
            if evidence_result.evidence:
                lines.append("  Top evidence:")
                for ev in evidence_result.evidence[:2]:
                    lines.append(f"    - {ev.source_name} ({ev.credibility_level})")
                    lines.append(f"      \"{ev.passage[:100]}...\"")

        # Overall summary
        total_sources = sum(v.evidence_count for v in verdicts)
        high_cred = sum(v.high_credibility_count for v in verdicts)

        lines.append(f"\nSummary:")
        lines.append(f"  Total sources analyzed: {total_sources}")
        lines.append(f"  High credibility sources: {high_cred}")
        lines.append(f"  Subclaims evaluated: {len(verdicts)}")

        if final_verdict == "SUPPORTED":
            lines.append(f"\nAll subclaims are supported by credible sources.")
        else:
            lines.append(f"\nOne or more subclaims could not be verified.")

        return "\n".join(lines)

    def to_dict(self, result: VerdictResult) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            'original_claim': result.original_claim,
            'subclaim_verdicts': [asdict(sv) for sv in result.subclaim_verdicts],
            'final_verdict': result.final_verdict,
            'overall_confidence': result.overall_confidence,
            'explanation': result.explanation,
            'metadata': result.metadata
        }
