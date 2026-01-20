"""
Explainable AI Agent - LIME/SHAP-inspired explanations for transparency
"""

from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from loguru import logger


@dataclass
class ExplanationResult:
    """Result from Explainable AI Agent"""
    verdict: str
    feature_importance: Dict[str, float]
    critical_evidence: List[str]
    counterfactual_analysis: str
    conflict_resolution: Dict[str, Any]
    explanation_quality: Dict[str, float]


class ExplainableAIAgent:
    """
    Agent 5: Explainable AI

    Provides LIME/SHAP-inspired explanations:
    1. Feature Importance - Which evidence mattered most?
    2. Counterfactual Analysis - What would change the verdict?
    3. Conflict Resolution - How were contradictions handled?

    Metrics:
    - Coverage: % of reasoning explained
    - Soundness: Logical consistency
    - Readability: Human comprehension
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Explainable AI Agent.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        self.enable_feature_importance = self.config.get('enable_feature_importance', True)
        self.enable_counterfactual = self.config.get('enable_counterfactual', True)

        logger.info("Explainable AI Agent initialized")

    def process(
        self,
        verdict_result: Any,
        evidence_results: List[Any]
    ) -> ExplanationResult:
        """
        Generate explanations for verdict.

        Args:
            verdict_result: VerdictResult from Verdict Prediction Agent
            evidence_results: List of EvidenceResult objects

        Returns:
            ExplanationResult with detailed explanations
        """
        logger.info(f"Generating explanations for verdict: {verdict_result.final_verdict}")

        # 1. Feature Importance
        feature_importance = self._calculate_feature_importance(
            verdict_result,
            evidence_results
        )

        # 2. Identify critical evidence
        critical_evidence = self._identify_critical_evidence(
            feature_importance,
            evidence_results
        )

        # 3. Counterfactual analysis
        counterfactual = self._counterfactual_analysis(
            verdict_result,
            feature_importance
        )

        # 4. Conflict resolution
        conflict_resolution = self._analyze_conflicts(evidence_results)

        # 5. Explanation quality metrics
        quality = self._calculate_explanation_quality(
            verdict_result,
            evidence_results,
            feature_importance
        )

        result = ExplanationResult(
            verdict=verdict_result.final_verdict,
            feature_importance=feature_importance,
            critical_evidence=critical_evidence,
            counterfactual_analysis=counterfactual,
            conflict_resolution=conflict_resolution,
            explanation_quality=quality
        )

        logger.info(f"Explanation quality - Coverage: {quality['coverage']:.2f}, Soundness: {quality['soundness']:.2f}")

        return result

    def _calculate_feature_importance(
        self,
        verdict_result: Any,
        evidence_results: List[Any]
    ) -> Dict[str, float]:
        """
        Calculate contribution of each evidence source to the verdict.

        Similar to LIME's local feature importance.
        """
        importance = {}

        for er in evidence_results:
            for evidence in er.evidence:
                # Calculate importance based on:
                # 1. Credibility score
                # 2. Passage length (relevance proxy)
                # 3. Position in results (earlier = more important)

                score = (
                    0.5 * evidence.credibility_score +
                    0.3 * min(len(evidence.passage) / 500, 1.0) +
                    0.2 * (1.0 if evidence.credibility_level == 'high' else 0.5)
                )

                importance[evidence.id] = score

        # Normalize to sum to 1.0
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}

        return importance

    def _identify_critical_evidence(
        self,
        feature_importance: Dict[str, float],
        evidence_results: List[Any]
    ) -> List[str]:
        """Identify evidence that was critical to the decision"""
        # Sort by importance
        sorted_evidence = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Top 3 most important
        critical_ids = [ev_id for ev_id, score in sorted_evidence[:3]]

        # Get full evidence details
        critical = []
        for er in evidence_results:
            for ev in er.evidence:
                if ev.id in critical_ids:
                    critical.append(f"{ev.source_name}: \"{ev.passage[:100]}...\"")

        return critical

    def _counterfactual_analysis(
        self,
        verdict_result: Any,
        feature_importance: Dict[str, float]
    ) -> str:
        """
        Analyze what would need to change to flip the verdict.

        Similar to SHAP's counterfactual explanations.
        """
        current_verdict = verdict_result.final_verdict

        if current_verdict == "SUPPORTED":
            # How many high-importance sources would need to be removed?
            sorted_importance = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )

            cumulative = 0
            count = 0
            for ev_id, score in sorted_importance:
                cumulative += score
                count += 1
                if cumulative > 0.6:  # Threshold
                    break

            return f"Removing the top {count} most important sources would likely change the verdict to NOT_SUPPORTED."

        else:
            return "Additional high-credibility sources supporting all subclaims would be needed to change the verdict to SUPPORTED."

    def _analyze_conflicts(self, evidence_results: List[Any]) -> Dict[str, Any]:
        """Analyze contradictory evidence and how it was resolved"""
        conflicts = []
        resolutions = []

        for er in evidence_results:
            # Check for credibility variance
            if er.evidence:
                cred_levels = [ev.credibility_level for ev in er.evidence]
                unique_levels = set(cred_levels)

                if len(unique_levels) > 1:
                    conflicts.append(f"Subclaim {er.subclaim_id}: Mixed credibility sources")
                    resolutions.append("Higher credibility sources weighted more heavily")

        return {
            'conflicts_detected': len(conflicts),
            'conflicts': conflicts,
            'resolution_methods': resolutions
        }

    def _calculate_explanation_quality(
        self,
        verdict_result: Any,
        evidence_results: List[Any],
        feature_importance: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate explanation quality metrics.

        Metrics:
        - Coverage: How much of the decision is explained?
        - Soundness: Logical consistency
        - Readability: Ease of understanding
        """
        # Coverage: Do we have feature importance for all evidence?
        total_evidence = sum(len(er.evidence) for er in evidence_results)
        explained_evidence = len(feature_importance)
        coverage = explained_evidence / total_evidence if total_evidence > 0 else 0

        # Soundness: Does the verdict match the evidence?
        # High if all subclaims have evidence
        has_evidence = [len(er.evidence) > 0 for er in evidence_results]
        soundness = sum(has_evidence) / len(has_evidence) if has_evidence else 0

        # Readability: Based on explanation length and structure
        explanation_length = len(verdict_result.explanation)
        readability = min(explanation_length / 1000, 1.0)  # Longer explanations score higher

        return {
            'coverage': coverage,
            'soundness': soundness,
            'readability': readability,
            'overall': (coverage + soundness + readability) / 3
        }

    def to_dict(self, result: ExplanationResult) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return asdict(result)
