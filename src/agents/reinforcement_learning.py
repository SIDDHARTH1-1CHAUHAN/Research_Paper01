"""
Reinforcement Learning Agent - Heuristic-based performance tracking and optimization
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from loguru import logger
from collections import defaultdict
import statistics


@dataclass
class PerformanceMetrics:
    """Performance metrics for a run"""
    run_id: int
    claim: str
    verdict: str
    ground_truth: Optional[str]
    accuracy: float
    evidence_quality: float
    efficiency: float
    processing_time: float


@dataclass
class RLResult:
    """Result from Reinforcement Learning Agent"""
    performance_score: float
    patterns: Dict[str, Any]
    suggestions: List[str]
    metrics: Dict[str, float]


class ReinforcementLearningAgent:
    """
    Agent 6: Reinforcement Learning (Heuristic-Based)

    Tracks performance over time and suggests improvements using
    statistical pattern analysis (not gradient-based RL).

    Performance scoring:
    - Accuracy: 1.0 weight
    - Evidence Quality: 0.3 weight
    - Efficiency: 0.2 weight
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize RL Agent.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Scoring weights
        self.accuracy_weight = self.config.get('accuracy_weight', 1.0)
        self.evidence_quality_weight = self.config.get('evidence_quality_weight', 0.3)
        self.efficiency_weight = self.config.get('efficiency_weight', 0.2)

        # Storage
        self.run_history: List[PerformanceMetrics] = []
        self.run_counter = 0

        # Pattern tracking
        self.query_success_rates: Dict[str, List[float]] = defaultdict(list)
        self.source_reliability: Dict[str, List[float]] = defaultdict(list)

        logger.info("Reinforcement Learning Agent initialized (heuristic-based)")

    def record_run(
        self,
        claim: str,
        verdict_result: Any,
        evidence_results: List[Any],
        ground_truth: Optional[str] = None
    ) -> PerformanceMetrics:
        """
        Record a fact-checking run for analysis.

        Args:
            claim: Original claim
            verdict_result: VerdictResult object
            evidence_results: List of EvidenceResult objects
            ground_truth: Optional ground truth label

        Returns:
            PerformanceMetrics for this run
        """
        self.run_counter += 1

        # Calculate accuracy
        if ground_truth:
            accuracy = 1.0 if verdict_result.final_verdict == ground_truth else 0.0
        else:
            accuracy = verdict_result.overall_confidence  # Use confidence as proxy

        # Calculate evidence quality
        total_evidence = sum(len(er.evidence) for er in evidence_results)
        high_cred_evidence = sum(er.high_credibility_count for er in evidence_results)
        evidence_quality = high_cred_evidence / total_evidence if total_evidence > 0 else 0

        # Calculate efficiency (inverse of processing time)
        processing_time = verdict_result.metadata.get('processing_time', 1.0)
        efficiency = 1.0 / max(processing_time, 0.1)  # Avoid division by zero

        metrics = PerformanceMetrics(
            run_id=self.run_counter,
            claim=claim,
            verdict=verdict_result.final_verdict,
            ground_truth=ground_truth,
            accuracy=accuracy,
            evidence_quality=evidence_quality,
            efficiency=efficiency,
            processing_time=processing_time
        )

        self.run_history.append(metrics)

        logger.info(f"Run {self.run_counter} recorded: accuracy={accuracy:.2f}, quality={evidence_quality:.2f}")

        return metrics

    def process(self) -> RLResult:
        """
        Analyze performance patterns and generate suggestions.

        Returns:
            RLResult with performance score, patterns, and suggestions
        """
        logger.info(f"Analyzing {len(self.run_history)} runs")

        if not self.run_history:
            return RLResult(
                performance_score=0.0,
                patterns={},
                suggestions=["No runs recorded yet. Process more claims to generate insights."],
                metrics={}
            )

        # Calculate overall performance score
        performance_score = self._calculate_performance_score()

        # Analyze patterns
        patterns = self._analyze_patterns()

        # Generate suggestions
        suggestions = self._generate_suggestions(patterns)

        # Aggregate metrics
        metrics = self._aggregate_metrics()

        result = RLResult(
            performance_score=performance_score,
            patterns=patterns,
            suggestions=suggestions,
            metrics=metrics
        )

        logger.info(f"Performance score: {performance_score:.2f}")

        return result

    def _calculate_performance_score(self) -> float:
        """Calculate weighted performance score"""
        if not self.run_history:
            return 0.0

        scores = []
        for run in self.run_history:
            score = (
                self.accuracy_weight * run.accuracy +
                self.evidence_quality_weight * run.evidence_quality +
                self.efficiency_weight * run.efficiency
            )
            scores.append(score)

        return statistics.mean(scores)

    def _analyze_patterns(self) -> Dict[str, Any]:
        """Analyze performance patterns"""
        patterns = {}

        if len(self.run_history) < 2:
            return patterns

        # Accuracy trend
        accuracies = [run.accuracy for run in self.run_history]
        patterns['accuracy_trend'] = {
            'mean': statistics.mean(accuracies),
            'stdev': statistics.stdev(accuracies) if len(accuracies) > 1 else 0,
            'recent_mean': statistics.mean(accuracies[-5:]) if len(accuracies) >= 5 else statistics.mean(accuracies)
        }

        # Evidence quality trend
        qualities = [run.evidence_quality for run in self.run_history]
        patterns['evidence_quality_trend'] = {
            'mean': statistics.mean(qualities),
            'stdev': statistics.stdev(qualities) if len(qualities) > 1 else 0,
            'recent_mean': statistics.mean(qualities[-5:]) if len(qualities) >= 5 else statistics.mean(qualities)
        }

        # Efficiency trend
        efficiencies = [run.efficiency for run in self.run_history]
        patterns['efficiency_trend'] = {
            'mean': statistics.mean(efficiencies),
            'recent_mean': statistics.mean(efficiencies[-5:]) if len(efficiencies) >= 5 else statistics.mean(efficiencies)
        }

        # Processing time trend
        times = [run.processing_time for run in self.run_history]
        patterns['processing_time_trend'] = {
            'mean': statistics.mean(times),
            'min': min(times),
            'max': max(times)
        }

        return patterns

    def _generate_suggestions(self, patterns: Dict[str, Any]) -> List[str]:
        """Generate improvement suggestions based on patterns"""
        suggestions = []

        # Accuracy suggestions
        if patterns.get('accuracy_trend'):
            acc_mean = patterns['accuracy_trend']['mean']
            if acc_mean < 0.7:
                suggestions.append(
                    f"Accuracy is low (mean: {acc_mean:.2f}). Consider:\n"
                    "  - Increasing queries per subclaim (current k=3, try k=4)\n"
                    "  - Raising credibility threshold to filter low-quality sources"
                )
            elif acc_mean > 0.85:
                suggestions.append(f"Excellent accuracy (mean: {acc_mean:.2f})! System is performing well.")

        # Evidence quality suggestions
        if patterns.get('evidence_quality_trend'):
            qual_mean = patterns['evidence_quality_trend']['mean']
            if qual_mean < 0.6:
                suggestions.append(
                    f"Evidence quality is low (mean: {qual_mean:.2f}). Consider:\n"
                    "  - Using stricter credibility threshold (current: medium, try: high)\n"
                    "  - Increasing max search results to find more high-quality sources"
                )

        # Efficiency suggestions
        if patterns.get('processing_time_trend'):
            time_mean = patterns['processing_time_trend']['mean']
            if time_mean > 10.0:
                suggestions.append(
                    f"Processing is slow (mean: {time_mean:.2f}s). Consider:\n"
                    "  - Reducing queries per subclaim\n"
                    "  - Enabling caching for repeated claims\n"
                    "  - Using parallel processing"
                )

        # Trend-based suggestions
        if patterns.get('accuracy_trend'):
            recent_acc = patterns['accuracy_trend'].get('recent_mean', 0)
            overall_acc = patterns['accuracy_trend']['mean']

            if recent_acc < overall_acc - 0.1:
                suggestions.append(
                    "Recent accuracy declining. Check if:\n"
                    "  - Claim difficulty has increased\n"
                    "  - Source availability has changed"
                )

        if not suggestions:
            suggestions.append("System is operating normally. Continue monitoring performance.")

        return suggestions

    def _aggregate_metrics(self) -> Dict[str, float]:
        """Aggregate metrics across all runs"""
        if not self.run_history:
            return {}

        return {
            'total_runs': len(self.run_history),
            'mean_accuracy': statistics.mean(run.accuracy for run in self.run_history),
            'mean_evidence_quality': statistics.mean(run.evidence_quality for run in self.run_history),
            'mean_processing_time': statistics.mean(run.processing_time for run in self.run_history),
            'total_processing_time': sum(run.processing_time for run in self.run_history)
        }

    def to_dict(self, result: RLResult) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return asdict(result)
