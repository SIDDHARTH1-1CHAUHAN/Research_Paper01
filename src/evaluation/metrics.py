"""
Evaluation Metrics - Comprehensive metrics for fact-checking performance
"""

from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import statistics
from collections import defaultdict


@dataclass
class ClassificationMetrics:
    """Standard classification metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int


@dataclass
class PerformanceMetrics:
    """System performance metrics"""
    mean_processing_time: float
    total_processing_time: float
    mean_queries_per_claim: float
    mean_evidence_per_claim: float
    mean_high_credibility_ratio: float


@dataclass
class ExplanationMetrics:
    """Explanation quality metrics"""
    mean_coverage: float
    mean_soundness: float
    mean_readability: float
    mean_overall_quality: float


@dataclass
class ComprehensiveMetrics:
    """All metrics combined"""
    classification: ClassificationMetrics
    performance: PerformanceMetrics
    explanation: ExplanationMetrics
    by_category: Dict[str, ClassificationMetrics]
    by_difficulty: Dict[str, ClassificationMetrics]


class MetricsCalculator:
    """Calculate comprehensive evaluation metrics"""

    @staticmethod
    def calculate_classification_metrics(
        predictions: List[str],
        ground_truths: List[str]
    ) -> ClassificationMetrics:
        """
        Calculate classification metrics (accuracy, precision, recall, F1).

        Args:
            predictions: List of predicted verdicts
            ground_truths: List of ground truth labels

        Returns:
            ClassificationMetrics object
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have same length")

        tp = tn = fp = fn = 0

        for pred, truth in zip(predictions, ground_truths):
            if pred == "SUPPORTED" and truth == "SUPPORTED":
                tp += 1
            elif pred == "NOT_SUPPORTED" and truth == "NOT_SUPPORTED":
                tn += 1
            elif pred == "SUPPORTED" and truth == "NOT_SUPPORTED":
                fp += 1
            elif pred == "NOT_SUPPORTED" and truth == "SUPPORTED":
                fn += 1

        total = len(predictions)
        accuracy = (tp + tn) / total if total > 0 else 0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return ClassificationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn
        )

    @staticmethod
    def calculate_performance_metrics(results: List[Dict[str, Any]]) -> PerformanceMetrics:
        """
        Calculate performance metrics (time, queries, evidence).

        Args:
            results: List of verification results

        Returns:
            PerformanceMetrics object
        """
        processing_times = []
        queries_per_claim = []
        evidence_per_claim = []
        high_cred_ratios = []

        for result in results:
            # Processing time
            if 'verdict' in result and 'metadata' in result['verdict']:
                processing_times.append(result['verdict']['metadata'].get('processing_time', 0))

            # Queries per claim
            if 'queries' in result:
                total_queries = sum(len(q['queries']) for q in result['queries'])
                queries_per_claim.append(total_queries)

            # Evidence per claim
            if 'evidence' in result:
                total_evidence = sum(e['total_sources'] for e in result['evidence'])
                high_cred_count = sum(e['high_credibility_count'] for e in result['evidence'])

                evidence_per_claim.append(total_evidence)

                if total_evidence > 0:
                    high_cred_ratios.append(high_cred_count / total_evidence)

        return PerformanceMetrics(
            mean_processing_time=statistics.mean(processing_times) if processing_times else 0,
            total_processing_time=sum(processing_times) if processing_times else 0,
            mean_queries_per_claim=statistics.mean(queries_per_claim) if queries_per_claim else 0,
            mean_evidence_per_claim=statistics.mean(evidence_per_claim) if evidence_per_claim else 0,
            mean_high_credibility_ratio=statistics.mean(high_cred_ratios) if high_cred_ratios else 0
        )

    @staticmethod
    def calculate_explanation_metrics(results: List[Dict[str, Any]]) -> ExplanationMetrics:
        """
        Calculate explanation quality metrics.

        Args:
            results: List of verification results

        Returns:
            ExplanationMetrics object
        """
        coverages = []
        soundnesses = []
        readabilities = []
        overalls = []

        for result in results:
            if 'explanation' in result and 'explanation_quality' in result['explanation']:
                quality = result['explanation']['explanation_quality']
                coverages.append(quality.get('coverage', 0))
                soundnesses.append(quality.get('soundness', 0))
                readabilities.append(quality.get('readability', 0))
                overalls.append(quality.get('overall', 0))

        return ExplanationMetrics(
            mean_coverage=statistics.mean(coverages) if coverages else 0,
            mean_soundness=statistics.mean(soundnesses) if soundnesses else 0,
            mean_readability=statistics.mean(readabilities) if readabilities else 0,
            mean_overall_quality=statistics.mean(overalls) if overalls else 0
        )

    @staticmethod
    def calculate_by_category(
        results: List[Dict[str, Any]],
        dataset: List[Dict[str, Any]]
    ) -> Dict[str, ClassificationMetrics]:
        """
        Calculate metrics grouped by claim category.

        Args:
            results: List of verification results
            dataset: List of dataset entries with categories

        Returns:
            Dictionary of category -> metrics
        """
        categories = defaultdict(lambda: {'preds': [], 'truths': []})

        for result, data_entry in zip(results, dataset):
            category = data_entry.get('category', 'unknown')
            pred = result.get('verdict', {}).get('final_verdict', 'NOT_SUPPORTED')
            truth = data_entry.get('ground_truth', 'NOT_SUPPORTED')

            categories[category]['preds'].append(pred)
            categories[category]['truths'].append(truth)

        metrics_by_category = {}
        for category, data in categories.items():
            metrics_by_category[category] = MetricsCalculator.calculate_classification_metrics(
                data['preds'],
                data['truths']
            )

        return metrics_by_category

    @staticmethod
    def calculate_by_difficulty(
        results: List[Dict[str, Any]],
        dataset: List[Dict[str, Any]]
    ) -> Dict[str, ClassificationMetrics]:
        """
        Calculate metrics grouped by difficulty level.

        Args:
            results: List of verification results
            dataset: List of dataset entries with difficulty levels

        Returns:
            Dictionary of difficulty -> metrics
        """
        difficulties = defaultdict(lambda: {'preds': [], 'truths': []})

        for result, data_entry in zip(results, dataset):
            difficulty = data_entry.get('difficulty', 'unknown')
            pred = result.get('verdict', {}).get('final_verdict', 'NOT_SUPPORTED')
            truth = data_entry.get('ground_truth', 'NOT_SUPPORTED')

            difficulties[difficulty]['preds'].append(pred)
            difficulties[difficulty]['truths'].append(truth)

        metrics_by_difficulty = {}
        for difficulty, data in difficulties.items():
            metrics_by_difficulty[difficulty] = MetricsCalculator.calculate_classification_metrics(
                data['preds'],
                data['truths']
            )

        return metrics_by_difficulty

    @staticmethod
    def calculate_comprehensive_metrics(
        results: List[Dict[str, Any]],
        dataset: List[Dict[str, Any]]
    ) -> ComprehensiveMetrics:
        """
        Calculate all metrics comprehensively.

        Args:
            results: List of verification results
            dataset: List of dataset entries

        Returns:
            ComprehensiveMetrics object with all metrics
        """
        # Extract predictions and ground truths
        predictions = [r.get('verdict', {}).get('final_verdict', 'NOT_SUPPORTED') for r in results]
        ground_truths = [d.get('ground_truth', 'NOT_SUPPORTED') for d in dataset]

        # Calculate all metric types
        classification = MetricsCalculator.calculate_classification_metrics(predictions, ground_truths)
        performance = MetricsCalculator.calculate_performance_metrics(results)
        explanation = MetricsCalculator.calculate_explanation_metrics(results)
        by_category = MetricsCalculator.calculate_by_category(results, dataset)
        by_difficulty = MetricsCalculator.calculate_by_difficulty(results, dataset)

        return ComprehensiveMetrics(
            classification=classification,
            performance=performance,
            explanation=explanation,
            by_category=by_category,
            by_difficulty=by_difficulty
        )

    @staticmethod
    def format_metrics_report(metrics: ComprehensiveMetrics) -> str:
        """
        Format metrics as human-readable report.

        Args:
            metrics: ComprehensiveMetrics object

        Returns:
            Formatted string report
        """
        lines = []
        lines.append("="*80)
        lines.append("COMPREHENSIVE EVALUATION METRICS")
        lines.append("="*80)

        # Classification metrics
        lines.append("\n📊 CLASSIFICATION METRICS")
        lines.append("-" * 40)
        lines.append(f"Accuracy:   {metrics.classification.accuracy:.4f} ({metrics.classification.accuracy*100:.2f}%)")
        lines.append(f"Precision:  {metrics.classification.precision:.4f}")
        lines.append(f"Recall:     {metrics.classification.recall:.4f}")
        lines.append(f"F1-Score:   {metrics.classification.f1_score:.4f}")
        lines.append(f"\nConfusion Matrix:")
        lines.append(f"  True Positives:  {metrics.classification.true_positives}")
        lines.append(f"  True Negatives:  {metrics.classification.true_negatives}")
        lines.append(f"  False Positives: {metrics.classification.false_positives}")
        lines.append(f"  False Negatives: {metrics.classification.false_negatives}")

        # Performance metrics
        lines.append("\n⚡ PERFORMANCE METRICS")
        lines.append("-" * 40)
        lines.append(f"Mean Processing Time:        {metrics.performance.mean_processing_time:.2f}s")
        lines.append(f"Total Processing Time:       {metrics.performance.total_processing_time:.2f}s")
        lines.append(f"Mean Queries per Claim:      {metrics.performance.mean_queries_per_claim:.1f}")
        lines.append(f"Mean Evidence per Claim:     {metrics.performance.mean_evidence_per_claim:.1f}")
        lines.append(f"Mean High Credibility Ratio: {metrics.performance.mean_high_credibility_ratio:.2%}")

        # Explanation metrics
        lines.append("\n💡 EXPLANATION QUALITY METRICS")
        lines.append("-" * 40)
        lines.append(f"Mean Coverage:    {metrics.explanation.mean_coverage:.4f}")
        lines.append(f"Mean Soundness:   {metrics.explanation.mean_soundness:.4f}")
        lines.append(f"Mean Readability: {metrics.explanation.mean_readability:.4f}")
        lines.append(f"Overall Quality:  {metrics.explanation.mean_overall_quality:.4f}")

        # By category
        if metrics.by_category:
            lines.append("\n📂 METRICS BY CATEGORY")
            lines.append("-" * 40)
            for category, cat_metrics in metrics.by_category.items():
                lines.append(f"{category.upper()}:")
                lines.append(f"  Accuracy: {cat_metrics.accuracy:.4f}, F1: {cat_metrics.f1_score:.4f}")

        # By difficulty
        if metrics.by_difficulty:
            lines.append("\n📈 METRICS BY DIFFICULTY")
            lines.append("-" * 40)
            for difficulty, diff_metrics in metrics.by_difficulty.items():
                lines.append(f"{difficulty.upper()}:")
                lines.append(f"  Accuracy: {diff_metrics.accuracy:.4f}, F1: {diff_metrics.f1_score:.4f}")

        lines.append("\n" + "="*80)

        return "\n".join(lines)
