"""
Evaluation Metrics - Comprehensive metrics for fact-checking performance

Supports:
- Macro F1-score for verdict correctness (quantitative)
- Mean Average Rank (MAR) for explanation quality using LLM-as-judge (qualitative)
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
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
class PerClassMetrics:
    """Metrics for a single class (used in Macro F1 calculation)"""
    class_label: str
    precision: float
    recall: float
    f1_score: float
    support: int  # Number of samples in this class


@dataclass
class MacroF1Result:
    """
    Macro F1-score result with per-class breakdown.

    Macro F1 averages F1 scores across classes equally,
    treating each class with equal importance regardless of size.
    """
    macro_f1: float
    per_class_metrics: List[PerClassMetrics]
    labels: List[str] = field(default_factory=lambda: ["supported", "not_supported"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "macro_f1": self.macro_f1,
            "per_class_metrics": [
                {
                    "class": m.class_label,
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1_score": m.f1_score,
                    "support": m.support
                }
                for m in self.per_class_metrics
            ],
            "labels": self.labels
        }


@dataclass
class ExplanationRank:
    """Rank assigned to an explanation by the LLM judge"""
    claim_id: str
    explanation_id: str
    rank: int  # 1 (best) to 4 (worst)
    criteria_scores: Dict[str, float] = field(default_factory=dict)
    # criteria_scores keys: factual_correctness, completeness,
    #                       faithfulness_to_evidence, clarity


@dataclass
class MeanAverageRankResult:
    """
    Mean Average Rank (MAR) result for explanation quality.

    MAR measures the average rank assigned by an LLM judge to explanations.
    Lower is better (1.0 = all explanations ranked first/best).
    """
    mean_average_rank: float
    rank_distribution: Dict[int, int]  # rank -> count
    total_evaluated: int
    criteria_averages: Dict[str, float] = field(default_factory=dict)
    per_claim_ranks: List[ExplanationRank] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_average_rank": self.mean_average_rank,
            "rank_distribution": {
                f"rank_{k}_count": v for k, v in self.rank_distribution.items()
            },
            "total_evaluated": self.total_evaluated,
            "criteria_averages": self.criteria_averages,
            "interpretation": "Lower is better (1.0 = perfect)"
        }


@dataclass
class EvaluationTaskResult:
    """
    Combined evaluation result with both Macro F1 and MAR metrics.
    Follows the evaluation task specification.
    """
    macro_f1_result: MacroF1Result
    mar_result: Optional[MeanAverageRankResult]
    dataset_name: str
    total_samples: int

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "dataset": self.dataset_name,
            "total_samples": self.total_samples,
            "macro_f1": self.macro_f1_result.to_dict()
        }
        if self.mar_result:
            result["mean_average_rank"] = self.mar_result.to_dict()
        return result


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
        lines.append("\n[CLASSIFICATION METRICS]")
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
        lines.append("\n[PERFORMANCE METRICS]")
        lines.append("-" * 40)
        lines.append(f"Mean Processing Time:        {metrics.performance.mean_processing_time:.2f}s")
        lines.append(f"Total Processing Time:       {metrics.performance.total_processing_time:.2f}s")
        lines.append(f"Mean Queries per Claim:      {metrics.performance.mean_queries_per_claim:.1f}")
        lines.append(f"Mean Evidence per Claim:     {metrics.performance.mean_evidence_per_claim:.1f}")
        lines.append(f"Mean High Credibility Ratio: {metrics.performance.mean_high_credibility_ratio:.2%}")

        # Explanation metrics
        lines.append("\n[EXPLANATION QUALITY METRICS]")
        lines.append("-" * 40)
        lines.append(f"Mean Coverage:    {metrics.explanation.mean_coverage:.4f}")
        lines.append(f"Mean Soundness:   {metrics.explanation.mean_soundness:.4f}")
        lines.append(f"Mean Readability: {metrics.explanation.mean_readability:.4f}")
        lines.append(f"Overall Quality:  {metrics.explanation.mean_overall_quality:.4f}")

        # By category
        if metrics.by_category:
            lines.append("\n[METRICS BY CATEGORY]")
            lines.append("-" * 40)
            for category, cat_metrics in metrics.by_category.items():
                lines.append(f"{category.upper()}:")
                lines.append(f"  Accuracy: {cat_metrics.accuracy:.4f}, F1: {cat_metrics.f1_score:.4f}")

        # By difficulty
        if metrics.by_difficulty:
            lines.append("\n[METRICS BY DIFFICULTY]")
            lines.append("-" * 40)
            for difficulty, diff_metrics in metrics.by_difficulty.items():
                lines.append(f"{difficulty.upper()}:")
                lines.append(f"  Accuracy: {diff_metrics.accuracy:.4f}, F1: {diff_metrics.f1_score:.4f}")

        lines.append("\n" + "="*80)

        return "\n".join(lines)

    @staticmethod
    def calculate_macro_f1(
        predictions: List[str],
        ground_truths: List[str],
        labels: List[str] = None
    ) -> MacroF1Result:
        """
        Calculate Macro F1-score with per-class breakdown.

        Macro F1 computes F1 for each class independently, then averages
        them equally. This gives equal weight to each class regardless
        of class imbalance.

        Args:
            predictions: List of predicted labels
            ground_truths: List of gold-standard labels
            labels: List of class labels (default: ["supported", "not_supported"])

        Returns:
            MacroF1Result with macro F1 and per-class metrics
        """
        if labels is None:
            labels = ["supported", "not_supported"]

        # Normalize labels to lowercase
        predictions = [p.lower().replace("_", "_") for p in predictions]
        ground_truths = [g.lower().replace("_", "_") for g in ground_truths]

        per_class_metrics = []
        f1_scores = []

        for label in labels:
            # Calculate TP, FP, FN for this class
            tp = sum(1 for p, g in zip(predictions, ground_truths)
                     if p == label and g == label)
            fp = sum(1 for p, g in zip(predictions, ground_truths)
                     if p == label and g != label)
            fn = sum(1 for p, g in zip(predictions, ground_truths)
                     if p != label and g == label)
            support = sum(1 for g in ground_truths if g == label)

            # Calculate precision, recall, F1 for this class
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            per_class_metrics.append(PerClassMetrics(
                class_label=label,
                precision=precision,
                recall=recall,
                f1_score=f1,
                support=support
            ))
            f1_scores.append(f1)

        # Macro F1 is the simple average of per-class F1 scores
        macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

        return MacroF1Result(
            macro_f1=macro_f1,
            per_class_metrics=per_class_metrics,
            labels=labels
        )

    @staticmethod
    def calculate_mean_average_rank(
        ranks: List[ExplanationRank]
    ) -> MeanAverageRankResult:
        """
        Calculate Mean Average Rank (MAR) from LLM judge rankings.

        MAR is the average of ranks assigned to explanations.
        Lower is better (1.0 = all explanations ranked best).

        Args:
            ranks: List of ExplanationRank objects with rankings 1-4

        Returns:
            MeanAverageRankResult with MAR and distribution
        """
        if not ranks:
            return MeanAverageRankResult(
                mean_average_rank=0.0,
                rank_distribution={1: 0, 2: 0, 3: 0, 4: 0},
                total_evaluated=0,
                criteria_averages={},
                per_claim_ranks=[]
            )

        # Calculate rank distribution
        rank_distribution = {1: 0, 2: 0, 3: 0, 4: 0}
        for r in ranks:
            if 1 <= r.rank <= 4:
                rank_distribution[r.rank] += 1

        # Calculate mean average rank
        total_rank = sum(r.rank for r in ranks)
        mar = total_rank / len(ranks)

        # Calculate average criteria scores if available
        criteria_keys = ["factual_correctness", "completeness",
                         "faithfulness_to_evidence", "clarity"]
        criteria_averages = {}

        for key in criteria_keys:
            scores = [r.criteria_scores.get(key, 0) for r in ranks
                      if key in r.criteria_scores]
            if scores:
                criteria_averages[key] = sum(scores) / len(scores)

        return MeanAverageRankResult(
            mean_average_rank=mar,
            rank_distribution=rank_distribution,
            total_evaluated=len(ranks),
            criteria_averages=criteria_averages,
            per_claim_ranks=ranks
        )

    @staticmethod
    def evaluate_with_llm_judge(
        claims: List[Dict[str, Any]],
        explanations: List[Dict[str, Any]],
        llm_interface: Any
    ) -> MeanAverageRankResult:
        """
        Evaluate explanations using an LLM as a judge.

        The LLM ranks explanations from 1 (best) to 4 (worst) based on:
        - Factual correctness
        - Completeness
        - Faithfulness to evidence
        - Clarity

        Args:
            claims: List of claims with evidence
            explanations: List of explanation candidates per claim
            llm_interface: LLM interface for judging

        Returns:
            MeanAverageRankResult with MAR and rankings
        """
        ranks = []

        judge_prompt_template = """
You are an expert judge evaluating the quality of fact-checking explanations.

CLAIM: {claim}

EVIDENCE:
{evidence}

EXPLANATION TO EVALUATE:
{explanation}

Rate this explanation on a scale of 1-4 where:
1 = Excellent (accurate, complete, faithful to evidence, clear)
2 = Good (mostly accurate with minor issues)
3 = Fair (some inaccuracies or incompleteness)
4 = Poor (significant issues)

Evaluate based on these criteria:
1. Factual Correctness: Does it accurately reflect the evidence? (Score 0-1)
2. Completeness: Does it cover all relevant aspects? (Score 0-1)
3. Faithfulness: Is it grounded in the retrieved evidence? (Score 0-1)
4. Clarity: Is it clear and understandable? (Score 0-1)

Respond in JSON format:
{{
    "overall_rank": <1-4>,
    "factual_correctness": <0-1>,
    "completeness": <0-1>,
    "faithfulness_to_evidence": <0-1>,
    "clarity": <0-1>
}}
"""

        for claim_data, explanation_data in zip(claims, explanations):
            claim_id = claim_data.get("claim_id", str(len(ranks)))
            claim_text = claim_data.get("claim_text", "")
            evidence = claim_data.get("evidence", "")

            for exp_candidate in explanation_data.get("explanation_candidates", []):
                exp_id = exp_candidate.get("explanation_id", "")
                exp_text = exp_candidate.get("explanation_text", "")

                prompt = judge_prompt_template.format(
                    claim=claim_text,
                    evidence=evidence,
                    explanation=exp_text
                )

                try:
                    if llm_interface:
                        response = llm_interface.generate(prompt)
                        # Parse JSON response
                        import json
                        result = json.loads(response)

                        ranks.append(ExplanationRank(
                            claim_id=claim_id,
                            explanation_id=exp_id,
                            rank=result.get("overall_rank", 4),
                            criteria_scores={
                                "factual_correctness": result.get("factual_correctness", 0),
                                "completeness": result.get("completeness", 0),
                                "faithfulness_to_evidence": result.get("faithfulness_to_evidence", 0),
                                "clarity": result.get("clarity", 0)
                            }
                        ))
                    else:
                        # Fallback: assign middle rank if no LLM available
                        ranks.append(ExplanationRank(
                            claim_id=claim_id,
                            explanation_id=exp_id,
                            rank=2,
                            criteria_scores={}
                        ))
                except Exception as e:
                    # On error, assign worst rank
                    ranks.append(ExplanationRank(
                        claim_id=claim_id,
                        explanation_id=exp_id,
                        rank=4,
                        criteria_scores={}
                    ))

        return MetricsCalculator.calculate_mean_average_rank(ranks)

    @staticmethod
    def run_evaluation_task(
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        explanations: Optional[List[Dict[str, Any]]] = None,
        llm_interface: Any = None,
        dataset_name: str = "unknown"
    ) -> EvaluationTaskResult:
        """
        Run the complete evaluation task as specified in evaluation_task.yaml.

        Computes:
        1. Macro F1-score for verdict correctness
        2. Mean Average Rank (MAR) for explanation quality (if explanations provided)

        Args:
            predictions: List of verdict predictions with claim_id, predicted_label
            ground_truths: List of gold labels with claim_id, gold_label
            explanations: Optional list of explanations for MAR evaluation
            llm_interface: LLM interface for MAR judging
            dataset_name: Name of the dataset being evaluated

        Returns:
            EvaluationTaskResult with both metrics
        """
        # Extract predicted and gold labels
        pred_labels = [p.get("predicted_label", "not_supported").lower()
                       for p in predictions]
        gold_labels = [g.get("gold_label", "not_supported").lower()
                       for g in ground_truths]

        # Calculate Macro F1
        macro_f1_result = MetricsCalculator.calculate_macro_f1(
            pred_labels, gold_labels
        )

        # Calculate MAR if explanations provided
        mar_result = None
        if explanations and llm_interface:
            # Build claims list for MAR evaluation
            claims_for_mar = []
            for pred, gt in zip(predictions, ground_truths):
                claims_for_mar.append({
                    "claim_id": pred.get("claim_id", ""),
                    "claim_text": pred.get("claim_text", ""),
                    "evidence": pred.get("evidence", "")
                })

            mar_result = MetricsCalculator.evaluate_with_llm_judge(
                claims_for_mar, explanations, llm_interface
            )

        return EvaluationTaskResult(
            macro_f1_result=macro_f1_result,
            mar_result=mar_result,
            dataset_name=dataset_name,
            total_samples=len(predictions)
        )

    @staticmethod
    def format_evaluation_report(result: EvaluationTaskResult) -> str:
        """
        Format evaluation task result as a human-readable report.

        Args:
            result: EvaluationTaskResult object

        Returns:
            Formatted string report
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"EVALUATION REPORT: {result.dataset_name.upper()}")
        lines.append("=" * 80)
        lines.append(f"Total samples evaluated: {result.total_samples}")

        # Macro F1 section
        lines.append("\n[MACRO F1-SCORE]")
        lines.append("-" * 40)
        lines.append(f"Macro F1: {result.macro_f1_result.macro_f1:.4f}")
        lines.append("\nPer-class breakdown:")
        for m in result.macro_f1_result.per_class_metrics:
            lines.append(f"  {m.class_label.upper()}:")
            lines.append(f"    Precision: {m.precision:.4f}")
            lines.append(f"    Recall:    {m.recall:.4f}")
            lines.append(f"    F1-Score:  {m.f1_score:.4f}")
            lines.append(f"    Support:   {m.support}")

        # MAR section (if available)
        if result.mar_result:
            lines.append("\n[MEAN AVERAGE RANK (MAR)]")
            lines.append("-" * 40)
            lines.append(f"MAR: {result.mar_result.mean_average_rank:.4f}")
            lines.append("(Lower is better, 1.0 = perfect)")
            lines.append("\nRank distribution:")
            for rank, count in sorted(result.mar_result.rank_distribution.items()):
                lines.append(f"  Rank {rank}: {count} explanations")

            if result.mar_result.criteria_averages:
                lines.append("\nCriteria averages:")
                for criterion, score in result.mar_result.criteria_averages.items():
                    lines.append(f"  {criterion}: {score:.4f}")

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)
