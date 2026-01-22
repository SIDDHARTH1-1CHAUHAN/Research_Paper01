# Evaluation Methodology

This document explains how the Multi-Agent Fact-Checker system evaluates accuracy and measures performance.

## Overview

The system uses **exact string matching** for evaluation - NOT LLM-as-judge or semantic similarity approaches.

| Aspect | Implementation |
|--------|---------------|
| **Matching Method** | Exact string comparison (`pred == truth`) |
| **LLM-as-Judge?** | No |
| **Semantic Similarity?** | No |
| **Classification Type** | Binary: "SUPPORTED" vs "NOT_SUPPORTED" |

## Why Exact String Matching?

The system uses exact string matching because:

1. **Binary Classification**: Verdicts are discrete labels ("SUPPORTED" or "NOT_SUPPORTED"), not continuous scores
2. **Deterministic Evaluation**: Reproducible results without variability from LLM judges
3. **Standard Practice**: Aligns with established fact-checking benchmarks (FEVEROUS, HoVer, SciFact)
4. **Simplicity**: No additional model inference required for evaluation

## How Accuracy is Calculated

### Core Implementation

From `src/evaluation/metrics.py`:

```python
def calculate_classification_metrics(
    predictions: List[str],
    ground_truths: List[str]
) -> ClassificationMetrics:
    """Calculate classification metrics (accuracy, precision, recall, F1)."""

    tp = tn = fp = fn = 0

    for pred, truth in zip(predictions, ground_truths):
        if pred == "SUPPORTED" and truth == "SUPPORTED":
            tp += 1  # True Positive
        elif pred == "NOT_SUPPORTED" and truth == "NOT_SUPPORTED":
            tn += 1  # True Negative
        elif pred == "SUPPORTED" and truth == "NOT_SUPPORTED":
            fp += 1  # False Positive
        elif pred == "NOT_SUPPORTED" and truth == "SUPPORTED":
            fn += 1  # False Negative

    total = len(predictions)
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
```

### Classification Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Accuracy** | (TP + TN) / Total | Overall correctness rate |
| **Precision** | TP / (TP + FP) | How many "SUPPORTED" predictions were correct |
| **Recall** | TP / (TP + FN) | How many actual "SUPPORTED" cases were found |
| **F1-Score** | 2 * (P * R) / (P + R) | Harmonic mean of precision and recall |

### Confusion Matrix

```
                    Predicted
                 SUPPORTED  NOT_SUPPORTED
Actual SUPPORTED     TP          FN
       NOT_SUPPORTED FP          TN
```

## Reinforcement Learning Agent Accuracy

The RL agent (`src/agents/reinforcement_learning.py`) records accuracy per run:

```python
# Calculate accuracy
if ground_truth:
    accuracy = 1.0 if verdict_result.final_verdict == ground_truth else 0.0
else:
    accuracy = verdict_result.overall_confidence  # Use confidence as proxy
```

- **Accuracy = 1.0**: System's verdict matches ground truth exactly
- **Accuracy = 0.0**: System's verdict doesn't match
- **No ground truth**: Uses confidence score as a proxy

## Performance Metrics

Beyond accuracy, the system tracks:

| Metric | Description |
|--------|-------------|
| Mean Processing Time | Average time to verify a claim |
| Queries per Claim | Number of search queries generated |
| Evidence per Claim | Number of evidence pieces retrieved |
| High Credibility Ratio | Proportion of high-credibility sources |

## Explanation Quality Metrics

Separate from accuracy (does not affect verdict evaluation):

| Metric | Description |
|--------|-------------|
| Coverage | Percentage of decision explained |
| Soundness | Logical consistency of explanation |
| Readability | Human comprehension score |
| Overall Quality | Combined explanation score |

## Comparison to Alternative Approaches

### LLM-as-Judge (Not Used)

LLM-as-judge would use a language model to compare predictions to ground truth:

```python
# NOT IMPLEMENTED - Example of what LLM-as-judge would look like
prompt = f"""
Compare the prediction '{prediction}' to the ground truth '{ground_truth}'.
Are they semantically equivalent? Score from 0-1.
"""
score = llm.evaluate(prompt)
```

**Why not used:**
- Adds latency and cost
- Introduces variability between evaluation runs
- Unnecessary for discrete binary labels

### Semantic Similarity (Not Used)

Semantic similarity would use embeddings to compare texts:

```python
# NOT IMPLEMENTED - Example of what semantic similarity would look like
pred_embedding = embedding_model.encode(prediction)
truth_embedding = embedding_model.encode(ground_truth)
similarity = cosine_similarity(pred_embedding, truth_embedding)
```

**Why not used:**
- Verdicts are discrete labels, not free-form text
- "SUPPORTED" and "NOT_SUPPORTED" have no semantic ambiguity
- Embedding comparison adds unnecessary complexity

## Key Files

| File | Purpose |
|------|---------|
| `src/evaluation/metrics.py` | Core metrics calculation |
| `src/agents/reinforcement_learning.py` | Per-run accuracy recording |
| `tests/test_ablation_studies.py` | Ablation study metrics |
| `src/agents/verdict_prediction.py` | Verdict determination logic |

## Ablation Study Methodology

Ablation studies compare full system performance to degraded variants:

| Configuration | Accuracy Impact |
|--------------|-----------------|
| Without FOL decomposition | -12% |
| Without query diversity (k=3) | -8% |
| Without credibility weighting | -15% |
| Without 3-stage pipeline | -10% |

See [ABLATION_STUDIES.md](../evaluation/ABLATION_STUDIES.md) for detailed results.

## Summary

The Multi-Agent Fact-Checker uses a straightforward, reproducible evaluation methodology:

1. **Exact match** between predicted and ground truth labels
2. **Standard metrics**: Accuracy, Precision, Recall, F1
3. **No LLM-as-judge** or semantic similarity - unnecessary for discrete labels
4. **Separate tracking** for performance and explanation quality
