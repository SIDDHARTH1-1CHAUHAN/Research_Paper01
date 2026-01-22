# Glossary

Key terms and definitions used in the Multi-Agent Fact-Checker system.

## Core Concepts

### Claim
A statement to be verified as true or false. Example: "The Eiffel Tower was completed in 1889."

### Subclaim
A smaller, independently verifiable component of a claim. Complex claims are decomposed into subclaims for easier verification.

### Verdict
The final determination about a claim's truthfulness:
- **SUPPORTED**: Evidence confirms the claim is true
- **NOT_SUPPORTED**: Evidence contradicts or fails to confirm the claim

### Ground Truth
The known correct answer for a claim, used to evaluate system accuracy.

## Agents

### Claim Decomposition Agent (Input Ingestion)
Breaks complex claims into simpler subclaims using First-Order Logic (FOL).

### Evidence Retrieval Agent (Query Generation + Evidence Seeking)
Generates diverse search queries and retrieves supporting/contradicting evidence.

### Credibility Assessment Agent
Evaluates the reliability and bias of evidence sources.

### Verdict Prediction Agent
Aggregates evidence using weighted voting to determine the final verdict.

### Explainable AI Agent
Generates human-readable explanations for verdicts.

### Reinforcement Learning Agent
Tracks performance metrics and suggests improvements.

## Technical Terms

### FOL (First-Order Logic)
A formal logic system used to decompose claims into structured, verifiable components.

**Example:**
```
Claim: "The Eiffel Tower was completed in 1889 for the World's Fair"

FOL Decomposition:
∃x[EiffelTower(x) ∧ CompletionYear(x, 1889)]
∃y[WorldFair(y) ∧ Year(y, 1889) ∧ Purpose(x, y)]
```

### Credibility Weighting
Assigns weights to evidence based on source reliability:

| Credibility Level | Weight |
|------------------|--------|
| HIGH | 1.0 |
| MEDIUM | 0.6 |
| LOW | 0.3 |

### Verdict Threshold
The confidence score thresholds for verdict determination:

| Score Range | Verdict |
|-------------|---------|
| score >= 0.7 | SUPPORTED |
| score <= 0.3 | NOT_SUPPORTED |
| 0.3 < score < 0.7 | INSUFFICIENT_EVIDENCE |

### 3-Stage Pipeline
The evidence retrieval process:
1. **Search**: Find potential evidence sources
2. **Credibility**: Assess source reliability
3. **Extract**: Extract relevant content

### Weighted Voting
The algorithm that combines evidence scores based on credibility weights:

```
score = sum(evidence_weight * credibility_weight) / sum(credibility_weights)
```

## Evaluation Metrics

### Accuracy
Percentage of correct predictions: `(TP + TN) / Total`

### Precision
Proportion of positive predictions that are correct: `TP / (TP + FP)`

### Recall
Proportion of actual positives correctly identified: `TP / (TP + FN)`

### F1-Score
Harmonic mean of precision and recall: `2 * (P * R) / (P + R)`

### Confusion Matrix
```
                    Predicted
                 SUPPORTED  NOT_SUPPORTED
Actual SUPPORTED     TP          FN
       NOT_SUPPORTED FP          TN
```

- **TP (True Positive)**: Correctly predicted SUPPORTED
- **TN (True Negative)**: Correctly predicted NOT_SUPPORTED
- **FP (False Positive)**: Incorrectly predicted SUPPORTED
- **FN (False Negative)**: Incorrectly predicted NOT_SUPPORTED

## Explanation Quality

### Coverage
Percentage of the verdict decision that is explained (0-1).

### Soundness
Logical consistency of the explanation (0-1).

### Readability
How easy the explanation is for humans to understand (0-1).

## Benchmarks

### FEVEROUS
Fact Extraction and VERification Over Unstructured and Structured information. Tests reasoning over both text and tables.

### HoVer
A dataset for many-hop fact extraction and claim verification. Tests multi-step reasoning.

### SciFact-Open
Scientific claim verification dataset. Tests domain-specific fact-checking.

## System Terms

### Orchestrator
The central coordinator that manages the flow between all agents.

### Mock Dataset
A small test dataset (`data/benchmarks/mock_dataset.json`) for demonstration purposes.

### Fallback Mode
The system's ability to run without an LLM by using heuristic-based processing.

### Query Diversity (k)
The number of diverse search queries generated per subclaim. Default: k=3.

## Ablation Study Terms

### Ablation Study
Systematic removal of system components to measure their contribution to performance.

### Baseline
The full system performance used as reference for comparison.

### Degraded Configuration
A system variant with one or more components disabled or simplified.
