# Multi-Agent Fact-Checking System - Demo Observations

**Date:** 2026-01-21 03:12:12

**Dataset:** Mock Benchmark (10 claims)

---

## Executive Summary

- **Total Claims Processed:** 10
- **Overall Accuracy:** 80.00%
- **F1-Score:** 0.8571
- **Precision:** 1.0000
- **Recall:** 0.7500
- **Explanation Quality:** 0.7833/1.0

---

## Detailed Evaluation Metrics

### Classification Metrics

- **Accuracy:** 80.00% (8/10 correct)
- **Precision:** 1.0000
- **Recall:** 0.7500
- **F1-Score:** 0.8571

**Confusion Matrix:**
- True Positives: 6
- True Negatives: 2
- False Positives: 0
- False Negatives: 2

### Performance Metrics

- **Mean Queries per Claim:** 5.4
- **Mean Evidence per Claim:** 5.4
- **High Credibility Ratio:** 66.67%

### Explanation Quality

- **Coverage:** 0.85 (85% of reasoning explained)
- **Soundness:** 0.82 (logically consistent)
- **Readability:** 0.68 (easily understood)
- **Overall:** 0.78

---

## Claim-by-Claim Results

### Claim 1: The Eiffel Tower was completed in 1889 and is located in Paris, France

- **Ground Truth:** SUPPORTED
- **Predicted Verdict:** SUPPORTED
- **Confidence:** 85.00%
- **Result:** ✓ CORRECT
- **Evidence:** 6 sources (4 high credibility)
- **Subclaims:** 2
- **Category:** simple
- **Difficulty:** easy

### Claim 2: Albert Einstein was born in Germany in 1879 and won the Nobel Prize in Physics in 1921

- **Ground Truth:** SUPPORTED
- **Predicted Verdict:** SUPPORTED
- **Confidence:** 85.00%
- **Result:** ✓ CORRECT
- **Evidence:** 6 sources (4 high credibility)
- **Subclaims:** 2
- **Category:** biographical
- **Difficulty:** easy

### Claim 3: The Great Wall of China is visible from space and was built in the 5th century BC

- **Ground Truth:** NOT_SUPPORTED
- **Predicted Verdict:** NOT_SUPPORTED
- **Confidence:** 85.00%
- **Result:** ✓ CORRECT
- **Evidence:** 6 sources (4 high credibility)
- **Subclaims:** 2
- **Category:** mixed
- **Difficulty:** medium

### Claim 4: Water boils at 100 degrees Celsius at sea level

- **Ground Truth:** SUPPORTED
- **Predicted Verdict:** SUPPORTED
- **Confidence:** 85.00%
- **Result:** ✓ CORRECT
- **Evidence:** 3 sources (2 high credibility)
- **Subclaims:** 1
- **Category:** scientific
- **Difficulty:** easy

### Claim 5: The Python programming language was created by Guido van Rossum and first released in 1991

- **Ground Truth:** SUPPORTED
- **Predicted Verdict:** NOT_SUPPORTED
- **Confidence:** 45.00%
- **Result:** ✗ INCORRECT
- **Evidence:** 6 sources (4 high credibility)
- **Subclaims:** 2
- **Category:** technology
- **Difficulty:** easy

### Claim 6: Mount Everest is the tallest mountain on Earth and is located in Nepal

- **Ground Truth:** NOT_SUPPORTED
- **Predicted Verdict:** NOT_SUPPORTED
- **Confidence:** 85.00%
- **Result:** ✓ CORRECT
- **Evidence:** 6 sources (4 high credibility)
- **Subclaims:** 2
- **Category:** geographical
- **Difficulty:** medium

### Claim 7: The Earth orbits the Sun and completes one orbit every 365.25 days

- **Ground Truth:** SUPPORTED
- **Predicted Verdict:** SUPPORTED
- **Confidence:** 85.00%
- **Result:** ✓ CORRECT
- **Evidence:** 6 sources (4 high credibility)
- **Subclaims:** 2
- **Category:** scientific
- **Difficulty:** easy

### Claim 8: William Shakespeare wrote Hamlet and was born in Stratford-upon-Avon in 1564

- **Ground Truth:** SUPPORTED
- **Predicted Verdict:** SUPPORTED
- **Confidence:** 85.00%
- **Result:** ✓ CORRECT
- **Evidence:** 6 sources (4 high credibility)
- **Subclaims:** 2
- **Category:** historical
- **Difficulty:** easy

### Claim 9: The COVID-19 pandemic started in 2019 and vaccines were developed within one year

- **Ground Truth:** SUPPORTED
- **Predicted Verdict:** SUPPORTED
- **Confidence:** 85.00%
- **Result:** ✓ CORRECT
- **Evidence:** 6 sources (4 high credibility)
- **Subclaims:** 2
- **Category:** contemporary
- **Difficulty:** medium

### Claim 10: The human brain contains approximately 100 billion neurons

- **Ground Truth:** SUPPORTED
- **Predicted Verdict:** NOT_SUPPORTED
- **Confidence:** 45.00%
- **Result:** ✗ INCORRECT
- **Evidence:** 3 sources (2 high credibility)
- **Subclaims:** 1
- **Category:** scientific
- **Difficulty:** medium

---

## System Architecture Demonstrated

This demo successfully executed all 6 agents:

1. **Input Ingestion Agent** - FOL-based claim decomposition
2. **Query Generation Agent** - Diverse search query creation (k=3)
3. **Evidence Seeking Agent** - 3-stage evidence retrieval
4. **Verdict Prediction Agent** - Weighted evidence aggregation
5. **Explainable AI Agent** - LIME/SHAP-inspired explanations
6. **Reinforcement Learning Agent** - Performance tracking

---

## Key Findings

1. **High Accuracy:** Achieved 80.0% accuracy on diverse claims
2. **Strong F1-Score:** 0.857 demonstrates balanced precision and recall
3. **Quality Evidence:** 66.7% high credibility sources
4. **Transparent Decisions:** Explanation quality score of 0.78/1.0
5. **Efficient Processing:** Mean of 5 queries per claim

---

## Comparison to Research Baseline

From the original paper (Trinh et al., 2025):

| Benchmark | System F1 | Baseline (FOLK) | Improvement |
|-----------|-----------|-----------------|-------------|
| HoVer 3-hop | 0.617 | 0.501 | +23.2% |
| FEVEROUS | 0.681 | 0.649 | +4.9% |
| SciFact | 0.770 | 0.737 | +4.5% |
| **Average** | - | - | **+12.3%** |

Our demo achieved F1=0.857 on mock data.

---

## Conclusion

The system achieved **80.0% accuracy** with an **F1-score of 0.857**.

### Strengths

- ✅ Modular agent-based architecture
- ✅ Comprehensive evaluation metrics
- ✅ Transparent, explainable decisions
- ✅ Free-tier implementation
- ✅ Based on peer-reviewed research (arXiv:2506.17878v1)

