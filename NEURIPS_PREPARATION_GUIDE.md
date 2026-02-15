# NeurIPS Preparation Guide

## Making Your Multi-Agent Fact-Checking System Ready for Top-Tier Conferences

This guide provides a comprehensive roadmap to transform your project into a NeurIPS-level publication.

---

## Current State Assessment

### What You Have

| Component | Status | Quality |
|-----------|--------|---------|
| Multi-agent architecture | Complete | Good |
| 6 specialized agents | Complete | Good |
| FOL decomposition | Complete | Good |
| Credibility weighting | Complete | Good |
| Explainable AI | Complete | Basic |
| Demo with mock data | Complete | Works |

### What's Missing for NeurIPS

| Requirement | Current State | Gap |
|-------------|--------------|-----|
| Benchmark evaluation | Mock dataset (10 samples) | Need FEVER, HoVer, FEVEROUS, SciFact |
| Baseline comparisons | None | Need 3-5 SOTA baselines |
| Statistical significance | None | Need p-values, confidence intervals |
| Ablation studies | Documented | Need rigorous quantitative analysis |
| Novel contribution | Exists but unclear | Need sharper positioning |
| Reproducibility | Partial | Need seeds, exact versions, config files |

---

## Step-by-Step NeurIPS Preparation

### Phase 1: Benchmark Evaluation (Priority: CRITICAL)

**Week 1-2: Setup Academic Benchmarks**

#### 1.1 Download Benchmark Datasets

```bash
# Create scripts directory
mkdir -p scripts/benchmarks

# Create benchmark download script
```

Create file `scripts/download_benchmarks.py`:

```python
"""
Download and prepare academic fact-checking benchmarks.
"""

import os
from pathlib import Path
from datasets import load_dataset
import json

def download_fever():
    """Download FEVER dataset from HuggingFace"""
    print("Downloading FEVER dataset...")
    dataset = load_dataset("fever", "v1.0")

    # Save validation set (use for evaluation)
    output_path = Path("data/benchmarks/fever")
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert to our format
    claims = []
    for item in dataset['paper_dev']:
        claims.append({
            'id': item['id'],
            'claim': item['claim'],
            'ground_truth': 'SUPPORTED' if item['label'] == 'SUPPORTS' else 'NOT_SUPPORTED',
            'evidence': item.get('evidence', [])
        })

    with open(output_path / "fever_dev.json", 'w') as f:
        json.dump(claims[:1000], f, indent=2)  # First 1000 for testing

    print(f"Saved {len(claims[:1000])} FEVER claims")

def download_hover():
    """Download HoVer dataset"""
    print("Downloading HoVer dataset...")
    dataset = load_dataset("hover", "hover_dev_release_v1.1")

    output_path = Path("data/benchmarks/hover")
    output_path.mkdir(parents=True, exist_ok=True)

    claims = []
    for item in dataset['validation']:
        claims.append({
            'id': item['uid'],
            'claim': item['claim'],
            'ground_truth': 'SUPPORTED' if item['label'] == 'SUPPORTED' else 'NOT_SUPPORTED',
            'num_hops': item.get('num_hops', 2)
        })

    with open(output_path / "hover_dev.json", 'w') as f:
        json.dump(claims[:500], f, indent=2)

    print(f"Saved {len(claims[:500])} HoVer claims")

def download_scifact():
    """Download SciFact dataset"""
    print("Downloading SciFact dataset...")
    dataset = load_dataset("allenai/scifact", "claims")

    output_path = Path("data/benchmarks/scifact")
    output_path.mkdir(parents=True, exist_ok=True)

    claims = []
    for item in dataset['validation']:
        claims.append({
            'id': item['id'],
            'claim': item['claim'],
            'ground_truth': 'SUPPORTED' if item['evidence'] else 'NOT_SUPPORTED'
        })

    with open(output_path / "scifact_dev.json", 'w') as f:
        json.dump(claims, f, indent=2)

    print(f"Saved {len(claims)} SciFact claims")

if __name__ == "__main__":
    download_fever()
    download_hover()
    download_scifact()
    print("\nAll benchmarks downloaded!")
```

#### 1.2 Create Evaluation Script

Create file `scripts/run_benchmark_evaluation.py`:

```python
"""
Run comprehensive benchmark evaluation.
"""

import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestrator import FactCheckingOrchestrator
from src.evaluation.metrics import MetricsCalculator

def evaluate_benchmark(benchmark_path: str, output_dir: str, max_samples: int = None):
    """Evaluate on a single benchmark"""

    with open(benchmark_path, 'r') as f:
        dataset = json.load(f)

    if max_samples:
        dataset = dataset[:max_samples]

    benchmark_name = Path(benchmark_path).stem
    print(f"\n{'='*60}")
    print(f"Evaluating on {benchmark_name}: {len(dataset)} claims")
    print(f"{'='*60}\n")

    orchestrator = FactCheckingOrchestrator()
    results = []

    start_time = time.time()

    for i, item in enumerate(dataset):
        print(f"Processing {i+1}/{len(dataset)}: {item['claim'][:50]}...")

        result = orchestrator.verify_claim(
            item['claim'],
            ground_truth=item['ground_truth'],
            enable_xai=True,
            enable_rl=True
        )
        results.append(result)

        # Progress update every 10 claims
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (len(dataset) - i - 1)
            print(f"  Progress: {i+1}/{len(dataset)} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")

    total_time = time.time() - start_time

    # Calculate metrics
    metrics = MetricsCalculator.calculate_comprehensive_metrics(results, dataset)

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / f"{benchmark_name}_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'benchmark': benchmark_name,
            'num_samples': len(dataset),
            'total_time_seconds': total_time,
            'metrics': {
                'accuracy': metrics.classification.accuracy,
                'precision': metrics.classification.precision,
                'recall': metrics.classification.recall,
                'f1_score': metrics.classification.f1_score
            },
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n{benchmark_name} Results:")
    print(f"  Accuracy: {metrics.classification.accuracy:.4f}")
    print(f"  Precision: {metrics.classification.precision:.4f}")
    print(f"  Recall: {metrics.classification.recall:.4f}")
    print(f"  F1-Score: {metrics.classification.f1_score:.4f}")
    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Results saved to: {results_file}")

    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, default='all',
                       choices=['fever', 'hover', 'scifact', 'all'])
    parser.add_argument('--max-samples', type=int, default=100)
    parser.add_argument('--output-dir', type=str, default='results/benchmarks')
    args = parser.parse_args()

    benchmarks = {
        'fever': 'data/benchmarks/fever/fever_dev.json',
        'hover': 'data/benchmarks/hover/hover_dev.json',
        'scifact': 'data/benchmarks/scifact/scifact_dev.json'
    }

    if args.benchmark == 'all':
        for name, path in benchmarks.items():
            if Path(path).exists():
                evaluate_benchmark(path, args.output_dir, args.max_samples)
            else:
                print(f"Benchmark not found: {path}")
    else:
        path = benchmarks[args.benchmark]
        evaluate_benchmark(path, args.output_dir, args.max_samples)

if __name__ == "__main__":
    main()
```

---

### Phase 2: Baseline Comparisons (Priority: HIGH)

**Week 2-3: Implement Baselines**

#### 2.1 Baselines to Implement

| Baseline | Description | Reference |
|----------|-------------|-----------|
| BM25 + LLM | Traditional retrieval + modern LLM | Standard baseline |
| DPR + LLM | Dense passage retrieval + LLM | Karpukhin et al. 2020 |
| FEVER Baseline | Original FEVER paper approach | Thorne et al. 2018 |
| ProgramFC | Recent program-based approach | Pan et al. 2023 |
| Claim-Only LLM | LLM without evidence | Ablation baseline |

Create file `src/baselines/bm25_baseline.py`:

```python
"""
BM25 + LLM Baseline for comparison.
"""

from rank_bm25 import BM25Okapi
from typing import List, Dict, Any

class BM25Baseline:
    """Simple BM25 retrieval + LLM baseline"""

    def __init__(self, llm_interface=None):
        self.llm = llm_interface
        self.corpus = []
        self.bm25 = None

    def index_documents(self, documents: List[str]):
        """Index documents for retrieval"""
        tokenized = [doc.lower().split() for doc in documents]
        self.corpus = documents
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """Retrieve top-k documents"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.corpus[i] for i in top_k]

    def verify(self, claim: str, documents: List[str]) -> Dict[str, Any]:
        """Verify claim using BM25 retrieval + LLM"""
        self.index_documents(documents)
        evidence = self.retrieve(claim)

        if self.llm:
            prompt = f"""Based on the following evidence, determine if the claim is SUPPORTED or NOT_SUPPORTED.

Claim: {claim}

Evidence:
{chr(10).join(evidence)}

Verdict (SUPPORTED or NOT_SUPPORTED):"""

            response = self.llm.generate(prompt)
            verdict = "SUPPORTED" if "SUPPORTED" in response.upper() else "NOT_SUPPORTED"
        else:
            # Heuristic fallback
            verdict = "SUPPORTED" if evidence else "NOT_SUPPORTED"

        return {
            'verdict': verdict,
            'evidence': evidence,
            'method': 'bm25_baseline'
        }
```

#### 2.2 Comparison Table Template

For your paper, create a results table:

```
Table 1: Comparison with Baseline Methods

| Method              | FEVER F1 | HoVer F1 | SciFact F1 | Avg |
|---------------------|----------|----------|------------|-----|
| BM25 + GPT-4        | 0.XX     | 0.XX     | 0.XX       | 0.XX|
| DPR + GPT-4         | 0.XX     | 0.XX     | 0.XX       | 0.XX|
| FEVER Baseline      | 0.65     | -        | -          | -   |
| ProgramFC           | 0.72     | 0.58     | 0.76       | 0.69|
| **Ours (Full)**     | **0.XX** | **0.XX** | **0.XX**   | **0.XX**|
| Ours (No Cred.)     | 0.XX     | 0.XX     | 0.XX       | 0.XX|
| Ours (No FOL)       | 0.XX     | 0.XX     | 0.XX       | 0.XX|
```

---

### Phase 3: Statistical Rigor (Priority: HIGH)

**Week 3-4: Add Statistical Analysis**

#### 3.1 Create Statistical Analysis Script

Create file `scripts/statistical_analysis.py`:

```python
"""
Statistical significance testing for benchmark results.
"""

import numpy as np
from scipy import stats
import json
from pathlib import Path

def paired_bootstrap_test(scores_a: list, scores_b: list, n_bootstrap: int = 10000):
    """
    Paired bootstrap test for statistical significance.

    Returns p-value for H0: scores_a <= scores_b
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)

    observed_diff = np.mean(scores_a) - np.mean(scores_b)

    n = len(scores_a)
    count = 0

    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n, size=n)
        bootstrap_diff = np.mean(scores_a[indices]) - np.mean(scores_b[indices])
        if bootstrap_diff <= 0:
            count += 1

    p_value = count / n_bootstrap
    return p_value, observed_diff

def mcnemar_test(pred_a: list, pred_b: list, ground_truth: list):
    """
    McNemar's test for comparing two classifiers.
    """
    correct_a = [p == g for p, g in zip(pred_a, ground_truth)]
    correct_b = [p == g for p, g in zip(pred_b, ground_truth)]

    # Count discordant pairs
    b = sum(1 for a, b in zip(correct_a, correct_b) if a and not b)  # A correct, B wrong
    c = sum(1 for a, b in zip(correct_a, correct_b) if not a and b)  # A wrong, B correct

    # McNemar's test with continuity correction
    if b + c == 0:
        return 1.0, 0

    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return p_value, statistic

def confidence_interval(scores: list, confidence: float = 0.95):
    """
    Calculate confidence interval using bootstrap.
    """
    scores = np.array(scores)
    n_bootstrap = 10000

    bootstrap_means = []
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, len(scores), size=len(scores))
        bootstrap_means.append(np.mean(scores[indices]))

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return lower, upper, np.mean(scores)

def generate_significance_report(results_dir: str):
    """Generate statistical significance report."""

    results_path = Path(results_dir)

    print("="*60)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("="*60)

    # Load all results
    all_results = {}
    for result_file in results_path.glob("*_results.json"):
        with open(result_file) as f:
            data = json.load(f)
            all_results[data['benchmark']] = data

    # Calculate confidence intervals
    print("\nConfidence Intervals (95%):")
    print("-"*40)

    for benchmark, data in all_results.items():
        acc = data['metrics']['accuracy']
        # Simulate individual predictions for CI (in real scenario, store actual predictions)
        n = data['num_samples']
        scores = [1] * int(acc * n) + [0] * (n - int(acc * n))
        lower, upper, mean = confidence_interval(scores)

        print(f"{benchmark}:")
        print(f"  Accuracy: {mean:.4f} [{lower:.4f}, {upper:.4f}]")
        print(f"  F1-Score: {data['metrics']['f1_score']:.4f}")

    print("\n" + "="*60)
    print("Report generated successfully!")

if __name__ == "__main__":
    generate_significance_report("results/benchmarks")
```

#### 3.2 Key Statistical Requirements for NeurIPS

1. **Multiple runs**: Run experiments 3-5 times with different random seeds
2. **Confidence intervals**: Report 95% CI for all metrics
3. **Significance tests**: Use paired bootstrap or McNemar's test
4. **Effect sizes**: Report Cohen's d or similar
5. **Variance reporting**: Include standard deviation

---

### Phase 4: Ablation Studies (Priority: HIGH)

**Week 4-5: Rigorous Ablation Analysis**

Create file `scripts/run_ablation_studies.py`:

```python
"""
Comprehensive ablation studies for component analysis.
"""

import json
import copy
from pathlib import Path
from datetime import datetime

# Ablation configurations
ABLATION_CONFIGS = {
    'full_system': {
        'description': 'Full system with all components',
        'enable_fol': True,
        'enable_credibility': True,
        'queries_per_subclaim': 3,
        'enable_3stage_pipeline': True
    },
    'no_fol': {
        'description': 'Without FOL decomposition',
        'enable_fol': False,
        'enable_credibility': True,
        'queries_per_subclaim': 3,
        'enable_3stage_pipeline': True
    },
    'no_credibility': {
        'description': 'Without credibility weighting',
        'enable_fol': True,
        'enable_credibility': False,
        'queries_per_subclaim': 3,
        'enable_3stage_pipeline': True
    },
    'single_query': {
        'description': 'Single query per subclaim (k=1)',
        'enable_fol': True,
        'enable_credibility': True,
        'queries_per_subclaim': 1,
        'enable_3stage_pipeline': True
    },
    'no_pipeline': {
        'description': 'Without 3-stage pipeline',
        'enable_fol': True,
        'enable_credibility': True,
        'queries_per_subclaim': 3,
        'enable_3stage_pipeline': False
    },
    'minimal': {
        'description': 'Minimal system (all components disabled)',
        'enable_fol': False,
        'enable_credibility': False,
        'queries_per_subclaim': 1,
        'enable_3stage_pipeline': False
    }
}

def run_ablation_study(benchmark_path: str, output_dir: str, max_samples: int = 100):
    """Run all ablation configurations."""

    from src.orchestrator import FactCheckingOrchestrator
    from src.evaluation.metrics import MetricsCalculator

    with open(benchmark_path, 'r') as f:
        dataset = json.load(f)[:max_samples]

    results = {}

    for config_name, config in ABLATION_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Running ablation: {config_name}")
        print(f"Description: {config['description']}")
        print(f"{'='*60}")

        # Create orchestrator with modified config
        orchestrator = FactCheckingOrchestrator()

        # Modify configuration based on ablation
        if not config['enable_fol']:
            orchestrator.input_ingestion.config['enable_fol_decomposition'] = False

        if not config['enable_credibility']:
            orchestrator.evidence_seeking.config['enable_credibility_filter'] = False

        orchestrator.query_generation.queries_per_subclaim = config['queries_per_subclaim']

        # Run evaluation
        eval_results = []
        for item in dataset:
            result = orchestrator.verify_claim(
                item['claim'],
                ground_truth=item['ground_truth'],
                enable_xai=False,
                enable_rl=False
            )
            eval_results.append(result)

        metrics = MetricsCalculator.calculate_comprehensive_metrics(eval_results, dataset)

        results[config_name] = {
            'config': config,
            'accuracy': metrics.classification.accuracy,
            'f1_score': metrics.classification.f1_score,
            'precision': metrics.classification.precision,
            'recall': metrics.classification.recall
        }

        print(f"Results: Accuracy={metrics.classification.accuracy:.4f}, F1={metrics.classification.f1_score:.4f}")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "ablation_results.json", 'w') as f:
        json.dump({
            'benchmark': benchmark_path,
            'num_samples': len(dataset),
            'results': results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    # Generate ablation table
    print("\n" + "="*60)
    print("ABLATION STUDY RESULTS")
    print("="*60)
    print(f"\n{'Configuration':<20} {'Accuracy':<12} {'F1-Score':<12} {'Delta Acc':<12}")
    print("-"*60)

    baseline_acc = results['full_system']['accuracy']
    for name, data in results.items():
        delta = data['accuracy'] - baseline_acc
        delta_str = f"{delta:+.4f}" if name != 'full_system' else "-"
        print(f"{name:<20} {data['accuracy']:<12.4f} {data['f1_score']:<12.4f} {delta_str:<12}")

    return results

if __name__ == "__main__":
    run_ablation_study(
        "data/benchmarks/mock_dataset.json",
        "results/ablation",
        max_samples=10
    )
```

---

### Phase 5: Paper Writing (Priority: CRITICAL)

**Week 5-8: Write the Paper**

#### 5.1 NeurIPS Paper Structure

```
1. Abstract (150 words max)
   - Problem statement
   - Key contribution
   - Main results

2. Introduction (1 page)
   - Motivation
   - Research gap
   - Contributions (3-4 bullet points)
   - Paper organization

3. Related Work (0.5-1 page)
   - Fact-checking systems
   - Multi-agent systems
   - Explainable AI for NLP
   - Evidence retrieval

4. Method (2-3 pages)
   4.1 Problem Formulation
   4.2 System Architecture
   4.3 Agent Descriptions
       - Input Ingestion (FOL)
       - Query Generation
       - Evidence Seeking
       - Verdict Prediction
       - Explainable AI
   4.4 Training/Optimization (if any)

5. Experiments (2-3 pages)
   5.1 Datasets
   5.2 Baselines
   5.3 Implementation Details
   5.4 Main Results
   5.5 Ablation Studies
   5.6 Analysis

6. Conclusion (0.5 page)
   - Summary
   - Limitations
   - Future work

7. References
```

#### 5.2 Key Claims to Support

For each claim, you need evidence:

| Claim | Evidence Needed |
|-------|-----------------|
| Multi-agent improves accuracy | Comparison with single-agent |
| FOL decomposition helps | Ablation: with/without FOL |
| Credibility weighting is crucial | Ablation: with/without weighting |
| System is explainable | User study or explanation quality metrics |
| Generalizes across domains | Results on 3+ benchmarks |

---

### Phase 6: Reproducibility (Priority: HIGH)

**Week 6-7: Ensure Reproducibility**

#### 6.1 Create Reproducibility Checklist

Create file `REPRODUCIBILITY.md`:

```markdown
# Reproducibility Checklist

## Environment
- [ ] Python version: 3.10.x
- [ ] All dependencies pinned in requirements.txt
- [ ] Random seeds documented: 42, 123, 456, 789, 1024

## Data
- [ ] Benchmark download scripts provided
- [ ] Data preprocessing documented
- [ ] Data splits specified

## Code
- [ ] All hyperparameters in config files
- [ ] Training/evaluation scripts provided
- [ ] Model checkpoints available

## Results
- [ ] Main results table reproducible
- [ ] Ablation results reproducible
- [ ] Statistical tests reproducible
```

#### 6.2 Pin Exact Versions

Update `requirements.txt`:

```
# Exact versions for reproducibility
loguru==0.7.2
pyyaml==6.0.1
requests==2.31.0
beautifulsoup4==4.12.2
lxml==5.1.0
tenacity==8.2.3
scikit-learn==1.3.2
scipy==1.11.4
numpy==1.26.2
pandas==2.1.3
```

#### 6.3 Set Random Seeds

Add to every script:

```python
import random
import numpy as np

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)  # If using PyTorch

set_seed(42)
```

---

## Timeline Summary

| Week | Task | Deliverable |
|------|------|-------------|
| 1-2 | Benchmark evaluation | Results on FEVER, HoVer, SciFact |
| 2-3 | Baseline implementation | Comparison with 3-5 methods |
| 3-4 | Statistical analysis | p-values, confidence intervals |
| 4-5 | Ablation studies | Component contribution analysis |
| 5-7 | Paper writing | Complete draft |
| 7-8 | Revision & polish | Camera-ready paper |

---

## Expected Results

Based on the architecture and methodology:

| Benchmark | Expected F1 | Baseline F1 | Expected Improvement |
|-----------|-------------|-------------|---------------------|
| FEVER | 0.68-0.72 | 0.65 | +5-10% |
| HoVer | 0.58-0.62 | 0.50 | +16-24% |
| SciFact | 0.75-0.80 | 0.74 | +1-8% |

---

## Submission Checklist

### NeurIPS 2026 Deadlines (Estimated)
- Abstract deadline: May 2026
- Paper deadline: May 2026
- Supplementary: June 2026
- Notification: September 2026

### Submission Requirements
- [ ] 9 pages (excluding references)
- [ ] Anonymous submission
- [ ] Supplementary material (code, data)
- [ ] Reproducibility checklist
- [ ] Broader impact statement

---

## Tips for Success

1. **Novel contribution**: Emphasize the multi-agent coordination and explainability
2. **Strong baselines**: Include recent 2024-2025 methods
3. **Thorough ablations**: Show each component matters
4. **Clear writing**: NeurIPS values clarity
5. **Solid experiments**: Multiple benchmarks, statistical tests
6. **Reproducibility**: Code and data availability

---

## Quick Start Commands

```bash
# 1. Install additional research dependencies
pip install datasets rank_bm25 scipy matplotlib seaborn

# 2. Download benchmarks
python scripts/download_benchmarks.py

# 3. Run benchmark evaluation
python scripts/run_benchmark_evaluation.py --benchmark all --max-samples 100

# 4. Run ablation studies
python scripts/run_ablation_studies.py

# 5. Generate statistical analysis
python scripts/statistical_analysis.py
```

---

Good luck with your NeurIPS submission!
