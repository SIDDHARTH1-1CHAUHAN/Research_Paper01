# Core Files - Quick Reference

This folder contains copies of the key files created/modified during the integration session.

## Files Overview

| File | Description |
|------|-------------|
| `evaluation_task.yaml` | Evaluation specification with Macro F1 and MAR metrics |
| `EXECUTION_FLOWCHART.md` | Complete system execution flow diagram |
| `SESSION_SUMMARY.md` | Session summary with accuracy report |
| `dataset_loader.py` | Unified loader for HOVER, FEVEROUS, SciFact |
| `metrics.py` | Enhanced metrics with Macro F1 and MAR support |

## Original Locations

```
core/evaluation_task.yaml    → config/evaluation_task.yaml
core/EXECUTION_FLOWCHART.md  → docs/EXECUTION_FLOWCHART.md
core/SESSION_SUMMARY.md      → docs/SESSION_SUMMARY.md
core/dataset_loader.py       → src/evaluation/dataset_loader.py
core/metrics.py              → src/evaluation/metrics.py
```

## Quick Start

```python
# Load datasets
from src.evaluation.dataset_loader import DatasetLoader
loader = DatasetLoader("data/datasets")
samples = loader.load_hover(split="dev", limit=100)

# Compute Macro F1
from src.evaluation.metrics import MetricsCalculator
result = MetricsCalculator.calculate_macro_f1(predictions, ground_truths)
print(f"Macro F1: {result.macro_f1:.4f}")
```
