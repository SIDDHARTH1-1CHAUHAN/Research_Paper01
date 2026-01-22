# Quick Start

Get the Multi-Agent Fact-Checker running in 5 minutes.

## 1. Prerequisites

- Python 3.9+
- pip (Python package manager)

## 2. Setup

```bash
# Clone or navigate to the project
cd multi-agent-fact-checker

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 3. Run Demo

```bash
python demo.py
```

That's it! The demo will:
- Process 10 fact-checking claims
- Show all 6 agents in action
- Generate evaluation metrics
- Save results to `DEMO_OBSERVATIONS.md`

## What You'll See

```
CLAIM 1/10
Text: The Eiffel Tower was completed in 1889...

[1/6] Input Ingestion Agent - Decomposing claim...
[2/6] Query Generation Agent - Creating search queries...
[3/6] Evidence Seeking Agent - Retrieving evidence...
[4/6] Verdict Prediction Agent - Aggregating evidence...
[5/6] Explainable AI Agent - Generating explanations...
[6/6] Reinforcement Learning Agent - Recording performance...

CLASSIFICATION METRICS
Accuracy:   0.8000 (80.00%)
F1-Score:   0.8421
```

## Next Steps

| Goal | Document |
|------|----------|
| Run full demo | [How to Run](../guides/HOW_TO_RUN.md) |
| Understand architecture | [Architecture](../technical/ARCHITECTURE.md) |
| Learn key terms | [Glossary](GLOSSARY.md) |
| View results | [Demo Observations](../evaluation/DEMO_OBSERVATIONS.md) |

## Troubleshooting

**Module not found**: Run `pip install -r requirements.txt`

**Python version error**: Ensure Python 3.9+ with `python --version`

**Config file not found**: Demo works without config files (uses defaults)
