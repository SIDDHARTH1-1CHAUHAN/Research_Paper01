# Documentation

Welcome to the Multi-Agent Fact-Checker documentation. This guide helps you find the right documentation for your needs.

## Quick Navigation

### By Role

| If you are a... | Start here |
|-----------------|------------|
| **New User** | [Quick Start](getting-started/QUICK_START.md) → [How to Run](guides/HOW_TO_RUN.md) |
| **Developer** | [Contributing](development/CONTRIBUTING.md) → [Architecture](technical/ARCHITECTURE.md) |
| **Researcher** | [Research Paper](RESEARCH_PAPER.md) → [Evaluation Methodology](technical/EVALUATION_METHODOLOGY.md) |
| **Presenter** | [Web App Instructions](guides/WEB_APP.md) → [Demo Observations](evaluation/DEMO_OBSERVATIONS.md) |

## Documentation Structure

```
docs/
├── README.md                    # You are here
├── RESEARCH_PAPER.md            # Academic paper & methodology
│
├── getting-started/             # Onboarding
│   ├── QUICK_START.md           # 5-minute setup guide
│   └── GLOSSARY.md              # Terms & definitions
│
├── technical/                   # Technical documentation
│   ├── ARCHITECTURE.md          # System design & data flows
│   └── EVALUATION_METHODOLOGY.md # How accuracy is measured
│
├── guides/                      # How-to guides
│   ├── HOW_TO_RUN.md            # Demo execution
│   └── WEB_APP.md               # Web presentation setup
│
├── evaluation/                  # Results & analysis
│   ├── ABLATION_STUDIES.md      # Component performance analysis
│   └── DEMO_OBSERVATIONS.md     # Demo results & metrics
│
└── development/                 # Contributor docs
    ├── CONTRIBUTING.md          # Development guidelines
    └── IMPROVEMENTS.md          # Enhancement strategies
```

## Document Summaries

### Core Documentation

| Document | Description |
|----------|-------------|
| [Research Paper](RESEARCH_PAPER.md) | Full academic paper with methodology, results, and analysis |

### Getting Started

| Document | Description |
|----------|-------------|
| [Quick Start](getting-started/QUICK_START.md) | Get the system running in 5 minutes |
| [Glossary](getting-started/GLOSSARY.md) | Key terms and definitions |

### Technical Documentation

| Document | Description |
|----------|-------------|
| [Architecture](technical/ARCHITECTURE.md) | System design, agent details, data flows |
| [Evaluation Methodology](technical/EVALUATION_METHODOLOGY.md) | How accuracy is measured (exact match, not LLM-as-judge) |

### How-To Guides

| Document | Description |
|----------|-------------|
| [How to Run](guides/HOW_TO_RUN.md) | Running demos and processing claims |
| [Web App](guides/WEB_APP.md) | Setting up the Next.js presentation app |

### Evaluation & Results

| Document | Description |
|----------|-------------|
| [Ablation Studies](evaluation/ABLATION_STUDIES.md) | Component contribution analysis |
| [Demo Observations](evaluation/DEMO_OBSERVATIONS.md) | Results from demo runs |

### Development

| Document | Description |
|----------|-------------|
| [Contributing](development/CONTRIBUTING.md) | Guidelines for contributors |
| [Improvements](development/IMPROVEMENTS.md) | Roadmap and enhancement strategies |

## Key Concepts

### The 6-Agent Architecture

1. **Claim Decomposition Agent** - Breaks claims into verifiable subclaims using First-Order Logic
2. **Evidence Retrieval Agent** - Generates diverse search queries and retrieves evidence
3. **Credibility Assessment Agent** - Evaluates source reliability and bias
4. **Verdict Prediction Agent** - Aggregates evidence using weighted voting
5. **Explainable AI Agent** - Generates human-readable explanations
6. **Reinforcement Learning Agent** - Tracks performance for continuous improvement

### Evaluation Approach

The system uses **exact string matching** for evaluation:
- Verdicts: "SUPPORTED" or "NOT_SUPPORTED"
- Metrics: Accuracy, Precision, Recall, F1-Score
- **NOT** LLM-as-judge or semantic similarity

See [Evaluation Methodology](technical/EVALUATION_METHODOLOGY.md) for details.

## Quick Links

- [Run a quick demo](guides/HOW_TO_RUN.md#quick-start)
- [Understand the architecture](technical/ARCHITECTURE.md)
- [View benchmark results](evaluation/ABLATION_STUDIES.md)
- [Contribute to the project](development/CONTRIBUTING.md)
