# Codebase Summary: Multi-Agent Fact-Checking Pipeline

## 1. Project Overview
This repository implements a research-oriented, multi-agent fact-checking system and a companion Next.js presentation app.

Top-level areas:
- `src/`: Core Python implementation (agents, orchestrator, utilities, evaluation, storage)
- `config/`: YAML configuration for agents, APIs, prompts, and benchmarks
- `scripts/` and `tests/`: Ablation-study runner and simulation-style experiment code
- `web-app/`: Next.js + Tailwind app for architecture/results presentation
- `docs/`: Research, architecture, setup, and evaluation documentation

Primary goal:
- Input a natural-language claim
- Decompose into verifiable subclaims
- Generate search queries
- Retrieve and credibility-filter evidence
- Aggregate evidence to verdict
- Generate explanations and performance insights

## 2. Core Python Architecture

### 2.1 Orchestrator
File: `src/orchestrator.py`

Main class: `FactCheckingOrchestrator`
- Loads agent config (`config/agent_config.yaml`)
- Initializes LLM interface (`src/utils/llm_interface.py`)
- Instantiates 6 agents and coordinates end-to-end flow via `verify_claim(...)`

Pipeline sequence:
1. Input Ingestion Agent
2. Query Generation Agent
3. Evidence Seeking Agent
4. Verdict Prediction Agent
5. Explainable AI Agent (optional)
6. Reinforcement Learning Agent (optional)

Also includes:
- `batch_verify(...)` for multiple claims
- `get_performance_analysis(...)` for RL analytics
- top-level convenience function `verify_claim(...)`

### 2.2 Agent Modules

#### A) Input Ingestion
File: `src/agents/input_ingestion.py`

Responsibilities:
- Decompose claim into subclaims through `FOLParser`
- Filter non-verifiable statements (opinion/vague/future)

Output object:
- `IngestionResult` with verifiable + filtered subclaims

#### B) Query Generation
File: `src/agents/query_generation.py`

Responsibilities:
- Generate `k` search queries per subclaim (`queries_per_subclaim`, default 3)
- Uses prompt templates from `config/agenta_prompt.yaml`

Output object:
- `QueryResult` per subclaim

#### C) Evidence Seeking
File: `src/agents/evidence_seeking.py`

Responsibilities:
- For each query, search web (currently mock URL generation)
- Check source credibility (`CredibilityChecker`)
- Scrape content and extract relevant passages (`WebScraper`)

Output object:
- `EvidenceResult` containing list of `Evidence` entries and counts

#### D) Verdict Prediction
File: `src/agents/verdict_prediction.py`

Responsibilities:
- Weighted evidence scoring by credibility:
  - high = 1.0, medium = 0.6, low = 0.3
- Per-subclaim verdict: `SUPPORTED`, `NOT_SUPPORTED`, or `INSUFFICIENT_EVIDENCE`
- Aggregate final claim verdict
- Build textual explanation summary

Output object:
- `VerdictResult`

#### E) Explainable AI
File: `src/agents/explainable_ai.py`

Responsibilities:
- Feature-importance style scoring across evidence
- Select critical evidence
- Counterfactual analysis
- Conflict analysis across source credibility
- Explanation quality metrics:
  - coverage, soundness, readability, overall

Output object:
- `ExplanationResult`

#### F) Reinforcement Learning (Heuristic)
File: `src/agents/reinforcement_learning.py`

Responsibilities:
- Record each run’s metrics (accuracy proxy, evidence quality, efficiency)
- Aggregate trends and generate operational suggestions

Output object:
- `RLResult`

## 3. Utility Layer

### 3.1 FOL Parser
File: `src/utils/fol_parser.py`

Purpose:
- Decompose claims into atomic predicate-like subclaims
- Classify verifiability type

Data models:
- `SubClaim`
- `VerifiabilityType` enum

### 3.2 LLM Interface
File: `src/utils/llm_interface.py`

Purpose:
- Unified provider layer for Ollama and OpenAI
- Config loading from `config/api_config.yaml`
- Retry logic via `tenacity`
- Freeform + structured JSON output helper (`generate_structured`)

### 3.3 Web Scraper
File: `src/utils/web_scraper.py`

Purpose:
- URL fetch with `requests` (or Selenium fallback)
- Main-content extraction with BeautifulSoup
- Relevance-ranked passage extraction

### 3.4 Credibility Checker
File: `src/utils/credibility_checker.py`

Purpose:
- Heuristic credibility assignment by domain patterns
- Optional MBFC API path if configured
- Threshold checking (`low/medium/high`)

## 4. Evaluation and Storage

### 4.1 Metrics
File: `src/evaluation/metrics.py`

Provides:
- Classification metrics: accuracy/precision/recall/F1/confusion matrix
- Performance metrics: processing time, query/evidence volume, credibility ratios
- Explanation metrics: coverage/soundness/readability
- Grouped metrics by category and difficulty
- Formatted text report builder

### 4.2 Evidence Store
File: `src/storage/evidence_store.py`

Purpose:
- JSON-backed persistence of evidence by claim id
- Save/load high-credibility subsets

## 5. Demo, Scripts, and Experimentation

### 5.1 Demo Runner
File: `demo.py`

Purpose:
- Loads mock dataset
- Runs orchestrator over claims
- Computes metrics report
- Collects RL analysis
- Writes `DEMO_OBSERVATIONS.md`

### 5.2 Observation Generator
File: `generate_observations.py`

Purpose:
- Standalone script to produce `DEMO_OBSERVATIONS.md` from simulated outputs

### 5.3 Ablation Study Components
Files:
- `tests/test_ablation_studies.py`
- `scripts/run_ablation_studies.py`

Purpose:
- Defines ablation configurations and simulated performance changes
- Runs all ablations, calculates metrics, exports JSON + markdown report

## 6. Configuration System

### `config/agent_config.yaml`
- Per-agent knobs: decomposition limits, query count, credibility threshold, verdict thresholds, XAI and RL toggles

### `config/api_config.yaml`
- LLM providers (Ollama primary, OpenAI fallback)
- Search provider settings
- Credibility and scraping settings
- Retry/rate-limiting parameters

### `config/benchmark_config.yaml`
- Benchmark definitions (FEVEROUS, HoVer, SciFact)
- Sampling/evaluation/reporting options

### `config/agenta_prompt.yaml`
- Prompt templates for decomposition, query generation, retrieval, verdicting, and explanation ranking

## 7. Web Application Summary

Location: `web-app/`
Stack: Next.js 14, React, Tailwind CSS, Recharts, Lucide icons

Main files:
- `web-app/app/page.tsx`: tabbed presentation pages (overview, architecture, results, comparison)
- `web-app/components/AgentFlowchart.tsx`: visual pipeline flow
- `web-app/components/MetricsVisualization.tsx`: confusion/performance/explanation charts
- `web-app/components/ComparisonChart.tsx`: baseline comparison graphs

Purpose:
- Presentation interface for professors/research stakeholders
- Visual storytelling of architecture + benchmark outcomes

## 8. Dependencies and Packaging

### `requirements.txt`
Includes:
- LLM/search/scraping stack
- ML/data stack
- plotting/notebook/test/dev utilities

### `setup.py`
- Package metadata for `multi-agent-fact-checker`
- `find_packages(where="src")`
- extra dependency groups (`dev`, `viz`, `paid_apis`, `full`)

## 9. Current Functional State (Important)

Observed implementation gaps that affect reliability/execution:
- `src/agents/query_generation.py` calls LLM with incorrect method usage and argument naming, and expects a dict response shape that does not match the current LLM interface return type.
- `src/utils/fol_parser.py` calls `generate_structured` with an unsupported argument and attempts fallback to an undefined `_decompose_heuristic` method.
- `setup.py` defines CLI entry point `fact-check=src.orchestrator:main`, but `main` is not implemented in `src/orchestrator.py`.
- Several scripts reference `data/benchmarks/mock_dataset.json`, but this dataset path is absent in the repository.
- `README.md` usage examples in prior version do not fully align with current API names.
- `pytest -q` currently discovers no executable tests.

## 10. Suggested Next Priorities
1. Stabilize LLM interfaces and agent call contracts (`fol_parser`, `query_generation`, `llm_interface`).
2. Add a deterministic non-LLM fallback path for decomposition/query generation.
3. Add real pytest test cases for orchestrator and each agent unit.
4. Add/restore dataset fixtures under `data/benchmarks/` or update scripts to configurable dataset paths.
5. Fix packaging entry point and provide a minimal CLI in `src/orchestrator.py`.
6. Align docs and README to actual executable paths and API behavior.

## 11. Quick File Map
- Core pipeline: `src/orchestrator.py`
- Agents: `src/agents/*.py`
- Utilities: `src/utils/*.py`
- Evaluation: `src/evaluation/metrics.py`
- Storage: `src/storage/evidence_store.py`
- Demo: `demo.py`
- Ablation: `tests/test_ablation_studies.py`, `scripts/run_ablation_studies.py`
- Config: `config/*.yaml`
- Frontend: `web-app/*`
- Docs: `docs/*`
