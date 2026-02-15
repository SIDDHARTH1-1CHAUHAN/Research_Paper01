# Multi-Agent Fact-Checking System - Execution Flowchart

## System Overview

```
+-----------------------------------------------------------------------------------+
|                        MULTI-AGENT FACT-CHECKING PIPELINE                         |
+-----------------------------------------------------------------------------------+

                                    USER INPUT
                                        |
                                        v
+-----------------------------------------------------------------------------------+
|                              ORCHESTRATOR                                          |
|                         (src/orchestrator.py)                                      |
|                                                                                    |
|   1. Load Configuration (config/agent_config.yaml)                                |
|   2. Initialize LLM Interface                                                     |
|   3. Initialize Multilingual Components                                           |
|   4. Coordinate 6-Agent Pipeline                                                  |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
                        +-------------------------------+
                        |    LANGUAGE DETECTION        |
                        |  (src/utils/language_detector.py)
                        +-------------------------------+
                                        |
                                        v
                        +-------------------------------+
                        |    TRANSLATION (if needed)   |
                        |  (src/utils/translator.py)   |
                        +-------------------------------+
                                        |
                                        v
================================================================================
                              6-AGENT PIPELINE
================================================================================


+-----------------------------------------------------------------------------------+
|  AGENT 1: INPUT INGESTION                                                         |
|  Location: src/agents/input_ingestion.py                                          |
+-----------------------------------------------------------------------------------+
|                                                                                    |
|  INPUT: Raw claim text (translated to English if necessary)                       |
|                                                                                    |
|  PROCESS:                                                                          |
|  +------------------+     +-------------------+     +----------------------+       |
|  | Claim Parsing    | --> | FOL Decomposition | --> | Verifiability Filter |       |
|  | (Extract text)   |     | (Split subclaims) |     | (Remove opinions)    |       |
|  +------------------+     +-------------------+     +----------------------+       |
|                                                                                    |
|  COMPONENTS USED:                                                                  |
|  - src/utils/fol_parser.py (First-Order Logic decomposition)                      |
|  - config/languages/keywords/*.yaml (Language-specific keywords)                  |
|                                                                                    |
|  OUTPUT: IngestionResult                                                           |
|  - verifiable_subclaims: List[SubClaim]                                           |
|  - entities: List[str]                                                            |
|  - logical_form: str (FOL representation)                                         |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
|  AGENT 2: QUERY GENERATION                                                        |
|  Location: src/agents/query_generation.py                                         |
+-----------------------------------------------------------------------------------+
|                                                                                    |
|  INPUT: List of verifiable subclaims                                              |
|                                                                                    |
|  PROCESS:                                                                          |
|  +-------------------+     +------------------+     +--------------------+         |
|  | Entity Extraction | --> | Query Expansion  | --> | Query Diversification|       |
|  | (NER from claim)  |     | (Paraphrasing)   |     | (k=3 per subclaim) |         |
|  +-------------------+     +------------------+     +--------------------+         |
|                                                                                    |
|  CONFIGURATION:                                                                    |
|  - queries_per_subclaim: 3 (from agent_config.yaml)                               |
|  - enable_entity_queries: true                                                    |
|  - enable_paraphrasing: true                                                      |
|                                                                                    |
|  OUTPUT: List[QueryResult]                                                         |
|  - subclaim_id: str                                                               |
|  - queries: List[str] (search queries)                                            |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
|  AGENT 3: EVIDENCE SEEKING                                                        |
|  Location: src/agents/evidence_seeking.py                                         |
+-----------------------------------------------------------------------------------+
|                                                                                    |
|  INPUT: List of search queries per subclaim                                       |
|                                                                                    |
|  3-STAGE PIPELINE:                                                                |
|                                                                                    |
|  STAGE 1: WEB SEARCH                                                              |
|  +------------------------+                                                        |
|  | DuckDuckGo Search API  |  (Primary - Free tier)                                |
|  | SerperAPI              |  (Fallback - Paid)                                    |
|  +------------------------+                                                        |
|            |                                                                       |
|            v                                                                       |
|  STAGE 2: CREDIBILITY CHECKING                                                    |
|  +------------------------+                                                        |
|  | Domain Analysis        |  .edu, .gov = HIGH                                    |
|  | Source Whitelist       |  Wikipedia, Reuters = HIGH                            |
|  | MBFC API               |  (Optional - fact-check bias)                         |
|  +------------------------+                                                        |
|  Components: src/utils/credibility_checker.py                                     |
|              config/languages/credibility/*.yaml                                  |
|            |                                                                       |
|            v                                                                       |
|  STAGE 3: CONTENT EXTRACTION                                                      |
|  +------------------------+                                                        |
|  | Web Scraper            |  (requests + BeautifulSoup)                           |
|  | Passage Extraction     |  (Relevant text extraction)                           |
|  | LLM Enhancement        |  (Optional - better extraction)                       |
|  +------------------------+                                                        |
|  Components: src/utils/web_scraper.py                                             |
|                                                                                    |
|  OUTPUT: List[EvidenceResult]                                                      |
|  - subclaim_id: str                                                               |
|  - evidence: List[Evidence] (passages, URLs, credibility scores)                  |
|  - total_sources: int                                                             |
|  - high_credibility_count: int                                                    |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
|  AGENT 4: VERDICT PREDICTION                                                      |
|  Location: src/agents/verdict_prediction.py                                       |
+-----------------------------------------------------------------------------------+
|                                                                                    |
|  INPUT: Claim + Evidence Results                                                  |
|                                                                                    |
|  PROCESS:                                                                          |
|  +----------------------+     +--------------------+     +-------------------+     |
|  | Evidence Aggregation | --> | Weighted Voting    | --> | Verdict Decision  |     |
|  | (Combine all sources)|     | (By credibility)   |     | (Threshold-based) |     |
|  +----------------------+     +--------------------+     +-------------------+     |
|                                                                                    |
|  CREDIBILITY WEIGHTS:                                                              |
|  - HIGH:   1.0                                                                    |
|  - MEDIUM: 0.6                                                                    |
|  - LOW:    0.3                                                                    |
|                                                                                    |
|  DECISION THRESHOLDS:                                                              |
|  - support_ratio > 0.7  --> SUPPORTED                                             |
|  - support_ratio < 0.3  --> NOT_SUPPORTED                                         |
|  - otherwise            --> NOT_SUPPORTED (conservative)                          |
|                                                                                    |
|  LLM-ENHANCED MODE:                                                                |
|  +----------------------+                                                          |
|  | LLM Reasoning        |  Uses prompts from config/prompts.yaml                 |
|  | Chain-of-Thought     |  For complex claims                                    |
|  +----------------------+                                                          |
|                                                                                    |
|  OUTPUT: VerdictResult                                                             |
|  - final_verdict: "SUPPORTED" | "NOT_SUPPORTED"                                   |
|  - overall_confidence: float [0, 1]                                               |
|  - explanation: str                                                               |
|  - subclaim_verdicts: Dict                                                        |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
|  AGENT 5: EXPLAINABLE AI                                                          |
|  Location: src/agents/explainable_ai.py                                           |
+-----------------------------------------------------------------------------------+
|                                                                                    |
|  INPUT: VerdictResult + EvidenceResults                                           |
|                                                                                    |
|  EXPLANATION METHODS:                                                              |
|  +------------------------+     +------------------------+                        |
|  | Feature Importance     |     | Counterfactual Analysis|                        |
|  | (LIME-inspired)        |     | (What would change?)   |                        |
|  +------------------------+     +------------------------+                        |
|            |                              |                                        |
|            +----------+-------------------+                                        |
|                       |                                                            |
|                       v                                                            |
|  +------------------------+                                                        |
|  | Conflict Resolution    |                                                        |
|  | (Handle contradictions)|                                                        |
|  +------------------------+                                                        |
|                                                                                    |
|  QUALITY METRICS:                                                                  |
|  - Coverage:    % of reasoning explained                                          |
|  - Soundness:   Logical consistency score                                         |
|  - Readability: Human comprehension score                                         |
|                                                                                    |
|  LLM JUDGE (for MAR evaluation):                                                   |
|  +------------------------+                                                        |
|  | Rank Explanations 1-4  |                                                        |
|  | Criteria: correctness, |                                                        |
|  |   completeness, faith- |                                                        |
|  |   fulness, clarity     |                                                        |
|  +------------------------+                                                        |
|                                                                                    |
|  OUTPUT: ExplanationResult                                                         |
|  - feature_importance: Dict                                                       |
|  - counterfactuals: List[str]                                                     |
|  - explanation_quality: {coverage, soundness, readability, overall}               |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
|  AGENT 6: REINFORCEMENT LEARNING                                                  |
|  Location: src/agents/reinforcement_learning.py                                   |
+-----------------------------------------------------------------------------------+
|                                                                                    |
|  INPUT: Claim, VerdictResult, EvidenceResults, GroundTruth (optional)             |
|                                                                                    |
|  PERFORMANCE TRACKING:                                                             |
|  +------------------------+     +------------------------+                        |
|  | Record Run Metrics     | --> | Analyze Patterns       |                        |
|  | (accuracy, quality)    |     | (sliding window)       |                        |
|  +------------------------+     +------------------------+                        |
|                       |                                                            |
|                       v                                                            |
|  +------------------------+                                                        |
|  | Generate Suggestions   |                                                        |
|  | (Improvement hints)    |                                                        |
|  +------------------------+                                                        |
|                                                                                    |
|  SCORING FORMULA:                                                                  |
|  score = 1.0 * accuracy + 0.3 * evidence_quality + 0.2 * efficiency              |
|                                                                                    |
|  OUTPUT: PerformanceMetrics                                                        |
|  - accuracy: float                                                                |
|  - evidence_quality: float                                                        |
|  - efficiency: float                                                              |
|  - suggestions: List[str]                                                         |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
================================================================================
                              EVALUATION PIPELINE
================================================================================


+-----------------------------------------------------------------------------------+
|  EVALUATION MODULE                                                                 |
|  Location: src/evaluation/metrics.py                                              |
+-----------------------------------------------------------------------------------+
|                                                                                    |
|  METRIC 1: MACRO F1-SCORE (Quantitative)                                          |
|  +----------------------------------------------------------------------+         |
|  |                                                                      |         |
|  |  For each class (supported, not_supported):                         |         |
|  |                                                                      |         |
|  |  Precision = TP / (TP + FP)                                         |         |
|  |  Recall    = TP / (TP + FN)                                         |         |
|  |  F1        = 2 * (P * R) / (P + R)                                  |         |
|  |                                                                      |         |
|  |  Macro F1  = (F1_supported + F1_not_supported) / 2                  |         |
|  |                                                                      |         |
|  +----------------------------------------------------------------------+         |
|                                                                                    |
|  METRIC 2: MEAN AVERAGE RANK (Qualitative - LLM-as-Judge)                         |
|  +----------------------------------------------------------------------+         |
|  |                                                                      |         |
|  |  For each claim:                                                    |         |
|  |  1. LLM receives: claim, evidence, explanation candidates           |         |
|  |  2. LLM ranks explanations 1 (best) to 4 (worst)                   |         |
|  |  3. Record rank of best explanation                                 |         |
|  |                                                                      |         |
|  |  MAR = Average(all_ranks)                                           |         |
|  |  Lower is better (1.0 = perfect)                                    |         |
|  |                                                                      |         |
|  +----------------------------------------------------------------------+         |
|                                                                                    |
|  DATASETS:                                                                         |
|  - HOVER:       data/datasets/hover/                                              |
|  - FEVEROUS:    data/datasets/feverous/                                           |
|  - SciFact-Open: data/datasets/scifact-open/                                      |
|                                                                                    |
|  OUTPUT: ComprehensiveMetrics                                                      |
|  - macro_f1: float                                                                |
|  - mean_average_rank: float                                                       |
|  - per_class_metrics: Dict                                                        |
|  - per_dataset_breakdown: Dict                                                    |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
                        +-------------------------------+
                        |      FINAL OUTPUT            |
                        |  - Verdict (SUPPORTED/NOT)   |
                        |  - Confidence Score          |
                        |  - Explanation               |
                        |  - Evidence Sources          |
                        |  - Quality Metrics           |
                        +-------------------------------+


================================================================================
                              DATA FLOW DIAGRAM
================================================================================

┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Claim     │───>│  Subclaims  │───>│   Queries   │───>│  Evidence   │
│   (Input)   │    │  (FOL)      │    │  (Search)   │    │  (Web)      │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                │
                                                                v
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Report    │<───│ Explanation │<───│   Verdict   │<───│ Aggregated  │
│   (Output)  │    │  (XAI)      │    │  (Decision) │    │  Evidence   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘


================================================================================
                              CONFIGURATION FILES
================================================================================

config/
├── agent_config.yaml      # Agent-specific parameters
├── api_config.yaml        # LLM and Search API settings
├── benchmark_config.yaml  # Dataset and evaluation settings
├── evaluation_task.yaml   # Evaluation metrics specification
├── prompts.yaml          # LLM prompts for all agents
└── languages/            # Multilingual support
    ├── keywords/         # Language-specific keywords
    ├── stopwords/        # Language-specific stopwords
    ├── prompts/          # Translated prompts
    └── credibility/      # Region-specific credibility rules


================================================================================
                              DIRECTORY STRUCTURE
================================================================================

multi-agent-fact-checker/
├── src/
│   ├── orchestrator.py           # Main coordinator
│   ├── agents/
│   │   ├── input_ingestion.py    # Agent 1
│   │   ├── query_generation.py   # Agent 2
│   │   ├── evidence_seeking.py   # Agent 3
│   │   ├── verdict_prediction.py # Agent 4
│   │   ├── explainable_ai.py     # Agent 5
│   │   └── reinforcement_learning.py # Agent 6
│   ├── evaluation/
│   │   └── metrics.py            # Evaluation metrics
│   ├── storage/
│   │   └── evidence_store.py     # Caching
│   └── utils/
│       ├── llm_interface.py      # LLM abstraction
│       ├── fol_parser.py         # FOL decomposition
│       ├── web_scraper.py        # Content extraction
│       ├── credibility_checker.py # Source validation
│       ├── language_detector.py  # Language detection
│       └── translator.py         # Translation
├── config/                       # All configuration
├── data/
│   ├── datasets/                 # Evaluation datasets
│   │   ├── hover/
│   │   ├── feverous/
│   │   └── scifact-open/
│   ├── outputs/                  # Generated outputs
│   │   ├── predictions/
│   │   ├── explanations/
│   │   └── reports/
│   └── cache/                    # Runtime cache
├── scripts/                      # Entry point scripts
├── tests/                        # Unit tests
├── docs/                         # Documentation
└── web-app/                      # Presentation layer
