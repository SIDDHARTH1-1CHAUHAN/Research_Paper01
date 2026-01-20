# Multi-Agent Fact-Checking System - Technical Architecture

## Overview

This document provides technical details about the system architecture, data flows, and implementation patterns for developers and researchers.

---

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Orchestrator                              │
│                     (LangGraph Coordinator)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   ┌────▼────┐          ┌────▼────┐          ┌────▼────┐
   │ Agent 1 │          │ Agent 2 │          │ Agent 3 │
   │  Input  │──────────│  Query  │──────────│Evidence │
   │Ingestion│          │   Gen   │          │ Seeking │
   └─────────┘          └─────────┘          └─────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                         ┌────▼────┐
                         │ Agent 4 │
                         │ Verdict │
                         │Prediction│
                         └─────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
           ┌────▼────┐                 ┌───▼────┐
           │ Agent 5 │                 │Agent 6 │
           │   XAI   │                 │   RL   │
           └─────────┘                 └────────┘
```

---

## Agent Details

### 1. Input Ingestion Agent

**Module:** `src/agents/input_ingestion.py`

**Input:** Raw natural language claim (string)

**Output:**
```python
{
    "original_claim": str,
    "verifiable_subclaims": [
        {
            "id": str,  # e.g., "SC1"
            "text": str,
            "logical_form": str,  # FOL predicate
            "verifiability": str,  # "verifiable" | "opinion" | "vague" | "future_prediction"
            "entities": List[str],
            "properties": List[str]
        }
    ],
    "filtered_count": int
}
```

**Dependencies:**
- `src/utils/fol_parser.py` - FOL decomposition logic
- `src/utils/llm_interface.py` - LLM for intelligent parsing

**Algorithm:**
1. Parse claim using LLM or heuristic decomposition
2. Extract entities and predicates
3. Classify verifiability
4. Filter non-verifiable subclaims

**Configuration:** `config/agent_config.yaml` → `input_ingestion`

---

### 2. Query Generation Agent

**Module:** `src/agents/query_generation.py`

**Input:** List of verifiable subclaims

**Output:**
```python
{
    "subclaim_id": str,
    "queries": List[str]  # length k (typically 3-4)
}
```

**Strategy:**
1. Entity extraction from subclaim
2. Keyword-based query generation
3. Paraphrasing for diversity
4. Temporal/contextual variants

**Configuration:** `config/agent_config.yaml` → `query_generation`

---

### 3. Evidence Seeking Agent

**Module:** `src/agents/evidence_seeking.py`

**Input:** Subclaims with queries

**Output:**
```python
{
    "subclaim_id": str,
    "evidence": [
        {
            "id": str,
            "source_url": str,
            "source_name": str,
            "passage": str,
            "credibility_score": float,  # 0.0 - 1.0
            "credibility_level": str,  # "high" | "medium" | "low"
            "retrieved_at": str,  # ISO 8601 timestamp
            "metadata": Dict[str, Any]
        }
    ]
}
```

**Three-Stage Pipeline:**

**Stage 1: Web Search**
- Module: `src/utils/web_scraper.py`
- Provider: DuckDuckGo (free) or SerperAPI (paid)
- Result limit: 10 per query

**Stage 2: Credibility Check**
- Module: `src/utils/credibility_checker.py`
- Methods:
  - Heuristic (free): Domain suffix analysis
  - MBFC API (paid): Factuality ratings

**Stage 3: Content Extraction**
- Full-page scraping with BeautifulSoup
- Main content extraction
- Passage relevance scoring

**Configuration:** `config/agent_config.yaml` → `evidence_seeking`

---

### 4. Verdict Prediction Agent

**Module:** `src/agents/verdict_prediction.py`

**Input:** Evidence from all subclaims

**Output:**
```python
{
    "subclaim_verdicts": {
        "SC1": {
            "verdict": str,  # "SUPPORTED" | "NOT_SUPPORTED"
            "confidence": float,  # 0.0 - 1.0
            "evidence_count": int
        }
    },
    "final_verdict": str,
    "explanation": str,
    "metadata": {
        "total_sources": int,
        "high_credibility_sources": int,
        "processing_time": float
    }
}
```

**Weighted Voting Algorithm:**
```python
def aggregate_evidence(evidences):
    scores = []
    for ev in evidences:
        weight = {
            "high": 1.0,
            "medium": 0.6,
            "low": 0.3
        }[ev.credibility_level]

        support = check_support(ev.passage, subclaim)
        scores.append(weight * support)

    final_score = sum(scores) / len(scores)

    if final_score > 0.7:
        return "SUPPORTED"
    elif final_score < 0.3:
        return "NOT_SUPPORTED"
    else:
        return "INSUFFICIENT_EVIDENCE"
```

**Configuration:** `config/agent_config.yaml` → `verdict_prediction`

---

### 5. Explainable AI Agent

**Module:** `src/agents/explainable_ai.py`

**Input:** Verdict + Evidence

**Output:**
```python
{
    "feature_importance": {
        "evidence_id": float  # contribution score
    },
    "counterfactual": {
        "critical_evidence": List[str],
        "what_would_change": str
    },
    "conflict_resolution": {
        "contradictions": List[Dict],
        "resolution_method": str
    },
    "metrics": {
        "coverage": float,  # 0.0 - 1.0
        "soundness": float,
        "readability": float
    }
}
```

**Methods:**

**Feature Importance (LIME-inspired):**
- Calculate influence of each evidence item
- Rank by contribution to verdict

**Counterfactual Analysis:**
- Identify minimal evidence changes that would flip verdict
- Highlight critical sources

**Conflict Resolution:**
- Document contradictory evidence
- Explain resolution logic (credibility weighting)

**Configuration:** `config/agent_config.yaml` → `explainable_ai`

---

### 6. Reinforcement Learning Agent

**Module:** `src/agents/reinforcement_learning.py`

**Input:** Run results (verdict + ground truth)

**Output:**
```python
{
    "performance_score": float,
    "patterns": {
        "query_success_rates": Dict[str, float],
        "source_reliability": Dict[str, float],
        "optimal_k": Dict[str, int]
    },
    "suggestions": List[str]
}
```

**Heuristic Performance Scoring:**
```python
score = (
    1.0 * is_correct +
    0.3 * evidence_quality +
    0.2 * efficiency
)
```

**Pattern Analysis:**
- Track metrics over sliding window (last N runs)
- Statistical aggregation (means, medians, trends)
- Rule-based suggestions

**Storage:** `src/storage/performance_tracker.py` (SQLite)

**Configuration:** `config/agent_config.yaml` → `reinforcement_learning`

---

## Utility Modules

### LLM Interface

**Module:** `src/utils/llm_interface.py`

**Purpose:** Unified API for multiple LLM providers

**Supported Providers:**
- Ollama (free, local)
- OpenAI (paid, API)
- Google Gemini (paid, API)

**Features:**
- Automatic fallback (Ollama → OpenAI)
- Retry logic with exponential backoff
- Structured output generation (JSON schema)

**Configuration:** `config/api_config.yaml` → `llm`

---

### FOL Parser

**Module:** `src/utils/fol_parser.py`

**Purpose:** First-Order Logic claim decomposition

**Methods:**
- LLM-based decomposition (intelligent)
- Heuristic decomposition (fast fallback)

**Output:** List of atomic predicates

---

### Credibility Checker

**Module:** `src/utils/credibility_checker.py`

**Purpose:** Assess source reliability

**Methods:**
1. **Heuristic (free):**
   - Domain suffix rules (.edu → HIGH)
   - Whitelist of known sources
   - HTTPS check

2. **MBFC API (paid):**
   - Factuality rating
   - Political bias check

**Output:** Credibility score (0.0 - 1.0) + level (high/medium/low)

---

### Web Scraper

**Module:** `src/utils/web_scraper.py`

**Purpose:** Extract content from URLs

**Backends:**
- **requests + BeautifulSoup** (fast, free)
- **Selenium** (slower, full JS support)

**Features:**
- Main content extraction (remove ads/nav)
- Passage extraction with relevance scoring
- Retry logic

---

## Storage Modules

### Evidence Store

**Module:** `src/storage/evidence_store.py`

**Format:** JSON

**Structure:**
```json
{
    "claim_id_1": [
        {
            "id": "EV1",
            "subclaim_id": "SC1",
            "source_url": "...",
            "passage": "...",
            "credibility_score": 0.9,
            "credibility_level": "high",
            "retrieved_at": "2026-01-21T02:00:00Z",
            "metadata": {}
        }
    ]
}
```

---

### Performance Tracker

**Module:** `src/storage/performance_tracker.py`

**Format:** SQLite

**Schema:**
```sql
CREATE TABLE runs (
    id INTEGER PRIMARY KEY,
    claim TEXT,
    verdict TEXT,
    ground_truth TEXT,
    accuracy REAL,
    evidence_quality REAL,
    processing_time REAL,
    timestamp TEXT
);
```

---

## Data Flow

### End-to-End Verification Pipeline

```
User Claim
    │
    ▼
┌─────────────────────┐
│ Input Ingestion     │
│ - FOL decomposition │
│ - Filter verifiable │
└──────────┬──────────┘
           │
    [Subclaims]
           │
           ▼
┌─────────────────────┐
│ Query Generation    │
│ - k queries/subclaim│
└──────────┬──────────┘
           │
    [Queries]
           │
           ▼
┌─────────────────────┐
│ Evidence Seeking    │
│ Stage 1: Search     │
│ Stage 2: Credibility│
│ Stage 3: Extract    │
└──────────┬──────────┘
           │
    [Evidence]
           │
           ▼
┌─────────────────────┐
│ Verdict Prediction  │
│ - Weighted voting   │
│ - Explanation gen   │
└──────────┬──────────┘
           │
    [Verdict]
           │
           ├─────────────────┐
           │                 │
           ▼                 ▼
    ┌───────────┐     ┌────────────┐
    │    XAI    │     │     RL     │
    │ Explain   │     │  Track     │
    │ verdict   │     │performance │
    └───────────┘     └────────────┘
```

---

## Configuration System

### Configuration Files

1. **`config/agent_config.yaml`**
   - Agent-specific parameters
   - Thresholds and weights
   - Enable/disable features

2. **`config/api_config.yaml`**
   - LLM provider settings
   - Search API configuration
   - Credibility check methods
   - Rate limiting

3. **`config/benchmark_config.yaml`**
   - Dataset paths
   - Evaluation metrics
   - Sample strategies
   - Baseline comparisons

### Environment Variables

`.env` file (optional):
- API keys for paid services
- Override configuration values
- Runtime settings

---

## Extension Points

### Adding New Agents

1. Create `src/agents/new_agent.py`
2. Implement agent interface:
   ```python
   class NewAgent:
       def __init__(self, config):
           pass

       def process(self, input_data):
           # Agent logic
           return output_data
   ```
3. Register in orchestrator
4. Add configuration to `agent_config.yaml`

### Adding New LLM Providers

1. Implement provider client in `src/utils/llm_interface.py`
2. Add to `LLMProvider` enum
3. Implement `_generate_<provider>()` method
4. Add configuration to `api_config.yaml`

### Adding New Benchmarks

1. Create dataset loader in `src/evaluation/benchmark_loader.py`
2. Add configuration to `benchmark_config.yaml`
3. Implement evaluation metrics in `src/evaluation/metrics.py`

---

## Performance Optimization

### Caching Strategy

- **Evidence cache:** Avoid re-scraping same URLs
- **LLM cache:** Cache responses for identical prompts
- **Benchmark cache:** Store evaluation results

### Parallel Processing

- Multiple queries processed concurrently
- Async web scraping (optional)
- Batch LLM requests

### Rate Limiting

- Configurable delays between requests
- Exponential backoff on failures
- Provider-specific rate limits

---

## Security & Privacy

### Data Handling

- No user data stored permanently
- Evidence cache can be cleared
- SQLite DB for local performance tracking only

### API Key Management

- Stored in `.env` (not version controlled)
- Environment variable substitution
- Optional - system works without keys

### Web Scraping Ethics

- Respects robots.txt
- User-agent identification
- Rate limiting to avoid overload
- Timeout handling

---

## Testing Strategy

### Unit Tests

Test individual components:
- FOL parser correctness
- Credibility scoring logic
- Evidence aggregation

### Integration Tests

Test agent interactions:
- End-to-end pipeline
- Error handling
- Fallback mechanisms

### Benchmark Evaluation

Test on academic datasets:
- FEVEROUS, HoVer, SciFact
- Statistical significance
- Ablation studies

---

## Deployment

### Local Development

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2:3b
python examples/simple_claim.py
```

### Production Deployment

- Docker containerization (future)
- API endpoint (future)
- Horizontal scaling (multiple workers)

---

## Monitoring & Debugging

### Logging

- Configured via `config/agent_config.yaml`
- Levels: DEBUG, INFO, WARNING, ERROR
- Output: Console + File

### Performance Metrics

Tracked in `src/storage/performance_tracker.py`:
- Accuracy over time
- Processing time per claim
- Evidence quality metrics

### Error Handling

- Graceful degradation (fallback to heuristics)
- Detailed error messages
- Retry logic with backoff

---

## Future Architecture Improvements

1. **Microservices:** Separate agents into independent services
2. **Message Queue:** Async communication between agents
3. **Distributed Cache:** Redis for shared evidence cache
4. **API Gateway:** RESTful API for external integrations
5. **Real-time Updates:** WebSocket connections for live results

---

## References

- LangGraph documentation: https://python.langchain.com/docs/langgraph
- Ollama API: https://github.com/ollama/ollama/blob/main/docs/api.md
- DuckDuckGo Search: https://github.com/deedy5/duckduckgo_search

---

## Contact & Support

For technical questions or contributions:
- GitHub Issues: https://github.com/SIDDHARTH1-1CHAUHAN/Research_Paper01/issues
- Documentation: See README.md and RESEARCH_PAPER.md
