"""
Agent modules for the Multi-Agent Fact-Checking System

Each agent handles a specific stage in the fact-verification pipeline:
1. InputIngestionAgent - Claim decomposition and verifiability filtering
2. QueryGenerationAgent - Search query creation for evidence retrieval
3. EvidenceSeekingAgent - Web search and content extraction
4. VerdictPredictionAgent - Evidence aggregation and verdict synthesis
5. ExplainableAIAgent - LIME/SHAP-inspired explanations and LLM judge
6. ReinforcementLearningAgent - Performance tracking and optimization
"""

from src.agents.input_ingestion import InputIngestionAgent, IngestionResult
from src.agents.query_generation import QueryGenerationAgent, QueryResult
from src.agents.evidence_seeking import EvidenceSeekingAgent, Evidence, EvidenceResult
from src.agents.verdict_prediction import VerdictPredictionAgent, VerdictResult, SubclaimVerdict
from src.agents.explainable_ai import ExplainableAIAgent, ExplanationResult
from src.agents.reinforcement_learning import ReinforcementLearningAgent, RLResult, PerformanceMetrics

__all__ = [
    # Agents
    'InputIngestionAgent',
    'QueryGenerationAgent',
    'EvidenceSeekingAgent',
    'VerdictPredictionAgent',
    'ExplainableAIAgent',
    'ReinforcementLearningAgent',

    # Results
    'IngestionResult',
    'QueryResult',
    'Evidence',
    'EvidenceResult',
    'VerdictResult',
    'SubclaimVerdict',
    'ExplanationResult',
    'RLResult',
    'PerformanceMetrics',
]
