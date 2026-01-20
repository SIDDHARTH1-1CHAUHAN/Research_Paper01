"""
Orchestrator - Coordinates all 6 agents in the fact-checking pipeline
"""

from typing import Dict, Any, Optional
import yaml
from pathlib import Path
from loguru import logger
import sys
from dataclasses import asdict

# Import all agents
from src.agents.input_ingestion import InputIngestionAgent
from src.agents.query_generation import QueryGenerationAgent
from src.agents.evidence_seeking import EvidenceSeekingAgent
from src.agents.verdict_prediction import VerdictPredictionAgent
from src.agents.explainable_ai import ExplainableAIAgent
from src.agents.reinforcement_learning import ReinforcementLearningAgent
from src.utils.llm_interface import LLMInterface


class FactCheckingOrchestrator:
    """
    Main orchestrator for the multi-agent fact-checking pipeline.

    Workflow:
    1. Input Ingestion → Decompose claim
    2. Query Generation → Create search queries
    3. Evidence Seeking → Retrieve and validate evidence
    4. Verdict Prediction → Aggregate evidence and decide
    5. Explainable AI → Generate explanations
    6. Reinforcement Learning → Track performance
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize orchestrator with all agents.

        Args:
            config_path: Path to agent_config.yaml
        """
        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize LLM interface
        try:
            self.llm = LLMInterface()
            logger.info("LLM interface initialized successfully")
        except Exception as e:
            logger.warning(f"LLM initialization failed: {e}. Using fallback mode.")
            self.llm = None

        # Initialize all agents
        logger.info("Initializing agents...")

        self.input_ingestion = InputIngestionAgent(
            config=self.config.get('input_ingestion', {}),
            llm_interface=self.llm
        )

        self.query_generation = QueryGenerationAgent(
            config=self.config.get('query_generation', {}),
            llm_interface=self.llm
        )

        self.evidence_seeking = EvidenceSeekingAgent(
            config=self.config.get('evidence_seeking', {})
        )

        self.verdict_prediction = VerdictPredictionAgent(
            config=self.config.get('verdict_prediction', {}),
            llm_interface=self.llm
        )

        self.explainable_ai = ExplainableAIAgent(
            config=self.config.get('explainable_ai', {})
        )

        self.reinforcement_learning = ReinforcementLearningAgent(
            config=self.config.get('reinforcement_learning', {})
        )

        logger.info("✓ All agents initialized successfully")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load agent configuration from YAML"""
        if not config_path:
            # Try to find config file
            possible_paths = [
                Path("config/agent_config.yaml"),
                Path("../config/agent_config.yaml"),
                Path("../../config/agent_config.yaml"),
            ]

            for path in possible_paths:
                if path.exists():
                    config_path = str(path)
                    break

        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return config
        else:
            logger.warning("Configuration file not found, using defaults")
            return {}

    def verify_claim(
        self,
        claim: str,
        ground_truth: Optional[str] = None,
        enable_xai: bool = True,
        enable_rl: bool = True
    ) -> Dict[str, Any]:
        """
        Verify a factual claim end-to-end.

        Args:
            claim: Natural language claim to verify
            ground_truth: Optional ground truth label for evaluation
            enable_xai: Enable Explainable AI agent
            enable_rl: Enable Reinforcement Learning agent

        Returns:
            Complete results dictionary with all agent outputs
        """
        logger.info("="*80)
        logger.info(f"VERIFYING CLAIM: {claim}")
        logger.info("="*80)

        results = {
            'claim': claim,
            'ground_truth': ground_truth
        }

        try:
            # AGENT 1: Input Ingestion
            logger.info("\n[1/6] Input Ingestion Agent - Decomposing claim...")
            ingestion_result = self.input_ingestion.process(claim)
            results['ingestion'] = self.input_ingestion.to_dict(ingestion_result)
            logger.info(f"✓ Found {len(ingestion_result.verifiable_subclaims)} verifiable subclaims")

            if not ingestion_result.verifiable_subclaims:
                logger.warning("No verifiable subclaims found!")
                results['verdict'] = {
                    'final_verdict': 'NOT_SUPPORTED',
                    'reason': 'No verifiable subclaims'
                }
                return results

            # AGENT 2: Query Generation
            logger.info("\n[2/6] Query Generation Agent - Creating search queries...")
            query_results = self.query_generation.process(ingestion_result.verifiable_subclaims)
            results['queries'] = [
                {
                    'subclaim_id': qr.subclaim_id,
                    'queries': qr.queries
                }
                for qr in query_results
            ]
            total_queries = sum(len(qr.queries) for qr in query_results)
            logger.info(f"✓ Generated {total_queries} search queries")

            # AGENT 3: Evidence Seeking
            logger.info("\n[3/6] Evidence Seeking Agent - Retrieving evidence...")
            logger.info("     Stage 1: Web Search")
            logger.info("     Stage 2: Credibility Check")
            logger.info("     Stage 3: Content Extraction")

            evidence_results = self.evidence_seeking.process(query_results)
            results['evidence'] = [
                {
                    'subclaim_id': er.subclaim_id,
                    'total_sources': er.total_sources,
                    'high_credibility_count': er.high_credibility_count,
                    'evidence': [asdict(ev) for ev in er.evidence]
                }
                for er in evidence_results
            ]
            total_evidence = sum(er.total_sources for er in evidence_results)
            logger.info(f"✓ Retrieved {total_evidence} evidence items")

            # AGENT 4: Verdict Prediction
            logger.info("\n[4/6] Verdict Prediction Agent - Aggregating evidence...")
            verdict_result = self.verdict_prediction.process(claim, evidence_results)
            results['verdict'] = self.verdict_prediction.to_dict(verdict_result)
            logger.info(f"✓ Verdict: {verdict_result.final_verdict} (confidence: {verdict_result.overall_confidence:.2f})")

            # AGENT 5: Explainable AI (optional)
            if enable_xai:
                logger.info("\n[5/6] Explainable AI Agent - Generating explanations...")
                xai_result = self.explainable_ai.process(verdict_result, evidence_results)
                results['explanation'] = self.explainable_ai.to_dict(xai_result)
                quality = xai_result.explanation_quality['overall']
                logger.info(f"✓ Explanation quality: {quality:.2f}")
            else:
                logger.info("\n[5/6] Explainable AI Agent - Skipped")

            # AGENT 6: Reinforcement Learning (optional)
            if enable_rl:
                logger.info("\n[6/6] Reinforcement Learning Agent - Recording performance...")
                run_metrics = self.reinforcement_learning.record_run(
                    claim,
                    verdict_result,
                    evidence_results,
                    ground_truth
                )
                results['performance'] = asdict(run_metrics)
                logger.info(f"✓ Run recorded (accuracy: {run_metrics.accuracy:.2f})")
            else:
                logger.info("\n[6/6] Reinforcement Learning Agent - Skipped")

            logger.info("\n" + "="*80)
            logger.info("VERIFICATION COMPLETE")
            logger.info("="*80)

            return results

        except Exception as e:
            logger.error(f"Error during verification: {e}", exc_info=True)
            results['error'] = str(e)
            return results

    def get_performance_analysis(self) -> Dict[str, Any]:
        """
        Get performance analysis from RL agent.

        Returns:
            Performance patterns and suggestions
        """
        logger.info("Generating performance analysis...")
        rl_result = self.reinforcement_learning.process()
        return self.reinforcement_learning.to_dict(rl_result)

    def batch_verify(
        self,
        claims: list[tuple[str, Optional[str]]],
        enable_xai: bool = False,
        enable_rl: bool = True
    ) -> list[Dict[str, Any]]:
        """
        Verify multiple claims in batch.

        Args:
            claims: List of (claim, ground_truth) tuples
            enable_xai: Enable XAI for each claim
            enable_rl: Enable RL tracking

        Returns:
            List of results dictionaries
        """
        logger.info(f"Batch verification: {len(claims)} claims")

        results = []
        for i, (claim, ground_truth) in enumerate(claims, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing claim {i}/{len(claims)}")
            logger.info(f"{'='*80}")

            result = self.verify_claim(
                claim,
                ground_truth=ground_truth,
                enable_xai=enable_xai,
                enable_rl=enable_rl
            )
            results.append(result)

        # Get performance analysis if RL was enabled
        if enable_rl:
            performance = self.get_performance_analysis()
            logger.info("\n" + "="*80)
            logger.info("BATCH PERFORMANCE ANALYSIS")
            logger.info("="*80)
            logger.info(f"Total runs: {performance['metrics'].get('total_runs', 0)}")
            logger.info(f"Mean accuracy: {performance['metrics'].get('mean_accuracy', 0):.2f}")
            logger.info(f"Mean evidence quality: {performance['metrics'].get('mean_evidence_quality', 0):.2f}")
            logger.info("\nSuggestions:")
            for suggestion in performance['suggestions']:
                logger.info(f"  - {suggestion}")

        return results


# Convenience function
def verify_claim(claim: str, ground_truth: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function for single claim verification.

    Args:
        claim: Claim to verify
        ground_truth: Optional ground truth label

    Returns:
        Results dictionary
    """
    orchestrator = FactCheckingOrchestrator()
    return orchestrator.verify_claim(claim, ground_truth)
