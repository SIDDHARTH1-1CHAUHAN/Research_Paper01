"""
Orchestrator - Coordinates all 6 agents in the fact-checking pipeline
Supports multilingual claims through automatic language detection and translation
Uses configurable prompts from prompts.yaml for LLM-enhanced verification
"""

from typing import Dict, Any, Optional, List
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
from src.utils.prompts_loader import get_prompts_loader

# Multilingual support
from src.utils.language_detector import LanguageDetector
from src.utils.translator import Translator
from src.utils.multilingual_config import get_multilingual_config


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

        # Initialize multilingual components
        self.ml_config = get_multilingual_config()
        self.language_detector = LanguageDetector()
        self.translator = Translator(self.llm)
        logger.info("Multilingual components initialized")

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
            config=self.config.get('evidence_seeking', {}),
            llm_interface=self.llm  # Added LLM for content extraction
        )

        self.verdict_prediction = VerdictPredictionAgent(
            config=self.config.get('verdict_prediction', {}),
            llm_interface=self.llm
        )

        self.explainable_ai = ExplainableAIAgent(
            config=self.config.get('explainable_ai', {}),
            llm_interface=self.llm  # Added LLM for judge functionality
        )

        # Load prompts configuration
        self.prompts_loader = get_prompts_loader()

        self.reinforcement_learning = ReinforcementLearningAgent(
            config=self.config.get('reinforcement_learning', {})
        )

        logger.info("[OK] All agents initialized successfully")

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
        language: Optional[str] = None,
        enable_xai: bool = True,
        enable_rl: bool = True
    ) -> Dict[str, Any]:
        """
        Verify a factual claim end-to-end.

        Args:
            claim: Natural language claim to verify
            ground_truth: Optional ground truth label for evaluation
            language: Language code (auto-detected if None)
            enable_xai: Enable Explainable AI agent
            enable_rl: Enable Reinforcement Learning agent

        Returns:
            Complete results dictionary with all agent outputs
        """
        logger.info("="*80)
        logger.info(f"VERIFYING CLAIM: {claim}")
        logger.info("="*80)

        # Detect language if not provided
        original_claim = claim
        if language is None:
            detection = self.language_detector.detect(claim)
            language = detection.primary_language
            detection_confidence = detection.primary_confidence
            logger.info(f"Detected language: {language} (confidence: {detection_confidence:.2f})")
        else:
            detection_confidence = 1.0

        # Translate to English if not English (translate-then-verify approach)
        working_claim = claim
        was_translated = False
        if language != 'en':
            logger.info(f"Translating claim from {language} to English...")
            translation_result = self.translator.translate(claim, language, 'en')
            working_claim = translation_result.translated_text
            was_translated = True
            logger.info(f"Translated claim: {working_claim}")

        results = {
            'claim': original_claim,
            'working_claim': working_claim,
            'ground_truth': ground_truth,
            'detected_language': language,
            'detection_confidence': detection_confidence,
            'was_translated': was_translated
        }

        try:
            # AGENT 1: Input Ingestion (use English working_claim)
            logger.info("\n[1/6] Input Ingestion Agent - Decomposing claim...")
            ingestion_result = self.input_ingestion.process(working_claim)
            results['ingestion'] = self.input_ingestion.to_dict(ingestion_result)
            logger.info(f"[OK] Found {len(ingestion_result.verifiable_subclaims)} verifiable subclaims")

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
            logger.info(f"[OK] Generated {total_queries} search queries")

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
            logger.info(f"[OK] Retrieved {total_evidence} evidence items")

            # AGENT 4: Verdict Prediction
            logger.info("\n[4/6] Verdict Prediction Agent - Aggregating evidence...")
            verdict_result = self.verdict_prediction.process(working_claim, evidence_results)
            results['verdict'] = self.verdict_prediction.to_dict(verdict_result)
            logger.info(f"[OK] Verdict: {verdict_result.final_verdict} (confidence: {verdict_result.overall_confidence:.2f})")

            # Translate explanation back to original language if needed
            if was_translated and language != 'en':
                logger.info(f"Translating explanation back to {language}...")
                translated_explanation = self.translator.translate(
                    verdict_result.explanation, 'en', language
                )
                results['verdict']['explanation_original_language'] = translated_explanation.translated_text

            # AGENT 5: Explainable AI (optional)
            if enable_xai:
                logger.info("\n[5/6] Explainable AI Agent - Generating explanations...")
                xai_result = self.explainable_ai.process(verdict_result, evidence_results)
                results['explanation'] = self.explainable_ai.to_dict(xai_result)
                quality = xai_result.explanation_quality['overall']
                logger.info(f"[OK] Explanation quality: {quality:.2f}")
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
                logger.info(f"[OK] Run recorded (accuracy: {run_metrics.accuracy:.2f})")
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


    def verify_claim_enhanced(
        self,
        claim: str,
        ground_truth: Optional[str] = None,
        language: Optional[str] = None,
        enable_xai: bool = True,
        enable_rl: bool = True
    ) -> Dict[str, Any]:
        """
        Verify a factual claim using LLM-enhanced methods for improved accuracy.

        This method uses the configurable prompts from prompts.yaml and
        LLM-based processing at each stage for better results.

        Args:
            claim: Natural language claim to verify
            ground_truth: Optional ground truth label for evaluation
            language: Language code (auto-detected if None)
            enable_xai: Enable Explainable AI agent
            enable_rl: Enable Reinforcement Learning agent

        Returns:
            Complete results dictionary with all agent outputs
        """
        logger.info("="*80)
        logger.info(f"VERIFYING CLAIM (LLM-ENHANCED): {claim}")
        logger.info("="*80)

        # Detect language if not provided
        original_claim = claim
        if language is None:
            detection = self.language_detector.detect(claim)
            language = detection.primary_language
            detection_confidence = detection.primary_confidence
            logger.info(f"Detected language: {language} (confidence: {detection_confidence:.2f})")
        else:
            detection_confidence = 1.0

        # Translate to English if not English
        working_claim = claim
        was_translated = False
        if language != 'en':
            logger.info(f"Translating claim from {language} to English...")
            translation_result = self.translator.translate(claim, language, 'en')
            working_claim = translation_result.translated_text
            was_translated = True
            logger.info(f"Translated claim: {working_claim}")

        results = {
            'claim': original_claim,
            'working_claim': working_claim,
            'ground_truth': ground_truth,
            'detected_language': language,
            'detection_confidence': detection_confidence,
            'was_translated': was_translated,
            'mode': 'llm_enhanced'
        }

        try:
            # AGENT 1: Input Ingestion with LLM Classification
            logger.info("\n[1/6] Input Ingestion Agent (LLM-Enhanced) - Decomposing claim...")
            if self.llm:
                ingestion_result = self.input_ingestion.process_with_classification(working_claim)
            else:
                ingestion_result = self.input_ingestion.process(working_claim)
            results['ingestion'] = self.input_ingestion.to_dict(ingestion_result)
            logger.info(f"[OK] Found {len(ingestion_result.verifiable_subclaims)} verifiable subclaims")

            if not ingestion_result.verifiable_subclaims:
                logger.warning("No verifiable subclaims found!")
                results['verdict'] = {
                    'final_verdict': 'NOT_SUPPORTED',
                    'reason': 'No verifiable subclaims'
                }
                return results

            # AGENT 2: Query Generation with LLM
            logger.info("\n[2/6] Query Generation Agent (LLM-Enhanced) - Creating search queries...")
            if self.llm:
                query_results = self.query_generation.process_with_llm(ingestion_result.verifiable_subclaims)
            else:
                query_results = self.query_generation.process(ingestion_result.verifiable_subclaims)
            results['queries'] = [
                {
                    'subclaim_id': qr.subclaim_id,
                    'queries': qr.queries
                }
                for qr in query_results
            ]
            total_queries = sum(len(qr.queries) for qr in query_results)
            logger.info(f"[OK] Generated {total_queries} search queries")

            # AGENT 3: Evidence Seeking with LLM Content Extraction
            logger.info("\n[3/6] Evidence Seeking Agent (LLM-Enhanced) - Retrieving evidence...")
            logger.info("     Stage 1: Web Search")
            logger.info("     Stage 2: Credibility Check")
            logger.info("     Stage 3: LLM Content Extraction")

            if self.llm:
                evidence_results = self.evidence_seeking.process_with_llm(query_results)
            else:
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
            logger.info(f"[OK] Retrieved {total_evidence} evidence items")

            # AGENT 4: Verdict Prediction with LLM
            logger.info("\n[4/6] Verdict Prediction Agent (LLM-Enhanced) - Aggregating evidence...")
            if self.llm:
                verdict_result = self.verdict_prediction.process_with_llm(
                    working_claim,
                    ingestion_result.verifiable_subclaims,
                    evidence_results
                )
            else:
                verdict_result = self.verdict_prediction.process(working_claim, evidence_results)
            results['verdict'] = self.verdict_prediction.to_dict(verdict_result)
            logger.info(f"[OK] Verdict: {verdict_result.final_verdict} (confidence: {verdict_result.overall_confidence:.2f})")

            # Translate explanation back to original language if needed
            if was_translated and language != 'en':
                logger.info(f"Translating explanation back to {language}...")
                translated_explanation = self.translator.translate(
                    verdict_result.explanation, 'en', language
                )
                results['verdict']['explanation_original_language'] = translated_explanation.translated_text

            # AGENT 5: Explainable AI with LLM Judge
            if enable_xai:
                logger.info("\n[5/6] Explainable AI Agent (LLM-Enhanced) - Generating explanations...")
                xai_result = self.explainable_ai.process(verdict_result, evidence_results)
                results['explanation'] = self.explainable_ai.to_dict(xai_result)

                # Generate detailed report
                detailed_report = self.explainable_ai.generate_detailed_report(
                    verdict_result, evidence_results, xai_result
                )
                results['detailed_report'] = detailed_report

                quality = xai_result.explanation_quality['overall']
                logger.info(f"[OK] Explanation quality: {quality:.2f}")
            else:
                logger.info("\n[5/6] Explainable AI Agent - Skipped")

            # AGENT 6: Reinforcement Learning
            if enable_rl:
                logger.info("\n[6/6] Reinforcement Learning Agent - Recording performance...")
                run_metrics = self.reinforcement_learning.record_run(
                    claim,
                    verdict_result,
                    evidence_results,
                    ground_truth
                )
                results['performance'] = asdict(run_metrics)
                logger.info(f"[OK] Run recorded (accuracy: {run_metrics.accuracy:.2f})")
            else:
                logger.info("\n[6/6] Reinforcement Learning Agent - Skipped")

            logger.info("\n" + "="*80)
            logger.info("VERIFICATION COMPLETE (LLM-ENHANCED)")
            logger.info("="*80)

            return results

        except Exception as e:
            logger.error(f"Error during verification: {e}", exc_info=True)
            results['error'] = str(e)
            return results

    def compare_methods(
        self,
        claim: str,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare heuristic vs LLM-enhanced verification methods.

        Args:
            claim: Claim to verify
            ground_truth: Optional ground truth label

        Returns:
            Comparison results with both methods and accuracy metrics
        """
        logger.info("="*80)
        logger.info("COMPARING VERIFICATION METHODS")
        logger.info("="*80)

        # Run heuristic method
        logger.info("\n--- HEURISTIC METHOD ---")
        heuristic_result = self.verify_claim(claim, ground_truth, enable_xai=True, enable_rl=False)

        # Run LLM-enhanced method
        logger.info("\n--- LLM-ENHANCED METHOD ---")
        enhanced_result = self.verify_claim_enhanced(claim, ground_truth, enable_xai=True, enable_rl=False)

        # Compare results
        comparison = {
            'claim': claim,
            'ground_truth': ground_truth,
            'heuristic': {
                'verdict': heuristic_result.get('verdict', {}).get('final_verdict'),
                'confidence': heuristic_result.get('verdict', {}).get('overall_confidence'),
                'subclaims_found': len(heuristic_result.get('ingestion', {}).get('verifiable_subclaims', [])),
                'evidence_count': sum(e.get('total_sources', 0) for e in heuristic_result.get('evidence', [])),
            },
            'llm_enhanced': {
                'verdict': enhanced_result.get('verdict', {}).get('final_verdict'),
                'confidence': enhanced_result.get('verdict', {}).get('overall_confidence'),
                'subclaims_found': len(enhanced_result.get('ingestion', {}).get('verifiable_subclaims', [])),
                'evidence_count': sum(e.get('total_sources', 0) for e in enhanced_result.get('evidence', [])),
            }
        }

        # Calculate accuracy if ground truth provided
        if ground_truth:
            comparison['heuristic']['correct'] = comparison['heuristic']['verdict'] == ground_truth
            comparison['llm_enhanced']['correct'] = comparison['llm_enhanced']['verdict'] == ground_truth

        # Use LLM judge to compare explanations if available
        if self.llm and 'explanation' in heuristic_result and 'explanation' in enhanced_result:
            explanations = {
                'Heuristic': {
                    'label': comparison['heuristic']['verdict'],
                    'explanation': heuristic_result.get('verdict', {}).get('explanation', '')
                },
                'LLM-Enhanced': {
                    'label': comparison['llm_enhanced']['verdict'],
                    'explanation': enhanced_result.get('verdict', {}).get('explanation', '')
                }
            }
            judge_result = self.explainable_ai.judge_explanations(claim, explanations)
            comparison['llm_judge'] = judge_result

        logger.info("\n" + "="*80)
        logger.info("COMPARISON COMPLETE")
        logger.info("="*80)
        logger.info(f"Heuristic verdict: {comparison['heuristic']['verdict']}")
        logger.info(f"LLM-Enhanced verdict: {comparison['llm_enhanced']['verdict']}")
        if ground_truth:
            logger.info(f"Ground truth: {ground_truth}")
            logger.info(f"Heuristic correct: {comparison['heuristic'].get('correct')}")
            logger.info(f"LLM-Enhanced correct: {comparison['llm_enhanced'].get('correct')}")

        return comparison


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


def verify_claim_enhanced(claim: str, ground_truth: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function for LLM-enhanced claim verification.

    Args:
        claim: Claim to verify
        ground_truth: Optional ground truth label

    Returns:
        Results dictionary
    """
    orchestrator = FactCheckingOrchestrator()
    return orchestrator.verify_claim_enhanced(claim, ground_truth)
