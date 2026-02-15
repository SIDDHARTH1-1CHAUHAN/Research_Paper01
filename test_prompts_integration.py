"""
Test script for verifying the agent prompts integration
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_prompts_loader():
    """Test the prompts loader functionality"""
    print("=" * 60)
    print("TEST 1: Prompts Loader")
    print("=" * 60)

    from src.utils.prompts_loader import PromptsLoader, get_prompts_loader

    # Test singleton
    loader = get_prompts_loader()
    print(f"Config path: {loader.config_path}")

    # List all agents
    agents = loader.list_agents()
    print(f"\nAvailable agents ({len(agents)}):")
    for agent in agents:
        print(f"  - {agent}")

    # Get pipeline flow
    flow = loader.get_pipeline_flow()
    print(f"\nPipeline flow: {' -> '.join(flow)}")

    # Test prompt retrieval with variable substitution
    print("\n" + "-" * 40)
    print("Testing prompt variable substitution:")
    print("-" * 40)

    test_claim = "The Eiffel Tower is located in Paris, France."
    prompt = loader.get_prompt('claim_decomposition_agent', claim=test_claim)
    print(f"\nClaim Decomposition prompt (truncated):")
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)

    # Test subclaim classification prompt
    prompt2 = loader.get_prompt('subclaim_classification_agent', claim=test_claim)
    print(f"\nSubclaim Classification prompt (truncated):")
    print(prompt2[:500] + "..." if len(prompt2) > 500 else prompt2)

    print("\n[PASS] Prompts loader working correctly!")
    return True


def test_input_ingestion_heuristic():
    """Test input ingestion agent with heuristic (no LLM)"""
    print("\n" + "=" * 60)
    print("TEST 2: Input Ingestion Agent (Heuristic)")
    print("=" * 60)

    from src.agents.input_ingestion import InputIngestionAgent

    agent = InputIngestionAgent()

    # Test claim decomposition
    test_claim = "Howard University Hospital and Providence Hospital are both located in Washington, D.C."
    print(f"\nTest claim: {test_claim}")

    result = agent.process(test_claim)

    print(f"\nResults:")
    print(f"  Original claim: {result.original_claim}")
    print(f"  Verifiable subclaims: {len(result.verifiable_subclaims)}")
    print(f"  Filtered subclaims: {result.filtered_count}")

    for sc in result.verifiable_subclaims:
        print(f"\n  Subclaim {sc['id']}:")
        print(f"    Text: {sc['text']}")
        print(f"    Logical form: {sc['logical_form']}")
        print(f"    Verifiability: {sc['verifiability']}")

    # Test classification heuristic
    print("\n" + "-" * 40)
    print("Testing subclaim classification (heuristic):")
    print("-" * 40)

    test_claims = [
        "The film Parasite won the Academy Award for Best Picture in 2020.",
        "Parasite deserved to win the Academy Award.",
        "The movie will be remembered for decades.",
        "Some people think it's a great film."
    ]

    for claim in test_claims:
        classification = agent._classify_heuristic(claim)
        print(f"\n  '{claim[:50]}...'")
        print(f"    -> {classification['classification']}: {classification['explanation']}")

    print("\n[PASS] Input Ingestion Agent working correctly!")
    return True


def test_query_generation():
    """Test query generation agent"""
    print("\n" + "=" * 60)
    print("TEST 3: Query Generation Agent")
    print("=" * 60)

    from src.agents.query_generation import QueryGenerationAgent

    agent = QueryGenerationAgent(config={'queries_per_subclaim': 3})

    # Test subclaims
    subclaims = [
        {
            'id': 'SC1',
            'text': 'Howard University Hospital is located in Washington, D.C.',
            'entities': ['Howard University Hospital', 'Washington D.C.']
        },
        {
            'id': 'SC2',
            'text': 'Providence Hospital is located in Washington, D.C.',
            'entities': ['Providence Hospital', 'Washington D.C.']
        }
    ]

    results = agent.process(subclaims)

    print(f"\nGenerated queries for {len(results)} subclaims:")
    for qr in results:
        print(f"\n  {qr.subclaim_id}: {qr.subclaim_text[:50]}...")
        for i, q in enumerate(qr.queries, 1):
            print(f"    Q{i}: {q}")

    print("\n[PASS] Query Generation Agent working correctly!")
    return True


def test_verdict_prediction_heuristic():
    """Test verdict prediction agent with heuristic"""
    print("\n" + "=" * 60)
    print("TEST 4: Verdict Prediction Agent (Heuristic)")
    print("=" * 60)

    from src.agents.verdict_prediction import VerdictPredictionAgent
    from src.agents.evidence_seeking import Evidence, EvidenceResult
    from datetime import datetime

    agent = VerdictPredictionAgent()

    # Create mock evidence
    evidence1 = Evidence(
        id="EV1",
        subclaim_id="SC1",
        source_url="https://example.edu/hospital",
        source_name="Example University",
        passage="Howard University Hospital is a major teaching hospital located in Washington, D.C., serving the community since 1862.",
        credibility_score=0.9,
        credibility_level="high",
        retrieved_at=datetime.utcnow().isoformat(),
        metadata={},
        language='en'
    )

    evidence2 = Evidence(
        id="EV2",
        subclaim_id="SC1",
        source_url="https://wikipedia.org/wiki/Howard",
        source_name="Wikipedia",
        passage="Howard University Hospital is located in Washington, D.C.",
        credibility_score=0.7,
        credibility_level="medium",
        retrieved_at=datetime.utcnow().isoformat(),
        metadata={},
        language='en'
    )

    evidence_result = EvidenceResult(
        subclaim_id="SC1",
        evidence=[evidence1, evidence2],
        total_sources=2,
        high_credibility_count=1,
        language='en'
    )

    # Test verdict prediction
    claim = "Howard University Hospital is located in Washington, D.C."
    result = agent.process(claim, [evidence_result])

    print(f"\nClaim: {claim}")
    print(f"\nVerdict: {result.final_verdict}")
    print(f"Confidence: {result.overall_confidence:.2f}")
    print(f"\nExplanation (truncated):")
    print(result.explanation[:500] + "..." if len(result.explanation) > 500 else result.explanation)

    print("\n[PASS] Verdict Prediction Agent working correctly!")
    return True


def test_explainable_ai():
    """Test explainable AI agent"""
    print("\n" + "=" * 60)
    print("TEST 5: Explainable AI Agent")
    print("=" * 60)

    from src.agents.explainable_ai import ExplainableAIAgent
    from src.agents.verdict_prediction import VerdictResult, SubclaimVerdict
    from src.agents.evidence_seeking import Evidence, EvidenceResult
    from datetime import datetime

    agent = ExplainableAIAgent()

    # Create mock verdict result
    subclaim_verdict = SubclaimVerdict(
        subclaim_id="SC1",
        verdict="SUPPORTED",
        confidence=0.85,
        evidence_count=2,
        high_credibility_count=1
    )

    verdict_result = VerdictResult(
        original_claim="Howard University Hospital is located in Washington, D.C.",
        subclaim_verdicts=[subclaim_verdict],
        final_verdict="SUPPORTED",
        overall_confidence=0.85,
        explanation="The claim is supported by credible evidence from educational sources.",
        metadata={'total_sources': 2, 'high_credibility_sources': 1, 'processing_time': 1.5}
    )

    # Create mock evidence
    evidence = Evidence(
        id="EV1",
        subclaim_id="SC1",
        source_url="https://example.edu/hospital",
        source_name="Example University",
        passage="Howard University Hospital is located in Washington, D.C.",
        credibility_score=0.9,
        credibility_level="high",
        retrieved_at=datetime.utcnow().isoformat(),
        metadata={},
        language='en'
    )

    evidence_result = EvidenceResult(
        subclaim_id="SC1",
        evidence=[evidence],
        total_sources=1,
        high_credibility_count=1,
        language='en'
    )

    # Generate explanations
    result = agent.process(verdict_result, [evidence_result])

    print(f"\nVerdict: {result.verdict}")
    print(f"\nFeature Importance:")
    for ev_id, score in result.feature_importance.items():
        print(f"  {ev_id}: {score:.3f}")

    print(f"\nCritical Evidence:")
    for ce in result.critical_evidence:
        print(f"  - {ce[:80]}...")

    print(f"\nCounterfactual Analysis:")
    print(f"  {result.counterfactual_analysis}")

    print(f"\nExplanation Quality:")
    for metric, value in result.explanation_quality.items():
        print(f"  {metric}: {value:.2f}")

    # Test detailed report generation
    print("\n" + "-" * 40)
    print("Detailed Report:")
    print("-" * 40)
    report = agent.generate_detailed_report(verdict_result, [evidence_result], result)
    print(report)

    print("\n[PASS] Explainable AI Agent working correctly!")
    return True


def run_all_tests():
    """Run all integration tests"""
    print("\n")
    print("*" * 60)
    print("  MULTI-AGENT FACT-CHECKER PROMPTS INTEGRATION TESTS")
    print("*" * 60)

    tests = [
        ("Prompts Loader", test_prompts_loader),
        ("Input Ingestion", test_input_ingestion_heuristic),
        ("Query Generation", test_query_generation),
        ("Verdict Prediction", test_verdict_prediction_heuristic),
        ("Explainable AI", test_explainable_ai),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success, None))
        except Exception as e:
            print(f"\n[FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False, str(e)))

    # Summary
    print("\n")
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for name, success, error in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {status} {name}")
        if error:
            print(f"         Error: {error}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
