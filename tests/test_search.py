import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.evidence_seeking import EvidenceSeekingAgent

def test_search():
    agent = EvidenceSeekingAgent(config={'api_keys': {'serper': 'test_key'}})
    
    print("Testing HoVER (Wikipedia via DuckDuckGo):")
    results = agent._search_web("Coffee", "HoVER")
    print(f"Results: {results}")
    
    print("\nTesting SciFact-Open (Scholar/PubMed via Serper):")
    results = agent._search_web("Quantum computing", "SciFact-Open")
    print(f"Results: {results}")
    
    print("\nTesting General (DuckDuckGo):")
    results = agent._search_web("Deep learning", "General")
    print(f"Results: {results}")

if __name__ == "__main__":
    test_search()
