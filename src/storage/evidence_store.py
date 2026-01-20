"""
Evidence Store - JSON-based storage for retrieved evidence with metadata
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from loguru import logger


@dataclass
class Evidence:
    """Evidence item with metadata"""
    id: str
    subclaim_id: str
    source_url: str
    passage: str
    credibility_score: float
    credibility_level: str
    retrieved_at: str
    metadata: Dict[str, Any]


class EvidenceStore:
    """Stores and retrieves evidence with full metadata"""

    def __init__(self, storage_path: str = "data/cache/evidence.json"):
        """
        Initialize evidence store.

        Args:
            storage_path: Path to JSON storage file
        """
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        self.evidence: Dict[str, List[Evidence]] = {}
        self._load()

    def _load(self):
        """Load evidence from storage"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for claim_id, evidences in data.items():
                        self.evidence[claim_id] = [
                            Evidence(**ev) for ev in evidences
                        ]
                logger.info(f"Loaded {len(self.evidence)} evidence groups")
            except Exception as e:
                logger.error(f"Failed to load evidence: {e}")

    def save(self):
        """Save evidence to storage"""
        try:
            data = {
                claim_id: [asdict(ev) for ev in evidences]
                for claim_id, evidences in self.evidence.items()
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info("Evidence saved successfully")
        except Exception as e:
            logger.error(f"Failed to save evidence: {e}")

    def store(self, claim_id: str, evidence_list: List[Evidence]):
        """Store evidence for a claim"""
        self.evidence[claim_id] = evidence_list
        self.save()

    def get(self, claim_id: str) -> List[Evidence]:
        """Retrieve evidence for a claim"""
        return self.evidence.get(claim_id, [])

    def get_high_credibility(self, claim_id: str) -> List[Evidence]:
        """Get only high credibility evidence"""
        all_evidence = self.get(claim_id)
        return [ev for ev in all_evidence if ev.credibility_level == "high"]
