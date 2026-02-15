"""
Dataset Loader - Load and preprocess evaluation datasets
Supports: HOVER, FEVEROUS, SciFact-Open
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass
import random


@dataclass
class EvaluationSample:
    """Standardized evaluation sample across all datasets"""
    claim_id: str
    claim_text: str
    gold_label: str  # "supported" or "not_supported"
    evidence: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class DatasetLoader:
    """Load and standardize evaluation datasets"""

    LABEL_MAPPING = {
        # HOVER labels
        "SUPPORTED": "supported",
        "NOT_SUPPORTED": "not_supported",
        # FEVEROUS labels
        "SUPPORTS": "supported",
        "REFUTES": "not_supported",
        "NOT ENOUGH INFO": "not_supported",
        "NOT_ENOUGH_INFO": "not_supported",
        # SciFact labels
        "SUPPORT": "supported",
        "CONTRADICT": "not_supported",
        "NEI": "not_supported",
    }

    def __init__(self, data_dir: str = "data/datasets"):
        self.data_dir = Path(data_dir)

    def _normalize_label(self, label: str) -> str:
        """Normalize label to standard format"""
        label_upper = label.upper().replace("_", " ").strip()
        return self.LABEL_MAPPING.get(label_upper, "not_supported")

    def load_hover(
        self,
        split: str = "dev",
        limit: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 42
    ) -> List[EvaluationSample]:
        """
        Load HOVER dataset.

        Args:
            split: "train" or "dev"
            limit: Max samples to load
            shuffle: Shuffle samples
            seed: Random seed for shuffling

        Returns:
            List of standardized EvaluationSample
        """
        file_map = {
            "train": "hover_train_release_v1.1.json",
            "dev": "hover_dev_release_v1.1.json"
        }

        file_path = self.data_dir / "hover" / file_map.get(split, file_map["dev"])

        if not file_path.exists():
            raise FileNotFoundError(f"HOVER dataset not found at {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        samples = []
        for item in data:
            sample = EvaluationSample(
                claim_id=str(item.get("uid", item.get("id", len(samples)))),
                claim_text=item.get("claim", ""),
                gold_label=self._normalize_label(item.get("label", "NOT_SUPPORTED")),
                evidence=[
                    {"article": sf[0], "sentence_idx": sf[1]}
                    for sf in item.get("supporting_facts", [])
                ],
                metadata={
                    "num_hops": item.get("num_hops", 2),
                    "dataset": "hover",
                    "split": split
                }
            )
            samples.append(sample)

        if shuffle:
            random.seed(seed)
            random.shuffle(samples)

        if limit:
            samples = samples[:limit]

        return samples

    def load_feverous(
        self,
        split: str = "dev",
        limit: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 42
    ) -> List[EvaluationSample]:
        """
        Load FEVEROUS dataset.

        Args:
            split: "train" or "dev"
            limit: Max samples to load
            shuffle: Shuffle samples
            seed: Random seed for shuffling

        Returns:
            List of standardized EvaluationSample
        """
        file_map = {
            "train": "feverous_train.jsonl",
            "dev": "feverous_dev.jsonl"
        }

        file_path = self.data_dir / "feverous" / file_map.get(split, file_map["dev"])

        if not file_path.exists():
            raise FileNotFoundError(f"FEVEROUS dataset not found at {file_path}")

        samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)

                # Extract evidence
                evidence_list = []
                for ev_set in item.get("evidence", []):
                    if isinstance(ev_set, list):
                        for ev in ev_set:
                            if isinstance(ev, dict):
                                evidence_list.append(ev)
                    elif isinstance(ev_set, dict):
                        evidence_list.append(ev_set)

                sample = EvaluationSample(
                    claim_id=str(item.get("id", len(samples))),
                    claim_text=item.get("claim", ""),
                    gold_label=self._normalize_label(item.get("label", "NOT ENOUGH INFO")),
                    evidence=evidence_list,
                    metadata={
                        "dataset": "feverous",
                        "split": split,
                        "challenge": item.get("challenge", "")
                    }
                )
                samples.append(sample)

                if limit and len(samples) >= limit:
                    break

        if shuffle:
            random.seed(seed)
            random.shuffle(samples)

        return samples

    def load_scifact(
        self,
        split: str = "dev",
        limit: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 42
    ) -> List[EvaluationSample]:
        """
        Load SciFact-Open dataset.

        Args:
            split: "train", "dev", or "test"
            limit: Max samples to load
            shuffle: Shuffle samples
            seed: Random seed for shuffling

        Returns:
            List of standardized EvaluationSample
        """
        file_map = {
            "train": "claims_train.jsonl",
            "dev": "claims_dev.jsonl",
            "test": "claims_test.jsonl"
        }

        file_path = self.data_dir / "scifact-open" / file_map.get(split, file_map["dev"])

        if not file_path.exists():
            raise FileNotFoundError(f"SciFact dataset not found at {file_path}")

        # Load corpus for evidence retrieval
        corpus = {}
        corpus_path = self.data_dir / "scifact-open" / "corpus.jsonl"
        if corpus_path.exists():
            with open(corpus_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        doc = json.loads(line)
                        corpus[str(doc.get("doc_id", doc.get("_id", "")))] = doc

        samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)

                # Extract evidence from corpus
                evidence_list = []
                evidence_info = item.get("evidence", {})

                # Determine label from evidence
                label = "not_supported"  # default
                if evidence_info:
                    for doc_id, sentences in evidence_info.items():
                        doc = corpus.get(str(doc_id), {})
                        if isinstance(sentences, list):
                            for sent_info in sentences:
                                if isinstance(sent_info, dict):
                                    sent_label = sent_info.get("label", "")
                                    if sent_label == "SUPPORT":
                                        label = "supported"
                                    evidence_list.append({
                                        "doc_id": doc_id,
                                        "title": doc.get("title", ""),
                                        "sentences": sent_info.get("sentences", [])
                                    })

                sample = EvaluationSample(
                    claim_id=str(item.get("id", len(samples))),
                    claim_text=item.get("claim", ""),
                    gold_label=label,
                    evidence=evidence_list,
                    metadata={
                        "dataset": "scifact",
                        "split": split,
                        "cited_doc_ids": item.get("cited_doc_ids", [])
                    }
                )
                samples.append(sample)

                if limit and len(samples) >= limit:
                    break

        if shuffle:
            random.seed(seed)
            random.shuffle(samples)

        return samples

    def load_dataset(
        self,
        name: str,
        split: str = "dev",
        limit: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 42
    ) -> List[EvaluationSample]:
        """
        Load any supported dataset by name.

        Args:
            name: Dataset name ("hover", "feverous", "scifact")
            split: Data split to load
            limit: Max samples
            shuffle: Shuffle samples
            seed: Random seed

        Returns:
            List of standardized EvaluationSample
        """
        loaders = {
            "hover": self.load_hover,
            "feverous": self.load_feverous,
            "scifact": self.load_scifact,
            "scifact-open": self.load_scifact
        }

        loader = loaders.get(name.lower())
        if not loader:
            raise ValueError(f"Unknown dataset: {name}. Supported: {list(loaders.keys())}")

        return loader(split=split, limit=limit, shuffle=shuffle, seed=seed)

    def load_all_datasets(
        self,
        split: str = "dev",
        limit_per_dataset: Optional[int] = None
    ) -> Dict[str, List[EvaluationSample]]:
        """
        Load all available datasets.

        Args:
            split: Data split to load
            limit_per_dataset: Max samples per dataset

        Returns:
            Dictionary mapping dataset name to samples
        """
        results = {}
        for name in ["hover", "feverous", "scifact"]:
            try:
                results[name] = self.load_dataset(
                    name, split=split, limit=limit_per_dataset
                )
            except FileNotFoundError as e:
                print(f"Warning: Could not load {name}: {e}")

        return results

    def get_dataset_stats(self, samples: List[EvaluationSample]) -> Dict[str, Any]:
        """Get statistics for a list of samples"""
        label_counts = {}
        for s in samples:
            label_counts[s.gold_label] = label_counts.get(s.gold_label, 0) + 1

        return {
            "total_samples": len(samples),
            "label_distribution": label_counts,
            "supported_ratio": label_counts.get("supported", 0) / len(samples) if samples else 0
        }


def to_prediction_format(samples: List[EvaluationSample]) -> List[Dict[str, Any]]:
    """Convert samples to prediction input format"""
    return [
        {
            "claim_id": s.claim_id,
            "claim_text": s.claim_text,
            "gold_label": s.gold_label
        }
        for s in samples
    ]


# Convenience function
def load_evaluation_data(
    dataset: str,
    split: str = "dev",
    limit: Optional[int] = None,
    data_dir: str = "data/datasets"
) -> List[EvaluationSample]:
    """Convenience function to load evaluation data"""
    loader = DatasetLoader(data_dir)
    return loader.load_dataset(dataset, split=split, limit=limit)
