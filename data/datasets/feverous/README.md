# FEVEROUS Dataset

## Description
Fact Extraction and VERification Over Unstructured and Structured information

## Dataset Statistics
- **Dev set**: 7,891 samples (49.5% supported, 50.5% not_supported)
- **Train set**: 87,026 samples
- **Source**: https://fever.ai/dataset/feverous.html

## Files
- `feverous_dev.jsonl` - Development set (17.0 MB)
- `feverous_train.jsonl` - Training set (167.4 MB)

## Format

```json
{
  "id": "unique_id",
  "claim": "claim text",
  "evidence": [[{"content": "...", "context": "..."}]],
  "label": "SUPPORTS" | "REFUTES" | "NOT ENOUGH INFO"
}
```

## Label Mapping
- SUPPORTS -> supported
- REFUTES -> not_supported
- NOT ENOUGH INFO -> not_supported

## Citation
```
@inproceedings{aly2021feverous,
  title={FEVEROUS: Fact Extraction and VERification Over Unstructured and Structured information},
  author={Aly, Rami and others},
  booktitle={NeurIPS},
  year={2021}
}
```
