# SciFact-Open Dataset

## Description
Scientific Claim Verification (Open Domain)

## Dataset Statistics
- **Dev set**: 300 samples (41.3% supported, 58.7% not_supported)
- **Train set**: 809 samples
- **Test set**: 300 samples
- **Corpus**: 5,183 scientific abstracts
- **Source**: https://github.com/allenai/scifact

## Files
- `claims_dev.jsonl` - Development claims (0.06 MB)
- `claims_train.jsonl` - Training claims (0.17 MB)
- `claims_test.jsonl` - Test claims (0.03 MB)
- `corpus.jsonl` - Scientific abstracts corpus (7.92 MB)
- `cross_validation/` - Cross-validation folds

## Format

```json
{
  "id": "unique_id",
  "claim": "scientific claim text",
  "evidence": {"doc_id": [{"sentences": [...], "label": "SUPPORT|CONTRADICT"}]},
  "cited_doc_ids": [doc_ids]
}
```

## Label Mapping
- SUPPORT -> supported
- CONTRADICT -> not_supported
- NEI (No Evidence Info) -> not_supported

## Citation
```
@inproceedings{wadden2020scifact,
  title={Fact or Fiction: Verifying Scientific Claims},
  author={Wadden, David and others},
  booktitle={EMNLP},
  year={2020}
}
```
