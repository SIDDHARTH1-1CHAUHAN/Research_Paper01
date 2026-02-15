# HOVER Dataset

## Description
Multi-hop Fact Verification with Wikipedia (HOppy VERification)

## Dataset Statistics
- **Dev set**: 4,000 samples (50% supported, 50% not_supported)
- **Train set**: ~18,000 samples
- **Source**: https://github.com/hover-nlp/hover

## Files
- `hover_dev_release_v1.1.json` - Development set (2.05 MB)
- `hover_train_release_v1.1.json` - Training set (8.78 MB)

## Format

```json
{
  "uid": "unique_id",
  "claim": "claim text",
  "supporting_facts": [["article_title", sentence_idx], ...],
  "label": "SUPPORTED" | "NOT_SUPPORTED",
  "num_hops": 2 | 3 | 4
}
```

## Label Mapping
- SUPPORTED -> supported
- NOT_SUPPORTED -> not_supported

## Citation
```
@inproceedings{jiang2020hover,
  title={HoVer: A Dataset for Many-Hop Fact Extraction And Claim Verification},
  author={Jiang, Yichen and others},
  booktitle={Findings of EMNLP},
  year={2020}
}
```
