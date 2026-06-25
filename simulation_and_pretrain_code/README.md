# Simulation and Pretrain Code

This folder contains the original research code used for **RNA masked language model pretraining**, **segment-based simulation**, and **offline feature extraction** (BigBird / Longformer).

It is not required for end-user prediction. For classification from FASTA, use the scripts in the repository root:

- `download_weights.py`
- `extract_nucleotide_embeddings.py`
- `predict_lncrna.py`

## Main scripts

| Script | Purpose |
| --- | --- |
| `pretrain.py` | RNA MLM pretraining (Longformer / BigBird) |
| `run.py` | Fine-tune pretrained models for classification |
| `feature.py` / `multi_feature.py` | Extract nucleotide embeddings from CSV/txt datasets |
| `pre_dataset.py` / `dataset.py` | Dataset and vocabulary builders |
| `split_sequence.py` | Split long sequences into training segments |
| `trainer.py` | Training loop utilities |
| `test.py` | Evaluation on held-out data |
| `model/` | Custom BigBird / Longformer / BERT model definitions |

## Weights

After running `python download_weights.py` from the repo root, RNA pretrain checkpoints are saved to:

```
simulation_and_pretrain_code/weights/pretrain/
```

## Example (pretraining)

Run from the repository root:

```bash
cd simulation_and_pretrain_code
python pretrain.py --model Longformer --writing_params_path weights/pretrain/Longformer.pretrain
```
