# GAR-Syn

Clean PyTorch implementation of **GAR-Syn**, a gated attention and residual aggregation model for anti-cancer drug synergy prediction.

This repository contains only the GAR-Syn model and the scripts needed for training, evaluation. Large datasets, checkpoints, logs, and generated result files are intentionally excluded.

## Model

GAR-Syn represents each drug-pair and cell-line sample as 10 modality tokens:

```text
[SYN]
Drug A: fingerprint, sequence, molecular graph
Drug B: fingerprint, sequence, molecular graph
Cell line: expression, mutation, CNV
```

The model includes:

- interaction-type-aware gated attention
- token-level modality gating
- depth-aware residual aggregation
- permutation-consistent drug-pair prediction
- gate regularization

## Default Configuration

The default configuration in `configs/garsyn_default.json` uses the sensitivity-analysis-selected setting:

```text
hidden_size = 512
dropout = 0.1
num_gar_layers = 5
lambda_perm = 0.1
batch_size = 1024
```

Other architecture and feature dimensions follow the original GAR-Syn setting.

## Installation

Create an environment with PyTorch and PyTorch Geometric that matches your CUDA version, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

If `torch-geometric` installation fails, install it using the official wheel command for your PyTorch/CUDA version.

## Data

Prepare a raw data directory containing the following files:

```text
Drug_map.npy
drug_feature_graph.npy
nv_zscore.csv
mutation.csv
Drug_use.csv
Cell_use_zscore.csv
drug_sequence_em.csv
data_to_split.csv
```

`data_to_split.csv` must include the identifier columns:

```text
drug_row, drug_col, depmap
```

and the label columns:

```text
S_mean, synergy_zip, synergy_loewe, synergy_hsa, synergy_bliss
```

## Quick Start

Run 5-fold random-split training on `S_mean`:

```bash
SOURCE_RAW_DIR=/path/to/rawData/drugcomb bash run_garsyn_random.sh
```

Common overrides:

```bash
GPU=1 FOLDS=1,2 LABELS=S_mean BATCH_SIZE=1024 \
SOURCE_RAW_DIR=/path/to/rawData/drugcomb \
bash run_garsyn_random.sh
```

## Manual Commands

Prepare random splits:

```bash
python run.py prepare \
  --dataset drugcomb \
  --split-mode random \
  --source-raw-dir /path/to/rawData/drugcomb \
  --output-dir ./data_random \
  --folds 1,2,3,4,5
```

Train:

```bash
python run.py train \
  --config ./configs/garsyn_default.json \
  --labels S_mean \
  --folds 1,2,3,4,5 \
  --data-dir ./data_random \
  --results-dir ./results_random \
  --gpu 0 \
  --gpu-cache all \
  --graph-cache-mode full_batch \
  --batch-size 1024
```

Evaluate:

```bash
python run.py evaluate \
  --results-dir ./results_random \
  --labels S_mean \
  --output ./results_random/summary_all.csv
```

## Outputs

Training and evaluation generate:

```text
results_random/GAR-Syn/checkpoints/repeatX_best.pth
results_random/GAR-Syn/predict/repeatX_predict.csv
results_random/GAR-Syn/metrics.csv
results_random/summary_all.csv
results_random/summary_all_mean_std.csv
```