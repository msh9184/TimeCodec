# TimeCodec

Time series tokenization framework based on VQ-VAE, starting from the [TOTEM](https://arxiv.org/pdf/2402.16412.pdf) codebase. TimeCodec learns discrete codebook representations of time series data through self-supervised Vector Quantized Variational Autoencoders, enabling tokenized analysis across multiple downstream tasks.

## Repository Structure

```
TimeCodec/
├── forecasting/                # Time series forecasting task
│   ├── data_provider/          #   Data loading utilities
│   ├── lib/models/             #   VQ-VAE, decoder, RevIN, metrics
│   ├── lib/utils/              #   Checkpointing, environment, time features
│   ├── scripts/                #   Per-dataset configs (.json) and run scripts (.sh)
│   ├── save_revin_data.py      #   Step 1: RevIN preprocessing
│   ├── train_vqvae.py          #   Step 2: VQ-VAE training
│   ├── extract_forecasting_data.py  # Step 3: Code extraction
│   ├── train_forecaster.py     #   Step 4: Transformer forecaster training
│   └── generalist_eval.py      #   Generalist model evaluation
├── anomaly_detection/          # Anomaly detection task
│   ├── data_provider/          #   Data loading utilities
│   ├── layers/                 #   RevIN layer
│   ├── lib/models/             #   VQ-VAE and core models
│   ├── scripts/                #   Per-dataset configs and run scripts
│   ├── save_chunked_data.py    #   Step 1: Chunk raw data into windows
│   ├── revin_data.py           #   Step 2: RevIN normalization
│   ├── train_vqvae.py          #   Step 3: VQ-VAE training
│   └── detect_anomaly.py       #   Step 4: Anomaly scoring and evaluation
├── imputation/                 # Time series imputation task
│   ├── data_provider/          #   Data loading utilities
│   ├── layers/                 #   RevIN layer
│   ├── lib/models/             #   VQ-VAE and core models
│   ├── scripts/                #   Per-dataset configs and run scripts
│   ├── save_notrevin_notrevinmasked_revinx_revinxmasked.py  # Step 1: Preprocessing
│   ├── train_vqvae.py          #   Step 2: VQ-VAE training with masking
│   └── imputation_performance.py  # Step 3: Evaluation
├── process_zero_shot_data/     # Zero-shot data processing utilities
├── scripts/                    # Project-level helper scripts
│   └── download_data.sh        #   Data download helper
├── docs/                       # Documentation and notes
├── requirements.txt            # Python dependencies
└── CLAUDE.md                   # Development notes
```

## Quick Start

### 1. Environment Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10, PyTorch 2.1, and CUDA-capable GPU.

### 2. Data Download

Download the benchmark datasets from [Google Drive](https://drive.google.com/drive/u/0/folders/1gI36rS8irRZ32ibzKBPGncDmMXQtEf1C) and set the `DATA_ROOT` environment variable:

```bash
export DATA_ROOT=/path/to/your/downloaded/data

# Or use the helper script for guidance:
bash scripts/download_data.sh /path/to/your/data
```

See the [Data Setup](#data-setup) section for details.

### 3. Run Experiments

Each task has per-dataset shell scripts that run the full pipeline:

```bash
# Forecasting (specialist, single dataset)
bash forecasting/scripts/ETTh1.sh

# Anomaly Detection (specialist, single dataset)
bash anomaly_detection/scripts/msl.sh

# Imputation (specialist, single dataset)
bash imputation/scripts/ETTh1.sh

# Generalist (all datasets combined)
bash forecasting/scripts/all.sh
bash anomaly_detection/scripts/all.sh
bash imputation/scripts/all.sh
```

## Tasks

### Forecasting

Predicts future time series values by first learning a VQ-VAE codebook, then training a Transformer-based decoder on the discrete token sequences. Supports prediction horizons of 96, 192, 336, and 720 steps.

See [forecasting/README.md](forecasting/README.md) for details.

### Anomaly Detection

Detects anomalies by measuring VQ-VAE reconstruction error. Windows with high reconstruction loss are flagged as anomalous. Uses a threshold based on configurable anomaly ratio.

See [anomaly_detection/README.md](anomaly_detection/README.md) for details.

### Imputation

Reconstructs missing (masked) time series values using a VQ-VAE trained with random masking. Evaluates at mask ratios of 12.5%, 25%, 37.5%, and 50%.

See [imputation/README.md](imputation/README.md) for details.

## Data Setup

All scripts read raw data from a directory specified by the `DATA_ROOT` environment variable. Set it before running any experiment:

```bash
export DATA_ROOT=/path/to/your/data
```

The expected directory layout depends on the task:

- **Forecasting**: `DATA_ROOT/` should contain CSV files (ETTh1.csv, ETTh2.csv, ETTm1.csv, ETTm2.csv, weather.csv, electricity.csv, traffic.csv)
- **Anomaly Detection**: `DATA_ROOT/` should contain subdirectories per dataset (MSL/, PSM/, SMAP/, SMD/, SWaT/) with train/test NPY files and labels
- **Imputation**: Uses the same CSV files as forecasting

If `DATA_ROOT` is not set, each script falls back to a placeholder path and will fail with a file-not-found error.

## Requirements

- Python 3.10.11
- PyTorch 2.1.0
- pandas 2.0.3
- scikit-learn 1.3.0
- comet-ml 3.33.6
- xarray 2023.8.0

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Citation

This codebase is built upon TOTEM. If you use this code, please cite:

```bibtex
@article{
  talukder2024totem,
  title={{TOTEM}: {TO}kenized Time Series {EM}beddings for General Time Series Analysis},
  author={Sabera J Talukder and Yisong Yue and Georgia Gkioxari},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2024},
  url={https://openreview.net/forum?id=QlTLkH6xRC}
}
```

## License

This project is for research purposes. Please refer to the original TOTEM repository for licensing terms.
