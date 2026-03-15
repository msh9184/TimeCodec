# Forecasting

Time series forecasting using VQ-VAE discrete token representations. The approach first tokenizes time series into discrete codes via a VQ-VAE, then trains a Transformer-based decoder to predict future token sequences, which are decoded back to continuous values.

## Architecture

```
Training Phase (VQ-VAE):
  Raw Time Series --> RevIN --> Encoder --> Vector Quantizer (Codebook) --> Decoder --> Reconstructed Series
                                                |
                                          Discrete Codes

Forecasting Phase:
  Input Codes [t-T..t] --> Transformer Decoder --> Predicted Codes [t+1..t+H] --> VQ-VAE Decoder --> Forecast
```

**VQ-VAE**: Convolutional encoder-decoder with residual blocks. The encoder compresses input windows by a compression factor (default 4x), and the vector quantizer maps continuous embeddings to the nearest codebook entry.

**Forecaster**: Transformer decoder that operates on discrete token sequences. Takes the codebook indices from the input window and autoregressively predicts the codebook indices for the forecast horizon.

## Pipeline

The full pipeline has four sequential steps:

### Step 1: RevIN Preprocessing (`save_revin_data.py`)

Applies Reversible Instance Normalization (RevIN) to the raw CSV data and saves the normalized train/val/test splits as tensors.

```bash
python -u forecasting/save_revin_data.py \
  --data ETTh1 \
  --root_path $DATA_ROOT \
  --data_path ETTh1.csv \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --label_len 0 \
  --enc_in 7 \
  --gpu 0 \
  --save_path "forecasting/data/ETTh1"
```

### Step 2: VQ-VAE Training (`train_vqvae.py`)

Trains the VQ-VAE on the RevIN-normalized data to learn the discrete codebook.

```bash
python forecasting/train_vqvae.py \
  --config_path forecasting/scripts/ETTh1.json \
  --model_init_num_gpus 0 \
  --data_init_cpu_or_gpu cpu \
  --save_path "forecasting/saved_models/ETTh1/" \
  --base_path "forecasting/data" \
  --batchsize 4096
```

### Step 3: Code Extraction (`extract_forecasting_data.py`)

Uses the trained VQ-VAE to encode the dataset into discrete codes for each prediction horizon.

```bash
for pred_len in 96 192 336 720; do
  python -u forecasting/extract_forecasting_data.py \
    --data ETTh1 \
    --root_path $DATA_ROOT \
    --data_path ETTh1.csv \
    --features M \
    --seq_len 96 \
    --pred_len $pred_len \
    --label_len 0 \
    --enc_in 7 \
    --gpu 0 \
    --save_path "forecasting/data/ETTh1/Tin96_Tout${pred_len}/" \
    --trained_vqvae_model_path "forecasting/saved_models/ETTh1/checkpoints/final_model.pth" \
    --compression_factor 4 \
    --classifiy_or_forecast "forecast"
done
```

### Step 4: Transformer Forecaster Training (`train_forecaster.py`)

Trains the Transformer decoder on the extracted codes across multiple seeds and prediction horizons.

```bash
for seed in 2021 1 13; do
  for Tout in 96 192 336 720; do
    python forecasting/train_forecaster.py \
      --data-type ETTh1 \
      --Tin 96 \
      --Tout $Tout \
      --cuda-id 0 \
      --seed $seed \
      --data_path "forecasting/data/ETTh1/Tin96_Tout${Tout}" \
      --codebook_size 256 \
      --checkpoint \
      --checkpoint_path "forecasting/saved_models/ETTh1/forecaster_checkpoints/ETTh1_Tin96_Tout${Tout}_seed${seed}" \
      --file_save_path "forecasting/results/ETTh1/"
  done
done
```

## Datasets

| Dataset | Channels (`enc_in`) | CSV File | Data Type |
|---------|---------------------|----------|-----------|
| ETTh1 | 7 | ETTh1.csv | ETTh1 |
| ETTh2 | 7 | ETTh2.csv | ETTh2 |
| ETTm1 | 7 | ETTm1.csv | ETTm1 |
| ETTm2 | 7 | ETTm2.csv | ETTm2 |
| Weather | 21 | weather.csv | custom |
| Electricity | 321 | electricity.csv | custom |
| Traffic | 862 | traffic.csv | custom |

## Specialist vs Generalist

- **Specialist**: Trains a separate VQ-VAE per dataset. Run individual scripts (e.g., `scripts/ETTh1.sh`).
- **Generalist**: Trains a single VQ-VAE on all datasets combined, then extracts codes and trains a shared forecaster. Run `scripts/all.sh`. Requires running all individual preprocessing steps first.

## Running

Set `DATA_ROOT` to the directory containing the raw CSV files, then run the per-dataset script:

```bash
export DATA_ROOT=/path/to/forecasting/csv/data

# Specialist (single dataset)
bash forecasting/scripts/ETTh1.sh

# Generalist (all datasets)
bash forecasting/scripts/all.sh
```

Each script runs all four pipeline steps end-to-end.

## VQ-VAE Configuration Parameters

Configuration is stored in JSON files under `scripts/`. Example (`ETTh1.json`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 1e-3 | VQ-VAE learning rate |
| `num_training_updates` | 15000 | Number of training iterations |
| `block_hidden_size` | 128 | Hidden size in encoder/decoder blocks |
| `num_residual_layers` | 2 | Number of residual layers per block |
| `res_hidden_size` | 64 | Hidden size within residual layers |
| `embedding_dim` | 64 | Dimension of codebook embeddings |
| `num_embeddings` | 256 | Number of codebook entries |
| `commitment_cost` | 0.25 | VQ commitment loss weight |
| `compression_factor` | 4 | Temporal compression ratio |

## Forecaster Parameters

| Parameter | Description |
|-----------|-------------|
| `--Tin` | Input sequence length (default: 96) |
| `--Tout` | Prediction horizon (96, 192, 336, 720) |
| `--codebook_size` | Must match VQ-VAE `num_embeddings` |
| `--seed` | Random seed (experiments use 2021, 1, 13) |

## Expected Results

Results are saved to `forecasting/results/<dataset>/` as CSV files. Metrics include MSE and MAE averaged over three seeds (2021, 1, 13) for each prediction horizon.
