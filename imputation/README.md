# Imputation

Masked time series reconstruction using VQ-VAE. The model is trained with random masking applied to input windows, forcing the VQ-VAE to learn robust representations that can reconstruct missing values. At evaluation time, the model imputes values at specified mask positions.

## Architecture

```
Training Phase:
  Raw Time Series --> Preprocess (RevIN + Masking variants) --> VQ-VAE Training with mask_ratio=0.5
                                                                  |
                                                               Codebook
                                                          (mask-robust patterns)

Evaluation Phase:
  Test Window --> Apply Mask (at target ratio) --> VQ-VAE Encoder --> Quantize --> Decoder --> Full Reconstruction
                     |                                                                            |
                     +--- Masked Positions ---- Compare (MSE, MAE) ------ Reconstructed Values ---+
```

The VQ-VAE is trained with 50% random masking, which teaches the codebook to represent time series patterns robustly even with missing data. At test time, the model is evaluated at multiple mask ratios (12.5%, 25%, 37.5%, 50%) to measure imputation quality.

## Pipeline

### Step 1: Preprocessing (`save_notrevin_notrevinmasked_revinx_revinxmasked.py`)

Generates four data variants: raw, raw-masked, RevIN-normalized, and RevIN-normalized-masked. These are used during training and evaluation.

```bash
python -u imputation/save_notrevin_notrevinmasked_revinx_revinxmasked.py \
  --data ETTh1 \
  --root_path $DATA_ROOT \
  --data_path ETTh1.csv \
  --features M \
  --seq_len 96 \
  --pred_len 0 \
  --label_len 0 \
  --enc_in 7 \
  --gpu 0 \
  --save_path "imputation/data/ETTh1"
```

### Step 2: VQ-VAE Training with Masking (`train_vqvae.py`)

Trains the VQ-VAE with `mask_ratio=0.5`. A single model trained at 50% masking is used for evaluation at all mask ratios.

```bash
for seed in 2021 13 1; do
  python imputation/train_vqvae.py \
    --config_path imputation/scripts/ETTh1.json \
    --model_init_num_gpus 0 \
    --data_init_cpu_or_gpu cpu \
    --save_path "imputation/saved_models/ETTh1/mask_ratio_0.5/" \
    --base_path "imputation/data" \
    --batchsize 8192 \
    --mask_ratio 0.5 \
    --revined_data 'False' \
    --seed $seed
done
```

### Step 3: Imputation Evaluation (`imputation_performance.py`)

Evaluates the trained model at different mask ratios, computing MSE and MAE on the masked positions only.

```bash
for mask_ratio_test in 0.125 0.25 0.375 0.5; do
  python imputation/imputation_performance.py \
    --dataset ETTh1 \
    --trained_vqvae_model_path "imputation/saved_models/ETTh1/<model_dir>/checkpoints/final_model.pth" \
    --compression_factor 4 \
    --gpu 0 \
    --base_path "imputation/data" \
    --mask_ratio $mask_ratio_test
done
```

Note: Replace `<model_dir>` with the actual directory name generated during training (e.g., `mask_ratio_0.5/CD64_CW512_CF4_BS8192_ITR15000_seed2021`).

## Datasets

| Dataset | Channels (`enc_in`) | CSV File | Data Type | Codebook Size |
|---------|---------------------|----------|-----------|---------------|
| ETTh1 | 7 | ETTh1.csv | ETTh1 | 512 |
| ETTh2 | 7 | ETTh2.csv | ETTh2 | 512 |
| ETTm1 | 7 | ETTm1.csv | ETTm1 | 512 |
| ETTm2 | 7 | ETTm2.csv | ETTm2 | 512 |
| Weather | 21 | weather.csv | custom | 512 |
| Electricity | 321 | electricity.csv | custom | 512 |

All datasets use `seq_len=96`, `compression_factor=4`, and are trained with `mask_ratio=0.5`.

## Mask Ratios

The model is evaluated at four mask ratios representing increasing amounts of missing data:

| Mask Ratio | Missing Data | Description |
|------------|-------------|-------------|
| 12.5% | ~12 of 96 timesteps | Light missingness |
| 25.0% | ~24 of 96 timesteps | Moderate missingness |
| 37.5% | ~36 of 96 timesteps | Heavy missingness |
| 50.0% | ~48 of 96 timesteps | Extreme missingness |

## Running

Set `DATA_ROOT` to the directory containing the raw CSV files (same files as forecasting), then run the per-dataset script:

```bash
export DATA_ROOT=/path/to/csv/data

# Individual datasets
bash imputation/scripts/ETTh1.sh
bash imputation/scripts/ETTh2.sh
bash imputation/scripts/ETTm1.sh
bash imputation/scripts/ETTm2.sh
bash imputation/scripts/weather.sh
bash imputation/scripts/electricity.sh

# Generalist (all datasets combined)
bash imputation/scripts/all.sh
```

Each script runs all three pipeline steps end-to-end.

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
| `num_embeddings` | 512 | Number of codebook entries |
| `commitment_cost` | 0.25 | VQ commitment loss weight |
| `compression_factor` | 4 | Temporal compression ratio |

## Evaluation Metrics

Imputation quality is measured on the masked positions only:

- **MSE** (Mean Squared Error): Average squared difference between imputed and true values at masked positions
- **MAE** (Mean Absolute Error): Average absolute difference between imputed and true values at masked positions

Results are reported separately for each mask ratio. Lower values indicate better imputation quality.
