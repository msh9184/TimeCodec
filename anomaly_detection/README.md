# Anomaly Detection

Reconstruction-based anomaly detection using VQ-VAE. The approach trains a VQ-VAE to reconstruct normal time series windows. At test time, windows with high reconstruction error are flagged as anomalous, since the codebook has only learned to represent normal patterns.

## Architecture

```
Training Phase:
  Raw Time Series --> Chunk into Windows (seq_len=100) --> RevIN --> VQ-VAE Training
                                                                      |
                                                                   Codebook
                                                                (normal patterns)

Detection Phase:
  Test Window --> RevIN --> VQ-VAE Encoder --> Quantize --> Decoder --> Reconstruction
                    |                                                       |
                    +-------------- Reconstruction Error ------------------>+
                                         |
                                   Threshold (anomaly_ratio) --> Anomaly Label
```

The VQ-VAE learns to reconstruct normal patterns from the training set. At test time, anomalous segments produce higher reconstruction error because the codebook cannot represent unseen abnormal patterns.

## Pipeline

### Step 1: Chunk Raw Data (`save_chunked_data.py`)

Segments the raw time series into fixed-length windows of `seq_len=100` timesteps.

```bash
python anomaly_detection/save_chunked_data.py \
    --data 'MSL' \
    --batch_size 128 \
    --task_name 'anomaly_detection' \
    --root_path "${DATA_ROOT}" \
    --seq_len 100 \
    --save_path "anomaly_detection/data/MSL/" \
    --num_vars 55
```

### Step 2: RevIN Normalization (`revin_data.py`)

Applies Reversible Instance Normalization to the chunked data.

```bash
python anomaly_detection/revin_data.py \
    --root_path "anomaly_detection/data/MSL/" \
    --seq_len 100 \
    --save_path "anomaly_detection/data/MSL/revin_data/" \
    --num_vars 55
```

### Step 3: VQ-VAE Training (`train_vqvae.py`)

Trains the VQ-VAE on the normalized training windows across multiple seeds.

```bash
for seed in 47 1 13; do
  python anomaly_detection/train_vqvae.py \
      --config_path anomaly_detection/scripts/msl.json \
      --model_init_num_gpus 0 \
      --data_init_cpu_or_gpu cpu \
      --save_path "anomaly_detection/saved_models/MSL/" \
      --base_path "anomaly_detection/data/MSL/revin_data/" \
      --batchsize 4096 \
      --seed $seed
done
```

### Step 4: Anomaly Detection and Evaluation (`detect_anomaly.py`)

Computes reconstruction error on test data, applies threshold, and evaluates against ground truth labels.

```bash
python anomaly_detection/detect_anomaly.py \
    --dataset "MSL" \
    --trained_vqvae_model_path "anomaly_detection/saved_models/MSL/CD64_CW1024_CF4_BS4096_ITR15000_seed47/checkpoints/final_model.pth" \
    --compression_factor 4 \
    --base_path "anomaly_detection/data/MSL/revin_data" \
    --labels_path "anomaly_detection/data/MSL" \
    --anomaly_ratio 2 \
    --gpu 0 \
    --num_vars 55 \
    --seq_len 100
```

## Datasets

| Dataset | Sensors (`num_vars`) | Codebook Size (`num_embeddings`) | Training Iterations | Anomaly Ratio |
|---------|---------------------|----------------------------------|---------------------|---------------|
| MSL | 55 | 1024 | 15000 | 2 |
| PSM | 25 | 1024 | 60000 | 1 |
| SMAP | 25 | 1024 | 15000 | 1 |
| SMD | 38 | 1024 | 60000 | 0.5 |
| SWaT | 51 | 1024 | 15000 | 1 |

Notes:
- All datasets use `seq_len=100` and `compression_factor=4`.
- PSM and SMD require 60K training iterations (vs 15K for others).
- The anomaly ratio controls the detection threshold percentile. Lower values produce tighter thresholds (fewer but more confident detections).

## Evaluation Metrics

The detection script reports:
- **Accuracy**: Overall classification accuracy
- **Precision**: Fraction of detected anomalies that are true anomalies
- **Recall**: Fraction of true anomalies that are detected
- **F-score**: Harmonic mean of precision and recall

## Running

Set `DATA_ROOT` to the directory containing the raw dataset files, then run the per-dataset script:

```bash
export DATA_ROOT=/path/to/anomaly_detection/data

# Individual datasets
bash anomaly_detection/scripts/msl.sh
bash anomaly_detection/scripts/psm.sh
bash anomaly_detection/scripts/smap.sh
bash anomaly_detection/scripts/smd.sh
bash anomaly_detection/scripts/swat.sh

# Generalist (all datasets combined)
bash anomaly_detection/scripts/all.sh
```

Each script runs all four pipeline steps end-to-end.

## VQ-VAE Configuration Parameters

Configuration is stored in JSON files under `scripts/`. Example (`msl.json`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 1e-3 | VQ-VAE learning rate |
| `num_training_updates` | 15000 | Number of training iterations (60000 for PSM, SMD) |
| `block_hidden_size` | 128 | Hidden size in encoder/decoder blocks |
| `num_residual_layers` | 2 | Number of residual layers per block |
| `res_hidden_size` | 64 | Hidden size within residual layers |
| `embedding_dim` | 64 | Dimension of codebook embeddings |
| `num_embeddings` | 1024 | Number of codebook entries |
| `commitment_cost` | 0.25 | VQ commitment loss weight |
| `compression_factor` | 4 | Temporal compression ratio |

## Data Format

Each anomaly detection dataset should be placed in a subdirectory under `DATA_ROOT`:

```
DATA_ROOT/
├── MSL/
│   ├── MSL_train.npy
│   ├── MSL_test.npy
│   └── MSL_test_label.npy
├── PSM/
│   ├── train.csv
│   ├── test.csv
│   └── test_label.csv
├── SMAP/
│   ├── SMAP_train.npy
│   ├── SMAP_test.npy
│   └── SMAP_test_label.npy
├── SMD/
│   ├── SMD_train.npy
│   ├── SMD_test.npy
│   └── SMD_test_label.npy
└── SWaT/
    ├── swat_train2.csv
    └── swat2.csv
```
