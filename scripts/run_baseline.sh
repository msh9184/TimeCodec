#!/bin/bash
# ==========================================================================
# TimeCodec Baseline Reproduction Runner
# ==========================================================================
# Usage:
#   bash scripts/run_baseline.sh --task forecasting --dataset ETTh1
#   bash scripts/run_baseline.sh --task forecasting --dataset all
#   bash scripts/run_baseline.sh --task anomaly_detection --dataset SMD
#   bash scripts/run_baseline.sh --task imputation --dataset ETTh1
#
# Options:
#   --task          Task name: forecasting, anomaly_detection, imputation
#   --dataset       Dataset name (or "all" for all datasets in that task)
#   --gpu           GPU device ID (default: 0)
#   --data-root     Root path to raw data (default: env DATA_ROOT)
#   --pretrained    Path to pretrained TOTEM tokenizer (optional, for comparison)
#   --skip-train    Skip VQ-VAE training, use existing model
#   --help          Show this help
#
# Output:
#   results/<task>/<dataset>/
#     ├── logs/              # Step-by-step execution logs
#     ├── report.txt         # Final summary report
#     └── metrics.csv        # Machine-readable metrics
# ==========================================================================

set -euo pipefail

# --------------------------------------------------
# Defaults
# --------------------------------------------------
TASK=""
DATASET=""
GPU=0
DATA_ROOT="${DATA_ROOT:-/group-volume/workspace/sunghwan.mun/ts-dataset/TOTEM/raw}"
PRETRAINED_ROOT="/group-volume/workspace/sunghwan.mun/ts-dataset/TOTEM/generatlist_pretrained_tokenizers"
SKIP_TRAIN=false
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# --------------------------------------------------
# Parse arguments
# --------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --task) TASK="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --pretrained) PRETRAINED_ROOT="$2"; shift 2 ;;
        --skip-train) SKIP_TRAIN=true; shift ;;
        --help)
            head -20 "$0" | grep -E "^#" | sed 's/^# //'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$TASK" || -z "$DATASET" ]]; then
    echo "Error: --task and --dataset are required"
    echo "Usage: bash scripts/run_baseline.sh --task forecasting --dataset ETTh1"
    exit 1
fi

cd "$SCRIPT_DIR"

# --------------------------------------------------
# Dataset registry
# --------------------------------------------------
declare -A FC_DATASETS=(
    [ETTh1]="7:ETTh1:ETTh1.csv"
    [ETTh2]="7:ETTh2:ETTh2.csv"
    [ETTm1]="7:ETTm1:ETTm1.csv"
    [ETTm2]="7:ETTm2:ETTm2.csv"
    [weather]="21:custom:weather.csv"
    [electricity]="321:custom:electricity.csv"
    [traffic]="862:custom:traffic.csv"
)

declare -A AD_DATASETS=(
    [MSL]="55:100:15000:2.0"
    [PSM]="25:100:30000:1.0"
    [SMAP]="25:100:30000:1.0"
    [SMD]="38:100:60000:0.5"
    [SWaT]="51:100:60000:1.0"
)

declare -A IM_DATASETS=(
    [ETTh1]="7:ETTh1:ETTh1.csv"
    [ETTh2]="7:ETTh2:ETTh2.csv"
    [ETTm1]="7:ETTm1:ETTm1.csv"
    [ETTm2]="7:ETTm2:ETTm2.csv"
    [weather]="21:custom:weather.csv"
    [electricity]="321:custom:electricity.csv"
)

# --------------------------------------------------
# Logging setup
# --------------------------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="results/${TASK}/${DATASET}/${TIMESTAMP}"
LOG_DIR="${RESULT_DIR}/logs"
mkdir -p "$LOG_DIR"

log() {
    local msg="[$(date '+%H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "${LOG_DIR}/master.log"
}

run_step() {
    local step_name="$1"
    local log_file="${LOG_DIR}/${step_name}.log"
    shift
    log "START: ${step_name}"
    local start_time=$(date +%s)
    if "$@" > "$log_file" 2>&1; then
        local end_time=$(date +%s)
        local elapsed=$((end_time - start_time))
        log "DONE:  ${step_name} (${elapsed}s)"
        return 0
    else
        local exit_code=$?
        local end_time=$(date +%s)
        local elapsed=$((end_time - start_time))
        log "FAIL:  ${step_name} (${elapsed}s, exit=${exit_code})"
        log "       See: ${log_file}"
        return $exit_code
    fi
}

# --------------------------------------------------
# Report generation
# --------------------------------------------------
generate_report() {
    local report_file="${RESULT_DIR}/report.txt"
    {
        echo "================================================================"
        echo "  TimeCodec Baseline Report"
        echo "  Task: ${TASK} | Dataset: ${DATASET}"
        echo "  Date: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "  GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'N/A')"
        echo "================================================================"
        echo ""

        if [[ "$TASK" == "forecasting" ]]; then
            echo "--- Scratch-Trained Specialist Results ---"
            echo ""
            printf "%-10s %-8s %-10s %-10s %-10s\n" "Dataset" "Horizon" "MSE" "MAE" "Corr"
            printf "%-10s %-8s %-10s %-10s %-10s\n" "-------" "-------" "--------" "--------" "--------"
            for f in $(ls forecasting/results/${DATASET}/*.txt 2>/dev/null); do
                if [[ -f "$f" ]]; then
                    cat "$f"
                fi
            done
            echo ""

            # Pretrained comparison if available
            if ls forecasting/results/${DATASET}_pretrained/*.txt >/dev/null 2>&1; then
                echo ""
                echo "--- Pretrained Generalist Results ---"
                echo ""
                printf "%-10s %-8s %-10s %-10s %-10s\n" "Dataset" "Horizon" "MSE" "MAE" "Corr"
                printf "%-10s %-8s %-10s %-10s %-10s\n" "-------" "-------" "--------" "--------" "--------"
                for f in $(ls forecasting/results/${DATASET}_pretrained/*.txt 2>/dev/null); do
                    if [[ -f "$f" ]]; then
                        cat "$f"
                    fi
                done
            fi

            echo ""
            echo "--- TOTEM Paper Reference (Specialist) ---"
            echo "(From Table 3 of arXiv:2402.16412)"
            echo "Note: Exact values may vary; check paper for precise numbers."

        elif [[ "$TASK" == "anomaly_detection" ]]; then
            echo "--- Anomaly Detection Results ---"
            echo ""
            if [[ -f "${LOG_DIR}/step4_detect.log" ]]; then
                grep -E "Accuracy|Precision|Recall|F-score|f_score" "${LOG_DIR}/step4_detect.log" || echo "(No metrics found)"
            fi

        elif [[ "$TASK" == "imputation" ]]; then
            echo "--- Imputation Results ---"
            echo ""
            printf "%-10s %-12s %-10s %-10s\n" "Dataset" "Mask_Ratio" "MSE" "MAE"
            printf "%-10s %-12s %-10s %-10s\n" "-------" "----------" "--------" "--------"
            for mr in 0.125 0.25 0.375 0.5; do
                if [[ -f "${LOG_DIR}/step3_eval_mr${mr}.log" ]]; then
                    local mse=$(grep -i "mse" "${LOG_DIR}/step3_eval_mr${mr}.log" | tail -1 || echo "N/A")
                    local mae=$(grep -i "mae" "${LOG_DIR}/step3_eval_mr${mr}.log" | tail -1 || echo "N/A")
                    printf "%-10s %-12s %-10s %-10s\n" "$DATASET" "$mr" "$mse" "$mae"
                fi
            done
        fi

        echo ""
        echo "================================================================"
        echo "  Execution Summary"
        echo "================================================================"
        grep -E "START|DONE|FAIL" "${LOG_DIR}/master.log" 2>/dev/null || true
        echo ""
        echo "Full logs: ${LOG_DIR}/"
        echo "================================================================"
    } > "$report_file"

    echo ""
    echo "================================================================"
    cat "$report_file"
    echo ""
    log "Report saved: ${report_file}"
}

# ==========================================================================
# FORECASTING PIPELINE
# ==========================================================================
run_forecasting() {
    local ds="$1"
    local info="${FC_DATASETS[$ds]}"
    local enc_in=$(echo "$info" | cut -d: -f1)
    local data_name=$(echo "$info" | cut -d: -f2)
    local data_file=$(echo "$info" | cut -d: -f3)
    local fc_data_root="${DATA_ROOT}/forecasting"

    log "=== Forecasting: ${ds} (enc_in=${enc_in}) ==="

    # Step 1: RevIN preprocessing
    run_step "step1_preprocess" \
        python -u forecasting/save_revin_data.py \
            --random_seed 2021 \
            --data "$data_name" \
            --root_path "$fc_data_root" \
            --data_path "$data_file" \
            --features M \
            --seq_len 96 --pred_len 96 --label_len 0 \
            --enc_in "$enc_in" \
            --gpu "$GPU" \
            --save_path "forecasting/data/${ds}"

    # Step 2: VQ-VAE training (scratch)
    if [[ "$SKIP_TRAIN" == false ]]; then
        # Determine JSON config (use dataset-specific if exists, else ETTh1 as template)
        local config_file="forecasting/scripts/${ds}.json"
        if [[ ! -f "$config_file" ]]; then
            config_file="forecasting/scripts/ETTh1.json"
            log "WARNING: No config for ${ds}, using ETTh1.json"
        fi

        run_step "step2_train_vqvae" \
            python forecasting/train_vqvae.py \
                --config_path "$config_file" \
                --model_init_num_gpus "$GPU" \
                --data_init_cpu_or_gpu cpu \
                --save_path "forecasting/saved_models/${ds}/" \
                --base_path "forecasting/data" \
                --batchsize 4096
    fi

    # Find the trained model path
    local vqvae_dir=$(ls -d forecasting/saved_models/${ds}/CD* 2>/dev/null | head -1)
    if [[ -z "$vqvae_dir" ]]; then
        log "ERROR: No trained VQ-VAE found in forecasting/saved_models/${ds}/"
        return 1
    fi
    local vqvae_path="${vqvae_dir}/checkpoints/final_model.pth"
    log "Using VQ-VAE: ${vqvae_path}"

    # Step 3: Extract codes for all prediction horizons
    for pred_len in 96 192 336 720; do
        run_step "step3_extract_Tout${pred_len}" \
            python -u forecasting/extract_forecasting_data.py \
                --random_seed 2021 \
                --data "$data_name" \
                --root_path "$fc_data_root" \
                --data_path "$data_file" \
                --features M \
                --seq_len 96 --pred_len "$pred_len" --label_len 0 \
                --enc_in "$enc_in" \
                --gpu "$GPU" \
                --save_path "forecasting/data/${ds}/Tin96_Tout${pred_len}/" \
                --trained_vqvae_model_path "$vqvae_path" \
                --compression_factor 4 \
                --classifiy_or_forecast "forecast"
    done

    # Step 4: Train forecaster (3 seeds x 4 horizons)
    mkdir -p "forecasting/results/${ds}"
    for seed in 2021 1 13; do
        for Tout in 96 192 336 720; do
            run_step "step4_forecaster_Tout${Tout}_seed${seed}" \
                python forecasting/train_forecaster.py \
                    --data-type "$ds" \
                    --Tin 96 --Tout "$Tout" \
                    --cuda-id "$GPU" \
                    --seed "$seed" \
                    --data_path "forecasting/data/${ds}/Tin96_Tout${Tout}" \
                    --codebook_size 256 \
                    --checkpoint \
                    --checkpoint_path "forecasting/saved_models/${ds}/forecaster_checkpoints/${ds}_Tin96_Tout${Tout}_seed${seed}" \
                    --file_save_path "forecasting/results/${ds}/"
        done
    done

    # Step 5: Pretrained comparison (if available)
    local pretrained_vqvae="${PRETRAINED_ROOT}/forecasting/CD64_CW256_CF4_BS4096_ITR120000/checkpoints/final_model.pth"
    if [[ -f "$pretrained_vqvae" ]]; then
        log "=== Pretrained Generalist Comparison ==="
        for pred_len in 96 192 336 720; do
            run_step "step5_pretrained_extract_Tout${pred_len}" \
                python -u forecasting/extract_forecasting_data.py \
                    --random_seed 2021 \
                    --data "$data_name" \
                    --root_path "$fc_data_root" \
                    --data_path "$data_file" \
                    --features M \
                    --seq_len 96 --pred_len "$pred_len" --label_len 0 \
                    --enc_in "$enc_in" \
                    --gpu "$GPU" \
                    --save_path "forecasting/data/${ds}_pretrained/Tin96_Tout${pred_len}/" \
                    --trained_vqvae_model_path "$pretrained_vqvae" \
                    --compression_factor 4 \
                    --classifiy_or_forecast "forecast"
        done

        mkdir -p "forecasting/results/${ds}_pretrained"
        for seed in 2021; do
            for Tout in 96 192 336 720; do
                run_step "step5_pretrained_forecaster_Tout${Tout}_seed${seed}" \
                    python forecasting/train_forecaster.py \
                        --data-type "$ds" \
                        --Tin 96 --Tout "$Tout" \
                        --cuda-id "$GPU" \
                        --seed "$seed" \
                        --data_path "forecasting/data/${ds}_pretrained/Tin96_Tout${Tout}" \
                        --codebook_size 256 \
                        --checkpoint \
                        --checkpoint_path "forecasting/saved_models/${ds}_pretrained/forecaster_checkpoints/${ds}_Tin96_Tout${Tout}_seed${seed}" \
                        --file_save_path "forecasting/results/${ds}_pretrained/"
            done
        done
    else
        log "Pretrained model not found at ${pretrained_vqvae}, skipping comparison"
    fi
}

# ==========================================================================
# ANOMALY DETECTION PIPELINE
# ==========================================================================
run_anomaly_detection() {
    local ds="$1"
    local info="${AD_DATASETS[$ds]}"
    local num_vars=$(echo "$info" | cut -d: -f1)
    local seq_len=$(echo "$info" | cut -d: -f2)
    local num_itr=$(echo "$info" | cut -d: -f3)
    local anomaly_ratio=$(echo "$info" | cut -d: -f4)
    local ad_data_root="${DATA_ROOT}/anomaly_detection/${ds}"

    log "=== Anomaly Detection: ${ds} (vars=${num_vars}, seq=${seq_len}) ==="

    # Step 1: Chunk data
    run_step "step1_chunk" \
        python anomaly_detection/save_chunked_data.py \
            --data "$ds" \
            --root_path "$ad_data_root" \
            --save_path "anomaly_detection/data/${ds}/" \
            --seq_len "$seq_len" \
            --num_vars "$num_vars" \
            --batch_size 256 \
            --task_name anomaly_detection \
            --gpu "$GPU"

    # Step 2: RevIN normalization
    run_step "step2_revin" \
        python anomaly_detection/revin_data.py \
            --root_path "anomaly_detection/data/${ds}/" \
            --save_path "anomaly_detection/data/${ds}/revin_data/" \
            --seq_len "$seq_len" \
            --num_vars "$num_vars" \
            --gpu "$GPU"

    # Step 3: Train VQ-VAE
    if [[ "$SKIP_TRAIN" == false ]]; then
        local config_file="anomaly_detection/scripts/${ds,,}.json"
        if [[ ! -f "$config_file" ]]; then
            config_file="anomaly_detection/scripts/smd.json"
            log "WARNING: No config for ${ds}, using smd.json"
        fi

        for seed in 47; do
            run_step "step3_train_vqvae_seed${seed}" \
                python anomaly_detection/train_vqvae.py \
                    --config_path "$config_file" \
                    --model_init_num_gpus "$GPU" \
                    --data_init_cpu_or_gpu cpu \
                    --save_path "anomaly_detection/saved_models/${ds}/" \
                    --base_path "anomaly_detection/data/${ds}/revin_data/" \
                    --batchsize 4096 \
                    --seed "$seed"
        done
    fi

    # Find trained model
    local vqvae_dir=$(ls -d anomaly_detection/saved_models/${ds}/CD* 2>/dev/null | head -1)
    if [[ -z "$vqvae_dir" ]]; then
        log "ERROR: No trained VQ-VAE found"
        return 1
    fi
    local vqvae_path="${vqvae_dir}/checkpoints/final_model.pth"

    # Step 4: Detect anomalies
    run_step "step4_detect" \
        python anomaly_detection/detect_anomaly.py \
            --dataset "$ds" \
            --trained_vqvae_model_path "$vqvae_path" \
            --compression_factor 4 \
            --base_path "anomaly_detection/data/${ds}/revin_data/" \
            --labels_path "anomaly_detection/data/${ds}/" \
            --num_vars "$num_vars" \
            --seq_len "$seq_len" \
            --anomaly_ratio "$anomaly_ratio" \
            --gpu "$GPU"
}

# ==========================================================================
# IMPUTATION PIPELINE
# ==========================================================================
run_imputation() {
    local ds="$1"
    local info="${IM_DATASETS[$ds]}"
    local enc_in=$(echo "$info" | cut -d: -f1)
    local data_name=$(echo "$info" | cut -d: -f2)
    local data_file=$(echo "$info" | cut -d: -f3)
    local im_data_root="${DATA_ROOT}/imputation"

    log "=== Imputation: ${ds} (enc_in=${enc_in}) ==="

    # Step 1: Preprocess
    run_step "step1_preprocess" \
        python -u imputation/save_notrevin_notrevinmasked_revinx_revinxmasked.py \
            --random_seed 2021 \
            --data "$data_name" \
            --root_path "$im_data_root" \
            --data_path "$data_file" \
            --features M \
            --seq_len 96 --pred_len 0 --label_len 0 \
            --enc_in "$enc_in" \
            --gpu "$GPU" \
            --save_path "imputation/data/${ds}"

    # Step 2: Train VQ-VAE with masking
    if [[ "$SKIP_TRAIN" == false ]]; then
        local config_file="imputation/scripts/${ds}.json"
        if [[ ! -f "$config_file" ]]; then
            config_file="imputation/scripts/ETTh1.json"
            log "WARNING: No config for ${ds}, using ETTh1.json"
        fi

        run_step "step2_train_vqvae" \
            python imputation/train_vqvae.py \
                --config_path "$config_file" \
                --model_init_num_gpus "$GPU" \
                --data_init_cpu_or_gpu cpu \
                --save_path "imputation/saved_models/${ds}/" \
                --base_path "imputation/data" \
                --batchsize 8192 \
                --mask_ratio 0.5 \
                --revined_data False \
                --seed 2021
    fi

    # Find trained model
    local vqvae_dir=$(ls -d imputation/saved_models/${ds}/CD* 2>/dev/null | head -1)
    if [[ -z "$vqvae_dir" ]]; then
        log "ERROR: No trained VQ-VAE found"
        return 1
    fi
    local vqvae_path="${vqvae_dir}/checkpoints/final_model.pth"

    # Step 3: Evaluate at different mask ratios
    for mr in 0.125 0.25 0.375 0.5; do
        run_step "step3_eval_mr${mr}" \
            python imputation/imputation_performance.py \
                --dataset "$ds" \
                --trained_vqvae_model_path "$vqvae_path" \
                --compression_factor 4 \
                --gpu "$GPU" \
                --base_path "imputation/data" \
                --mask_ratio "$mr"
    done
}

# ==========================================================================
# Main execution
# ==========================================================================
log "================================================================"
log "TimeCodec Baseline Runner"
log "Task: ${TASK} | Dataset: ${DATASET} | GPU: ${GPU}"
log "Data root: ${DATA_ROOT}"
log "Working dir: ${SCRIPT_DIR}"
log "Results dir: ${RESULT_DIR}"
log "================================================================"

# Expand "all" to dataset list
if [[ "$DATASET" == "all" ]]; then
    case "$TASK" in
        forecasting)
            DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 weather electricity traffic) ;;
        anomaly_detection)
            DATASETS=(SMD MSL PSM SMAP SWaT) ;;
        imputation)
            DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 weather electricity) ;;
        *)
            echo "Unknown task: $TASK"; exit 1 ;;
    esac
else
    DATASETS=("$DATASET")
fi

# Run pipeline for each dataset
OVERALL_START=$(date +%s)
FAILED=()

for ds in "${DATASETS[@]}"; do
    log ""
    log "======== Processing: ${ds} ========"
    DATASET="$ds"  # Update for report
    RESULT_DIR="results/${TASK}/${ds}/${TIMESTAMP}"
    LOG_DIR="${RESULT_DIR}/logs"
    mkdir -p "$LOG_DIR"

    case "$TASK" in
        forecasting)
            if ! run_forecasting "$ds"; then
                FAILED+=("$ds")
                log "FAILED: ${ds} — continuing with next dataset"
            fi
            ;;
        anomaly_detection)
            if ! run_anomaly_detection "$ds"; then
                FAILED+=("$ds")
                log "FAILED: ${ds} — continuing with next dataset"
            fi
            ;;
        imputation)
            if ! run_imputation "$ds"; then
                FAILED+=("$ds")
                log "FAILED: ${ds} — continuing with next dataset"
            fi
            ;;
        *)
            echo "Unknown task: $TASK"
            exit 1
            ;;
    esac

    generate_report
done

OVERALL_END=$(date +%s)
OVERALL_ELAPSED=$((OVERALL_END - OVERALL_START))

echo ""
echo "================================================================"
echo "  ALL DONE"
echo "  Total time: ${OVERALL_ELAPSED}s ($((OVERALL_ELAPSED/60))m $((OVERALL_ELAPSED%60))s)"
echo "  Failed: ${#FAILED[@]} (${FAILED[*]:-none})"
echo "  Results: results/${TASK}/*/${TIMESTAMP}/"
echo "================================================================"
