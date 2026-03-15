#!/bin/bash
# =============================================================================
# TimeCodec Data Download Helper
# =============================================================================
#
# Downloads and organizes benchmark datasets for the TimeCodec project.
#
# Usage:
#   bash scripts/download_data.sh [DATA_ROOT]
#
# Arguments:
#   DATA_ROOT   Target directory for datasets (optional)
#               Default: /group-volume/workspace/sunghwan.mun/ts-dataset/TOTEM
#
# The datasets must be downloaded manually from Google Drive:
#   https://drive.google.com/drive/u/0/folders/1gI36rS8irRZ32ibzKBPGncDmMXQtEf1C
#
# =============================================================================

set -euo pipefail

DATA_ROOT=${1:-/group-volume/workspace/sunghwan.mun/ts-dataset/TOTEM}

# -----------------------------------------------------------------------------
# Color output helpers
# -----------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC}   $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# -----------------------------------------------------------------------------
# Create directory structure
# -----------------------------------------------------------------------------
create_directories() {
    info "Creating directory structure under: ${DATA_ROOT}"

    mkdir -p "${DATA_ROOT}/raw/forecasting"
    mkdir -p "${DATA_ROOT}/raw/anomaly_detection/MSL"
    mkdir -p "${DATA_ROOT}/raw/anomaly_detection/PSM"
    mkdir -p "${DATA_ROOT}/raw/anomaly_detection/SMAP"
    mkdir -p "${DATA_ROOT}/raw/anomaly_detection/SMD"
    mkdir -p "${DATA_ROOT}/raw/anomaly_detection/SWaT"

    # Imputation uses the same CSV files as forecasting
    if [ ! -e "${DATA_ROOT}/raw/imputation" ]; then
        ln -sf "${DATA_ROOT}/raw/forecasting" "${DATA_ROOT}/raw/imputation"
        info "Created symlink: raw/imputation -> raw/forecasting"
    else
        info "raw/imputation already exists, skipping symlink"
    fi

    success "Directory structure created"
    echo ""
}

# -----------------------------------------------------------------------------
# Print download instructions
# -----------------------------------------------------------------------------
print_download_instructions() {
    echo "============================================================================="
    echo " DATASET DOWNLOAD INSTRUCTIONS"
    echo "============================================================================="
    echo ""
    echo " All datasets are available on Google Drive:"
    echo ""
    echo "   https://drive.google.com/drive/u/0/folders/1gI36rS8irRZ32ibzKBPGncDmMXQtEf1C"
    echo ""
    echo " Download the files and place them in the directories listed below."
    echo " If you are behind a proxy, gdown may not work; use a browser instead."
    echo ""
    echo "-----------------------------------------------------------------------------"
    echo " Forecasting / Imputation (CSV files)"
    echo "   Target: ${DATA_ROOT}/raw/forecasting/"
    echo "-----------------------------------------------------------------------------"
    echo "   - ETTh1.csv"
    echo "   - ETTh2.csv"
    echo "   - ETTm1.csv"
    echo "   - ETTm2.csv"
    echo "   - weather.csv"
    echo "   - electricity.csv"
    echo "   - traffic.csv"
    echo ""
    echo "-----------------------------------------------------------------------------"
    echo " Anomaly Detection"
    echo "   Target: ${DATA_ROOT}/raw/anomaly_detection/<dataset>/"
    echo "-----------------------------------------------------------------------------"
    echo ""
    echo "   MSL/ (55 sensors)"
    echo "     - MSL_train.npy"
    echo "     - MSL_test.npy"
    echo "     - MSL_test_label.npy"
    echo ""
    echo "   PSM/ (25 sensors)"
    echo "     - train.csv"
    echo "     - test.csv"
    echo "     - test_label.csv"
    echo ""
    echo "   SMAP/ (25 sensors)"
    echo "     - SMAP_train.npy"
    echo "     - SMAP_test.npy"
    echo "     - SMAP_test_label.npy"
    echo ""
    echo "   SMD/ (38 sensors)"
    echo "     - SMD_train.npy"
    echo "     - SMD_test.npy"
    echo "     - SMD_test_label.npy"
    echo ""
    echo "   SWaT/ (51 sensors)"
    echo "     - swat_train2.csv"
    echo "     - swat2.csv"
    echo ""
    echo "============================================================================="
    echo ""
}

# -----------------------------------------------------------------------------
# Verify downloaded files
# -----------------------------------------------------------------------------
verify_files() {
    info "Verifying downloaded files..."
    echo ""

    local all_ok=true

    # Forecasting CSV files
    local forecasting_files=(
        "ETTh1.csv"
        "ETTh2.csv"
        "ETTm1.csv"
        "ETTm2.csv"
        "weather.csv"
        "electricity.csv"
        "traffic.csv"
    )

    echo "  Forecasting / Imputation:"
    for f in "${forecasting_files[@]}"; do
        if [ -f "${DATA_ROOT}/raw/forecasting/${f}" ]; then
            success "  ${f}"
        else
            warn "  ${f} -- MISSING"
            all_ok=false
        fi
    done
    echo ""

    # Anomaly detection datasets
    local -A ad_files
    ad_files["MSL"]="MSL_train.npy MSL_test.npy MSL_test_label.npy"
    ad_files["PSM"]="train.csv test.csv test_label.csv"
    ad_files["SMAP"]="SMAP_train.npy SMAP_test.npy SMAP_test_label.npy"
    ad_files["SMD"]="SMD_train.npy SMD_test.npy SMD_test_label.npy"
    ad_files["SWaT"]="swat_train2.csv swat2.csv"

    echo "  Anomaly Detection:"
    for dataset in MSL PSM SMAP SMD SWaT; do
        echo "    ${dataset}/:"
        for f in ${ad_files[$dataset]}; do
            if [ -f "${DATA_ROOT}/raw/anomaly_detection/${dataset}/${f}" ]; then
                success "    ${f}"
            else
                warn "    ${f} -- MISSING"
                all_ok=false
            fi
        done
    done
    echo ""

    if [ "$all_ok" = true ]; then
        success "All expected files are present."
    else
        warn "Some files are missing. Please download them from Google Drive."
    fi
    echo ""
}

# -----------------------------------------------------------------------------
# Create DATA_ROOT README
# -----------------------------------------------------------------------------
create_data_readme() {
    cat > "${DATA_ROOT}/README.md" << 'DATAREADME'
# TimeCodec Datasets

This directory contains the benchmark datasets for the TimeCodec project.

## Source

All datasets are from the TOTEM benchmark:
https://drive.google.com/drive/u/0/folders/1gI36rS8irRZ32ibzKBPGncDmMXQtEf1C

## Directory Layout

```
raw/
├── forecasting/          # CSV files for forecasting and imputation tasks
│   ├── ETTh1.csv         # Electricity Transformer (hourly, 7 channels)
│   ├── ETTh2.csv         # Electricity Transformer (hourly, 7 channels)
│   ├── ETTm1.csv         # Electricity Transformer (15-min, 7 channels)
│   ├── ETTm2.csv         # Electricity Transformer (15-min, 7 channels)
│   ├── weather.csv       # Weather (21 channels)
│   ├── electricity.csv   # Electricity consumption (321 channels)
│   └── traffic.csv       # Road occupancy (862 channels)
├── imputation/           # Symlink to forecasting/ (same CSV files)
└── anomaly_detection/    # Sensor anomaly datasets
    ├── MSL/              # Mars Science Lab (55 sensors)
    ├── PSM/              # Pooled Server Metrics (25 sensors)
    ├── SMAP/             # Soil Moisture Active Passive (25 sensors)
    ├── SMD/              # Server Machine Dataset (38 sensors)
    └── SWaT/             # Secure Water Treatment (51 sensors)
```

## Usage

Set the DATA_ROOT environment variable to use these datasets:

```bash
# For forecasting and imputation (point to raw/forecasting/)
export DATA_ROOT=/path/to/this/directory/raw/forecasting

# For anomaly detection (point to raw/anomaly_detection/<dataset>/)
export DATA_ROOT=/path/to/this/directory/raw/anomaly_detection/MSL
```

See individual task READMEs for specific instructions.
DATAREADME

    success "Created ${DATA_ROOT}/README.md"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    echo ""
    echo "============================================================================="
    echo " TimeCodec Data Setup"
    echo " Target: ${DATA_ROOT}"
    echo "============================================================================="
    echo ""

    create_directories
    print_download_instructions
    create_data_readme
    verify_files

    echo "-----------------------------------------------------------------------------"
    echo " After downloading, set DATA_ROOT in your shell:"
    echo ""
    echo "   # For forecasting / imputation:"
    echo "   export DATA_ROOT=${DATA_ROOT}/raw/forecasting"
    echo ""
    echo "   # For anomaly detection (per dataset):"
    echo "   export DATA_ROOT=${DATA_ROOT}/raw/anomaly_detection/MSL"
    echo ""
    echo "   # Then run experiments:"
    echo "   bash forecasting/scripts/ETTh1.sh"
    echo "   bash anomaly_detection/scripts/msl.sh"
    echo "   bash imputation/scripts/ETTh1.sh"
    echo "-----------------------------------------------------------------------------"
    echo ""
}

main
