#!/bin/bash

GPUID=$1
# exit on first error
set -e
# directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
parentdir="$(dirname "$SCRIPT_DIR")"
SSD_DIR="${SCRIPT_DIR}"
ATTACKS_DIR="${parentdir}/Attackset"
TARGETED_ATTACKS_DIR="${SCRIPT_DIR}/sample_targeted_attacks_temp"
DEFENSES_DIR="${SCRIPT_DIR}/sample_defenses_temp"
DATASET_DIR="${parentdir}/Originset"
DATASET_METADATA_FILE="${parentdir}/dev_dataset.csv"
MAX_EPSILON=16

# Prepare working directory and copy all necessary files.
# In particular copy attacks defenses and dataset, so originals won't
# be overwritten.
WORKING_DIR="${parentdir}/Advset"
echo "Running attacks and defenses"
python "${SCRIPT_DIR}/run_attacks.py" \
  --attacks_dir="${ATTACKS_DIR}" \
  --targeted_attacks_dir="${WORKING_DIR}/targeted_attacks" \
  --defenses_dir="${WORKING_DIR}/defenses" \
  --dataset_dir="${DATASET_DIR}" \
  --intermediate_results_dir="${WORKING_DIR}" \
  --dataset_metadata="${DATASET_METADATA_FILE}" \
  --output_dir="${WORKING_DIR}/output_dir${SETID}" \
  --epsilon="${MAX_EPSILON}" \
  --save_all_classification \
  --models=all \
  --gpu \
  --use_existing 0 \
  --gpuid="${GPUID}" \


