#!/bin/bash
set -e

# ---------------------------------------------------------------------------
# Debug run: single GPU, 1 scan, minimal batches, no WandB.
# Must be run from the train/ directory.
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$TRAIN_DIR"

# -------------------------------- config -----------------------------------
MODEL_CONFIG="configs/models/vggt_omega.yaml"
DATA_CONFIG="configs/data/hypersim/hypersim_default_train.yaml:configs/data/tartanair/tartanair_default_train.yaml:configs/data/blendedmvg/blendedmvg_default_train.yaml:configs/data/matrix_city/matrix_city_default_train.yaml:configs/data/vkitti/vkitti_default_train.yaml:configs/data/dynamic_replica/dynamic_replica_default_train.yaml:configs/data/mvssynth/mvssynth_default_train.yaml:configs/data/sailvos3d/sailvos3d_default_train.yaml"
VAL_DATA_CONFIG="configs/data/scannet/scannet_default_val.yaml"

# -------------------------------- hardware ---------------------------------
NUM_GPUS=1
export CUDA_VISIBLE_DEVICES=0

# -------------------------------- launch -----------------------------------
ARGS=(
    --config_file          "$MODEL_CONFIG"
    --data_config_file     "$DATA_CONFIG"
    --val_data_config_file "$VAL_DATA_CONFIG"
    --name                 debug
    --gpus                 "$NUM_GPUS"
    --batch_size           1
    --val_batch_size       1
    --num_workers          0
    --val_batches          2
    --val_interval         10
    --log_interval         1
    --max_epochs           1
    --num_sanity_val_steps 1
    --debug
)

echo "========================================================"
echo "  DEBUG RUN"
echo "  GPU  : $CUDA_VISIBLE_DEVICES"
echo "========================================================"

python train.py "${ARGS[@]}"
