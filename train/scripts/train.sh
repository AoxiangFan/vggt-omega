#!/bin/bash
set -e

# ---------------------------------------------------------------------------
# Usage:
#   bash scripts/train.sh [experiment_name] [pretrained_checkpoint]
#
# Must be run from the train/ directory so that data_splits/ and configs/
# are found relative to $PWD.
#
# Examples:
#   bash scripts/train.sh
#   bash scripts/train.sh vggt_omega_scannet
#   bash scripts/train.sh vggt_omega_finetune /path/to/pretrained.ckpt
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Must run from train/ so relative paths (data_splits/, configs/) resolve
cd "$TRAIN_DIR"

# -------------------------------- args -------------------------------------
NAME="${1:-vggt_omega}"
PRETRAINED_CKPT="${2:-}"

# -------------------------------- config -----------------------------------
MODEL_CONFIG="configs/models/vggt_omega.yaml"
DATA_CONFIG="configs/data/hypersim/hypersim_default_train.yaml:configs/data/tartanair/tartanair_default_train.yaml:configs/data/blendedmvg/blendedmvg_default_train.yaml:configs/data/matrix_city/matrix_city_default_train.yaml:configs/data/vkitti/vkitti_default_train.yaml:configs/data/dynamic_replica/dynamic_replica_default_train.yaml:configs/data/mvssynth/mvssynth_default_train.yaml:configs/data/sailvos3d/sailvos3d_default_train.yaml"
VAL_DATA_CONFIG="configs/data/scannet/scannet_default_val.yaml"

# -------------------------------- hardware ---------------------------------
NUM_GPUS=4

# -------------------------------- launch -----------------------------------
ARGS=(
    --config_file        "$MODEL_CONFIG"
    --data_config_file   "$DATA_CONFIG"
    --val_data_config_file "$VAL_DATA_CONFIG"
    --name               "$NAME"
    --gpus               "$NUM_GPUS"
)

if [ -n "$PRETRAINED_CKPT" ]; then
    ARGS+=(--lazy_load_weights_from_checkpoint "$PRETRAINED_CKPT")
fi

echo "========================================================"
echo "  Experiment : $NAME"
echo "  GPUs       : $CUDA_VISIBLE_DEVICES"
echo "  Model cfg  : $MODEL_CONFIG"
echo "  Data cfg   : $DATA_CONFIG"
echo "  Val cfg    : $VAL_DATA_CONFIG"
[ -n "$PRETRAINED_CKPT" ] && echo "  Checkpoint : $PRETRAINED_CKPT"
echo "========================================================"

python train.py "${ARGS[@]}"
