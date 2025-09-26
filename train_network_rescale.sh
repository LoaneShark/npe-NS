#!/bin/bash
#SBATCH --nodes 1
#SBATCH --time 48:00:00
#SBATCH --mem 300G
#SBATCH --job-name npe-rescale
#SBATCH --output ./logs/npe/%J.txt
#SBATCH -p GravityTheory               # GravityTheory, physics, secondary, or IllinoisComputes
#SBATCH --ntasks-per-node=1            # Number of tasks per node (1 per node for parallel execution)

DATASET_NAME=${1:-"dataset_NS_6"}
NETWORK_NAME=${2:-"npe-NS-6"}
RUN_TYPE=${3:-"NS"}
DATA_SEED=${4:-1234}
TRAIN_SEED=${5:--1}
NUM_EPOCHS=${6:-50}

#BASE_NETWORK_NAME="${NETWORK_NAME}_100-epochs"
#BASE_NETWORK_NAME="npe-${RUN_TYPE}_100-epochs"
#BASE_NETWORK_NAME="${NETWORK_NAME}"
#BASE_NETWORK_NAME="npe-NS-5PN_100-epochs"
BASE_NETWORK_NAME="npe-NS-5PN_200-epochs"
#BASE_NETWORK_NAME="npe-NS-5PN_400-epochs"

#EXTRA_ARGS="--include-tidal --include-tidal-data --data-dim 640"
#EXTRA_ARGS="--include-tidal --include-tidal-data --data-dim 640 --num-epochs $NUM_EPOCHS"
EXTRA_ARGS="--include-tidal --include-tidal-data --data-dim 640 --rescale-2p5and4-only --num-epochs $NUM_EPOCHS"
#EXTRA_ARGS="--include-tidal --include-tidal-data --data-dim 640 --num-epochs 200"
#EXTRA_ARGS="--data-dim 1280"
#EXTRA_ARGS=""

python npe/train_network_rescale.py \
  --dataset-rootdir $DATASET_NAME \
  --rescale-rootdir ${DATASET_NAME}_rescale \
  --base-network-path network/checkpoints/${NETWORK_NAME}/${BASE_NETWORK_NAME}.pt \
  --output-rootdir network \
  --run-title $NETWORK_NAME \
  --run-type $RUN_TYPE \
  --dataset-seed $DATA_SEED \
  --training-seed $TRAIN_SEED $EXTRA_ARGS