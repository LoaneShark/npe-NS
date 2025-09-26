#!/bin/bash
#SBATCH --nodes 1
#SBATCH --time 48:00:00
#SBATCH --mem 50G
#SBATCH --job-name npe
#SBATCH --output ./logs/npe/%J.txt
#SBATCH -p secondary                   # GravityTheory, physics, secondary, or test
#SBATCH --ntasks-per-node=1            # Number of tasks per node (1 per node for parallel execution)

content=${1:-"NS"}
dataset_seed=${2:-1234}
training_seed=${3:--1}
num_epochs=${4:-50}

N_FREQS=640
#N_FREQS=1280
#TIDAL_TERMS="--include-tidal"
#TIDAL_TERMS="--include-tidal-full"
#TIDAL_TERMS="--include-tidal --include-tidal-data"
TIDAL_TERMS="--no-include-tidal"

DATASET=dataset_${content}
#DATASET=dataset_${content}_3.5PN
#DATASET=dataset_${content}_5PN
#DATASET=dataset_${content}_5PN_tidal_full

TITLE=npe-$content
#TITLE=npe-$content-3.5PN
#TITLE=npe-$content-5PN

python npe/train_network.py \
  --dataset-rootdir $DATASET \
  --output-rootdir network \
  --run-title $TITLE \
  --run-type $content \
  --dataset_seed $dataset_seed \
  --training_seed $training_seed \
  --data-dim $N_FREQS $TIDAL_TERMS \
  --num-epochs $num_epochs
