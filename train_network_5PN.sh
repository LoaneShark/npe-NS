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
TIDAL_TERMS="--include-tidal --include-tidal-data"
#TIDAL_TERMS="--include-tidal --include-tidal-data --rescale-2p5and4"
#TIDAL_TERMS="--include-tidal --include-tidal-data --no-include-tidal-data-2p5and4"
#TIDAL_TERMS="--no-include-tidal"

#DATASET=dataset_${content}
#DATASET=dataset_${content}_3.5PN
DATASET=dataset_${content}_5PN
#DATASET=dataset_${content}_5PN_v2
#DATASET=dataset_${content}_5PN_test
#DATASET=dataset_${content}_5PN_test8
#DATASET=dataset_${content}_5PN_tidal_full

TITLE=npe-$content
#TITLE=npe-$content-3.5PN
#TITLE=npe-$content-5PN
#TITLE=npe-$content-5PN-v2
#TITLE=npe-$content-5PN-v2-200
#TITLE=npe-$content-5PN-no2p5and4
#TITLE=npe-$content-5PN-nodeg
#TITLE=npe-$content-5PN-test-19
#TITLE=npe-$content-5PN-200
#TITLE=npe-$content-5PN-200-test
#TITLE=npe-$content-5PN-400
#TITLE=npe-$content-5PN-400-no2p5and4
#TITLE=npe-$content-5PN-400-test

#TRAIN_LR=1e-4
TRAIN_LR=5e-4
#TRAIN_LR=1e-3

TRAIN_WD=1e-4

TRAIN_GAMMA=0.9
#TRAIN_GAMMA=0.95

TRAIN_KL_COEFF=1e-6
#TRAIN_KL_COEFF=1e-7
#TRAIN_KL_COEFF=1e-8

python npe/train_network.py \
  --dataset-rootdir $DATASET \
  --output-rootdir network \
  --run-title $TITLE \
  --run-type $content \
  --dataset_seed $dataset_seed \
  --training_seed $training_seed \
  --data-dim $N_FREQS $TIDAL_TERMS \
  --num-epochs $num_epochs \
  --train-lr $TRAIN_LR --train-wd $TRAIN_WD \
  --train-gamma $TRAIN_GAMMA --train-kl-coeff $TRAIN_KL_COEFF
