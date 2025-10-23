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
num_epochs_shape=${4:-50}
num_epochs_scale=${5:-50}
num_epochs_recon=${5:-50}

N_FREQS=640
#N_FREQS=1280
#TIDAL_TERMS="--include-tidal-params"
#TIDAL_TERMS="--include-tidal-params-full"
#TIDAL_TERMS="--include-tidal-params --include-tidal-data"
#TIDAL_TERMS="--include-tidal-params --include-tidal-data --rescale-2p5and4"
#TIDAL_TERMS="--include-tidal-params --include-tidal-data --no-include-tidal-data-2p5and4"
#TIDAL_TERMS="--include-tidal-params --include-tidal-data --penalize-highPN"
#TIDAL_TERMS="--penalize-highPN"
#TIDAL_TERMS="--no-penalize-highPN"
#TIDAL_TERMS="--include-tidal-params --penalize-highPN"
#TIDAL_TERMS="--no-include-tidal-params --no-include-tidal-data --no-penalize-highPN"
#TIDAL_TERMS="--no-include-tidal-params --no-include-tidal-data --penalize-highPN --no-train-recon-phase"
#TIDAL_TERMS="--no-include-tidal-params --no-include-tidal-data --penalize-highPN --train-recon-phase"
#TIDAL_TERMS="--include-tidal-params --no-include-tidal-data --no-penalize-highPN"
#TIDAL_TERMS="--no-include-tidal-params --penalize-highPN"
TIDAL_TERMS="--no-include-tidal-params --no-include-tidal-data --no-penalize-highPN --no-train-recon-phase"

#DATASET=dataset_${content}
#DATASET=dataset_${content}_test
#DATASET=dataset_${content}_3.5PN
#DATASET=dataset_${content}_5PN
#DATASET=dataset_${content}_5PN_v2
#DATASET=dataset_${content}_5PN_test
#DATASET=dataset_${content}_5PN_test8
#DATASET=dataset_${content}_5PN_tidal_full
DATASET=dataset_${content}_mc

#TITLE=npe-$content
#TITLE=npe-$content-nohighPN-test-3
#TITLE=npe-$content-nohighPN-test-m1-2
#TITLE=npe-$content-nohighPN-test-m1e4
#TITLE=npe-$content-nohighPN-test-p1-2
#TITLE=npe-$content-nohighPN-test-p1e4
#TITLE=npe-$content-nohighPN-v2-p1e4
#TITLE=npe-$content-nohighPN-v2-p1e4
#TITLE=npe-$content-new
#TITLE=npe-$content-new-longscale-2
#TITLE=npe-$content-new6cond-longscale-2
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
#TITLE=npe-$content-5PN-nohighPN
#TITLE=npe-$content-5PN-nohighPN-test
#TITLE=npe-$content-nohighPN-p1e4
#TITLE=npe-$content-new-nohighPN-p1e4
#TITLE=npe-$content-new-nohighPN-p1e4-longscale
#TITLE=npe-$content-new-nohighPN-p1e4-longscale-2
#TITLE=npe-$content-new-nohighPN-p1e4-longscale-3
#TITLE=npe-$content-new-nohighPN-p1e4-recon
#TITLE=npe-$content-new-nohighPN-p1e4-recon-longscale
#TITLE=npe-$content-new-nohighPN-p1e4-recon-test
#TITLE=npe-$content-new-nohighPN-m1
#TITLE=npe-$content-new-nohighPN-m1-longscale-2
#TITLE=npe-$content-new-nohighPN-m1e4
#TITLE=npe-$content-new-nohighPN-m1e4-longscale-2
TITLE=npe-$content-mc
#TITLE=npe-$content-mc-test

TRAIN_LR=1e-4
#TRAIN_LR=5e-4
#TRAIN_LR=1e-3

TRAIN_WD=1e-4

TRAIN_GAMMA=0.9
#TRAIN_GAMMA=0.95

TRAIN_KL_COEFF=1e-6
#TRAIN_KL_COEFF=1e-7
#TRAIN_KL_COEFF=1e-8

#TRAIN_HIGHPN_COEFF=1.0
TRAIN_HIGHPN_COEFF=1e-4
#TRAIN_HIGHPN_COEFF=-1.0
#TRAIN_HIGHPN_COEFF=-1e-4
#TRAIN_HIGHPN_COEFF=0.0

python npe/train_network.py \
  --dataset-rootdir $DATASET \
  --output-rootdir network \
  --run-title $TITLE \
  --run-type $content \
  --dataset_seed $dataset_seed \
  --training_seed $training_seed \
  --data-dim $N_FREQS $TIDAL_TERMS \
  --num-epochs-shape $num_epochs_shape \
  --num-epochs-scale $num_epochs_scale \
  --num-epochs-recon $num_epochs_recon \
  --train-lr $TRAIN_LR --train-wd $TRAIN_WD \
  --train-gamma $TRAIN_GAMMA --train-kl-coeff $TRAIN_KL_COEFF \
  --train-highPN-coeff " ${TRAIN_HIGHPN_COEFF}"
