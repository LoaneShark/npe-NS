#!/bin/bash
#SBATCH --nodes 1
#SBATCH --time 48:00:00
#SBATCH --mem 50G
#SBATCH --job-name npe
#SBATCH --output ./logs/npe/%J.txt
#SBATCH -p secondary                   # GravityTheory, physics, secondary, or test
#SBATCH --ntasks-per-node=1            # Number of tasks per node (1 per node for parallel execution)

content=${1:-"BH"}
seed=${2:-1234}

python npe/train_network.py \
  --dataset-rootdir dataset_$content \
  --output-rootdir network \
  --run-title npe-$content \
  --run-type $content \
  --seed $seed