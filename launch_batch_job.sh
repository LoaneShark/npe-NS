#!/bin/bash
#SBATCH --nodes 1
#SBATCH --time 48:00:00
#SBATCH --mem 50G
#SBATCH --job-name npe
#SBATCH --output ./logs/%J.txt
#SBATCH -p secondary                   # GravityTheory, physics, secondary, or test
#SBATCH --ntasks-per-node=1            # Number of tasks per node (1 per node for parallel execution)

## set NUMEXPR_MAX_THREADS
#export NUMEXPR_MAX_THREADS=416

module load anaconda3/2024.06-Jun
module load texlive

export PYTHONPATH=/home/loane2/config/python:${PYTHONPATH}
source /usr/local/anaconda/2024.06/3/etc/profile.d/conda.sh
conda activate npe

systype=${1:-"BH"}
nsample=${2:-100000} # reproduces the paper

if [[ $nsample -gt 1 ]]
then
    bash ./generate_dataset.sh $systype $nsample
fi

bash ./train_network.sh $systype