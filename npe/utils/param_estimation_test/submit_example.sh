#!/bin/bash -l
#SBATCH --job-name=npe-pe
#SBATCH --partition=<PARTITION>
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=2G
#SBATCH --output=job_%j.out
##SBATCH --mail-user=<EMAIL>
##SBATCH --mail-type=FAIL,END

export OMP_NUM_THREADS=1
export MPI_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NPE_PROJECT_DIR="<ROOTDIR>"

conda activate igwn-py310
python $NPE_PROJECT_DIR/npe/utils/param_estimation_test/bilby_script.py \
    --label $SLURM_JOB_NAME \
    --npool $SLURM_CPUS_PER_TASK \
    --check-point-delta-t 3600 \
    --b-inj -5 \
    --beta-rel-inj 0.5
