#!/bin/bash

#SBATCH --job-name=foltr
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0
#SBATCH --cpus-per-task=64

conda init bash
source ~/.bashrc
conda activate pyserini

srun zsh -c "
    python get_ctrs.py \
        --job-id=${SLURM_ARRAY_TASK_ID} \
        --job-count=${SLURM_ARRAY_TASK_COUNT}
"
