#!/bin/bash

#SBATCH --job-name=foltr
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0

export JAVA_HOME=~/jdk-21.0.6
export PATH=$JAVA_HOME/bin:$PATH

conda init bash
source ~/.bashrc
conda activate pyserini

srun zsh -c "
    python ./create_metadata.py \
        --job-id=${SLURM_ARRAY_TASK_ID} \
        --job-count=${SLURM_ARRAY_TASK_COUNT}
"