#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0
#SBATCH --cpus-per-task=64

conda init bash
source ~/.bashrc
conda activate pyserini

python exp.py