#!/bin/bash

#SBATCH --job-name=foltr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0

conda init bash
source ~/.bashrc
conda activate pyserini

rm -f dataset/letor.txt
python write_letor.py --ds dataset/metadata.csv --output dataset/letor.txt
