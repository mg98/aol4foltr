#!/bin/bash

#SBATCH --job-name=foltr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0
#SBATCH --cpus-per-task=64

export JAVA_HOME=~/jdk-21.0.6
export PATH=$JAVA_HOME/bin:$PATH

conda init bash
source ~/.bashrc
conda activate pyserini

head -n 1 aol_0_12.csv > output.csv
tail -n +2 -q aol_*_*.csv >> output.csv

rm aol_*_*.csv
mv output.csv aol_dataset_raw.csv

python postprocess.py
