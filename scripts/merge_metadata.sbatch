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

head -n 1 dataset/metadata_0_12.csv > dataset/temp.csv
tail -n +2 -q dataset/metadata_*_*.csv >> dataset/temp.csv

rm dataset/metadata_*_*.csv
mv dataset/temp.csv dataset/metadata_raw.csv

python postprocess_metadata.py
