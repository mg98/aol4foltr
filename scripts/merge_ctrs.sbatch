#!/bin/bash

#SBATCH --job-name=foltr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0

export JAVA_HOME=~/jdk-21.0.6
export PATH=$JAVA_HOME/bin:$PATH

conda init bash
source ~/.bashrc
conda activate pyserini

rm -rf dataset/ctrs.lmdb
python merge_ctrs.py
