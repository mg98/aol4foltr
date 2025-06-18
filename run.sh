#!/bin/bash

sbatch create_index.sbatch
sbatch --dependency=singleton create_dataset.sbatch
sbatch --dependency=singleton postprocess.sbatch
sbatch --dependency=singleton create_ctrs.sbatch
sbatch --dependency=singleton merge_ctrs.sbatch
