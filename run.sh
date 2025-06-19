#!/bin/bash

sbatch create_index.sbatch
sbatch --dependency=singleton reconstruct_results.sbatch
sbatch --dependency=singleton postprocess_metadata.sbatch
sbatch --dependency=singleton create_ctrs.sbatch
sbatch --dependency=singleton merge_ctrs.sbatch
sbatch --dependency=singleton write_letor.sbatch
