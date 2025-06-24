#!/bin/bash

sbatch scripts/create_index.sbatch
sbatch --dependency=singleton scripts/create_metadata.sbatch
sbatch --dependency=singleton scripts/merge_metadata.sbatch
sbatch --dependency=singleton scripts/create_ctrs.sbatch
sbatch --dependency=singleton scripts/merge_ctrs.sbatch
sbatch --dependency=singleton scripts/write_letor.sbatch
