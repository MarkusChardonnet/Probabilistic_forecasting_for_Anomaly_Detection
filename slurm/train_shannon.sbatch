#!/bin/bash
#SBATCH --job-name=njode_shannon_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --open-mode=append
# Submit from the repository root through the helper, which adds the Slurm account (SLURM_ACC)
# and the CPU-type request:
#   SLURM_ACC=<account> slurm/submit_with_fallback.sh slurm/train_shannon.sbatch
# Resubmitting after a wall-time abort resumes model id 1 from its last checkpoint.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}/src"
../.venv/bin/python run.py \
    --params=param_list_microbial_novel_alpha_div_shannon \
    --NB_JOBS=1 --NB_CPUS=1 --SEND=False --USE_GPU=False --first_id=1 \
    --get_overview=overview_dict_microbial_novel_alpha_div_shannon
