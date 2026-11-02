#!/bin/bash
#SBATCH --job-name=njode_faith_pd_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --open-mode=append
# Retrains the published configuration (id-57 of saved_models_microbial_novel_alpha_div2) alone.
# Submit from the repository root through the helper, which adds the Slurm account (SLURM_ACC)
# and the CPU-type request:
#   SLURM_ACC=<account> slurm/submit_with_fallback.sh slurm/train_faith_pd.sbatch
# Resubmitting after a wall-time abort resumes model id 57 from its last checkpoint, so the
# folder must not hold an id-57 of an earlier training.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}/src"
../.venv/bin/python run.py \
    --params=param_list_microbial_novel_alpha_div2_id57 \
    --NB_JOBS=1 --NB_CPUS=1 --SEND=False --USE_GPU=False --first_id=57 \
    --get_overview=overview_dict_microbial_novel_alpha_div2_id57
