#!/bin/bash
#SBATCH --job-name=njode_bench
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --open-mode=append
# Per-epoch timing of the published configuration (5 epochs, no pre-training, Faith PD
# dataset) with 1 and with 2 threads; results in data/saved_models_x_bench/id-{1,2}/.
# Submit from the repository root through the helper, which adds the Slurm account (SLURM_ACC)
# and the CPU-type request:
#   SLURM_ACC=<account> slurm/submit_with_fallback.sh slurm/bench_epoch_time.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}/src"
for threads in 1 2; do
    ../.venv/bin/python run.py --params=param_list_microbial_novel_alpha_div_bench \
        --NB_JOBS=1 --NB_CPUS=$threads --SEND=False --USE_GPU=False --first_id=$threads
done
