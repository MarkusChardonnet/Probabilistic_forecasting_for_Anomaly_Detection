#!/bin/bash
# Usage, from the repository root: SLURM_ACC=<account> slurm/submit_with_fallback.sh slurm/<job>.sbatch
# Submits the job, charged to the Slurm account SLURM_ACC, pinned to the newest CPU type. If it
# has not started one hour later, cancels it and resubmits with the CPU type as a soft preference
# only (any node may run it). Run it inside tmux: it waits for up to one hour.
set -euo pipefail
: "${SLURM_ACC:?set SLURM_ACC to the Slurm account to charge}"
mkdir -p slurm/logs
script=$1
name=$(basename "$script" .sbatch)
jobid=$(sbatch --parsable -A "$SLURM_ACC" --constraint=EPYC_9654 "$script")
echo "$(date) submitted $jobid ($name, constraint EPYC_9654)"
sleep 3600
if [ "$(squeue -j "$jobid" -h -o %T)" = "PENDING" ]; then
    scancel "$jobid"
    jobid=$(sbatch --parsable -A "$SLURM_ACC" --prefer=EPYC_9654 "$script")
    echo "$(date) still pending after one hour: resubmitted as $jobid ($name, prefer EPYC_9654)"
fi
echo "$jobid" > "slurm/logs/last_jobid_$name"
