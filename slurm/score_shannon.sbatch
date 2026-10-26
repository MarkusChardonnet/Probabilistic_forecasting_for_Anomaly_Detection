#!/bin/bash
#SBATCH --job-name=njode_shannon_score
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --open-mode=append
# Submit from the repository root once the training job has completed, charged to the Slurm
# account SLURM_ACC (soft CPU preference, the job is short):
#   mkdir -p slurm/logs && sbatch -A "$SLURM_ACC" --prefer=EPYC_9654 slurm/score_shannon.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}/src"
PY=../.venv/bin/python
COMMON=(--NB_CPUS=2
        --forecast_model_ids=AD_microbial_novel_alpha_div_shannon_ids
        --forecast_saved_models_path=AD_microbial_novel_alpha_div_shannon)

# 1. z-scores of the unexposed validation infants
$PY Microbial_AD_eval.py "${COMMON[@]}" --ad_params=param_list_AD_microbial_novel_alpha_div2_scaling_factors --compute_scores=True --evaluate_scores=False --compute_zscore_scaling_factors=False
# 2. scaling factors from the z-scores
$PY Microbial_AD_eval.py "${COMMON[@]}" --ad_params=param_list_AD_microbial_novel_alpha_div2_scaling_factors2 --compute_scores=False --evaluate_scores=False --compute_zscore_scaling_factors=True
# 3. anomaly scores for the 1st, 2nd and 3rd exposure cut-off, with conditional-moment tables
$PY Microbial_AD_eval.py "${COMMON[@]}" --ad_params=param_list_AD_microbial_novel_alpha_div2 --compute_scores=True --evaluate_scores=False --compute_zscore_scaling_factors=False --save_cond_moments=True
# 4. one-step scores (no cut-off) and reliability evaluation; the notebook needs the one-step scores
$PY Microbial_AD_eval.py "${COMMON[@]}" --ad_params=param_list_AD_microbial_novel_alpha_div2_reliability_eval --compute_scores=True --evaluate_scores=False --compute_zscore_scaling_factors=False
# 5. built-in quick evaluation of the scores
$PY Microbial_AD_eval.py "${COMMON[@]}" --ad_params=param_list_AD_microbial_novel_alpha_div2_ev --compute_scores=False --evaluate_scores=True --compute_zscore_scaling_factors=False
# 6. score-evaluation notebook for the Shannon model, executed in place with the environment's own kernel
cd ../results
$PY -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 evaluate_scores_shannon.ipynb
