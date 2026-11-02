#!/bin/bash
#SBATCH --job-name=njode_faith_pd_score
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --open-mode=append
# Scores of the published model id-57 (README, Compute Anomaly Detection Scores) and the
# score-evaluation notebook. Submit from the repository root after the training job, charged to
# the Slurm account SLURM_ACC (soft CPU preference, the job is short):
#   mkdir -p slurm/logs && sbatch -A "$SLURM_ACC" --prefer=EPYC_9654 --dependency=afterany:<training job id> slurm/score_faith_pd.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}/src"
PY=../.venv/bin/python
COMMON=(--NB_CPUS=2 --forecast_saved_models_path=AD_microbial_novel_alpha_div2)

# 0. stop unless all epochs were trained (afterany: the training job fails in its final overview step)
$PY check_training_complete.py ../data/saved_models_microbial_novel_alpha_div2 57

# 1. z-scores of the unexposed validation infants
$PY Microbial_AD_eval.py "${COMMON[@]}" --forecast_model_ids=AD_microbial_novel_alpha_div_ids2 --ad_params=param_list_AD_microbial_novel_alpha_div2_scaling_factors --compute_scores=True --evaluate_scores=False --compute_zscore_scaling_factors=False
# 2. scaling factors from the z-scores
$PY Microbial_AD_eval.py "${COMMON[@]}" --forecast_model_ids=AD_microbial_novel_alpha_div_ids2 --ad_params=param_list_AD_microbial_novel_alpha_div2_scaling_factors2 --compute_scores=False --evaluate_scores=False --compute_zscore_scaling_factors=True
# 3. anomaly scores for the 1st, 2nd and 3rd exposure cut-off, with conditional-moment tables
$PY Microbial_AD_eval.py "${COMMON[@]}" --forecast_model_ids=AD_microbial_novel_alpha_div_ids2_1 --ad_params=param_list_AD_microbial_novel_alpha_div2 --compute_scores=True --evaluate_scores=False --compute_zscore_scaling_factors=False --save_cond_moments=True
# 4. one-step scores (no cut-off) and reliability evaluation; the evaluation and the notebooks need the one-step scores
$PY Microbial_AD_eval.py "${COMMON[@]}" --forecast_model_ids=AD_microbial_novel_alpha_div_ids2_1 --ad_params=param_list_AD_microbial_novel_alpha_div2_reliability_eval --compute_scores=True --evaluate_scores=False --compute_zscore_scaling_factors=False
# 5. built-in quick evaluation of the scores
$PY Microbial_AD_eval.py "${COMMON[@]}" --forecast_model_ids=AD_microbial_novel_alpha_div_ids2_1 --ad_params=param_list_AD_microbial_novel_alpha_div2_ev --compute_scores=False --evaluate_scores=True --compute_zscore_scaling_factors=False
# 6. reference scores without scaling factors
$PY Microbial_AD_eval.py "${COMMON[@]}" --forecast_model_ids=AD_microbial_novel_alpha_div_ids2_1 --ad_params=param_list_AD_microbial_novel_alpha_div2_nsf --compute_scores=True --evaluate_scores=False --compute_zscore_scaling_factors=False
$PY Microbial_AD_eval.py "${COMMON[@]}" --forecast_model_ids=AD_microbial_novel_alpha_div_ids2_1 --ad_params=param_list_AD_microbial_novel_alpha_div2_reliability_eval_nsf --compute_scores=True --evaluate_scores=False --compute_zscore_scaling_factors=False
# 7. score-evaluation notebook, executed in place with the environment's own kernel
cd ../results
$PY -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 evaluate_scores.ipynb
# 8. learned conditional moments of validation paths (README, Training PD-NJODE)
cd ../src
$PY run.py --plot_paths=plot_paths_novel_alpha_div2
