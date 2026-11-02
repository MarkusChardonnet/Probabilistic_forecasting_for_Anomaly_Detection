#!/bin/bash
#SBATCH --job-name=njode_exposure_shift_score
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --open-mode=append
# Scores for the sensitivity to the recording resolution of the exposure ages (README section of
# the same name) of the published model id-57 and of the model with the country of residence as
# additional covariate. Submit from the repository root after slurm/score_faith_pd.sbatch and
# slurm/score_country.sbatch (it needs their scaling factors):
#   mkdir -p slurm/logs && sbatch -A "$SLURM_ACC" --prefer=EPYC_9654 --dependency=afterany:<scoring job ids> slurm/score_exposure_shift.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}/src"
PY=../.venv/bin/python
$PY check_training_complete.py ../data/saved_models_microbial_novel_alpha_div2 57
$PY check_training_complete.py ../data/saved_models_microbial_novel_alpha_div_country 1

# shifted exposure and feature tables (data/original_data/*_shift_<token>.tsv)
$PY make_shifted_exposure_tables.py --shift_months=-0.5
$PY make_shifted_exposure_tables.py --shift_months=0.5

# per shift: dataset with the published split, model folder with copies of the checkpoint and the
# scaling factors, and the scores with the exposure cut-offs 1 to 3 and the one-step scores
#   score_shifts <dataset> <model folder> <model id> <AD models path> <AD params>
score_shifts () {
    local dataset=$1 folder=$2 id=$3 ad_path=$4 ad_params=$5
    for tok in m05 p05; do
        $PY make_microbial_dataset.py --dataset_config=config_${dataset#microbial_}_shift_$tok
        for split in all no_abx; do
            cp ../data/training_data/$dataset/$split/*_idx.npy ../data/training_data/${dataset}_shift_$tok/$split/
        done
        D=../data/${folder}_shift_$tok
        mkdir -p $D/id-$id/anomaly_detection
        cp ../data/$folder/model_overview.csv $D/
        cp -r ../data/$folder/id-$id/best_checkpoint $D/id-$id/
        cp -r ../data/$folder/id-$id/anomaly_detection/zscore_scaling_factors_best $D/id-$id/anomaly_detection/
        COMMON=(--NB_CPUS=2 --forecast_model_ids=${ad_path}_shift_${tok}_ids
                --forecast_saved_models_path=${ad_path}_shift_${tok}
                --compute_scores=True --evaluate_scores=False --compute_zscore_scaling_factors=False)
        $PY Microbial_AD_eval.py "${COMMON[@]}" --ad_params=${ad_params}_shift_${tok} --save_cond_moments=True
        $PY Microbial_AD_eval.py "${COMMON[@]}" --ad_params=${ad_params}_reliability_eval_shift_${tok}
    done
}
score_shifts microbial_novel_alpha_faith_pd saved_models_microbial_novel_alpha_div2 57 \
    AD_microbial_novel_alpha_div param_list_AD_microbial_novel_alpha_div2
score_shifts microbial_novel_alpha_faith_pd_w_geo saved_models_microbial_novel_alpha_div_country 1 \
    AD_microbial_novel_alpha_div_country param_list_AD_microbial_novel_alpha_div_country
