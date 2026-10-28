import os
import sys
import unittest

# since this is not a package - add src to path
src_dir = os.path.abspath("src")
sys.path.append(src_dir)

from Microbial_AD_eval import evaluate_scores

forecast_saved_models_path = "data/saved_models_microbial_novel_alpha_div/"
forecast_model_id = "56_test"
dataset = "ft_vat19_anomaly_v20240806_entero_genus.tsv"


evaluate_scores(
forecast_saved_models_path=forecast_saved_models_path,
forecast_model_id=forecast_model_id,
load_best=True,
dataset=dataset,
send=True,
)
