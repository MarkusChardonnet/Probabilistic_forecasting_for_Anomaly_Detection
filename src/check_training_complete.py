"""
Exits with an error unless a model was trained for all epochs of its
configuration. The Slurm scoring scripts call it first, because they follow
the training job with afterany: the training job ends as FAILED when only its
final overview step fails. Run from src/:

    python check_training_complete.py ../data/saved_models_microbial_novel_alpha_div2 57
"""

import json
import os
import sys

import pandas as pd


def trained_epochs(saved_models_path, model_id):
    """Epochs trained and epochs configured for model `model_id`."""
    overview = pd.read_csv(os.path.join(saved_models_path, "model_overview.csv"), index_col=0)
    description = overview.loc[overview["id"] == model_id, "description"].item()
    metric = pd.read_csv(
        os.path.join(saved_models_path, f"id-{model_id}", f"metric_id-{model_id}.csv")
    )
    return int(metric["epoch"].max()), int(json.loads(description)["epochs"])


if __name__ == "__main__":
    path, model_id = sys.argv[1], int(sys.argv[2])
    trained, configured = trained_epochs(path, model_id)
    if trained < configured:
        sys.exit(f"model id-{model_id} in {path}: {trained} of {configured} epochs trained")
    print(f"model id-{model_id} in {path}: all {configured} epochs trained")
