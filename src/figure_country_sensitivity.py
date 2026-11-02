"""
Supplementary figure of the sensitivity of the delineated anomaly dynamics to the
country of residence as additional covariate: the score box plots of Fig. 4b after
the 1st to 3rd exposure for the main model (without the country, left) and the model
additionally conditioned on it (right), drawn with the row layout of figure_exposure_shift.

Run from src/:
    uv run python figure_country_sensitivity.py --output=supp_country_sensitivity.pdf
Flags: the two model names (defaults: the models of the manuscript) and the data,
scaling-factor and output flags of figure_score_delineation.
"""

import os

from absl import app, flags

from figure_exposure_shift import (
    compose_exposure_shift_figure,
    legend_hides_stars,
    load_variant,
    save_figure,
    variant_names,
)
from figure_score_delineation import evaluation_folder_name, scores_folder_name

COLUMN_LABELS = ("without country of residence", "with country of residence")

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "without_country_model",
    "saved_models_microbial_novel_alpha_div2/id-57",
    "main model, without the country of residence as covariate",
)
flags.DEFINE_string(
    "with_country_model",
    "saved_models_microbial_novel_alpha_div_country/id-1",
    "model with the country of residence as additional covariate",
)


def compose_country_sensitivity_figure(without_country, with_country):
    """
    Figure with one row per exposure and the two models as columns; each argument
    is the pair (abx_scores_flat, abx_df) of a model.
    """
    return compose_exposure_shift_figure(
        [(COLUMN_LABELS[0], *without_country), (COLUMN_LABELS[1], *with_country)]
    )


def main(argv):
    del argv
    folder_name = scores_folder_name(
        FLAGS.stddev_type, FLAGS.moving_avg, FLAGS.duplicates, FLAGS.lower_bound
    )
    names = variant_names(0.0)
    models = []
    for model_name in (FLAGS.without_country_model, FLAGS.with_country_model):
        models.append(
            load_variant(
                FLAGS.data_path,
                model_name,
                FLAGS.point_to_evaluate,
                folder_name,
                names["ft_name"],
                names["abx_ts_name"],
                FLAGS.limit_months,
            )
        )
    fig = compose_country_sensitivity_figure(*models)
    if legend_hides_stars(fig):
        print("WARNING: a legend overlaps significance stars")
    output = FLAGS.output or os.path.join(
        FLAGS.data_path,
        FLAGS.without_country_model,
        "anomaly_detection",
        evaluation_folder_name(
            FLAGS.point_to_evaluate, FLAGS.stddev_type, FLAGS.moving_avg, FLAGS.duplicates
        ),
        "supp_country_sensitivity.pdf",
    )
    save_figure(fig, output)
    print("Saved figure to", output)


if __name__ == "__main__":
    app.run(main)
