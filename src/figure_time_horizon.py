"""
Composes Supplementary Fig. S4: distributions of the anomaly scores of the
validation set (unexposed infants) prior and after random cut-offs, pooled over
all cut-offs, with the pipeline of results/analyse_t_horizon_reliability.ipynb
("all cutoffs pooled") and its panel function, drawn at the printed size of the
figure (figure_style). The model and scaling-factor flags are those of
figure_score_delineation.py. Run from src/:

    uv run python figure_time_horizon.py --output=supp_time_horizon.pdf
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from absl import app, flags

from figure_score_delineation import scores_folder_name
from figure_style import (
    BASE_FONTSIZE,
    STAR_FONTSIZE,
    TEXT_WIDTH_IN,
    legend_above_footnote,
    rc_params,
    save_figure,
)
from utils_eval_score import (
    PanelStyle,
    _get_step_n_indicator,
    _group_samples_prior_to_cutoff,
    _plot_score_after_nth_abx_exposure,
)
from utils_t_horizon import (
    enrich_scores,
    select_last_score_per_host_per_bin,
    transform_cutoff_scores,
)

DAYS_PER_MONTH = 30.437
# cut-off days of the scored validation set, as in the notebook
CUTOFF_DAYS = range(0, 1141, 30)
MIN_SAMPLES, MAX_SAMPLES = -3.0, 12.0
FIGURE_SIZE_IN = (TEXT_WIDTH_IN, 3.0)
RC_PARAMS = rc_params()
# no legend in the panels: one legend below them, with stars only if a bin is
# significant (none is in the published data), above the footnote
PANEL_STYLE = PanelStyle(
    fontsize=BASE_FONTSIZE,
    ylim_factor=1.45,
    star_rows=(1.32, 1.14),
    star_fontsize=STAR_FONTSIZE,
    star_va="center",
    legend="none",
)
GRID = dict(left=0.08, right=0.99, top=0.93, bottom=0.225, hspace=0.18)
COUNT_YLABEL = "# samples"
HEIGHT_RATIOS = [1, 0.5]
XLABEL = "Months since cut-off"
FOOTNOTE = "(Here -1.0 is last sample prior to cut-off since -3.0 months)"

FLAGS = flags.FLAGS  # defined by figure_score_delineation


def cutoff_scores_folder(data_path, model_name, folder_name):
    """Folder of the scores of the validation set computed for random cut-offs."""
    return os.path.join(
        data_path,
        model_name,
        "anomaly_detection",
        "reliability_eval-val-noabx_best_normal",
        folder_name,
        "",
    )


def reliability_folder_name(stddev_type, moving_avg, duplicates):
    """Evaluation folder of the notebook (grouped samples, monthly resolution)."""
    return (
        f"gT_maxresF_{stddev_type}_ma{moving_avg}"
        f"{duplicates.replace('-', '_').lower()}"
    )


def load_cutoff_scores(cutoff_scores_path, cutoff_days=CUTOFF_DAYS, limit_months=24.0):
    """
    Long scores of all cut-off files with the months since the cut-off, limited
    to samples and cut-offs within `limit_months` of age.
    """
    frames = []
    for day in cutoff_days:
        c_scores = pd.read_csv(
            f"{cutoff_scores_path}val_noabx_ad_scores_{day}_coord-0.csv"
        )
        c_scores_t = transform_cutoff_scores(c_scores, DAYS_PER_MONTH)
        frames.append(enrich_scores(c_scores_t, max_resolution=False))
    scores = pd.concat(frames, ignore_index=True)
    if limit_months is not None:
        scores = scores[
            np.logical_and(
                scores["month5_bin"] <= limit_months,
                scores["cutoff_month"] < limit_months,
            )
        ]
    return scores


def bin_months_since_cutoff(scores, min_samples, max_samples):
    """
    Scores within `min_samples` and `max_samples` months of the cut-off with the
    months since the cut-off in monthly bins labelled by their left edge.
    """
    subset = scores.loc[
        np.logical_and(
            scores["months_since_cutoff"] >= min_samples,
            scores["months_since_cutoff"] <= max_samples,
        ),
        :,
    ].copy()
    bins = np.arange(min_samples, max_samples + 1.0, 1.0)
    subset.loc[:, "months_since_cutoff"] = pd.cut(
        subset["months_since_cutoff"], bins=bins, labels=bins[:-1], right=False
    )
    return subset


def bin_scores_around_cutoff(scores, min_samples=MIN_SAMPLES, max_samples=MAX_SAMPLES):
    """
    Scores within `min_samples` and `max_samples` months of the cut-off in
    monthly bins (labelled by their left edge), one score per host, cut-off and
    bin, with the samples before the cut-off grouped into the last one.
    """
    subset = bin_months_since_cutoff(scores, min_samples, max_samples)
    subset = select_last_score_per_host_per_bin(subset)
    group_step, last_bin_indicator = _get_step_n_indicator(False)
    return _group_samples_prior_to_cutoff(
        subset,
        "months_since_cutoff",
        ["host_id", "cutoff_month"],
        min_samples,
        group_step,
        last_bin_indicator,
    )


def compose_time_horizon(scores):
    """Box plot and count plot of the pooled scores around the cut-off."""
    with plt.rc_context(RC_PARAMS):
        fig = plt.figure(figsize=FIGURE_SIZE_IN)
        grid = fig.add_gridspec(2, 1, height_ratios=HEIGHT_RATIOS, **GRID)
        ax_box = fig.add_subplot(grid[0])
        ax_count = fig.add_subplot(grid[1], sharex=ax_box)
        _plot_score_after_nth_abx_exposure(
            scores,
            x_axis="months_since_cutoff",
            y_axis="score",
            n=0,
            min_samples=MIN_SAMPLES,
            max_samples=MAX_SAMPLES,
            max_resolution=False,
            grouped_samples=True,
            uniqueness_var_ls=["use_obs_until_day"],
            axs=(ax_box, ax_count),
            style=PANEL_STYLE,
            xlabel_note=False,
        )
        ax_count.set_ylabel(COUNT_YLABEL, fontsize=BASE_FONTSIZE)
        ax_count.set_xlabel(XLABEL, fontsize=BASE_FONTSIZE)
        stars = any("*" in t.get_text() for t in ax_box.texts)
        centre = (ax_count.get_position().x0 + ax_count.get_position().x1) / 2
        legend_above_footnote(fig, centre, FOOTNOTE, stars=stars)
    return fig


def main(argv):
    del argv
    folder_name = scores_folder_name(
        FLAGS.stddev_type, FLAGS.moving_avg, FLAGS.duplicates, FLAGS.lower_bound
    )
    scores = load_cutoff_scores(
        cutoff_scores_folder(FLAGS.data_path, FLAGS.model_name, folder_name),
        limit_months=FLAGS.limit_months,
    )
    binned = bin_scores_around_cutoff(scores)
    print(binned.groupby("months_since_cutoff", observed=True)["score"].count())
    fig = compose_time_horizon(binned)
    output = FLAGS.output or os.path.join(
        FLAGS.data_path,
        FLAGS.model_name,
        "anomaly_detection",
        "reliability_evaluation",
        reliability_folder_name(FLAGS.stddev_type, FLAGS.moving_avg, FLAGS.duplicates),
        "supp_time_horizon.pdf",
    )
    with plt.rc_context(RC_PARAMS):
        save_figure(fig, output)
    print("Saved figure to", output)


if __name__ == "__main__":
    app.run(main)
