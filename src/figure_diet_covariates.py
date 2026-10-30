"""
Composes Supplementary Fig. S13: the diet covariates (weaning, milk diet) of
the samples around the first three antibiotic exposures, i.e. of the samples
shown in Fig. 4b, as stacked monthly counts in a grid of two covariate rows and
three exposure columns. The samples are selected as in
results/evaluate_scores.ipynb and the figure is drawn at its printed size
(figure_style). The model and scaling-factor flags are those of
figure_score_delineation.py. Run from src/:

    uv run python figure_diet_covariates.py --output=diet_covariates_abx_exp.pdf
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from absl import app, flags

from figure_score_delineation import (
    evaluation_folder_name,
    scores_folder_name,
)
from figure_style import (
    BASE_FONTSIZE,
    LETTER_FONTSIZE,
    TEXT_WIDTH_IN,
    add_panel_letters,
    rc_params,
    save_figure,
)
from utils_color_maps import all_color_maps
from utils_eval_score import (
    _get_ordinal_suffix,
    _select_samples_around_nth_abx_exposure,
    get_scores_n_abx_info,
)

EXPOSURES = [1, 2, 3]
MIN_SAMPLES, MAX_SAMPLES = -3.0, 6.0
# monthly bins of the grouped samples: -1 (last sample before the exposure)
# and the months 0 to 5 after it
BINS = np.arange(-1.0, 6.0, 1.0)
DIET_ORDER = {
    "diet_weaning": ["no", "yes", "unknown"],
    "diet_milk": ["bd", "mixed", "fd", "unknown"],
}
LEGEND_TITLES = {"diet_weaning": "diet weaning", "diet_milk": "diet milk"}
XLABEL = "Months since abx exposure"
YLABEL = "# samples"
FOOTNOTE = "(Here -1.0 is last sample prior to abx since -3.0 months)"
FIGURE_SIZE_IN = (TEXT_WIDTH_IN, 3.4)
RC_PARAMS = rc_params()
GRID = dict(left=0.075, right=0.845, top=0.91, bottom=0.15, wspace=0.12, hspace=0.3)
LETTERS = ["a", "b", "c"]
LETTER_Y = 0.985
# one x label and the footnote, centred under the bottom row
XLABEL_Y = 0.09
FOOTNOTE_Y = 0.01
BAR_WIDTH = 0.6
LEGEND_KWS = dict(
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    frameon=False,
    handlelength=1.0,
    labelspacing=0.3,
    borderaxespad=0.0,
)

FLAGS = flags.FLAGS  # defined by figure_score_delineation


def select_exposure_samples(abx_scores_flat, abx_df, n):
    """Samples around the n-th exposure, grouped as for the score panels of Fig. 4b."""
    return _select_samples_around_nth_abx_exposure(
        abx_scores_flat,
        abx_df,
        n=n,
        min_samples=MIN_SAMPLES,
        max_samples=MAX_SAMPLES,
        group_samples=True,
        score_var=f"score_{n}",
        max_resolution=False,
    )


def diet_counts(samples, diet_var, x_axis="diff_age_nth_abx", bins=BINS):
    """
    Samples per bin (rows, all of `bins`) and category of `diet_var` (columns
    in the order of DIET_ORDER, missing values counted as 'unknown').
    """
    frame = samples[[x_axis, diet_var]].copy()
    frame[x_axis] = frame[x_axis].astype(float)
    frame[diet_var] = frame[diet_var].astype(object).fillna("unknown")
    counts = frame.groupby([x_axis, diet_var]).size().unstack(fill_value=0)
    counts = counts.reindex(index=bins, columns=DIET_ORDER[diet_var], fill_value=0)
    counts.index.name = x_axis
    return counts


def _draw_stacked_bars(ax, counts, diet_var):
    colors = all_color_maps[diet_var]
    bottom = np.zeros(len(counts))
    x = np.arange(len(counts))
    for level in counts.columns:
        ax.bar(
            x, counts[level], BAR_WIDTH, bottom=bottom, color=colors[level], label=level
        )
        bottom += counts[level].values
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:.0f}" for b in counts.index])
    zero_index = int(np.where(counts.index == 0.0)[0][0])
    ax.axvline(zero_index - 0.5, color="darkred")
    ax.tick_params(axis="both", labelsize=BASE_FONTSIZE)


def compose_diet_covariates(samples_by_exposure):
    """
    Grid of the diet covariate counts: rows = weaning and milk diet, columns =
    the exposures of `samples_by_exposure` (n -> samples, see
    `select_exposure_samples`), with the y-axis shared within each row and one
    legend per row.
    """
    exposures = list(samples_by_exposure)
    diet_vars = list(DIET_ORDER)
    with plt.rc_context(RC_PARAMS):
        fig = plt.figure(figsize=FIGURE_SIZE_IN)
        grid = fig.add_gridspec(len(diet_vars), len(exposures), **GRID)
        axes = []
        for row, diet_var in enumerate(diet_vars):
            row_axes = []
            for col, n in enumerate(exposures):
                ax = fig.add_subplot(grid[row, col])
                _draw_stacked_bars(ax, diet_counts(samples_by_exposure[n], diet_var), diet_var)
                if row == 0:
                    ax.set_title(f"{n}{_get_ordinal_suffix(n)} abx exposure", fontsize=BASE_FONTSIZE)
                ax.set_ylabel(YLABEL if col == 0 else "", fontsize=BASE_FONTSIZE)
                row_axes.append(ax)
            ymax = max(ax.get_ylim()[1] for ax in row_axes)
            for col, ax in enumerate(row_axes):
                ax.set_ylim(0, ymax)
                if col > 0:
                    ax.tick_params(labelleft=False)
            row_axes[-1].legend(
                title=LEGEND_TITLES[diet_var],
                fontsize=BASE_FONTSIZE,
                title_fontsize=BASE_FONTSIZE,
                **LEGEND_KWS,
            )
            axes += row_axes
        letter_x = [ax.get_position().x0 - 0.06 for ax in axes[: len(exposures)]]
        add_panel_letters(
            fig, dict(zip(LETTERS, [(x, LETTER_Y) for x in letter_x])), LETTER_FONTSIZE
        )
        bottom_row = axes[-len(exposures) :]
        centre = (bottom_row[0].get_position().x0 + bottom_row[-1].get_position().x1) / 2
        fig.text(centre, XLABEL_Y, XLABEL, fontsize=BASE_FONTSIZE, ha="center", va="top")
        fig.text(centre, FOOTNOTE_Y, FOOTNOTE, fontsize=BASE_FONTSIZE, ha="center", va="bottom")
    return fig


def main(argv):
    del argv
    folder_name = scores_folder_name(
        FLAGS.stddev_type, FLAGS.moving_avg, FLAGS.duplicates, FLAGS.lower_bound
    )
    scores_path = os.path.join(
        FLAGS.data_path,
        FLAGS.model_name,
        "anomaly_detection",
        f"scores_{FLAGS.point_to_evaluate}_normal",
        folder_name,
        "",
    )
    _, _, abx_scores_flat, abx_df, _ = get_scores_n_abx_info(
        scores_path,
        FLAGS.ft_name,
        FLAGS.limit_months,
        FLAGS.abx_ts_name,
        no_filter=True,
        path_to_data=os.path.join(FLAGS.data_path, "original_data", ""),
    )
    samples = {n: select_exposure_samples(abx_scores_flat, abx_df, n) for n in EXPOSURES}
    for n, frame in samples.items():
        for diet_var in DIET_ORDER:
            print(f"{n}: {diet_var}")
            print(diet_counts(frame, diet_var).to_string())
    fig = compose_diet_covariates(samples)
    output = FLAGS.output or os.path.join(
        FLAGS.data_path,
        FLAGS.model_name,
        "anomaly_detection",
        evaluation_folder_name(
            FLAGS.point_to_evaluate, FLAGS.stddev_type, FLAGS.moving_avg, FLAGS.duplicates
        ),
        "diet_covariates_abx_exp.pdf",
    )
    with plt.rc_context(RC_PARAMS):
        save_figure(fig, output)
    print("Saved figure to", output)


if __name__ == "__main__":
    app.run(main)
