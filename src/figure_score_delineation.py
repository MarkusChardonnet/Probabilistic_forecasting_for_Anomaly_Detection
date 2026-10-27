"""
Composes the score-delineation figure of the manuscript (Fig. 4): individual
anomaly score trajectories (a), anomaly score distributions (b) and
covariate-matched alpha diversity differences in percent of the matched mean
(c) around the first three antibiotic exposures. The panels are drawn with the
functions used by the notebooks results/evaluate_scores.ipynb and
results/evaluate_matched_alpha.ipynb. Run from src/:

    uv run python figure_score_delineation.py --output=score_delineation.pdf
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from absl import app, flags

from utils_eval_score import (
    REL_MATCHED_DIFF_COL,
    PanelStyle,
    _plot_score_after_nth_abx_exposure,
    _select_samples_around_nth_abx_exposure,
    calculate_matched_metric_n_diff,
    calculate_relative_matched_diff,
    get_scores_n_abx_info,
    plot_trajectory,
)

# page size of the published figure (547.52 x 621.682 pt)
FIGURE_SIZE_IN = (547.52 / 72, 621.682 / 72)
BASE_FONTSIZE = 6
STAR_FONTSIZE = 10
LETTER_FONTSIZE = 13
# the published panels were drawn at 9 x 6 in and scaled by about 0.4 on the page
PANEL_SCALE = 0.4
SCORE_PANEL_STYLE = PanelStyle(
    fontsize=BASE_FONTSIZE,
    ylim_factor=1.2,
    star_rows=(1.0, 0.9),
    star_fontsize=STAR_FONTSIZE,
    legend_markersize=12 * PANEL_SCALE,
    legend_linewidth=3 * PANEL_SCALE,
    star_va="center",
)
MATCHED_PANEL_STYLE = PanelStyle(
    fontsize=BASE_FONTSIZE,
    ylim_factor=1.5,
    star_rows=(1.15, 1.1),
    star_fontsize=STAR_FONTSIZE,
    legend_markersize=12 * PANEL_SCALE,
    legend_linewidth=3 * PANEL_SCALE,
    star_va="center",
)
# y label of the matched panels in two lines, so that it fits the axis height
MATCHED_YLABEL = "Δ matched alpha\ndiversity [%]"
# compact legends; the corners of the published hosts are those of the former
# figure (the automatic placement is misled by the vertical event lines)
TRAJECTORY_LEGEND_SPACING = 0.2
TRAJECTORY_LEGEND_LOC = {
    "P015091": "upper left",
    "E004898": "upper left",
    "E021822": "upper right",
}
RC_PARAMS = {
    "font.family": "DejaVu Sans",
    "font.size": BASE_FONTSIZE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # line widths, marker sizes, ticks and paddings of the scaled panels
    "lines.linewidth": 1.5 * PANEL_SCALE,
    "lines.markersize": 6 * PANEL_SCALE,
    "boxplot.flierprops.markersize": 6 * PANEL_SCALE,
    "patch.linewidth": 1.0 * PANEL_SCALE,
    "axes.linewidth": 0.8 * PANEL_SCALE,
    "grid.linewidth": 0.8 * PANEL_SCALE,
    "xtick.major.width": 0.8 * PANEL_SCALE,
    "ytick.major.width": 0.8 * PANEL_SCALE,
    "xtick.major.size": 3.5 * PANEL_SCALE,
    "ytick.major.size": 3.5 * PANEL_SCALE,
    "xtick.major.pad": 3.5 * PANEL_SCALE,
    "ytick.major.pad": 3.5 * PANEL_SCALE,
    "axes.titlepad": 6.0 * PANEL_SCALE,
    "axes.labelpad": 4.0 * PANEL_SCALE,
}

# definition of the figure, as in the notebooks
HOSTS = ["P015091", "E004898", "E021822"]
SCORE_COLS = ["score_1", "score_2", "score_3"]
EXPOSURES = [1, 2, 3]
MIN_SAMPLES, MAX_SAMPLES = -3.0, 6.0
METRIC = "div_alpha_faith_pd"
MATCHING_VARS = ["month_bin", "delivery_mode", "diet_milk", "diet_weaning"]

# layout in figure fractions: row a on top, rows b (left) and c (right) below
TRAJECTORY_GRID = dict(left=0.05, right=0.995, top=0.975, bottom=0.83, wspace=0.28)
PAIR_GRID = dict(
    left=0.065, right=0.995, top=0.755, bottom=0.065, wspace=0.22, hspace=0.32
)
PAIR_HEIGHT_RATIOS = [1, 0.5]
PAIR_HSPACE = 0.15
LETTER_POSITIONS = {"a": (0.004, 0.998), "b": (0.004, 0.785), "c": (0.5, 0.785)}

FLAGS = flags.FLAGS
flags.DEFINE_string("data_path", "../data/", "data directory, relative to src/")
flags.DEFINE_string(
    "model_name",
    "saved_models_microbial_novel_alpha_div2/id-57",
    "model folder in data_path",
)
flags.DEFINE_string("point_to_evaluate", "best", "model version: 'best' or 'last'")
flags.DEFINE_string(
    "ft_name",
    "ft_vat19_anomaly_v20240806_entero_genus",
    "feature table used for the model",
)
flags.DEFINE_string(
    "abx_ts_name", "ts_vat19_abx_v20240806", "abx time series used for the model"
)
flags.DEFINE_float("limit_months", 24.0, "evaluate scores up to this age in months")
flags.DEFINE_string("stddev_type", "nc_std", "scaling factors: 'nc_std' or 'std'")
flags.DEFINE_integer("moving_avg", 10, "scaling factors: moving average window")
flags.DEFINE_string(
    "duplicates", "--RD-True", "scaling factors: '--RD-True' or '--RD-False'"
)
flags.DEFINE_string("lower_bound", "", "scaling factors: 'lower_bound-1' or ''")
flags.DEFINE_list("hosts", HOSTS, "hosts of the individual trajectories in row a")
flags.DEFINE_string(
    "output",
    None,
    "output PDF (default: score_delineation.pdf in the evaluation folder)",
)


def scores_folder_name(stddev_type, moving_avg, duplicates, lower_bound):
    """Folder of the scores computed with scaling factors, as in the notebooks."""
    return (
        f"using-SF_{stddev_type}_z_scores--moving_avg-{moving_avg}-cummax"
        f"{lower_bound}{duplicates}"
    )


def evaluation_folder_name(point_to_evaluate, stddev_type, moving_avg, duplicates):
    """Evaluation folder of the notebooks (grouped samples, monthly resolution)."""
    return (
        f"evaluation_{point_to_evaluate}_overall_gT_maxresF_{stddev_type}_ma{moving_avg}"
        f"{duplicates.replace('-', '_').lower()}"
    )


def load_inputs(
    data_path,
    model_name,
    point_to_evaluate,
    folder_name,
    ft_name,
    abx_ts_name,
    limit_months,
):
    """Scores of the exposed infants with the matched differences, and their abx events."""
    scores_path = os.path.join(
        data_path,
        model_name,
        "anomaly_detection",
        f"scores_{point_to_evaluate}_normal",
        folder_name,
        "",
    )
    noabx_train, noabx_val, abx_scores_flat, abx_df, _ = get_scores_n_abx_info(
        scores_path,
        ft_name,
        limit_months,
        abx_ts_name,
        no_filter=True,
        path_to_data=os.path.join(data_path, "original_data", ""),
    )
    noabx = pd.concat([noabx_train, noabx_val])
    abx_scores_flat = calculate_matched_metric_n_diff(
        METRIC, abx_scores_flat, noabx, MATCHING_VARS
    )
    return calculate_relative_matched_diff(abx_scores_flat, METRIC), abx_df


def trajectory_legend_kws(host):
    """Legend keyword arguments of the trajectory panel of `host`."""
    return dict(
        labelspacing=TRAJECTORY_LEGEND_SPACING,
        loc=TRAJECTORY_LEGEND_LOC.get(host, "best"),
    )


def _draw_pair(
    fig, cell, abx_scores_flat, abx_df, n, y_axis, style, color, xlabel_note
):
    inner = cell.subgridspec(2, 1, height_ratios=PAIR_HEIGHT_RATIOS, hspace=PAIR_HSPACE)
    ax_box = fig.add_subplot(inner[0])
    ax_count = fig.add_subplot(inner[1], sharex=ax_box)
    selected = _select_samples_around_nth_abx_exposure(
        abx_scores_flat,
        abx_df,
        n=n,
        min_samples=MIN_SAMPLES,
        max_samples=MAX_SAMPLES,
        group_samples=True,
        score_var=y_axis,
        max_resolution=False,
    )
    _plot_score_after_nth_abx_exposure(
        selected,
        "diff_age_nth_abx",
        y_axis,
        n,
        min_samples=MIN_SAMPLES,
        max_samples=MAX_SAMPLES,
        grouped_samples=True,
        max_resolution=False,
        boxplot_color=color,
        axs=(ax_box, ax_count),
        style=style,
        xlabel_note=xlabel_note,
    )
    return ax_box, ax_count


def compose_score_delineation(
    abx_scores_flat, abx_df, hosts, matched_col=REL_MATCHED_DIFF_COL
):
    """
    Composes the figure: trajectories of `hosts` (a), score distributions (b)
    and matched differences `matched_col` (c) around the first three exposures.

    Args:
        abx_scores_flat (pd.DataFrame): Scores and metadata of the exposed
        infants with the matched differences (see `load_inputs`).
        abx_df (pd.DataFrame): Start age of each abx exposure per host.
        hosts (list of str): Hosts shown in row a.
        matched_col (str): Column shown in row c.

    Returns:
        matplotlib.figure.Figure
    """
    with plt.rc_context(RC_PARAMS):
        fig = plt.figure(figsize=FIGURE_SIZE_IN)
        trajectories = fig.add_gridspec(1, len(hosts), **TRAJECTORY_GRID)
        for i, host in enumerate(hosts):
            ax = fig.add_subplot(trajectories[0, i])
            plot_trajectory(
                abx_scores_flat,
                abx_df,
                host,
                SCORE_COLS,
                ax=ax,
                fontsize=BASE_FONTSIZE,
                legend_kws=trajectory_legend_kws(host),
            )

        pairs = fig.add_gridspec(len(EXPOSURES), 2, **PAIR_GRID)
        columns = [
            (lambda n: f"score_{n}", SCORE_PANEL_STYLE, "skyblue"),
            (lambda n: matched_col, MATCHED_PANEL_STYLE, "purple"),
        ]
        for col, (y_axis_of, style, color) in enumerate(columns):
            column_axes = []
            for row, n in enumerate(EXPOSURES):
                ax_box, ax_count = _draw_pair(
                    fig,
                    pairs[row, col],
                    abx_scores_flat,
                    abx_df,
                    n,
                    y_axis_of(n),
                    style,
                    color,
                    xlabel_note=(n == EXPOSURES[-1]),
                )
                if y_axis_of(n) == matched_col:
                    ax_box.set_ylabel(MATCHED_YLABEL, fontsize=BASE_FONTSIZE)
                column_axes += [ax_box, ax_count]
            fig.align_ylabels(column_axes)

        for letter, (x, y) in LETTER_POSITIONS.items():
            fig.text(
                x,
                y,
                letter,
                fontsize=LETTER_FONTSIZE,
                fontweight="bold",
                ha="left",
                va="top",
            )
    return fig


def save_figure(fig, output):
    """Saves the figure as PDF at its page size with TrueType fonts embedded."""
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with plt.rc_context(RC_PARAMS):
        fig.savefig(output, format="pdf")


def main(argv):
    del argv
    folder_name = scores_folder_name(
        FLAGS.stddev_type, FLAGS.moving_avg, FLAGS.duplicates, FLAGS.lower_bound
    )
    abx_scores_flat, abx_df = load_inputs(
        FLAGS.data_path,
        FLAGS.model_name,
        FLAGS.point_to_evaluate,
        folder_name,
        FLAGS.ft_name,
        FLAGS.abx_ts_name,
        FLAGS.limit_months,
    )
    fig = compose_score_delineation(abx_scores_flat, abx_df, FLAGS.hosts)
    output = FLAGS.output or os.path.join(
        FLAGS.data_path,
        FLAGS.model_name,
        "anomaly_detection",
        evaluation_folder_name(
            FLAGS.point_to_evaluate,
            FLAGS.stddev_type,
            FLAGS.moving_avg,
            FLAGS.duplicates,
        ),
        "score_delineation.pdf",
    )
    save_figure(fig, output)
    print("Saved figure to", output)


if __name__ == "__main__":
    app.run(main)
