"""
Composes the score-delineation figure of the manuscript (Fig. 4): individual
anomaly score trajectories (a), anomaly score distributions (b) and
covariate-matched alpha diversity differences in percent of the matched mean
(c) around the first three antibiotic exposures. The panels are drawn with the
functions used by the notebooks results/evaluate_scores.ipynb and
results/evaluate_matched_alpha.ipynb at the printed size of the figure
(figure_style). Run from src/:

    uv run python figure_score_delineation.py --output=score_delineation.pdf
"""

import os
from dataclasses import replace

import matplotlib.pyplot as plt
import pandas as pd
from absl import app, flags
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory, offset_copy

from figure_style import (
    BASE_FONTSIZE,
    LETTER_FONTSIZE,
    STAR_FONTSIZE,
    TEXT_WIDTH_IN,
    add_panel_letters,
    align_ylabels_at_distance,
    legend_above_footnote,
    rc_params,
    star_layout,
)
from figure_style import save_figure as _save_figure
from utils_eval_score import (
    REL_MATCHED_DIFF_COL,
    PanelStyle,
    _get_ordinal_suffix,
    _plot_score_after_nth_abx_exposure,
    _select_samples_around_nth_abx_exposure,
    box_ylim,
    calculate_matched_metric_n_diff,
    calculate_relative_matched_diff,
    get_scores_n_abx_info,
    plot_trajectory,
)

FIGURE_SIZE_IN = (TEXT_WIDTH_IN, 5.8)
RC_PARAMS = rc_params()
# box and count plots without legends (panels b and c share one); y-limits and
# star rows are set per box plot by `pair_style`
PANEL_STYLE = PanelStyle(
    fontsize=BASE_FONTSIZE,
    star_fontsize=STAR_FONTSIZE,
    star_va="center",
    legend="none",
)
# one title per panel; each box plot names its exposure above the red line
COLUMN_TITLES = [
    "Score before/after n-th abx exposure",
    "Matched diversity before/after n-th abx exposure",
]
MATCHED_YLABEL = "Δ diversity\n[%]"
COUNT_YLABEL = "n"
XLABEL = "Months since abx exposure"
# y labels of a column one point from its widest tick labels
YLABEL_PAD_PT = 1.0
# the note on the grouped pre-exposure bin is written once below panels b and c
FOOTNOTE = "(Here -1.0 is last sample prior to abx since -3.0 months)"
EXPOSURE_COLOR = "darkred"
EXPOSURE_LABEL_OFFSET_PT = 1.0
TITLE_OFFSET_PT = 2.0

# definition of the figure, as in the notebooks
HOSTS = ["P015091", "E004898", "E021822"]
SCORE_COLS = ["score_1", "score_2", "score_3"]
EXPOSURES = [1, 2, 3]
MIN_SAMPLES, MAX_SAMPLES = -3.0, 6.0
METRIC = "div_alpha_faith_pd"
MATCHING_VARS = ["month_bin", "delivery_mode", "diet_milk", "diet_weaning"]

# layout in figure fractions: row a with its legend below it, panels b (left)
# and c (right) below, their shared legend above the footnote at the bottom;
# the right margin leaves room for the title of panel c, which is wider than
# its subplots
# row a spans the subplots of panels b and c, so that its legend is centred
# under both
TRAJECTORY_GRID = dict(left=0.065, right=0.962, top=0.95, bottom=0.86, wspace=0.3)
PAIR_GRID = dict(
    left=0.065, right=0.962, top=0.72, bottom=0.118, wspace=0.27, hspace=0.31
)
PAIR_HEIGHT_RATIOS = [1, 0.45]
PAIR_HSPACE = 0.25
LETTER_A = (0.005, 0.998)
# letters b and c end this far left of their panel (y labels and title)
LETTER_OFFSET_PT = 8.0
# legend of panel a: centred below its subplots like the legend of b and c
TRAJECTORY_LEGEND_GAP_PT = 1.5
TRAJECTORY_LEGEND = dict(
    loc="upper center",
    ncol=4,
    frameon=False,
    borderaxespad=0.0,
    handlelength=1.4,
    columnspacing=0.9,
    handletextpad=0.4,
)

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
    "output PDF (default: the figure's file name in the evaluation folder)",
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


def trajectory_title(host):
    """Title of the trajectory panel of `host` (fits the printed panel width)."""
    return f"Host ID: {host}"


def trajectory_legend_handles(ax, score_cols):
    """Handles and labels of the shared trajectory legend, taken from one panel."""
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    handles = [by_label[col] for col in score_cols]
    handles.append(Line2D([0], [0], color=EXPOSURE_COLOR))
    return handles, list(score_cols) + ["abx event"]


def pair_style(ax_box, values, base=PANEL_STYLE):
    """
    Style of a box plot: y-limits from `values` with the star rows of
    `star_layout` between the largest value and the axes top.
    """
    height_pt = ax_box.get_window_extent().height * 72 / ax_box.figure.dpi
    y_min, y_max = box_ylim(values, 1.0)
    top, star_rows = star_layout(y_min, y_max, height_pt)
    return replace(base, ylim=(y_min, top), ylim_factor=top / y_max, star_rows=star_rows)


def label_exposure_line(ax, n):
    """Writes the ordinal of the n-th exposure above the red exposure line of `ax`."""
    (line,) = [line for line in ax.lines if line.get_color() == EXPOSURE_COLOR]
    transform = offset_copy(
        blended_transform_factory(ax.transData, ax.transAxes),
        fig=ax.figure,
        y=EXPOSURE_LABEL_OFFSET_PT,
        units="points",
    )
    return ax.text(
        line.get_xdata()[0],
        1.0,
        f"{n}{_get_ordinal_suffix(n)}",
        transform=transform,
        ha="center",
        va="bottom",
        color=EXPOSURE_COLOR,
        fontsize=BASE_FONTSIZE,
    )


def draw_pair(
    fig,
    cell,
    abx_scores_flat,
    abx_df,
    n,
    y_axis,
    color,
    height_ratios=PAIR_HEIGHT_RATIOS,
    hspace=PAIR_HSPACE,
):
    """Box plot and count plot of `y_axis` around the n-th exposure in `cell`."""
    inner = cell.subgridspec(2, 1, height_ratios=height_ratios, hspace=hspace)
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
        style=pair_style(ax_box, selected[y_axis]),
        xlabel_note=False,
    )
    return ax_box, ax_count


def _fraction(fig, x_px, y_px):
    """Figure fractions of a point in pixels."""
    return x_px / fig.bbox.width, y_px / fig.bbox.height


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
        trajectory_axes = []
        for i, host in enumerate(hosts):
            ax = fig.add_subplot(trajectories[0, i])
            plot_trajectory(
                abx_scores_flat,
                abx_df,
                host,
                SCORE_COLS,
                ax=ax,
                fontsize=BASE_FONTSIZE,
                title=trajectory_title(host),
                show_legend=False,
            )
            trajectory_axes.append(ax)

        pairs = fig.add_gridspec(len(EXPOSURES), 2, **PAIR_GRID)
        columns = [
            (lambda n: f"score_{n}", "skyblue", None),
            (lambda n: matched_col, "purple", MATCHED_YLABEL),
        ]
        column_axes = []
        for col, (y_axis_of, color, ylabel) in enumerate(columns):
            axes = []
            for row, n in enumerate(EXPOSURES):
                ax_box, ax_count = draw_pair(
                    fig, pairs[row, col], abx_scores_flat, abx_df, n, y_axis_of(n), color
                )
                ax_box.set_title("")
                if ylabel is not None:
                    ax_box.set_ylabel(ylabel, fontsize=BASE_FONTSIZE)
                label_exposure_line(ax_box, n)
                ax_count.set_ylabel(COUNT_YLABEL, fontsize=BASE_FONTSIZE)
                ax_count.set_xlabel(
                    XLABEL if n == EXPOSURES[-1] else "", fontsize=BASE_FONTSIZE
                )
                axes += [ax_box, ax_count]
            column_axes.append(axes)
        for axes in column_axes:
            align_ylabels_at_distance(fig, axes, pad_pt=YLABEL_PAD_PT)

        # one title per panel, centred over its subplots, above the exposure
        # ordinals of its top row; the letters at the panels' left
        fig.draw_without_rendering()
        letters = {"a": LETTER_A}
        title_gap_px = TITLE_OFFSET_PT * fig.dpi / 72
        offset_px = LETTER_OFFSET_PT * fig.dpi / 72
        for letter, title, axes in zip("bc", COLUMN_TITLES, column_axes):
            top_box = axes[0].get_window_extent()
            ordinal_top = max(t.get_window_extent().y1 for t in axes[0].texts)
            x, y = _fraction(fig, (top_box.x0 + top_box.x1) / 2, ordinal_top + title_gap_px)
            text = fig.text(x, y, title, fontsize=BASE_FONTSIZE, ha="center", va="bottom")
            labels_left = min(ax.yaxis.label.get_window_extent().x0 for ax in axes)
            left = min(labels_left, text.get_window_extent().x0)
            letter_x, letter_y = _fraction(fig, left - offset_px, text.get_window_extent().y1)
            letters[letter] = (max(letter_x, LETTER_A[0]), letter_y)

        # legends of panel a (below its subplots) and of panels b and c (above
        # the footnote), centred under the subplots of panels b and c
        centre_px = (
            column_axes[0][0].get_window_extent().x0
            + column_axes[1][0].get_window_extent().x1
        ) / 2
        centre, _ = _fraction(fig, centre_px, 0)
        xlabels_bottom = min(ax.xaxis.label.get_window_extent().y0 for ax in trajectory_axes)
        _, legend_top = _fraction(
            fig, 0, xlabels_bottom - TRAJECTORY_LEGEND_GAP_PT * fig.dpi / 72
        )
        handles, labels = trajectory_legend_handles(trajectory_axes[0], SCORE_COLS)
        fig.legend(
            handles,
            labels,
            fontsize=BASE_FONTSIZE,
            bbox_to_anchor=(centre, legend_top),
            **TRAJECTORY_LEGEND,
        )
        legend_above_footnote(fig, centre, FOOTNOTE)
        add_panel_letters(fig, letters, LETTER_FONTSIZE)
    return fig


def save_figure(fig, output):
    """Saves the figure as PDF at its page size with TrueType fonts embedded."""
    with plt.rc_context(RC_PARAMS):
        _save_figure(fig, output)


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
