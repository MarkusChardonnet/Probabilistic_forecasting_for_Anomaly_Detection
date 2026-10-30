"""
Composes the figures of the anomaly score distributions around an antibiotic
exposure split by exposure characteristics: by exposure duration and by age at
exposure for the 1st and 2nd exposure (Fig. 5 of the manuscript), by type of
antibiotic for the 2nd exposure (Supplementary Fig. S6) and by milk diet for
1st exposures in the first year of life (Supplementary Fig. S7). The groups are
defined as in results/evaluate_scores.ipynb and drawn with its panel function
at the printed size of the figures (figure_style); all box plots of a figure
share one y-axis range, as do all count plots. The model and scaling-factor
flags are those of figure_score_delineation.py. Run from src/:

    uv run python figure_score_splits.py --output_dir=<directory>
"""

import os
from dataclasses import dataclass, replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from absl import app, flags

from figure_score_delineation import evaluation_folder_name, scores_folder_name
from figure_style import (  # noqa: F401  (star gaps: used by the tests)
    BASE_FONTSIZE,
    LETTER_FONTSIZE,
    PAIRED_STAR_GAP_PT,
    STAR_FONTSIZE,
    STAR_TOP_GAP_PT,
    TEXT_WIDTH_IN,
    UNPAIRED_STAR_GAP_PT,
    add_panel_letters,
    align_ylabels_at_distance,
    legend_above_footnote,
    rc_params,
    star_layout,
)
from figure_style import save_figure as _save_figure
from utils_eval_score import (
    PanelStyle,
    _plot_score_after_nth_abx_exposure,
    _select_samples_around_nth_abx_exposure,
    box_ylim,
    get_scores_n_abx_info,
)

RC_PARAMS = rc_params()
MIN_SAMPLES, MAX_SAMPLES = -3.0, 6.0
PAIR_HEIGHT_RATIOS = [1, 0.5]
# gap between box and count plot, which the y labels of both may reach into
PAIR_HSPACE = 0.3
# panels without legends (the figure draws one); headroom and star rows are set
# per figure by `star_layout` (figure_style)
SPLIT_PANEL_STYLE = PanelStyle(
    fontsize=BASE_FONTSIZE,
    star_fontsize=STAR_FONTSIZE,
    star_va="center",
    legend="none",
)
COUNT_YLABEL = "# samples"
# the note on the grouped pre-exposure bin is written once at the bottom of the
# figure, the legend centred right above it, both centred under the panels
FOOTNOTE = "(Here -1.0 is last sample prior to abx since -3.0 months)"
# vertical space between the two pairs of a group (tick labels and the
# one-line title of the lower pair); the x label is written under the lowest
# pair of a group only
GROUP_PAIR_HSPACE = 0.37
# panel letters: left of and above the two-line title of the group
LETTER_OFFSET = (0.075, 0.055)

DURATION_BINS = [-np.inf, 7, np.inf]
DURATION_LABELS = ["< 7 days", ">= 7 days"]
AGE_BINS = [-np.inf, 12, np.inf]
AGE_LABELS = ["<= 12 months", "12 - 24 months"]
TOP_ABX_TYPE = "Penicillin"
MILK_DIETS = [["mixed", "bd"], ["fd"]]

FLAGS = flags.FLAGS
flags.DEFINE_list("figures", None, "figures to draw (default: all)")
flags.DEFINE_string(
    "output_dir", None, "output directory (default: the evaluation folder)"
)


def split_by_duration(selected):
    """
    Samples of the hosts whose last exposure lasted less than or at least 7
    days, as in the notebook: a host enters every group one of its samples
    falls in, which keeps the pre-exposure samples with their host.
    """
    category = pd.cut(
        selected["abx_any_last_dur_days"],
        DURATION_BINS,
        labels=DURATION_LABELS,
        right=False,
    )
    groups = []
    for label in DURATION_LABELS:
        hosts = selected.loc[category == label, "host_id"].unique()
        frame = selected[selected["host_id"].isin(hosts)].copy()
        groups.append((f"duration: {label}", frame))
    return groups


def split_by_age(selected):
    """Samples by the age of the infant at the exposure (up to or above 12 months)."""
    category = pd.cut(selected["age_nth_abx"], AGE_BINS, labels=AGE_LABELS, right=True)
    return [(f"age: {label}", selected[category == label].copy()) for label in AGE_LABELS]


def split_by_abx_type(selected):
    """Samples by the type of the antibiotic of the exposure: penicillin or others."""
    is_top = selected["abx_type"] == TOP_ABX_TYPE
    return [
        (f"Abx type: {TOP_ABX_TYPE}", selected[is_top].copy()),
        ("Abx type: Others", selected[~is_top].copy()),
    ]


def split_by_milk_diet_in_first_year(selected):
    """
    Samples of exposures in the first year of life by the milk diet at the
    sample: breast milk dominant or mixed, and formula dominant.
    """
    first_year = selected[selected["age_nth_abx"] <= AGE_BINS[1]]
    return [
        (
            f"age: {AGE_LABELS[0]}, diet: {diets}",
            first_year[first_year["diet_milk"].isin(diets)].copy(),
        )
        for diets in MILK_DIETS
    ]


@dataclass(frozen=True)
class SplitGroup:
    """Pairs of box and count plots of the `split` groups around the n-th exposure."""

    n: int
    split: callable
    letter: str = None


@dataclass(frozen=True)
class FigureSpec:
    """A figure of `groups` in `ncols` columns, `figsize` in inches."""

    file_name: str
    figsize: tuple
    groups: tuple
    ncols: int
    grid: dict


FIGURES = {
    "scoresplit": FigureSpec(
        file_name="scoresplit.pdf",
        # 5.7 in: figure and caption fit the text height of a page
        figsize=(TEXT_WIDTH_IN, 5.7),
        groups=(
            SplitGroup(1, split_by_duration, "a"),
            SplitGroup(2, split_by_duration, "b"),
            SplitGroup(1, split_by_age, "c"),
            SplitGroup(2, split_by_age, "d"),
        ),
        ncols=2,
        grid=dict(left=0.085, right=0.99, top=0.95, bottom=0.118, wspace=0.26, hspace=0.285),
    ),
    "penicillin_2nd": FigureSpec(
        file_name="penicillin_2nd.pdf",
        figsize=(TEXT_WIDTH_IN, 4.4),
        groups=(SplitGroup(2, split_by_abx_type),),
        ncols=1,
        grid=dict(left=0.085, right=0.99, top=0.93, bottom=0.152),
    ),
    "scoresplit_age_diet": FigureSpec(
        file_name="scoresplit_age_diet.pdf",
        figsize=(TEXT_WIDTH_IN, 4.4),
        groups=(SplitGroup(1, split_by_milk_diet_in_first_year),),
        ncols=1,
        grid=dict(left=0.085, right=0.99, top=0.93, bottom=0.152),
    ),
}


def load_scores(
    data_path, model_name, point_to_evaluate, folder_name, ft_name, abx_ts_name, limit_months
):
    """Scores and metadata of the exposed infants, and their abx events."""
    scores_path = os.path.join(
        data_path,
        model_name,
        "anomaly_detection",
        f"scores_{point_to_evaluate}_normal",
        folder_name,
        "",
    )
    _, _, abx_scores_flat, abx_df, _ = get_scores_n_abx_info(
        scores_path,
        ft_name,
        limit_months,
        abx_ts_name,
        no_filter=True,
        path_to_data=os.path.join(data_path, "original_data", ""),
    )
    return abx_scores_flat, abx_df


def group_subsets(abx_scores_flat, abx_df, group):
    """Score column of `group` and its samples per compared subset."""
    y_axis = f"score_{group.n}"
    selected = _select_samples_around_nth_abx_exposure(
        abx_scores_flat,
        abx_df,
        n=group.n,
        min_samples=MIN_SAMPLES,
        max_samples=MAX_SAMPLES,
        group_samples=True,
        score_var=y_axis,
        max_resolution=False,
    )
    return y_axis, group.split(selected)


def two_line_title(ax, tag):
    """Moves `tag` of the panel title to a second line."""
    head = ax.get_title()[: -len(tag)].rstrip()
    ax.set_title(f"{head}\n{tag}", fontsize=BASE_FONTSIZE)


def _group_axes(fig, cell, n_subsets):
    """Box and count axes of the `n_subsets` pairs of a group in `cell`."""
    inner = cell.subgridspec(n_subsets, 1, hspace=GROUP_PAIR_HSPACE)
    axes = []
    for i in range(n_subsets):
        pair = inner[i].subgridspec(
            2, 1, height_ratios=PAIR_HEIGHT_RATIOS, hspace=PAIR_HSPACE
        )
        ax_box = fig.add_subplot(pair[0])
        axes.append((ax_box, fig.add_subplot(pair[1], sharex=ax_box)))
    return axes


def compose_split_figure(abx_scores_flat, abx_df, spec):
    """
    Composes the figure `spec`: its groups of box and count plots in `spec.ncols`
    columns, with one y-axis range for all box plots and one for all count
    plots, the two-line title on the upper pair of a group and the split only
    on the lower ones, the x label under the lowest pair of each group, the y
    labels at one distance from their axes, the panel letters of the groups,
    and at the bottom the footnote with one significance legend (with stars
    only if a star is drawn) centred above it.

    Args:
        abx_scores_flat (pd.DataFrame): Scores and metadata of the exposed
        infants (see `load_scores`).
        abx_df (pd.DataFrame): Start age, type and reason of each abx exposure
        per host.
        spec (FigureSpec): Figure definition, see `FIGURES`.

    Returns:
        matplotlib.figure.Figure
    """
    nrows = int(np.ceil(len(spec.groups) / spec.ncols))
    drawn = [group_subsets(abx_scores_flat, abx_df, group) for group in spec.groups]
    scores = pd.concat([frame[y_axis] for y_axis, subsets in drawn for _, frame in subsets])
    with plt.rc_context(RC_PARAMS):
        fig = plt.figure(figsize=spec.figsize)
        cells = fig.add_gridspec(nrows, spec.ncols, **spec.grid)
        group_axes = [
            _group_axes(fig, cells[divmod(i, spec.ncols)], len(subsets))
            for i, (_, subsets) in enumerate(drawn)
        ]
        # all box plots are as high as the first one
        height_pt = group_axes[0][0][0].get_window_extent().height * 72 / fig.dpi
        y_min, y_max = box_ylim(scores, 1.0)
        top, star_rows = star_layout(y_min, y_max, height_pt)
        style = replace(
            SPLIT_PANEL_STYLE,
            ylim=(y_min, top),
            ylim_factor=top / y_max,
            star_rows=star_rows,
        )
        columns = {col: [] for col in range(spec.ncols)}
        letters = {}
        for i, (group, (y_axis, subsets), axes) in enumerate(
            zip(spec.groups, drawn, group_axes)
        ):
            for j, ((tag, frame), (ax_box, ax_count)) in enumerate(zip(subsets, axes)):
                _plot_score_after_nth_abx_exposure(
                    frame,
                    "diff_age_nth_abx",
                    y_axis,
                    group.n,
                    tag=tag,
                    min_samples=MIN_SAMPLES,
                    max_samples=MAX_SAMPLES,
                    grouped_samples=True,
                    max_resolution=False,
                    axs=(ax_box, ax_count),
                    style=style,
                    xlabel_note=False,
                )
                if j == 0:
                    two_line_title(ax_box, tag)
                else:
                    ax_box.set_title(tag, fontsize=BASE_FONTSIZE)
                ax_count.set_ylabel(COUNT_YLABEL, fontsize=BASE_FONTSIZE)
                if j < len(subsets) - 1:
                    ax_count.set_xlabel("")
                columns[i % spec.ncols] += [ax_box, ax_count]
            if group.letter is not None:
                position = cells[divmod(i, spec.ncols)].get_position(fig)
                letters[group.letter] = (
                    max(position.x0 - LETTER_OFFSET[0], 0.0),
                    min(position.y1 + LETTER_OFFSET[1], 1.0),
                )
        count_axes = [ax_count for axes in group_axes for _, ax_count in axes]
        count_top = max(ax.get_ylim()[1] for ax in count_axes)
        for ax in count_axes:
            ax.set_ylim(0, count_top)
        all_axes = [ax for axes in columns.values() for ax in axes]
        align_ylabels_at_distance(fig, all_axes)
        boxes = [ax_box for axes in group_axes for ax_box, _ in axes]
        stars = any("*" in t.get_text() for ax in boxes for t in ax.texts)
        left = min(ax.get_position().x0 for ax in all_axes)
        right = max(ax.get_position().x1 for ax in all_axes)
        legend_above_footnote(fig, (left + right) / 2, FOOTNOTE, stars=stars)
        add_panel_letters(fig, letters, LETTER_FONTSIZE)
    return fig


def save_figure(fig, output):
    """Saves the figure as PDF at its page size with TrueType fonts embedded."""
    with plt.rc_context(RC_PARAMS):
        _save_figure(fig, output)


def save_figures(abx_scores_flat, abx_df, output_dir, names=None):
    """Draws and saves the figures `names` (default: all) into `output_dir`."""
    paths = []
    for name in names or FIGURES:
        spec = FIGURES[name]
        fig = compose_split_figure(abx_scores_flat, abx_df, spec)
        path = os.path.join(output_dir, spec.file_name)
        save_figure(fig, path)
        plt.close(fig)
        paths.append(path)
    return paths


def main(argv):
    del argv
    folder_name = scores_folder_name(
        FLAGS.stddev_type, FLAGS.moving_avg, FLAGS.duplicates, FLAGS.lower_bound
    )
    abx_scores_flat, abx_df = load_scores(
        FLAGS.data_path,
        FLAGS.model_name,
        FLAGS.point_to_evaluate,
        folder_name,
        FLAGS.ft_name,
        FLAGS.abx_ts_name,
        FLAGS.limit_months,
    )
    output_dir = FLAGS.output_dir or os.path.join(
        FLAGS.data_path,
        FLAGS.model_name,
        "anomaly_detection",
        evaluation_folder_name(
            FLAGS.point_to_evaluate, FLAGS.stddev_type, FLAGS.moving_avg, FLAGS.duplicates
        ),
    )
    for path in save_figures(abx_scores_flat, abx_df, output_dir, FLAGS.figures):
        print("Saved figure to", path)


if __name__ == "__main__":
    app.run(main)
