"""
Composes the supplementary figures of anomaly score distributions that were
assembled from notebook panels before:

- `baseline.pdf` (Supplementary Fig. S5): scores of the validation set
  (unexposed infants) around cut-offs drawn to match the exposure months of the
  1st to 3rd exposure in the test set (results/analyse_t_horizon_reliability.ipynb),
- `priorexposure.pdf` (Supplementary Fig. S14): scores in monthly bins around
  the 1st to 3rd exposure without grouping the samples before the exposure (a)
  and around the 3rd exposure split by the time since the 2nd (b)
  (results/evaluate_scores.ipynb with group_samples = False),
- `supp_shannon_scores.pdf` (Supplementary Fig. S17): scores of the model
  retrained on Shannon diversity around the 1st to 3rd exposure
  (results/evaluate_scores_shannon.ipynb).

The panels are selected and drawn with the notebooks' functions at the printed
size of the figures (figure_style), each box plot with its own y range, and one
significance legend at the bottom above the footnote. The model and
scaling-factor flags are those of figure_score_delineation.py. Run from src/:

    uv run python figure_supplementary_scores.py --output_dir=<directory>
"""

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from absl import app, flags

import figure_score_splits  # noqa: F401  (defines the figures and output_dir flags)
from figure_score_delineation import pair_style, scores_folder_name
from figure_style import (
    BASE_FONTSIZE,
    LETTER_FONTSIZE,
    TEXT_WIDTH_IN,
    add_panel_letters,
    align_ylabels_at_distance,
    legend_above_footnote,
    rc_params,
)
from figure_style import save_figure as _save_figure
from figure_time_horizon import (
    CUTOFF_DAYS,
    bin_months_since_cutoff,
    load_cutoff_scores,
)
from utils_eval_score import (
    _get_age_at_1st_2nd_3rd_abx_exposure,
    _get_ordinal_suffix,
    _get_step_n_indicator,
    _group_samples_prior_to_cutoff,
    _plot_score_after_nth_abx_exposure,
    _select_samples_around_nth_abx_exposure,
    get_cutoff_value_sample_sizes,
    get_scores_n_abx_info,
)
from utils_t_horizon import select_last_score_per_host_per_bin

RC_PARAMS = rc_params()
MIN_SAMPLES, MAX_SAMPLES = -3.0, 6.0
EXPOSURES = (1, 2, 3)
# seed of the cut-off subsampling and boundary of the time since the 2nd
# exposure, as in the notebooks
CUTOFF_SEED = 8
TIME_SINCE_SECOND_MONTHS = 4
PAIR_HEIGHT_RATIOS = [1, 0.5]
PAIR_HSPACE = 0.25
COUNT_YLABEL = "# samples"
# letters end this far left of the subplots, level with the top of the title
LETTER_OFFSET_PT = 30.0


@dataclass(frozen=True)
class Pair:
    """Box and count plot of `y_axis` per bin of `x_axis` around exposure `n` (0: cut-off)."""

    data: pd.DataFrame
    x_axis: str
    y_axis: str
    n: int
    title: str
    grouped: bool = True
    uniqueness_var_ls: tuple = None


@dataclass(frozen=True)
class FigureSpec:
    """
    Page size, grid, x label of the lowest pair of each column, footnote
    (None: none) and panel letters ("column", "pair" or None) of a figure.
    """

    file_name: str
    figsize: tuple
    grid: dict
    xlabels: tuple
    footnote: str = None
    letters: str = None


FIGURES = {
    "baseline": FigureSpec(
        file_name="baseline.pdf",
        figsize=(TEXT_WIDTH_IN, 5.0),
        grid=dict(left=0.085, right=0.99, top=0.955, bottom=0.135, hspace=0.42),
        xlabels=("Months since cut-off",),
        footnote="(Here -1.0 is last sample prior to cut-off since -3.0 months)",
    ),
    "priorexposure": FigureSpec(
        file_name="priorexposure.pdf",
        figsize=(TEXT_WIDTH_IN, 5.6),
        grid=dict(left=0.085, right=0.99, top=0.935, bottom=0.095, wspace=0.28, hspace=0.6),
        xlabels=("Months since abx exposure", "Months since 3rd abx exposure"),
        letters="column",
    ),
    "supp_shannon_scores": FigureSpec(
        file_name="supp_shannon_scores.pdf",
        figsize=(TEXT_WIDTH_IN, 5.0),
        grid=dict(left=0.085, right=0.99, top=0.955, bottom=0.135, hspace=0.42),
        xlabels=("Months since abx exposure",),
        footnote="(Here -1.0 is last sample prior to abx since -3.0 months)",
        letters="pair",
    ),
}

# model and scaling-factor flags: figure_score_delineation; figures and
# output_dir: figure_score_splits
FLAGS = flags.FLAGS
flags.DEFINE_string(
    "shannon_model_name",
    "saved_models_microbial_novel_alpha_div_shannon/id-1",
    "model retrained on Shannon diversity (Supplementary Fig. S17)",
)
flags.DEFINE_string(
    "cutoff_ft_name",
    "ft_vat19_anomaly_v20240806_entero_family",
    "feature table of the cut-off benchmark, as in its notebook",
)


def exposure_pairs(abx_scores_flat, abx_df, grouped, exposures=EXPOSURES):
    """Pairs of the scores around the 1st to 3rd exposure (as the notebooks' cell 10)."""
    pairs = []
    for n in exposures:
        y_axis = f"score_{n}"
        data = _select_samples_around_nth_abx_exposure(
            abx_scores_flat,
            abx_df,
            n=n,
            min_samples=MIN_SAMPLES,
            max_samples=MAX_SAMPLES,
            group_samples=grouped,
            score_var=y_axis,
            max_resolution=False,
        )
        title = f"{y_axis.capitalize()} before/after {n}{_get_ordinal_suffix(n)} abx exposure"
        pairs.append(Pair(data, "diff_age_nth_abx", y_axis, n, title, grouped))
    return pairs


def time_since_second_pairs(abx_scores_flat, abx_df, threshold=TIME_SINCE_SECOND_MONTHS):
    """
    Pairs of the scores around the 3rd exposure (samples before it not grouped)
    of the infants whose 3rd exposure followed the 2nd within `threshold`
    months or later, as in results/evaluate_scores.ipynb.
    """
    ages = _get_age_at_1st_2nd_3rd_abx_exposure(abx_df)
    labels = [f"<= {threshold} months", f"> {threshold} months"]
    category = pd.cut(
        ages["age_3rd_abx"] - ages["age_2nd_abx"],
        bins=[-np.inf, threshold, np.inf],
        labels=labels,
        right=True,
    )
    (third,) = exposure_pairs(abx_scores_flat, abx_df, grouped=False, exposures=(3,))
    pairs = []
    for label in labels:
        hosts = category.index[category == label]
        data = third.data[third.data["host_id"].isin(hosts)].copy()
        title = f"{third.title}\ntime since 2nd abx exposure {label}"
        pairs.append(Pair(data, third.x_axis, third.y_axis, 3, title, grouped=False))
    return pairs


def sample_cutoffs(mapping, sample_sizes, seed=CUTOFF_SEED):
    """
    Infant cut-offs drawn per cut-off month, `sample_sizes[month]` of them with
    `seed` (utils_t_horizon.sample_from_each_group as applied by the notebook).
    """
    chosen = []
    for month, group in mapping.groupby("cutoff_month"):
        size = sample_sizes.get(month, 0)
        if len(group) < size:
            raise ValueError(f"{size} cut-offs needed in month {month}, {len(group)} available")
        chosen.append(group.sample(n=size, random_state=seed))
    return pd.concat(chosen)


def matched_cutoff_pairs(abx_scores_flat, abx_df, cutoff_scores, seed=CUTOFF_SEED):
    """
    Pairs of the validation scores around cut-offs drawn to match the months of
    the 1st to 3rd exposure in the test set (analyse_t_horizon_reliability.ipynb).
    """
    binned = bin_months_since_cutoff(cutoff_scores, MIN_SAMPLES, MAX_SAMPLES)
    mapping = binned[["host_id", "cutoff_month"]].drop_duplicates()
    group_step, indicator = _get_step_n_indicator(False)
    pairs = []
    for n in EXPOSURES:
        _, sample_sizes = get_cutoff_value_sample_sizes(
            abx_scores_flat, abx_df, n, MIN_SAMPLES, MAX_SAMPLES, True
        )
        chosen = sample_cutoffs(mapping, sample_sizes, seed)
        sampled = pd.merge(binned, chosen, on=["host_id", "cutoff_month"], how="inner")
        sampled = select_last_score_per_host_per_bin(sampled)
        sampled = _group_samples_prior_to_cutoff(
            sampled, "months_since_cutoff", ["host_id", "cutoff_month"], MIN_SAMPLES, group_step, indicator
        )
        title = f"Score before/after cut-off with {n}{_get_ordinal_suffix(n)} abx matched cutoffs"
        pairs.append(Pair(sampled, "months_since_cutoff", "score", 0, title, True, ("use_obs_until_day",)))
    return pairs


def _draw_pair(fig, cell, pair):
    inner = cell.subgridspec(2, 1, height_ratios=PAIR_HEIGHT_RATIOS, hspace=PAIR_HSPACE)
    ax_box = fig.add_subplot(inner[0])
    ax_count = fig.add_subplot(inner[1], sharex=ax_box)
    _plot_score_after_nth_abx_exposure(
        pair.data,
        pair.x_axis,
        pair.y_axis,
        pair.n,
        min_samples=MIN_SAMPLES,
        max_samples=MAX_SAMPLES,
        grouped_samples=pair.grouped,
        max_resolution=False,
        uniqueness_var_ls=list(pair.uniqueness_var_ls) if pair.uniqueness_var_ls else None,
        axs=(ax_box, ax_count),
        style=pair_style(ax_box, pair.data[pair.y_axis]),
        xlabel_note=False,
    )
    ax_box.set_title(pair.title, fontsize=BASE_FONTSIZE)
    ax_count.set_ylabel(COUNT_YLABEL, fontsize=BASE_FONTSIZE)
    return ax_box, ax_count


def compose_pair_figure(columns, spec):
    """
    Composes a figure of `columns` (lists of Pair, top to bottom) as defined by
    `spec`: each pair a box plot over its count plot, the x label under the
    lowest pair of each column, the y labels at one distance from their axes,
    the panel letters and, centred under the columns, one significance legend
    above the footnote.
    """
    nrows = max(len(pairs) for pairs in columns)
    with plt.rc_context(RC_PARAMS):
        fig = plt.figure(figsize=spec.figsize)
        outer = fig.add_gridspec(nrows, len(columns), **spec.grid)
        boxes, axes, letter_boxes, rows = [], [], [], {}
        for col, pairs in enumerate(columns):
            for row, pair in enumerate(pairs):
                ax_box, ax_count = _draw_pair(fig, outer[row, col], pair)
                ax_count.set_xlabel(
                    spec.xlabels[col] if row == len(pairs) - 1 else "", fontsize=BASE_FONTSIZE
                )
                boxes.append(ax_box)
                axes += [ax_box, ax_count]
                rows.setdefault(row, []).append(ax_box)
                if spec.letters == "pair" or (spec.letters == "column" and row == 0):
                    letter_boxes.append((row, ax_box))
        align_ylabels_at_distance(fig, axes)
        fig.draw_without_rendering()
        offset_px = LETTER_OFFSET_PT * fig.dpi / 72
        letters = {}
        # letters at the left of their subplots, level with the highest title
        # of their row
        for letter, (row, ax) in zip("abcdefgh", letter_boxes):
            x = max(ax.get_window_extent().x0 - offset_px, 0.005 * fig.bbox.width)
            y = max(box.title.get_window_extent().y1 for box in rows[row])
            letters[letter] = (x / fig.bbox.width, y / fig.bbox.height)
        add_panel_letters(fig, letters, LETTER_FONTSIZE)
        stars = any("*" in t.get_text() for ax in boxes for t in ax.texts)
        left = min(ax.get_position().x0 for ax in axes)
        right = max(ax.get_position().x1 for ax in axes)
        legend_above_footnote(fig, (left + right) / 2, spec.footnote, stars=stars)
    return fig


def save_figure(fig, output):
    """Saves the figure as PDF at its page size with TrueType fonts embedded."""
    with plt.rc_context(RC_PARAMS):
        _save_figure(fig, output)


def _scores(model_name, ft_name):
    folder = scores_folder_name(FLAGS.stddev_type, FLAGS.moving_avg, FLAGS.duplicates, FLAGS.lower_bound)
    scores_path = os.path.join(
        FLAGS.data_path, model_name, "anomaly_detection", f"scores_{FLAGS.point_to_evaluate}_normal", folder, ""
    )
    _, _, abx_scores_flat, abx_df, _ = get_scores_n_abx_info(
        scores_path,
        ft_name,
        FLAGS.limit_months,
        FLAGS.abx_ts_name,
        no_filter=True,
        path_to_data=os.path.join(FLAGS.data_path, "original_data", ""),
    )
    return abx_scores_flat, abx_df


def figure_columns(name):
    """Columns of pairs of the figure `name` from the data of the flags."""
    if name == "baseline":
        abx_scores_flat, abx_df = _scores(FLAGS.model_name, FLAGS.cutoff_ft_name)
        folder = scores_folder_name(FLAGS.stddev_type, FLAGS.moving_avg, FLAGS.duplicates, FLAGS.lower_bound)
        cutoff_path = os.path.join(
            FLAGS.data_path, FLAGS.model_name, "anomaly_detection",
            "reliability_eval-val-noabx_best_normal", folder, "",
        )
        cutoffs = load_cutoff_scores(cutoff_path, CUTOFF_DAYS, FLAGS.limit_months)
        return [matched_cutoff_pairs(abx_scores_flat, abx_df, cutoffs)]
    if name == "priorexposure":
        abx_scores_flat, abx_df = _scores(FLAGS.model_name, FLAGS.ft_name)
        return [
            exposure_pairs(abx_scores_flat, abx_df, grouped=False),
            time_since_second_pairs(abx_scores_flat, abx_df),
        ]
    abx_scores_flat, abx_df = _scores(FLAGS.shannon_model_name, FLAGS.ft_name)
    return [exposure_pairs(abx_scores_flat, abx_df, grouped=True)]


def main(argv):
    del argv
    output_dir = FLAGS.output_dir or os.path.join(FLAGS.data_path, FLAGS.model_name, "anomaly_detection")
    for name in FLAGS.figures or FIGURES:
        spec = FIGURES[name]
        fig = compose_pair_figure(figure_columns(name), spec)
        path = os.path.join(output_dir, spec.file_name)
        save_figure(fig, path)
        plt.close(fig)
        print("Saved figure to", path)


if __name__ == "__main__":
    app.run(main)
