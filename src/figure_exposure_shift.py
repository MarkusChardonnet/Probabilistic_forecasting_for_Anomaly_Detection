"""
Composes the supplementary figure of the sensitivity analysis for the recording
resolution of the antibiotic exposure ages: anomaly score distributions around
the first three antibiotic exposures (rows a-c, box plots as Fig. 4b without
the count panels) with all recorded exposure ages shifted by -15 days, as
recorded and shifted by +15 days (columns). The columns use the scores of the
published model id-57 and of its re-scorings on the shifted datasets (see the
README). The script also writes the per-bin summary and the months of maximal
median score. Run from src/:

    uv run python figure_exposure_shift.py --output=supp_exposure_shift.pdf

The flags are those of figure_score_delineation.py (data_path,
point_to_evaluate, limit_months, the scaling-factor options and output).
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from absl import app, flags

from figure_score_delineation import evaluation_folder_name, scores_folder_name
from figure_style import legend_above_footnote, rc_params
from make_shifted_exposure_tables import shift_token
from utils_eval_score import (
    PanelStyle,
    _plot_score_after_nth_abx_exposure,
    _select_samples_around_nth_abx_exposure,
    get_scores_n_abx_info,
    perform_significance_tests,
)

DAYS_PER_MONTH = 30.437
SHIFTS = [-0.5, 0.0, 0.5]
EXPOSURES = [1, 2, 3]
EXPOSURE_LABELS = ["1st exposure", "2nd exposure", "3rd exposure"]
ROW_LETTERS = ["a", "b", "c"]
MIN_SAMPLES, MAX_SAMPLES = -3.0, 6.0
REFERENCE_BIN = -1.0

# manuscript text width, 6 pt text, stars and letters slightly larger; line
# widths, ticks and paddings of the published figure (scaled by 0.4)
FIGURE_SIZE_IN = (5.17, 5.4)
BASE_FONTSIZE = 6
STAR_FONTSIZE = 8
LETTER_FONTSIZE = 9
RC_PARAMS = rc_params(BASE_FONTSIZE, 0.4)
# star colours of the panel helper (utils_eval_score._create_subplot) and the
# heights of the unpaired and paired stars as multiples of the largest value of
# a row of the grid; the upper y-limit of a row is YLIM_FACTOR times that value
# (as Fig. 4b)
UNPAIRED_STAR_COLOR, PAIRED_STAR_COLOR = "sandybrown", "darkgreen"
STAR_HEIGHTS = {UNPAIRED_STAR_COLOR: 1.2, PAIRED_STAR_COLOR: 1.1}
YLIM_FACTOR = 1.7
# no legend in the panels: one legend below the grid
STYLE = PanelStyle(
    fontsize=BASE_FONTSIZE,
    ylim_factor=YLIM_FACTOR,
    star_rows=(STAR_HEIGHTS[UNPAIRED_STAR_COLOR], STAR_HEIGHTS[PAIRED_STAR_COLOR]),
    star_fontsize=STAR_FONTSIZE,
    star_va="center",
    legend="none",
)
GRID = dict(left=0.1, right=0.99, top=0.95, bottom=0.13, wspace=0.25, hspace=0.15)
LETTER_X, LABEL_X = 0.005, 0.03
XLABEL = "Months since exposure"
# the note on the grouped pre-exposure bin is written once below the grid, the
# legend centred right above it
FOOTNOTE = f"Bin {REFERENCE_BIN} is the last sample prior to the exposure since {MIN_SAMPLES} months"

SUMMARY_COLUMNS = [
    "variant",
    "exposure",
    "bin",
    "n_samples",
    "median_score",
    "p_unpaired",
    "n_unpaired",
    "p_paired",
    "n_paired",
]
PEAK_COLUMNS = [
    "variant",
    "exposure",
    "peak_bin",
    "n_peak",
    "significant_bins_unpaired_005",
    "significant_bins_unpaired_01",
    "significant_bins_paired_005",
    "significant_bins_paired_01",
    "n_month0",
    "median_month0",
]

FLAGS = flags.FLAGS  # defined by figure_score_delineation


def shift_label(shift_months):
    """Row label of a shift: 'as recorded' or the shift in days."""
    if shift_months == 0:
        return "as recorded"
    days = round(shift_months * DAYS_PER_MONTH)
    return f"exposure ages {days:+d} d".replace("-", "−")


def variant_names(
    shift_months,
    model_folder="saved_models_microbial_novel_alpha_div2",
    model_id=57,
    ft_name="ft_vat19_anomaly_v20240806_entero_genus",
    abx_ts_name="ts_vat19_abx_v20240806",
):
    """Label, model folder and table names of the variant with `shift_months`."""
    suffix = "" if shift_months == 0 else f"_shift_{shift_token(shift_months)}"
    return dict(
        label=shift_label(shift_months),
        model_name=f"{model_folder}{suffix}/id-{model_id}",
        ft_name=f"{ft_name}{suffix}",
        abx_ts_name=f"{abx_ts_name}{suffix}",
    )


def load_variant(
    data_path, model_name, point_to_evaluate, folder_name, ft_name, abx_ts_name, limit_months
):
    """Scores of the exposed infants and the exposure table of one variant."""
    scores_path = (
        f"{data_path}{model_name}/anomaly_detection/"
        f"scores_{point_to_evaluate}_normal/{folder_name}/"
    )
    _, _, abx_scores_flat, abx_df, _ = get_scores_n_abx_info(
        scores_path,
        ft_name,
        limit_months,
        abx_ts_name,
        no_filter=True,
        path_to_data=f"{data_path}original_data/",
    )
    return abx_scores_flat, abx_df


def select_exposure(abx_scores_flat, abx_df, n):
    """Samples around the n-th exposure in the monthly bins of Fig. 4b."""
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


def _draw_panel(fig, cell, abx_scores_flat, abx_df, n):
    """
    Box plot of the scores around the n-th exposure in `cell`, drawn by the
    panel helper of Fig. 4b without its count panel; returns the axes and the
    largest plotted score.
    """
    ax = fig.add_subplot(cell)
    counts = fig.add_axes([0.0, 0.0, 1e-3, 1e-3])  # the helper's count panel, discarded
    data = select_exposure(abx_scores_flat, abx_df, n)
    _plot_score_after_nth_abx_exposure(
        data,
        "diff_age_nth_abx",
        f"score_{n}",
        n,
        min_samples=MIN_SAMPLES,
        max_samples=MAX_SAMPLES,
        grouped_samples=True,
        max_resolution=False,
        boxplot_color="skyblue",
        axs=(ax, counts),
        style=STYLE,
        xlabel_note=False,
    )
    counts.remove()
    # the helper hides the x tick labels of a box plot above its count panel,
    # and seaborn then its x label: both are shown again
    ax.tick_params(labelbottom=True)
    ax.xaxis.label.set_visible(True)
    return ax, data[f"score_{n}"].max()


def _label_row(fig, ax, letter, label):
    top, bottom = ax.get_position().y1, ax.get_position().y0
    fig.text(
        LETTER_X, top, letter, fontsize=LETTER_FONTSIZE, fontweight="bold", ha="left", va="top"
    )
    fig.text(
        LABEL_X,
        (top + bottom) / 2,
        label,
        fontsize=BASE_FONTSIZE,
        rotation=90,
        ha="center",
        va="center",
    )


def _star_texts(ax):
    return [t for t in ax.texts if t.get_text() in ("*", "**")]


def _set_row_ylim(axes, maximum, factor):
    bottom = min(ax.get_ylim()[0] for ax in axes)
    for ax in axes:
        ax.set_ylim(bottom, factor * maximum)


def _align_row(axes, maximum):
    """Stars of a row at common heights above its largest value; shared y range."""
    for ax in axes:
        for star in _star_texts(ax):
            star.set_y(STAR_HEIGHTS[star.get_color()] * maximum)
    _set_row_ylim(axes, maximum, YLIM_FACTOR)


def compose_exposure_shift_figure(variants):
    """
    Composes the figure: one row per exposure (a-c: first to third), one column
    per variant `(label, abx_scores_flat, abx_df)` in the given order; each
    panel the score box plot of Fig. 4b, with common star heights and y range
    per row, and below the grid one legend centred above the footnote.

    Returns:
        matplotlib.figure.Figure
    """
    with plt.rc_context({**RC_PARAMS, "font.size": BASE_FONTSIZE}):
        fig = plt.figure(figsize=FIGURE_SIZE_IN)
        grid = fig.add_gridspec(len(EXPOSURES), len(variants), **GRID)
        last_row = len(EXPOSURES) - 1
        for row, n in enumerate(EXPOSURES):
            axes, maxima = [], []
            for col, (label, abx_scores_flat, abx_df) in enumerate(variants):
                ax, maximum = _draw_panel(fig, grid[row, col], abx_scores_flat, abx_df, n)
                ax.set_title(label if row == 0 else "", fontsize=BASE_FONTSIZE)
                ax.set_xlabel(XLABEL if row == last_row else "", fontsize=BASE_FONTSIZE)
                if col != 0:
                    ax.set_ylabel("")
                axes.append(ax)
                maxima.append(maximum)
            _label_row(fig, axes[0], ROW_LETTERS[row], EXPOSURE_LABELS[row])
            _align_row(axes, max(maxima))
        legend_above_footnote(
            fig, (GRID["left"] + GRID["right"]) / 2, FOOTNOTE, fontsize=BASE_FONTSIZE
        )
    return fig


def legend_hides_stars(fig, axes=None):
    """True if a legend of `axes` (default: all axes of `fig`) overlaps a significance star."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes = fig.axes if axes is None else axes
    boxes = [
        ax.get_legend().get_window_extent(renderer) for ax in axes if ax.get_legend() is not None
    ]
    stars = [t for ax in axes for t in _star_texts(ax)]
    return any(
        box.overlaps(star.get_window_extent(renderer)) for box in boxes for star in stars
    )


def summarize_variants(variants):
    """
    Per variant, exposure and bin: sample count, median score and the one-sided
    tests against the pre-exposure bin that are drawn as stars in the panels.
    """
    rows = []
    for label, abx_scores_flat, abx_df in variants:
        for n in EXPOSURES:
            score_col = f"score_{n}"
            selected = select_exposure(abx_scores_flat, abx_df, n)
            bins = sorted(float(b) for b in selected["diff_age_nth_abx"].unique())
            tests = pd.DataFrame()
            if REFERENCE_BIN in bins:
                tests = perform_significance_tests(
                    selected,
                    REFERENCE_BIN,
                    list(bins),
                    metric_to_evaluate=score_col,
                    x_axis="diff_age_nth_abx",
                )
            for b in bins:
                in_bin = selected.loc[selected["diff_age_nth_abx"] == b, score_col]
                stats = tests.loc[b] if b in tests.index else pd.Series(dtype=float)
                rows.append(
                    dict(
                        variant=label,
                        exposure=n,
                        bin=b,
                        n_samples=int(in_bin.shape[0]),
                        median_score=in_bin.median(),
                        p_unpaired=stats.get("P-value unpaired", float("nan")),
                        n_unpaired=stats.get("# samples unpaired", float("nan")),
                        p_paired=stats.get("P-value paired", float("nan")),
                        n_paired=stats.get("# samples paired", float("nan")),
                    )
                )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _bins_below(post, column, threshold):
    return ",".join(str(b) for b in post.loc[post[column] < threshold, "bin"])


def peak_months(summary):
    """
    Per variant and exposure: the post-exposure month with the maximal median
    score (and its sample count) and the months significant by each test.
    """
    rows = []
    for (variant, n), group in summary.groupby(["variant", "exposure"], sort=False):
        post = group[group["bin"] >= 0].sort_values("bin")
        month0 = post[post["bin"] == 0.0]
        peak = post.loc[post["median_score"].idxmax()] if len(post) else None
        rows.append(
            dict(
                variant=variant,
                exposure=n,
                peak_bin=peak["bin"] if peak is not None else float("nan"),
                n_peak=int(peak["n_samples"]) if peak is not None else 0,
                significant_bins_unpaired_005=_bins_below(post, "p_unpaired", 0.05),
                significant_bins_unpaired_01=_bins_below(post, "p_unpaired", 0.1),
                significant_bins_paired_005=_bins_below(post, "p_paired", 0.05),
                significant_bins_paired_01=_bins_below(post, "p_paired", 0.1),
                n_month0=int(month0["n_samples"].iloc[0]) if len(month0) else 0,
                median_month0=month0["median_score"].iloc[0] if len(month0) else float("nan"),
            )
        )
    return pd.DataFrame(rows, columns=PEAK_COLUMNS)


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
    variants = []
    for shift in SHIFTS:
        names = variant_names(shift)
        abx_scores_flat, abx_df = load_variant(
            FLAGS.data_path,
            names["model_name"],
            FLAGS.point_to_evaluate,
            folder_name,
            names["ft_name"],
            names["abx_ts_name"],
            FLAGS.limit_months,
        )
        variants.append((names["label"], abx_scores_flat, abx_df))

    output = FLAGS.output or os.path.join(
        FLAGS.data_path,
        variant_names(0.0)["model_name"],
        "anomaly_detection",
        evaluation_folder_name(
            FLAGS.point_to_evaluate, FLAGS.stddev_type, FLAGS.moving_avg, FLAGS.duplicates
        ),
        "supp_exposure_shift.pdf",
    )
    fig = compose_exposure_shift_figure(variants)
    if legend_hides_stars(fig):
        print("WARNING: a legend overlaps significance stars")
    save_figure(fig, output)
    out_dir = os.path.dirname(os.path.abspath(output))
    summary = summarize_variants(variants)
    summary.to_csv(os.path.join(out_dir, "exposure_shift_summary.csv"), index=False)
    peaks = peak_months(summary)
    peaks.to_csv(os.path.join(out_dir, "exposure_shift_peaks.csv"), index=False)
    print("Saved figure to", output)
    print(peaks.to_string(index=False))


if __name__ == "__main__":
    app.run(main)
