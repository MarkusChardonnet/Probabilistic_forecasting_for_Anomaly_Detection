"""
Alluvial figure of the sensitivity analysis for the recording resolution of the
antibiotic exposure ages (manuscript revision, reviewer 2, point 3): where the
samples of the published analysis (Fig. 4b, recorded exposure ages) end up when
all recorded exposure ages are shifted by a fixed number of months. One panel
per exposure (1st to 3rd); each band connects the monthly bin of a set of
samples as recorded (left) with their bin under the shifted dating (right) and
is coloured by what happens to them. Bin -1 is the last sample within 3 months
before the dated exposure and after the previous one; 'not used' collects the
samples outside the selection under one of the two datings (one sample per
infant and bin, right censoring at the next exposure, 6-month window,
24-month limit). Requires the scores of the shifted re-scoring run (see the
README). Run from src/:

    uv run python figure_exposure_shift_flows.py --flow_shift=-0.5

The remaining flags are those of figure_score_delineation.py (data_path,
point_to_evaluate, limit_months, the scaling-factor options and output).
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from absl import app, flags
from matplotlib.patches import Patch, PathPatch, Rectangle
from matplotlib.path import Path

from figure_exposure_shift import (
    DAYS_PER_MONTH,
    EXPOSURE_LABELS,
    EXPOSURES,
    ROW_LETTERS,
    load_variant,
    save_figure,  # noqa: F401  (re-exported for callers and tests)
    select_exposure,
    variant_names,
)
from figure_score_delineation import RC_PARAMS, evaluation_folder_name, scores_folder_name
from make_shifted_exposure_tables import shift_token

NODES = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, "not used"]
FATES = [
    "same bin",
    "neighbouring bin",
    "crosses the cut-off",
    "leaves the selection",
    "enters the selection",
]
COLORS = {
    "same bin": "#d2d1ca",
    "neighbouring bin": "#2a78d6",
    "crosses the cut-off": "#eb6834",
    "leaves the selection": "#1baf7a",
    "enters the selection": "#eda100",
}
# horizontal position of a band's count label and the smallest band that is labelled
LABEL_X = {
    "same bin": 0.5,
    "neighbouring bin": 0.3,
    "crosses the cut-off": 0.5,
    "leaves the selection": 0.72,
    "enters the selection": 0.72,
}
MIN_LABEL = {
    "same bin": 3,
    "neighbouring bin": 2,
    "crosses the cut-off": 2,
    "leaves the selection": 4,
    "enters the selection": 4,
}
INK, INK_SECONDARY = "#0b0b0b", "#52514e"
FIGURE_SIZE_IN = (7.2, 7.4)
BASE_FONTSIZE = 7
GRID = dict(wspace=0.6, left=0.09, right=0.94, top=0.94, bottom=0.11)
NODE_WIDTH = 0.1  # in band-length units (bands run from x=0 to x=1)
NODE_GAP = 4.0  # in samples

FLAGS = flags.FLAGS  # the data and output flags are defined by figure_score_delineation
flags.DEFINE_float(
    "flow_shift", -0.5, "shift of the exposure ages of the compared re-scoring run, in months"
)


def node_name(node):
    """ASCII name of a node for tables: '-1', '0', ..., 'not used'."""
    return node if node == "not used" else f"{int(node)}"


def node_label(node):
    """Label of a node in the figure."""
    return "not used" if node == "not used" else f"bin {node_name(node)}".replace("-", "−")


def fate(src, tgt):
    """What happens to a sample that moves from node `src` (recorded) to node `tgt`."""
    if src == "not used":
        return "enters the selection"
    if tgt == "not used":
        return "leaves the selection"
    if src == tgt:
        return "same bin"
    if (src == -1.0) != (tgt == -1.0):
        return "crosses the cut-off"
    return "neighbouring bin"


def flow_matrix(recorded, shifted):
    """
    Counts of samples per (bin as recorded, bin under the shift), samples identified
    by host and day; `recorded` and `shifted` are selections around the same n-th
    exposure as returned by `select_exposure` for the two datings.
    """
    key = ["host_id", "day"]
    bins_rec = recorded.set_index(key)["diff_age_nth_abx"].astype(float)
    bins_sh = shifted.set_index(key)["diff_age_nth_abx"].astype(float)
    matrix = pd.DataFrame(0, index=NODES, columns=NODES, dtype=int)
    for sample, b in bins_rec.items():
        target = bins_sh[sample] if sample in bins_sh.index else "not used"
        matrix.loc[b, target] += 1
    for sample in bins_sh.index.difference(bins_rec.index):
        matrix.loc["not used", bins_sh[sample]] += 1
    return matrix


def flows_to_long(matrices):
    """Non-zero cells of the matrices (keyed by exposure) as one table."""
    rows = [
        dict(exposure=n, bin_recorded=node_name(src), bin_shifted=node_name(tgt), n=int(count))
        for n, matrix in sorted(matrices.items())
        for src in NODES
        for tgt in NODES
        for count in [matrix.loc[src, tgt]]
        if count > 0
    ]
    return pd.DataFrame(rows, columns=["exposure", "bin_recorded", "bin_shifted", "n"])


def band_centre(x, c0, c1):
    """Height of the centre line of a band at horizontal position x in [0, 1]."""
    lo, hi = 0.0, 1.0
    for _ in range(40):
        t = (lo + hi) / 2
        x_t = 1.5 * (1 - t) ** 2 * t + 1.5 * (1 - t) * t**2 + t**3
        lo, hi = (t, hi) if x_t < x else (lo, t)
    t = (lo + hi) / 2
    weight = (1 - t) ** 3 + 3 * (1 - t) ** 2 * t
    return c0 * weight + c1 * (1 - weight)


def _ribbon(ax, top0, bot0, top1, bot1, color):
    verts = [(0, top0), (0.5, top0), (0.5, top1), (1, top1), (1, bot1), (0.5, bot1), (0.5, bot0), (0, bot0), (0, top0)]
    codes = [Path.MOVETO] + [Path.CURVE4] * 3 + [Path.LINETO] + [Path.CURVE4] * 3 + [Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", alpha=0.75, zorder=1))


def _node_positions(totals):
    positions, y = {}, 0.0
    for node in NODES:
        height = totals[node]
        if height == 0:
            continue
        positions[node] = (y, y - height)  # top, bottom; stacked downwards
        y -= height + NODE_GAP
    return positions


def draw_flows(ax, matrix, title, shifted_header):
    """One alluvial panel: nodes as recorded (left) and under the shift (right)."""
    out_totals, in_totals = matrix.sum(axis=1), matrix.sum(axis=0)
    left, right = _node_positions(out_totals), _node_positions(in_totals)
    for node, (top, bot) in left.items():
        ax.add_patch(Rectangle((-NODE_WIDTH, bot), NODE_WIDTH, top - bot, facecolor=INK_SECONDARY, edgecolor="none", zorder=3))
        name = "enter" if node == "not used" else node_label(node)
        ax.text(-NODE_WIDTH - 0.03, (top + bot) / 2, f"{name}  {int(out_totals[node])}", ha="right", va="center")
    for node, (top, bot) in right.items():
        ax.add_patch(Rectangle((1, bot), NODE_WIDTH, top - bot, facecolor=INK_SECONDARY, edgecolor="none", zorder=3))
        name = "leave" if node == "not used" else node_label(node)
        ax.text(1 + NODE_WIDTH + 0.03, (top + bot) / 2, f"{int(in_totals[node])}  {name}", ha="left", va="center")
    out_cursor = {node: top for node, (top, _) in left.items()}
    in_cursor = {node: top for node, (top, _) in right.items()}
    for src in NODES:
        for tgt in NODES:
            count = matrix.loc[src, tgt]
            if count == 0:
                continue
            top0, bot0 = out_cursor[src], out_cursor[src] - count
            top1, bot1 = in_cursor[tgt], in_cursor[tgt] - count
            out_cursor[src], in_cursor[tgt] = bot0, bot1
            kind = fate(src, tgt)
            _ribbon(ax, top0, bot0, top1, bot1, COLORS[kind])
            if count >= MIN_LABEL[kind]:
                x = LABEL_X[kind]
                ax.text(x, band_centre(x, (top0 + bot0) / 2, (top1 + bot1) / 2), str(int(count)), ha="center", va="center",
                        fontsize=BASE_FONTSIZE - 1.5, color=INK, zorder=5,
                        bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.85))
    height = out_totals.sum() + NODE_GAP * (len(left) - 1)
    ax.set_xlim(-0.8, 1.8)
    ax.set_ylim(-height - 2, 6)
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=BASE_FONTSIZE + 1, fontweight="bold", pad=8)
    ax.text(-NODE_WIDTH, 2.5, "as recorded", ha="right", va="bottom", color=INK_SECONDARY)
    ax.text(1 + NODE_WIDTH, 2.5, shifted_header, ha="left", va="bottom", color=INK_SECONDARY)


def compose_flow_figure(matrices, shift_months):
    """Three alluvial panels (exposures 1 to 3) for the shift of `shift_months`."""
    days = round(shift_months * DAYS_PER_MONTH)
    header = f"dated {days:+d} d".replace("-", "−")
    direction = "earlier" if shift_months < 0 else "later"
    with plt.rc_context({**RC_PARAMS, "font.size": BASE_FONTSIZE, "text.color": INK}):
        fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE_IN, gridspec_kw=GRID)
        for ax, n in zip(axes, EXPOSURES):
            draw_flows(ax, matrices[n], f"{ROW_LETTERS[n - 1]}   {EXPOSURE_LABELS[n - 1]}", header)
        handles = [Patch(facecolor=COLORS[k], label=k) for k in FATES]
        fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=BASE_FONTSIZE - 0.5,
                   handlelength=1.2, columnspacing=1.2, bbox_to_anchor=(0.5, 0.005))
        fig.text(
            0.5, 0.062,
            f"Each band is a set of samples of Fig. 4b (left) and the bin it falls into when all exposures are dated "
            f"{abs(days)} days {direction} (right).\nBin −1: last sample within 3 months before the dated exposure and "
            "after the previous one. Enter/leave: not selected under the other dating.\nBands are labelled with their "
            "size when they hold at least 2 (moving), 3 (same bin) or 4 (enter/leave) samples; all sizes are in the CSV.",
            ha="center", va="center", fontsize=BASE_FONTSIZE - 1, color=INK_SECONDARY,
        )
    return fig


def main(argv):
    del argv
    folder_name = scores_folder_name(FLAGS.stddev_type, FLAGS.moving_avg, FLAGS.duplicates, FLAGS.lower_bound)
    shift = FLAGS.flow_shift
    variants = {}
    for s in (0.0, shift):
        names = variant_names(s)
        variants[s] = load_variant(FLAGS.data_path, names["model_name"], FLAGS.point_to_evaluate, folder_name,
                                   names["ft_name"], names["abx_ts_name"], FLAGS.limit_months)
    matrices = {
        n: flow_matrix(select_exposure(*variants[0.0], n), select_exposure(*variants[shift], n)) for n in EXPOSURES
    }
    output = FLAGS.output or os.path.join(
        FLAGS.data_path,
        variant_names(0.0)["model_name"],
        "anomaly_detection",
        evaluation_folder_name(FLAGS.point_to_evaluate, FLAGS.stddev_type, FLAGS.moving_avg, FLAGS.duplicates),
        f"exposure_shift_flows_{shift_token(shift)}.pdf",
    )
    save_figure(compose_flow_figure(matrices, shift), output)
    table = flows_to_long(matrices)
    table.to_csv(os.path.splitext(output)[0] + ".csv", index=False)
    print("Saved figure to", output)
    print(table.to_string(index=False))


if __name__ == "__main__":
    app.run(main)
