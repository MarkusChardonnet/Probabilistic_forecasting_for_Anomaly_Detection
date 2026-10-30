"""
Shared style of the manuscript figures composed by the figure_*.py scripts:
figures are drawn at their printed width (the text width of the manuscript),
with one font size for all axis text, bold panel letters and TrueType fonts
embedded in the PDF.
"""

import os

from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

# manuscript text width in inches (372 pt)
TEXT_WIDTH_IN = 5.17
# one size for titles, axis labels, tick labels, legends and footnotes
BASE_FONTSIZE = 7
LETTER_FONTSIZE = 10
STAR_FONTSIZE = 9
FONT_FAMILY = "DejaVu Sans"
# line widths, marker sizes, ticks and paddings relative to matplotlib's defaults
LINE_SCALE = 0.7
# colours of the significance stars and count outlines (utils_eval_score)
UNPAIRED_COLOR, PAIRED_COLOR = "sandybrown", "darkgreen"
SIGNIFICANCE_LABELS = ["unpaired to -1.0", "paired to -1.0"]
# star rows of a box plot in points: the paired stars stand above the largest
# value, the unpaired ones above those, and the text box of the unpaired stars
# (about as high as the star font size) ends well below the axes top
PAIRED_STAR_GAP_PT = 3.0
UNPAIRED_STAR_GAP_PT = 5.0
STAR_TOP_GAP_PT = 8.0
# legends drawn over data are see-through
LEGEND_FRAME_ALPHA = 0.4
# gap between the y labels and the widest y tick labels of their axes
YLABEL_PAD_PT = 2.0
# the footnote stands at the bottom of a figure, the significance legend
# centred right above it
FOOTNOTE_Y = 0.006
LEGEND_FOOTNOTE_GAP_PT = 1.0
BOTTOM_LEGEND_KWS = dict(
    loc="lower center",
    ncol=2,
    frameon=False,
    borderaxespad=0.0,
    handlelength=1.4,
    columnspacing=0.9,
    handletextpad=0.4,
)


def rc_params(fontsize=BASE_FONTSIZE, line_scale=LINE_SCALE):
    """Matplotlib rc parameters of a figure drawn at its printed size."""
    return {
        "font.family": FONT_FAMILY,
        "font.size": fontsize,
        "axes.titlesize": fontsize,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "legend.fontsize": fontsize,
        "legend.title_fontsize": fontsize,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "lines.linewidth": 1.5 * line_scale,
        "lines.markersize": 6 * line_scale,
        "boxplot.flierprops.markersize": 6 * line_scale,
        "patch.linewidth": 1.0 * line_scale,
        "axes.linewidth": 0.8 * line_scale,
        "grid.linewidth": 0.8 * line_scale,
        "xtick.major.width": 0.8 * line_scale,
        "ytick.major.width": 0.8 * line_scale,
        "xtick.major.size": 3.5 * line_scale,
        "ytick.major.size": 3.5 * line_scale,
        "xtick.major.pad": 3.5 * line_scale,
        "ytick.major.pad": 3.5 * line_scale,
        "axes.titlepad": 6.0 * line_scale,
        "axes.labelpad": 4.0 * line_scale,
    }


def add_panel_letters(fig, positions, fontsize=LETTER_FONTSIZE):
    """Bold panel letters at `positions` (letter -> (x, y) in figure fractions)."""
    return {
        letter: fig.text(
            x, y, letter, fontsize=fontsize, fontweight="bold", ha="left", va="top"
        )
        for letter, (x, y) in positions.items()
    }


def significance_handles(stars=True, markersize=5, linewidth=1.2):
    """
    Legend handles of the two tests: star (box plots) and line (outlines of the
    count plots) per test, or the lines only when no star is drawn.
    """
    handles = []
    for color in (UNPAIRED_COLOR, PAIRED_COLOR):
        line = Line2D([0], [0], color=color, lw=linewidth)
        if stars:
            star = Line2D([0], [0], color=color, marker="*", markersize=markersize, ls="None")
            handles.append((star, line))
        else:
            handles.append(line)
    return handles


def significance_legend(
    parent, stars=True, fontsize=BASE_FONTSIZE, markersize=5, linewidth=1.2, **legend_kws
):
    """
    One legend of the significance tests for all box and count plots of a
    figure, drawn on `parent` (a figure or an axes); `stars=False` when no star
    is drawn.
    """
    return parent.legend(
        significance_handles(stars, markersize, linewidth),
        SIGNIFICANCE_LABELS,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)},
        fontsize=fontsize,
        **legend_kws,
    )


def legend_above_footnote(
    fig,
    centre_x,
    footnote,
    stars=True,
    footnote_y=FOOTNOTE_Y,
    fontsize=BASE_FONTSIZE,
    markersize=5,
    linewidth=1.2,
):
    """
    Writes `footnote` (None: no footnote) centred at `centre_x` (figure
    fraction) at the bottom of `fig` and the significance legend centred right
    above it; returns the legend and the footnote text.
    """
    legend_y = footnote_y
    text = None
    if footnote is not None:
        text = fig.text(
            centre_x, footnote_y, footnote, fontsize=fontsize, ha="center", va="bottom"
        )
        fig.draw_without_rendering()
        gap_px = LEGEND_FOOTNOTE_GAP_PT * fig.dpi / 72
        legend_y = (text.get_window_extent().y1 + gap_px) / fig.bbox.height
    legend = significance_legend(
        fig,
        stars=stars,
        fontsize=fontsize,
        markersize=markersize,
        linewidth=linewidth,
        bbox_to_anchor=(centre_x, legend_y),
        **BOTTOM_LEGEND_KWS,
    )
    return legend, text


def star_layout(y_min, y_max, height_pt):
    """
    Upper y-limit and star rows (unpaired, paired; multiples of `y_max`) of a
    box plot `height_pt` points high: the paired stars stand PAIRED_STAR_GAP_PT
    above `y_max`, the unpaired ones UNPAIRED_STAR_GAP_PT above those and
    STAR_TOP_GAP_PT below the axes top.
    """
    gaps_pt = PAIRED_STAR_GAP_PT + UNPAIRED_STAR_GAP_PT + STAR_TOP_GAP_PT
    span = (y_max - y_min) * height_pt / (height_pt - gaps_pt)
    per_pt = span / height_pt
    paired = y_max + PAIRED_STAR_GAP_PT * per_pt
    unpaired = paired + UNPAIRED_STAR_GAP_PT * per_pt
    return y_min + span, (unpaired / y_max, paired / y_max)


def drawn_tick_labels(axis):
    """Labels of the major ticks of `axis` inside its view interval (the drawn ones)."""
    low, high = sorted(axis.get_view_interval())
    return [
        tick.label1
        for tick in axis.get_major_ticks()
        if low <= tick.get_loc() <= high and tick.label1.get_visible()
    ]


def align_ylabels_at_distance(fig, axes, pad_pt=YLABEL_PAD_PT):
    """
    Places the y labels of `axes` at one distance from their axes, as close as
    the widest y tick labels of all `axes` allow, and returns it in points.
    """
    fig.draw_without_rendering()
    widest = 0.0
    for ax in axes:
        left = ax.get_window_extent().x0
        for label in drawn_tick_labels(ax.yaxis):
            widest = max(widest, left - label.get_window_extent().x0)
    distance = widest + pad_pt * fig.dpi / 72
    for ax in axes:
        ax.yaxis.set_label_coords(-distance / ax.get_window_extent().width, 0.5)
    return distance * 72 / fig.dpi


def save_figure(fig, output):
    """Saves `fig` as PDF at its page size, creating the output folder."""
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    fig.savefig(output, format="pdf")
