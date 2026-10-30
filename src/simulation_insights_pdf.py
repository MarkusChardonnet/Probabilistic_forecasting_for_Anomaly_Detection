"""
Reads the panel data of the figure of the anomalies detected in the simulated
time series (Fig. 2 of the manuscript) back from its PDF and redraws the
figure in the manuscript style with figure_simulation_insights.py, for use
when the trained AD modules and anomalous datasets behind the figure are not at
hand. Run from src/:

    uv run python simulation_insights_pdf.py \\
        --source_pdf=<submitted Fig. 2 PDF> --output=simulation_insights_1.pdf

The aggregation weights are read from the colours of the heat maps (viridis,
256 levels) and the value ranges of their colour bars, exact to one colour
level (1/256 of their range); the true paths, the scores, the true anomalies
(orange overlay) and the predicted anomaly regions (red shading) from the
vector paths of the panels. Path and score values are exact at the vertices of
the PDF and interpolated linearly between them on the grid of the paths, where
matplotlib's path simplification dropped vertices; the redrawn curves follow
the published ones within a fraction of a point. The anomaly labels reproduce
the plotted regions.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags
from matplotlib import colormaps
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTChar, LTContainer, LTCurve, LTImage
from pdfminer.pdftypes import resolve1

from figure_simulation_insights import (
    ANOMALY_TYPES,
    NEIGHBOURING_SCORES,
    RC_PARAMS,
    STEPS_AHEAD,
    AnomalyPanel,
    compose_simulation_insights,
)
from figure_style import save_figure

# grid of the synthetic paths: t_k = k / 400 on [0, 1] (dt 0.0025)
GRID_STEPS = 400
# colours of the panels (evaluate_AD.plot_one_path_with_pred)
TRUE_PATH_RGB = (0.0, 0.4196, 0.6431)
ANOMALY_RGB = (1.0, 0.502, 0.0549)
SCORE_RGB = (0.0, 0.502, 0.0)
REGION_RGB = (1.0, 0.0, 0.0)
THRESHOLD_RGB = (1.0, 0.0, 0.0)
WHITE, BLACK = (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)
LEGEND_EDGE_RGB = (0.8, 0.8, 0.8)
COLOR_TOL = 0.01
# geometric tolerances in points and grid steps
POSITION_TOL_PT = 0.6
TICK_LENGTH_PT = (0.5, 6.0)
TICK_LABEL_SPAN_PT = 30.0
GRID_TOL = 0.05
# 256 levels of the colormap of the heat maps as matplotlib writes them (RGB
# bytes, truncated), and the largest accepted distance of a heat-map colour to
# its level
VIRIDIS = (colormaps["viridis"](np.arange(256))[:, :3] * 255).astype(np.uint8).astype(float)
COLORMAP_TOL = 4.0
HEATMAP_SHAPE = (len(STEPS_AHEAD), 2 * NEIGHBOURING_SCORES + 1)

FLAGS = flags.FLAGS
flags.DEFINE_string("source_pdf", None, "PDF of the figure to read the panels from")


def _flatten(container):
    items = []
    for item in container:
        items.append(item)
        if isinstance(item, LTContainer):
            items += _flatten(item)
    return items


def layout_items(pdf_path):
    """All layout objects of the first page of `pdf_path`, nested ones included."""
    return _flatten(next(extract_pages(pdf_path)))


def _rgb(color):
    """RGB triple of a PDF colour given as RGB or as grey level (None otherwise)."""
    if isinstance(color, (int, float)):
        return (float(color),) * 3
    if isinstance(color, (tuple, list)) and len(color) == 1:
        return (float(color[0]),) * 3
    if isinstance(color, (tuple, list)) and len(color) == 3:
        return tuple(float(c) for c in color)
    return None


def _has_color(color, target):
    rgb = _rgb(color)
    return rgb is not None and np.allclose(rgb, target, atol=COLOR_TOL)


def _inside(bbox, frame, margin=1.0):
    return (
        bbox[0] >= frame[0] - margin
        and bbox[1] >= frame[1] - margin
        and bbox[2] <= frame[2] + margin
        and bbox[3] <= frame[3] + margin
    )


def _subpaths(item):
    """End points of the segments of each subpath of a path object."""
    paths, current = [], []
    for op, *points in item.original_path:
        if op == "m":
            if current:
                paths.append(np.array(current))
            current = [points[-1]]
        elif points:
            current.append(points[-1])
    if current:
        paths.append(np.array(current))
    return paths


def parse_number(text):
    """Value of a tick label (matplotlib writes the minus as U+2212)."""
    return float(text.replace("−", "-"))


def linear_map(points):
    """Slope and intercept of the value of (position, value) tick points."""
    positions, values = np.array(points, dtype=float).T
    slope, intercept = np.polyfit(positions, values, 1)
    residual = np.abs(slope * positions + intercept - values).max()
    if residual > 1e-3 * max(np.ptp(values), 1e-12):
        raise ValueError(f"tick values are not linear in their positions: {points}")
    return slope, intercept


def grid_index(t):
    """Grid indices of the times `t`, which must lie on the grid."""
    k = np.rint(np.asarray(t) * GRID_STEPS)
    if np.any(np.abs(np.asarray(t) * GRID_STEPS - k) > GRID_TOL):
        raise ValueError("times off the grid of the synthetic paths")
    return k.astype(int)


def shrink_runs(mask):
    """Labels whose extension by one grid point per side (as plotted) gives `mask`."""
    mask = np.asarray(mask, dtype=int)
    labels = mask.copy()
    padded = np.r_[0, mask, 0]
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1) - 1
    for start, end in zip(starts, ends):
        if start > 0:
            labels[start] = 0
        if end < len(mask) - 1:
            labels[end] = 0
        if labels[start : end + 1].sum() == 0 and 0 < start and end < len(mask) - 1:
            raise ValueError("plotted region too short to come from extended labels")
    return labels


def tick_values(items, x_edge, y_range):
    """
    (position, value) of the ticks drawn outward from the vertical edge
    `x_edge` within `y_range`, read from the tick labels to their right.
    """
    ticks = sorted(
        item.bbox[1]
        for item in items
        if isinstance(item, LTCurve)
        and item.bbox[3] - item.bbox[1] < 0.1
        and abs(item.bbox[0] - x_edge) < POSITION_TOL_PT
        and TICK_LENGTH_PT[0] < item.bbox[2] - item.bbox[0] < TICK_LENGTH_PT[1]
        and y_range[0] - POSITION_TOL_PT <= item.bbox[1] <= y_range[1] + POSITION_TOL_PT
    )
    chars = [
        item
        for item in items
        if isinstance(item, LTChar)
        and item.upright
        and x_edge < item.bbox[0] < x_edge + TICK_LABEL_SPAN_PT
    ]
    points = []
    for y in ticks:
        label = sorted(
            (c for c in chars if abs((c.bbox[1] + c.bbox[3]) / 2 - y) < 0.7 * c.size),
            key=lambda c: c.bbox[0],
        )
        if label:
            points.append((y, parse_number("".join(c.get_text() for c in label))))
    if len(points) < 2:
        raise ValueError(f"fewer than two labelled ticks at x = {x_edge:.1f}")
    return points


def _colorbar_range(items, image):
    """Values at the bottom and top of the colour bar right of a heat map."""
    x1, y0, y1 = image.bbox[2], image.bbox[1], image.bbox[3]
    outlines = [
        item
        for item in items
        if isinstance(item, LTCurve)
        and item.stroke
        and _has_color(item.stroking_color, BLACK)
        and x1 <= item.bbox[0] < x1 + 40
        and item.bbox[3] - item.bbox[1] > 3 * (item.bbox[2] - item.bbox[0])
        and item.bbox[1] < y1
        and item.bbox[3] > y0
    ]
    if not outlines:
        raise ValueError("no colour bar right of a heat map")
    outline = min(outlines, key=lambda item: item.bbox[0])
    slope, intercept = linear_map(
        tick_values(items, outline.bbox[2], (outline.bbox[1], outline.bbox[3]))
    )
    return slope * outline.bbox[1] + intercept, slope * outline.bbox[3] + intercept


def image_rgb(image):
    """Pixels of an 8-bit RGB or palette (Indexed) image as array (height, width, 3)."""
    width, height = image.srcsize
    if image.stream.attrs.get("BitsPerComponent") != 8:
        raise ValueError("only 8-bit images are read")
    data = np.frombuffer(image.stream.get_data(), dtype=np.uint8)
    space = resolve1(image.stream.attrs.get("ColorSpace"))
    if isinstance(space, list) and getattr(space[0], "name", None) == "Indexed":
        lookup = resolve1(space[3])
        table = lookup.get_data() if hasattr(lookup, "get_data") else lookup
        palette = np.frombuffer(table, dtype=np.uint8).reshape(-1, 3)
        return palette[data[: width * height].reshape(height, width)].astype(float)
    return data[: width * height * 3].reshape(height, width, 3).astype(float)


def heatmap_weights(items, image):
    """Aggregation weights of a heat map image, rows = steps ahead, columns = neighbours."""
    width, height = image.srcsize
    pixels = image_rgb(image)
    rows, cols = HEATMAP_SHAPE
    centres = pixels[
        ((np.arange(rows) + 0.5) * height / rows).astype(int)[:, None],
        ((np.arange(cols) + 0.5) * width / cols).astype(int)[None, :],
    ]
    distances = np.linalg.norm(centres[:, :, None, :] - VIRIDIS[None, None], axis=-1)
    nearest = distances.min(axis=-1, keepdims=True)
    if nearest.max() > COLORMAP_TOL:
        raise ValueError("heat-map colours are not viridis levels")
    # adjacent levels can share their byte colour: the mean of the matching levels
    matching = distances <= nearest + 1e-9
    levels = (matching * np.arange(256)).sum(axis=-1) / matching.sum(axis=-1)
    vmin, vmax = _colorbar_range(items, image)
    return vmin + (levels + 0.5) / 256 * (vmax - vmin)


def _largest(items, rgb, frame, legends):
    candidates = [
        item
        for item in items
        if isinstance(item, LTCurve)
        and item.stroke
        and _has_color(item.stroking_color, rgb)
        and _inside(item.bbox, frame)
        and not any(_inside(item.bbox, legend, 0.0) for legend in legends)
    ]
    return candidates


def _region_mask(ranges_t):
    mask = np.zeros(GRID_STEPS + 1, dtype=int)
    for start, end in ranges_t:
        k0, k1 = grid_index(np.array([start, end]))
        mask[k0 : k1 + 1] = 1
    return mask


def path_panel_data(items, frame):
    """Grid, true path, anomaly labels, scores, predicted labels and threshold of a path panel."""
    legends = [
        item.bbox
        for item in items
        if isinstance(item, LTCurve)
        and item.fill
        and _has_color(item.stroking_color, LEGEND_EDGE_RGB)
        and _inside(item.bbox, frame)
    ]
    true_path = max(_largest(items, TRUE_PATH_RGB, frame, legends), key=lambda i: len(i.pts))
    (vertices,) = _subpaths(true_path)
    x_first, x_last = vertices[0, 0], vertices[-1, 0]

    def to_t(x):
        return (np.asarray(x) - x_first) / (x_last - x_first)

    def to_score(y):
        return (np.asarray(y) - frame[1]) / (frame[3] - frame[1])

    slope, intercept = linear_map(tick_values(items, frame[2], (frame[1], frame[3])))
    grid = np.linspace(0.0, 1.0, GRID_STEPS + 1)
    t_path = to_t(vertices[:, 0])
    grid_index(t_path)
    path = np.interp(grid, t_path, slope * vertices[:, 1] + intercept)

    score_curve = max(_largest(items, SCORE_RGB, frame, legends), key=lambda i: len(i.pts))
    (score_vertices,) = _subpaths(score_curve)
    t_scores = to_t(score_vertices[:, 0])
    grid_index(t_scores)
    scores = np.interp(grid, t_scores, to_score(score_vertices[:, 1]), left=np.nan, right=np.nan)

    overlays = [
        sub for item in _largest(items, ANOMALY_RGB, frame, legends) for sub in _subpaths(item)
    ]
    anomaly = _region_mask([(to_t(sub[0, 0]), to_t(sub[-1, 0])) for sub in overlays])
    regions = [
        item.bbox
        for item in items
        if isinstance(item, LTCurve)
        and item.fill
        and _has_color(item.non_stroking_color, REGION_RGB)
        and abs(item.bbox[1] - frame[1]) < POSITION_TOL_PT
        and abs(item.bbox[3] - frame[3]) < POSITION_TOL_PT
        and _inside(item.bbox, frame)
    ]
    predicted = _region_mask([(to_t(x0), to_t(x1)) for x0, _, x1, _ in regions])
    thresholds = [
        item.bbox[1]
        for item in items
        if isinstance(item, LTCurve)
        and item.stroke
        and _has_color(item.stroking_color, THRESHOLD_RGB)
        and item.bbox[3] - item.bbox[1] < 0.1
        and abs(item.bbox[0] - frame[0]) < POSITION_TOL_PT
        and abs(item.bbox[2] - frame[2]) < POSITION_TOL_PT
        and frame[1] < item.bbox[1] < frame[3]
    ]
    return dict(
        t=grid,
        true_path=path,
        anomaly_labels=shrink_runs(anomaly),
        scores=scores,
        predicted_labels=shrink_runs(predicted),
        threshold=float(to_score(thresholds[0])),
    )


def _path_frames(items):
    """Axes frames of the path panels, top to bottom: the smallest white
    rectangles around the true paths."""
    white = [
        item.bbox
        for item in items
        if isinstance(item, LTCurve)
        and item.fill
        and len(item.pts) <= 5
        and _has_color(item.non_stroking_color, WHITE)
    ]
    frames = []
    for curve in items:
        if not (
            isinstance(curve, LTCurve)
            and curve.stroke
            and _has_color(curve.stroking_color, TRUE_PATH_RGB)
            and len(curve.pts) > 50
        ):
            continue
        around = [frame for frame in white if _inside(curve.bbox, frame, 0.0)]
        frames.append(min(around, key=lambda f: (f[2] - f[0]) * (f[3] - f[1])))
    return sorted(set(frames), key=lambda f: -f[3])


def panels_from_pdf(pdf_path):
    """AnomalyPanels of the drift, diffusion, noise and spike rows of the figure."""
    items = layout_items(pdf_path)
    heatmaps = sorted(
        (item for item in items if isinstance(item, LTImage) and item.bbox[2] - item.bbox[0] > 40),
        key=lambda item: -item.bbox[3],
    )
    frames = _path_frames(items)
    if len(heatmaps) != len(ANOMALY_TYPES) or len(frames) != len(ANOMALY_TYPES):
        raise ValueError(
            f"expected {len(ANOMALY_TYPES)} heat maps and path panels, "
            f"found {len(heatmaps)} and {len(frames)}"
        )
    return [
        AnomalyPanel(
            name=name,
            weights=heatmap_weights(items, image),
            steps_ahead=STEPS_AHEAD,
            neighbors=list(range(-NEIGHBOURING_SCORES, NEIGHBOURING_SCORES + 1)),
            **path_panel_data(items, frame),
        )
        for name, image, frame in zip(ANOMALY_TYPES.values(), heatmaps, frames)
    ]


def main(argv):
    del argv
    fig = compose_simulation_insights(panels_from_pdf(FLAGS.source_pdf))
    output = FLAGS.output or os.path.join(
        os.path.dirname(os.path.abspath(FLAGS.source_pdf)), "simulation_insights_restyled.pdf"
    )
    with plt.rc_context(RC_PARAMS):
        save_figure(fig, output)
    print("Saved figure to", output)


if __name__ == "__main__":
    app.run(main)
