import os
import sys
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(src_dir)

from figure_style import (  # noqa: E402
    BASE_FONTSIZE,
    LEGEND_FRAME_ALPHA,
    LETTER_FONTSIZE,
    LINE_SCALE,
    PAIRED_STAR_GAP_PT,
    STAR_FONTSIZE,
    STAR_TOP_GAP_PT,
    TEXT_WIDTH_IN,
    UNPAIRED_STAR_GAP_PT,
    YLABEL_PAD_PT,
    FOOTNOTE_Y,
    add_panel_letters,
    align_ylabels_at_distance,
    legend_above_footnote,
    drawn_tick_labels,
    rc_params,
    save_figure,
    significance_legend,
    star_layout,
)

# rc parameters of the published Fig. 4 script (6 pt text, line widths x 0.4)
FORMER_FIG4_RC_PARAMS = {
    "font.family": "DejaVu Sans",
    "font.size": 6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "lines.linewidth": 1.5 * 0.4,
    "lines.markersize": 6 * 0.4,
    "boxplot.flierprops.markersize": 6 * 0.4,
    "patch.linewidth": 1.0 * 0.4,
    "axes.linewidth": 0.8 * 0.4,
    "grid.linewidth": 0.8 * 0.4,
    "xtick.major.width": 0.8 * 0.4,
    "ytick.major.width": 0.8 * 0.4,
    "xtick.major.size": 3.5 * 0.4,
    "ytick.major.size": 3.5 * 0.4,
    "xtick.major.pad": 3.5 * 0.4,
    "ytick.major.pad": 3.5 * 0.4,
    "axes.titlepad": 6.0 * 0.4,
    "axes.labelpad": 4.0 * 0.4,
}


def _text_of(artist):
    return getattr(artist, "get_text", lambda: type(artist).__name__)()


def label_artists(fig):
    """
    Texts and legends that must stay on the page and apart from each other:
    figure texts and legends, axes titles, axis labels, drawn tick labels,
    axes texts except significance stars (the two star rows may share a text
    box) and axes legends.
    """
    artists = list(fig.texts) + list(fig.legends)
    for ax in fig.axes:
        artists += [ax.title, ax.xaxis.label, ax.yaxis.label]
        artists += drawn_tick_labels(ax.xaxis) + drawn_tick_labels(ax.yaxis)
        artists += [t for t in ax.texts if "*" not in t.get_text()]
        if ax.get_legend() is not None:
            artists.append(ax.get_legend())
    return [a for a in artists if a.get_visible() and _text_of(a) != ""]


def assert_texts_inside(fig):
    """Every text and legend of `fig` lies inside the page after drawing."""
    fig.canvas.draw()
    page = fig.bbox
    outside = []
    for artist in label_artists(fig):
        box = artist.get_window_extent()
        if (
            box.x0 < page.x0 - 0.5
            or box.x1 > page.x1 + 0.5
            or box.y0 < page.y0 - 0.5
            or box.y1 > page.y1 + 0.5
        ):
            outside.append((_text_of(artist), box.extents))
    if outside:
        raise AssertionError(f"text outside the page: {outside}")


def assert_texts_do_not_overlap(fig, tolerance_px=0.5):
    """No two texts or legends of `fig` overlap after drawing."""
    fig.canvas.draw()
    items = [(a, a.get_window_extent()) for a in label_artists(fig)]
    overlaps = []
    for i, (a, box_a) in enumerate(items):
        for b, box_b in items[i + 1 :]:
            dx = min(box_a.x1, box_b.x1) - max(box_a.x0, box_b.x0)
            dy = min(box_a.y1, box_b.y1) - max(box_a.y0, box_b.y0)
            if dx > tolerance_px and dy > tolerance_px:
                overlaps.append((_text_of(a), _text_of(b)))
    if overlaps:
        raise AssertionError(f"overlapping texts: {overlaps}")


def assert_stars_clear_of_the_top(fig, axes, gap_pt=2.0):
    """The text box of every star of `axes` ends at least `gap_pt` below the axes top."""
    fig.canvas.draw()
    gap_px = gap_pt * fig.dpi / 72
    touching = []
    for ax in axes:
        top = ax.get_window_extent().y1
        for star in [t for t in ax.texts if "*" in t.get_text()]:
            if top - star.get_window_extent().y1 < gap_px - 1e-6:
                touching.append((star.get_text(), star.get_position()))
    if touching:
        raise AssertionError(f"stars at the axes top: {touching}")


class TestConstants(unittest.TestCase):
    def test_print_size_and_font_sizes(self):
        self.assertEqual(TEXT_WIDTH_IN, 5.17)
        self.assertEqual(BASE_FONTSIZE, 7)
        self.assertEqual(LETTER_FONTSIZE, 10)
        self.assertEqual(STAR_FONTSIZE, 9)
        self.assertEqual(LINE_SCALE, 0.7)
        self.assertEqual(LEGEND_FRAME_ALPHA, 0.4)

    def test_star_text_box_ends_two_points_below_the_axes_top(self):
        # the text box of a star is about as high as its font size
        self.assertGreaterEqual(STAR_TOP_GAP_PT - 1.02 * STAR_FONTSIZE / 2, 2.0)


class TestRcParams(unittest.TestCase):
    def test_former_fig4_parameters_are_6pt_with_scale_0_4(self):
        rc = rc_params(6, 0.4)
        self.assertEqual({k: rc[k] for k in FORMER_FIG4_RC_PARAMS}, FORMER_FIG4_RC_PARAMS)

    def test_all_text_sizes_follow_the_font_size(self):
        rc = rc_params(7)
        for key in (
            "axes.titlesize",
            "axes.labelsize",
            "xtick.labelsize",
            "ytick.labelsize",
            "legend.fontsize",
            "legend.title_fontsize",
        ):
            self.assertEqual(rc[key], 7, key)

    def test_defaults_are_7pt_truetype_with_scaled_lines(self):
        rc = rc_params()
        self.assertEqual(rc["font.size"], 7)
        self.assertEqual(rc["font.family"], "DejaVu Sans")
        self.assertEqual(rc["pdf.fonttype"], 42)
        self.assertAlmostEqual(rc["lines.linewidth"], 1.5 * 0.7)
        self.assertAlmostEqual(rc["axes.linewidth"], 0.8 * 0.7)


class TestStarLayout(unittest.TestCase):
    def test_rows_and_top_keep_the_gaps_in_points(self):
        y_min, y_max, height_pt = -1.0, 8.5, 40.0
        top, (unpaired, paired) = star_layout(y_min, y_max, height_pt)
        per_pt = (top - y_min) / height_pt
        self.assertAlmostEqual((paired * y_max - y_max) / per_pt, PAIRED_STAR_GAP_PT)
        self.assertAlmostEqual((unpaired - paired) * y_max / per_pt, UNPAIRED_STAR_GAP_PT)
        self.assertAlmostEqual((top - unpaired * y_max) / per_pt, STAR_TOP_GAP_PT)

    def test_shorter_panels_need_more_headroom(self):
        tall, _ = star_layout(-1.0, 8.5, 80.0)
        short, _ = star_layout(-1.0, 8.5, 40.0)
        self.assertGreater(short, tall)
        self.assertGreater(tall, 8.5)


class TestFigureHelpers(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_panel_letters_are_bold_at_the_given_positions(self):
        fig = plt.figure()
        texts = add_panel_letters(fig, {"a": (0.0, 1.0), "b": (0.5, 0.6)})
        self.assertEqual(sorted(texts), ["a", "b"])
        self.assertEqual(texts["b"].get_position(), (0.5, 0.6))
        for text in texts.values():
            self.assertEqual(text.get_fontweight(), "bold")
            self.assertEqual(text.get_fontsize(), LETTER_FONTSIZE)
            self.assertEqual(text.get_va(), "top")
            self.assertEqual(text.get_ha(), "left")
        self.assertEqual([t.get_text() for t in fig.texts], ["a", "b"])

    @staticmethod
    def _colours(legend):
        colours = []
        for handle in legend.legend_handles:
            artists = handle if isinstance(handle, (list, tuple)) else [handle]
            colours.append({a.get_color() for a in artists})
        return colours

    def test_significance_legend_lists_unpaired_then_paired(self):
        fig = plt.figure()
        legend = significance_legend(fig, loc="lower center", ncol=2)
        self.assertEqual(
            [t.get_text() for t in legend.get_texts()],
            ["unpaired to -1.0", "paired to -1.0"],
        )
        self.assertEqual({t.get_fontsize() for t in legend.get_texts()}, {7})
        self.assertEqual(self._colours(legend), [{"sandybrown"}, {"darkgreen"}])
        self.assertEqual(fig.legends, [legend])

    def test_significance_legend_without_stars_shows_the_count_outlines_only(self):
        fig, ax = plt.subplots()
        legend = significance_legend(ax, stars=False, loc="upper right")
        self.assertIs(ax.get_legend(), legend)
        self.assertEqual(fig.legends, [])
        markers = [h.get_marker() for h in legend.legend_handles]
        self.assertEqual(markers, ["None", "None"])
        self.assertEqual(self._colours(legend), [{"sandybrown"}, {"darkgreen"}])

    def test_legend_stands_centred_right_above_the_centred_footnote(self):
        fig = plt.figure(figsize=(4, 3))
        legend, footnote = legend_above_footnote(fig, 0.4, "(Here -1.0 is last sample)")
        fig.canvas.draw()
        self.assertEqual((footnote.get_ha(), footnote.get_va()), ("center", "bottom"))
        self.assertEqual(footnote.get_position(), (0.4, FOOTNOTE_Y))
        box, note = legend.get_window_extent(), footnote.get_window_extent()
        self.assertGreaterEqual(box.y0, note.y1)
        self.assertLess(box.y0 - note.y1, 3 * fig.dpi / 72)
        centre = 0.4 * fig.bbox.width
        self.assertAlmostEqual((box.x0 + box.x1) / 2, centre, delta=1.0)
        self.assertAlmostEqual((note.x0 + note.x1) / 2, centre, delta=1.0)
        self.assertEqual(fig.legends, [legend])
        self.assertEqual([t.get_text() for t in legend.get_texts()], ["unpaired to -1.0", "paired to -1.0"])

    def test_legend_and_footnote_take_a_font_size(self):
        fig = plt.figure(figsize=(4, 3))
        legend, footnote = legend_above_footnote(fig, 0.5, "note", fontsize=6)
        self.assertEqual(footnote.get_fontsize(), 6)
        self.assertEqual({t.get_fontsize() for t in legend.get_texts()}, {6})

    def test_legend_handle_sizes_pass_through(self):
        fig = plt.figure(figsize=(4, 3))
        legend, _ = legend_above_footnote(fig, 0.5, "note", markersize=12, linewidth=3)
        star = legend.legend_handles[0]
        self.assertEqual(star.get_markersize(), 12)

    def test_legend_without_footnote_stands_at_the_bottom(self):
        fig = plt.figure(figsize=(4, 3))
        legend, footnote = legend_above_footnote(fig, 0.5, None, stars=False)
        fig.canvas.draw()
        self.assertIsNone(footnote)
        self.assertEqual(fig.texts, [])
        self.assertAlmostEqual(legend.get_window_extent().y0, FOOTNOTE_Y * fig.bbox.height, delta=1.0)
        self.assertEqual([h.get_marker() for h in legend.legend_handles], ["None", "None"])

    def test_y_labels_share_one_distance_clear_of_the_widest_tick_labels(self):
        fig, (narrow, wide) = plt.subplots(1, 2, figsize=(4, 2))
        narrow.set_ylim(0, 10)
        wide.set_ylim(-100, 100)
        narrow.set_ylabel("score")
        wide.set_ylabel("Δ diversity\n[%]")
        distance_pt = align_ylabels_at_distance(fig, [narrow, wide])
        fig.canvas.draw()
        distances = [
            ax.get_window_extent().x0 - ax.yaxis.label.get_window_extent().x1
            for ax in (narrow, wide)
        ]
        self.assertAlmostEqual(distances[0], distances[1], delta=0.5)
        widest = max(
            ax.get_window_extent().x0 - t.get_window_extent().x0
            for ax in (narrow, wide)
            for t in drawn_tick_labels(ax.yaxis)
        )
        pad = YLABEL_PAD_PT * fig.dpi / 72
        self.assertAlmostEqual(distances[0], widest + pad, delta=0.5)
        self.assertAlmostEqual(distance_pt, distances[0] * 72 / fig.dpi, delta=0.5)

    def test_overlap_helper_rejects_overlapping_texts(self):
        fig = plt.figure(figsize=(2, 2))
        fig.text(0.1, 0.5, "first")
        fig.text(0.1, 0.8, "second")
        assert_texts_do_not_overlap(fig)
        fig.text(0.12, 0.5, "third")
        with self.assertRaises(AssertionError):
            assert_texts_do_not_overlap(fig)

    def test_bounds_helper_rejects_text_outside_the_page(self):
        fig = plt.figure(figsize=(2, 2))
        fig.text(0.5, 0.5, "inside")
        assert_texts_inside(fig)
        fig.text(0.99, 0.5, "far outside the page")
        with self.assertRaises(AssertionError):
            assert_texts_inside(fig)

    def test_save_figure_writes_pdf_into_new_folder(self):
        fig = plt.figure()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "new", "figure.pdf")
            save_figure(fig, path)
            with open(path, "rb") as f:
                self.assertEqual(f.read(4), b"%PDF")


if __name__ == "__main__":
    unittest.main()
