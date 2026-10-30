import os
import sys
import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(src_dir)

from figure_diet_covariates import (  # noqa: E402
    BINS,
    DIET_ORDER,
    EXPOSURES,
    FIGURE_SIZE_IN,
    FOOTNOTE,
    XLABEL,
    compose_diet_covariates,
    diet_counts,
    select_exposure_samples,
)
from figure_style import BASE_FONTSIZE, LETTER_FONTSIZE  # noqa: E402
from tests.test_figure_style import (  # noqa: E402
    assert_texts_do_not_overlap,
    assert_texts_inside,
)
from tests.test_figure_score_delineation import synthetic_inputs  # noqa: E402
from utils_color_maps import all_color_maps  # noqa: E402


def _texts_of(ax):
    texts = [ax.title, ax.xaxis.label, ax.yaxis.label]
    texts += ax.get_xticklabels() + ax.get_yticklabels()
    if ax.get_legend() is not None:
        texts += ax.get_legend().get_texts() + [ax.get_legend().get_title()]
    return [t for t in texts if t.get_text() != ""]


class TestCounts(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(EXPOSURES, [1, 2, 3])
        np.testing.assert_allclose(BINS, np.arange(-1.0, 6.0, 1.0))
        self.assertEqual(DIET_ORDER["diet_weaning"], ["no", "yes", "unknown"])
        self.assertEqual(DIET_ORDER["diet_milk"], ["bd", "mixed", "fd", "unknown"])
        np.testing.assert_allclose(FIGURE_SIZE_IN, (5.17, 3.4))

    def test_counts_fill_missing_categories_and_bins_with_zero(self):
        samples = pd.DataFrame(
            {
                "diff_age_nth_abx": [-1.0, -1.0, 0.0, 0.0, 2.0],
                "diet_milk": ["bd", "mixed", np.nan, "bd", "mixed"],
            }
        )
        counts = diet_counts(samples, "diet_milk")
        self.assertEqual(list(counts.columns), ["bd", "mixed", "fd", "unknown"])
        np.testing.assert_allclose(counts.index, BINS)
        self.assertEqual(counts.loc[-1.0].tolist(), [1, 1, 0, 0])
        self.assertEqual(counts.loc[0.0].tolist(), [1, 0, 0, 1])
        self.assertEqual(counts.loc[1.0].tolist(), [0, 0, 0, 0])
        self.assertEqual(counts.loc[2.0].tolist(), [0, 1, 0, 0])
        self.assertEqual(counts.loc[5.0].sum(), 0)

    def test_counts_accept_categorical_bins_of_the_selection(self):
        exposed, abx_df = synthetic_inputs()
        selected = select_exposure_samples(exposed, abx_df, 2)
        counts = diet_counts(selected, "diet_weaning")
        self.assertEqual(counts.values.sum(), len(selected))
        self.assertEqual(list(counts.columns), DIET_ORDER["diet_weaning"])


class TestComposedFigure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        exposed, abx_df = synthetic_inputs()
        # no formula-dominant samples around the 3rd exposure
        samples = {n: select_exposure_samples(exposed, abx_df, n) for n in EXPOSURES}
        samples[1].loc[samples[1].index[:3], "diet_milk"] = "fd"
        cls.fig = compose_diet_covariates(samples)
        cls.bar_axes = cls.fig.axes

    @classmethod
    def tearDownClass(cls):
        plt.close("all")

    def test_grid_of_two_covariate_rows_and_three_exposure_columns(self):
        np.testing.assert_allclose(self.fig.get_size_inches(), FIGURE_SIZE_IN)
        self.assertEqual(len(self.bar_axes), 6)
        self.assertEqual(
            [ax.get_title() for ax in self.bar_axes[:3]],
            ["1st abx exposure", "2nd abx exposure", "3rd abx exposure"],
        )
        self.assertEqual([ax.get_title() for ax in self.bar_axes[3:]], [""] * 3)
        self.assertEqual(
            [ax.get_ylabel() for ax in self.bar_axes],
            ["# samples", "", "", "# samples", "", ""],
        )
        self.assertEqual([ax.get_xlabel() for ax in self.bar_axes], [""] * 6)

    def test_one_x_label_centred_under_the_bottom_row(self):
        self.assertEqual(XLABEL, "Months since abx exposure")
        (label,) = [t for t in self.fig.texts if t.get_text() == XLABEL]
        self.assertEqual(label.get_ha(), "center")
        self.fig.canvas.draw()
        box = label.get_window_extent()
        left = self.bar_axes[3].get_window_extent().x0
        right = self.bar_axes[5].get_window_extent().x1
        self.assertAlmostEqual((box.x0 + box.x1) / 2, (left + right) / 2, delta=1.0)
        ticks = min(t.get_window_extent().y0 for ax in self.bar_axes[3:] for t in ax.get_xticklabels())
        self.assertLess(box.y1, ticks)
        footnote = [t for t in self.fig.texts if t.get_text() == FOOTNOTE][0]
        self.assertEqual(footnote.get_ha(), "center")
        self.assertLess(footnote.get_window_extent().y1, box.y0)

    def test_rows_share_their_y_range(self):
        for row in (self.bar_axes[:3], self.bar_axes[3:]):
            self.assertEqual(len({ax.get_ylim() for ax in row}), 1)
            tallest = max(
                sum(
                    p.get_height()
                    for p in ax.patches
                    if abs(p.get_x() + p.get_width() / 2 - x) < 0.1
                )
                for ax in row
                for x in range(len(BINS))
            )
            self.assertGreaterEqual(row[0].get_ylim()[1], tallest)
            self.assertLess(row[0].get_ylim()[1], 1.2 * tallest)

    def test_one_legend_per_covariate_row_with_the_category_colours(self):
        legends = [ax.get_legend() for ax in self.bar_axes if ax.get_legend()]
        self.assertEqual(len(legends), 2)
        self.assertEqual(
            [leg.get_title().get_text() for leg in legends], ["diet weaning", "diet milk"]
        )
        self.assertEqual(
            [[t.get_text() for t in leg.get_texts()] for leg in legends],
            [DIET_ORDER["diet_weaning"], DIET_ORDER["diet_milk"]],
        )
        milk = all_color_maps["diet_milk"]
        for handle, level in zip(legends[1].legend_handles, DIET_ORDER["diet_milk"]):
            np.testing.assert_allclose(
                handle.get_facecolor(), matplotlib.colors.to_rgba(milk[level])
            )

    def test_bar_colours_follow_the_category_also_where_a_category_is_absent(self):
        milk = all_color_maps["diet_milk"]
        for ax in self.bar_axes[3:]:
            colours = {tuple(np.round(p.get_facecolor(), 6)) for p in ax.patches}
            expected = {tuple(np.round(matplotlib.colors.to_rgba(milk[l]), 6)) for l in DIET_ORDER["diet_milk"]}
            self.assertTrue(colours <= expected, (colours, expected))
        fd = tuple(np.round(matplotlib.colors.to_rgba(milk["fd"]), 6))
        self.assertIn(fd, {tuple(np.round(p.get_facecolor(), 6)) for p in self.bar_axes[3].patches})

    def test_letters_footnote_and_font_sizes(self):
        letters = {t.get_text(): t for t in self.fig.texts if t.get_text() in "abc"}
        self.assertEqual(sorted(letters), ["a", "b", "c"])
        self.assertEqual({t.get_fontsize() for t in letters.values()}, {LETTER_FONTSIZE})
        notes = [t for t in self.fig.texts if t.get_text() == FOOTNOTE]
        self.assertEqual(len(notes), 1)
        self.assertEqual(FOOTNOTE, "(Here -1.0 is last sample prior to abx since -3.0 months)")
        sizes = {t.get_fontsize() for ax in self.bar_axes for t in _texts_of(ax)}
        self.assertEqual(sizes, {BASE_FONTSIZE})

    def test_all_texts_and_legends_stay_on_the_page(self):
        assert_texts_inside(self.fig)

    def test_texts_do_not_overlap(self):
        assert_texts_do_not_overlap(self.fig)

    def test_tick_labels_are_integers_that_do_not_touch(self):
        ax = self.bar_axes[3]
        self.assertEqual(
            [t.get_text() for t in ax.get_xticklabels()], ["-1", "0", "1", "2", "3", "4", "5"]
        )
        self.fig.canvas.draw()
        boxes = [t.get_window_extent() for t in ax.get_xticklabels()]
        for left, right in zip(boxes, boxes[1:]):
            self.assertGreater(right.x0 - left.x1, 2.0)

    def test_exposure_line_before_bin_zero_in_every_panel(self):
        for ax in self.bar_axes:
            lines = [l for l in ax.lines if l.get_color() == "darkred"]
            self.assertEqual(len(lines), 1)
            self.assertEqual(lines[0].get_xdata()[0], 0.5)


if __name__ == "__main__":
    unittest.main()
