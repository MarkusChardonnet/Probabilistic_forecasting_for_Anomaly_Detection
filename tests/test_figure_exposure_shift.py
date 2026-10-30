import os
import sys
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(src_dir)

from figure_exposure_shift import (  # noqa: E402
    BASE_FONTSIZE,
    EXPOSURE_LABELS,
    FIGURE_SIZE_IN,
    FOOTNOTE,
    compose_exposure_shift_figure,
    legend_hides_stars,
    peak_months,
    save_figure,
    shift_label,
    summarize_variants,
    variant_names,
)

DAYS_PER_MONTH = 30.437
EVENTS = {
    "H1": [6.0],
    "H2": [5.0, 9.0],
    "H3": [4.0, 8.0, 12.0],
    "H4": [7.0, 10.0, 13.0],
    "H5": [3.0, 6.0, 9.0],
    "H6": [5.5, 11.0, 15.0],
    "H7": [4.5, 9.5],
    "H8": [6.0, 12.0, 16.0],
}
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
NOTE = "(Here -1.0 is last sample prior"


def _samples(host_id, events, rng):
    rows = []
    for month in np.arange(1.0, 20.0, 0.5):
        cum = sum(1 for e in events if e <= month)
        bump = 1.5 * cum
        rows.append(
            dict(
                host_id=host_id,
                day=int(round(month * DAYS_PER_MONTH)),
                month5_bin=month,
                month_bin=int(round(month)),
                abx=True,
                abx_any_cumcount=cum,
                score_1=rng.gamma(2.0, 0.4) + bump,
                score_2=rng.gamma(2.0, 0.4) + bump,
                score_3=rng.gamma(2.0, 0.4) + bump,
            )
        )
    return rows


def synthetic_variant(shift_months, seed=0):
    """Scores and exposure table of the synthetic cohort with shifted exposure ages."""
    rng = np.random.default_rng(seed)
    events = {h: [e + shift_months for e in ev] for h, ev in EVENTS.items()}
    scores = pd.DataFrame(
        [row for h, ev in events.items() for row in _samples(h, ev, rng)]
    )
    abx_df = pd.DataFrame(
        [dict(host_id=h, abx_start_age_months=e) for h, ev in events.items() for e in ev]
    )
    return scores, abx_df


def synthetic_variants():
    return [
        (shift_label(s), *synthetic_variant(s, seed=i))
        for i, s in enumerate([-0.5, 0.0, 0.5])
    ]


def _star_texts(ax):
    return [t for t in ax.texts if t.get_text() in ("*", "**")]


class TestNames(unittest.TestCase):
    def test_shift_label(self):
        self.assertEqual(shift_label(0.0), "as recorded")
        self.assertEqual(shift_label(-0.5), "exposure ages −15 d")
        self.assertEqual(shift_label(0.5), "exposure ages +15 d")

    def test_variant_names(self):
        recorded = variant_names(0.0)
        self.assertEqual(
            recorded["model_name"], "saved_models_microbial_novel_alpha_div2/id-57"
        )
        self.assertEqual(recorded["ft_name"], "ft_vat19_anomaly_v20240806_entero_genus")
        self.assertEqual(recorded["abx_ts_name"], "ts_vat19_abx_v20240806")
        self.assertEqual(recorded["label"], "as recorded")
        minus = variant_names(-0.5)
        self.assertEqual(
            minus["model_name"], "saved_models_microbial_novel_alpha_div2_shift_m05/id-57"
        )
        self.assertEqual(
            minus["ft_name"], "ft_vat19_anomaly_v20240806_entero_genus_shift_m05"
        )
        self.assertEqual(minus["abx_ts_name"], "ts_vat19_abx_v20240806_shift_m05")
        self.assertEqual(minus["label"], "exposure ages −15 d")
        self.assertTrue(variant_names(0.5)["model_name"].endswith("_shift_p05/id-57"))


def _plotted_max(ax):
    """Largest y value drawn in `ax`: boxes, whiskers, medians and fliers."""
    values = [np.nanmax(line.get_ydata()) for line in ax.lines if len(line.get_ydata())]
    values += [
        patch.get_path().transformed(patch.get_patch_transform()).vertices[:, 1].max()
        for patch in ax.patches
    ]
    return max(values)


class TestFigure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.variants = synthetic_variants()
        cls.fig = compose_exposure_shift_figure(cls.variants)
        # row-major: rows are the exposures, columns the datings
        cls.axes = cls.fig.axes
        cls.rows = [cls.axes[row * 3 : (row + 1) * 3] for row in range(3)]

    def test_grid_of_nine_panels_without_count_panels(self):
        self.assertEqual(len(self.axes), 9)
        self.assertFalse(any(ax.get_ylabel() == "# samples" for ax in self.axes))
        np.testing.assert_allclose(self.fig.get_size_inches(), FIGURE_SIZE_IN)

    def test_rows_are_exposures_and_columns_datings(self):
        self.assertEqual(
            [row[0].get_ylabel() for row in self.rows], ["score_1", "score_2", "score_3"]
        )
        self.assertEqual([ax.get_ylabel() for row in self.rows for ax in row[1:]], [""] * 6)
        titles = [ax.get_title() for ax in self.axes]
        self.assertEqual(titles[:3], [v[0] for v in self.variants])
        self.assertEqual(titles[3:], [""] * 6)
        texts = [t.get_text() for t in self.fig.texts]
        for expected in ["a", "b", "c"] + EXPOSURE_LABELS:
            self.assertIn(expected, texts)

    def test_one_legend_centred_above_the_centred_footnote_below_the_grid(self):
        self.assertTrue(all(ax.get_legend() is None for ax in self.axes))
        (legend,) = self.fig.legends
        self.assertEqual(
            [t.get_text() for t in legend.get_texts()], ["unpaired to -1.0", "paired to -1.0"]
        )
        self.fig.canvas.draw()
        (note,) = [t for t in self.fig.texts if t.get_text() == FOOTNOTE]
        self.assertEqual(note.get_ha(), "center")
        box, text = legend.get_window_extent(), note.get_window_extent()
        centre = (self.axes[0].get_window_extent().x0 + self.axes[2].get_window_extent().x1) / 2
        self.assertAlmostEqual((box.x0 + box.x1) / 2, centre, delta=1.0)
        self.assertAlmostEqual((text.x0 + text.x1) / 2, centre, delta=1.0)
        self.assertGreaterEqual(box.y0, text.y1)
        self.assertLessEqual(box.y1, min(ax.xaxis.label.get_window_extent().y0 for ax in self.rows[2]))
        self.assertEqual({t.get_fontsize() for t in legend.get_texts()}, {BASE_FONTSIZE})
        self.assertEqual(note.get_fontsize(), BASE_FONTSIZE)

    def test_legends_do_not_hide_stars(self):
        self.assertTrue(any(_star_texts(ax) for ax in self.axes))
        self.assertFalse(legend_hides_stars(self.fig))

    def test_stars_above_all_plotted_values_and_aligned_within_a_row(self):
        for row in self.rows:
            stars = [t for ax in row for t in _star_texts(ax)]
            self.assertTrue(stars)
            plotted = max(_plotted_max(ax) for ax in row)
            self.assertGreater(min(t.get_position()[1] for t in stars), plotted)
            for color in ("sandybrown", "darkgreen"):
                heights = {round(t.get_position()[1], 9) for t in stars if t.get_color() == color}
                self.assertLessEqual(len(heights), 1)
            self.assertEqual({t.get_color() for t in stars} - {"sandybrown", "darkgreen"}, set())

    def test_panels_of_a_row_share_the_y_range(self):
        for row in self.rows:
            self.assertEqual(len({ax.get_ylim() for ax in row}), 1)

    def test_x_labels_in_the_bottom_row_only_and_footnote_once(self):
        xlabels = [ax.get_xlabel() for ax in self.axes]
        self.assertEqual(xlabels[:6], [""] * 6)
        self.assertTrue(all("Months since" in label for label in xlabels[6:]))
        self.assertFalse(any(NOTE in label for label in xlabels))
        for ax in self.axes:
            self.assertTrue(any(label.get_visible() for label in ax.get_xticklabels()))
        self.assertTrue(all(ax.xaxis.label.get_visible() for ax in self.rows[2]))
        texts = [t.get_text() for t in self.fig.texts]
        self.assertEqual(texts.count(FOOTNOTE), 1)
        self.assertIn("-1.0 is the last sample prior", FOOTNOTE)

    def test_save_figure_writes_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "supp_exposure_shift.pdf")
            save_figure(self.fig, output)
            with open(output, "rb") as f:
                self.assertTrue(f.read(4) == b"%PDF")


class TestSummary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        variants = synthetic_variants()
        label, scores, abx_df = variants[1]
        first = abx_df.groupby("host_id")["abx_start_age_months"].first()
        diff = scores["month5_bin"] - scores["host_id"].map(first)
        bumped = (diff >= 2.0) & (diff < 3.0) & (scores["abx_any_cumcount"] == 1)
        scores = scores.copy()
        scores.loc[bumped, "score_1"] += 50.0
        variants[1] = (label, scores, abx_df)
        cls.summary = summarize_variants(variants)
        cls.peaks = peak_months(cls.summary)
        cls.label = label

    def test_summary_columns_and_rows(self):
        self.assertEqual(list(self.summary.columns), SUMMARY_COLUMNS)
        keys = self.summary[["variant", "exposure", "bin"]]
        self.assertFalse(keys.duplicated().any())
        self.assertEqual(sorted(self.summary["exposure"].unique()), [1, 2, 3])
        self.assertEqual(self.summary["variant"].nunique(), 3)

    def test_p_values_nan_at_reference_bin_only(self):
        reference = self.summary["bin"] == -1.0
        self.assertTrue(self.summary.loc[reference, "p_unpaired"].isna().all())
        self.assertTrue(np.isfinite(self.summary.loc[~reference, "p_unpaired"]).all())
        self.assertTrue((self.summary["n_samples"] > 0).all())

    def test_peak_month_of_bumped_exposure(self):
        self.assertEqual(list(self.peaks.columns), PEAK_COLUMNS)
        row = self.peaks[(self.peaks["variant"] == self.label) & (self.peaks["exposure"] == 1)]
        self.assertEqual(len(row), 1)
        self.assertEqual(row["peak_bin"].iloc[0], 2.0)
        self.assertGreater(row["n_peak"].iloc[0], 0)
        for col in ("significant_bins_unpaired_005", "significant_bins_paired_005"):
            self.assertIsInstance(row[col].iloc[0], str)
            self.assertIn("2.0", row[col].iloc[0])
        month0 = self.summary[
            (self.summary["variant"] == self.label)
            & (self.summary["exposure"] == 1)
            & (self.summary["bin"] == 0.0)
        ]
        self.assertEqual(row["n_month0"].iloc[0], month0["n_samples"].iloc[0])


if __name__ == "__main__":
    unittest.main()
