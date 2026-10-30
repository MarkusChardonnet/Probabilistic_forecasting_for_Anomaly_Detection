import os
import sys
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(src_dir)

from figure_style import BASE_FONTSIZE  # noqa: E402
from tests.test_figure_style import assert_texts_inside  # noqa: E402
from figure_time_horizon import (  # noqa: E402
    CUTOFF_DAYS,
    FIGURE_SIZE_IN,
    FOOTNOTE,
    MAX_SAMPLES,
    MIN_SAMPLES,
    bin_months_since_cutoff,
    bin_scores_around_cutoff,
    compose_time_horizon,
    load_cutoff_scores,
)

DAYS_PER_MONTH = 30.437
HOSTS = [f"H{i}" for i in range(12)]
OBSERVED_DAYS = list(range(0, 760, 20))


def _wide_scores(cutoff_day, rng):
    """One cut-off file as written by the scoring: one column per day."""
    rows = []
    for host in HOSTS:
        row = {"host_id": host, "abx": False, "use_obs_until_day": cutoff_day}
        for day in OBSERVED_DAYS:
            observed = rng.random() < 0.6
            row[f"ad_score_day-{day}"] = rng.gamma(2.0, 0.5) if observed else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _write_cutoff_files(folder, cutoff_days, seed=0):
    rng = np.random.default_rng(seed)
    for day in cutoff_days:
        _wide_scores(day, rng).to_csv(
            os.path.join(folder, f"val_noabx_ad_scores_{day}_coord-0.csv"), index=False
        )


def _long_scores(seed=0, shift_after=0.0):
    """Long scores of several cut-offs with months_since_cutoff of 0.5 resolution."""
    rng = np.random.default_rng(seed)
    rows = []
    for cutoff_month in (2, 6, 10):
        for host in HOSTS:
            for months in np.arange(-3.0, 12.5, 0.5):
                if rng.random() < 0.7:
                    rows.append(
                        dict(
                            host_id=host,
                            cutoff_month=cutoff_month,
                            use_obs_until_day=int(round(cutoff_month * DAYS_PER_MONTH)),
                            months_since_cutoff=months,
                            month5_bin=cutoff_month + months,
                            score=rng.gamma(2.0, 0.5) + (shift_after if months >= 0 else 0.0),
                        )
                    )
    return pd.DataFrame(rows)


def _texts_of(ax):
    texts = [ax.title, ax.xaxis.label, ax.yaxis.label]
    texts += ax.get_xticklabels() + ax.get_yticklabels()
    if ax.get_legend() is not None:
        texts += ax.get_legend().get_texts()
    return [t for t in texts if t.get_text() != ""]


class TestLoading(unittest.TestCase):
    def test_constants_follow_the_notebook(self):
        self.assertEqual(list(CUTOFF_DAYS), list(range(0, 1141, 30)))
        self.assertEqual((MIN_SAMPLES, MAX_SAMPLES), (-3.0, 12.0))
        np.testing.assert_allclose(FIGURE_SIZE_IN, (5.17, 3.0))

    def test_cutoff_files_are_read_into_long_scores_with_months_since_cutoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_cutoff_files(tmp, [0, 30, 60])
            scores = load_cutoff_scores(tmp + "/", cutoff_days=[0, 30, 60])
        self.assertEqual(sorted(scores["cutoff_month"].unique()), [0, 1, 2])
        self.assertFalse(scores["score"].isna().any())
        np.testing.assert_allclose(
            scores["months_since_cutoff"],
            np.round(scores["month5_bin"] - scores["cutoff_month"], 2),
        )
        self.assertLessEqual(scores["month5_bin"].max(), 24.0)
        self.assertIn("use_obs_until_day", scores.columns)

    def test_limit_months_drops_late_samples_and_late_cutoffs(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_cutoff_files(tmp, [0, 750])
            scores = load_cutoff_scores(tmp + "/", cutoff_days=[0, 750], limit_months=12.0)
        self.assertLessEqual(scores["month5_bin"].max(), 12.0)
        self.assertEqual(scores["cutoff_month"].unique().tolist(), [0])


class TestBinning(unittest.TestCase):
    def test_months_since_cutoff_are_binned_by_their_left_edge_within_the_range(self):
        scores = _long_scores()
        binned = bin_months_since_cutoff(scores, -3.0, 6.0)
        bins = sorted(binned["months_since_cutoff"].dropna().astype(float).unique())
        self.assertEqual(bins, [float(b) for b in range(-3, 6)])
        # as in the notebook: samples exactly at the upper limit get no bin (and
        # are dropped when one score per bin is kept)
        upper = scores.loc[binned.index, "months_since_cutoff"] == 6.0
        self.assertTrue(binned.loc[upper, "months_since_cutoff"].isna().all())
        self.assertFalse(binned.loc[~upper, "months_since_cutoff"].isna().any())

    def test_monthly_bins_with_one_score_per_host_cutoff_and_bin(self):
        binned = bin_scores_around_cutoff(_long_scores())
        bins = sorted(binned["months_since_cutoff"].astype(float).unique())
        self.assertEqual(bins, [float(b) for b in range(-1, 12)])
        self.assertFalse(
            binned.duplicated(["host_id", "cutoff_month", "months_since_cutoff"]).any()
        )

    def test_pre_cutoff_samples_are_grouped_into_the_last_sample_before_cutoff(self):
        scores = _long_scores()
        binned = bin_scores_around_cutoff(scores)
        pre = binned[binned["months_since_cutoff"].astype(float) == -1.0]
        # one pre-cutoff score per host and cut-off, the last one before the cut-off
        self.assertFalse(pre.duplicated(["host_id", "cutoff_month"]).any())
        host_pre = scores[(scores["host_id"] == "H0") & (scores["cutoff_month"] == 2)]
        host_pre = host_pre[host_pre["months_since_cutoff"] < 0]
        expected = host_pre.sort_values("months_since_cutoff").iloc[-1]["score"]
        got = pre[(pre["host_id"] == "H0") & (pre["cutoff_month"] == 2)]["score"].iloc[0]
        self.assertEqual(got, expected)


class TestComposedFigure(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_single_pair_without_stars_has_a_line_legend_above_the_footnote(self):
        fig = compose_time_horizon(bin_scores_around_cutoff(_long_scores()))
        np.testing.assert_allclose(fig.get_size_inches(), FIGURE_SIZE_IN)
        self.assertEqual(len(fig.axes), 2)
        box, count = fig.axes
        self.assertEqual(box.get_title(), "Score before/after cut-off")
        self.assertEqual(box.get_ylabel(), "score")
        self.assertEqual(count.get_ylabel(), "# samples")
        self.assertEqual(count.get_xlabel(), "Months since cut-off")
        self.assertEqual(
            FOOTNOTE, "(Here -1.0 is last sample prior to cut-off since -3.0 months)"
        )
        self.assertEqual([t for t in box.texts if "*" in t.get_text()], [])
        self.assertIsNone(box.get_legend())
        self.assertIsNone(count.get_legend())
        (legend,) = fig.legends
        self.assertEqual(
            [t.get_text() for t in legend.get_texts()], ["unpaired to -1.0", "paired to -1.0"]
        )
        self.assertEqual([h.get_marker() for h in legend.legend_handles], ["None", "None"])
        fig.canvas.draw()
        (note,) = [t for t in fig.texts if t.get_text() == FOOTNOTE]
        box_l, box_n = legend.get_window_extent(), note.get_window_extent()
        centre = (count.get_window_extent().x0 + count.get_window_extent().x1) / 2
        self.assertAlmostEqual((box_l.x0 + box_l.x1) / 2, centre, delta=1.0)
        self.assertAlmostEqual((box_n.x0 + box_n.x1) / 2, centre, delta=1.0)
        self.assertGreaterEqual(box_l.y0, box_n.y1)
        self.assertLessEqual(box_l.y1, count.xaxis.label.get_window_extent().y0)
        sizes = {t.get_fontsize() for ax in fig.axes for t in _texts_of(ax)}
        self.assertEqual(sizes, {BASE_FONTSIZE})

    def test_all_texts_and_legends_stay_on_the_page(self):
        assert_texts_inside(compose_time_horizon(bin_scores_around_cutoff(_long_scores())))

    def test_stars_and_their_legend_appear_for_significant_bins(self):
        fig = compose_time_horizon(bin_scores_around_cutoff(_long_scores(shift_after=3.0)))
        box = fig.axes[0]
        self.assertGreater(len([t for t in box.texts if "*" in t.get_text()]), 0)
        (legend,) = fig.legends
        self.assertEqual([h.get_marker() for h in legend.legend_handles], ["*", "*"])


if __name__ == "__main__":
    unittest.main()
