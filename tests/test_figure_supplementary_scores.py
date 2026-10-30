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

from figure_style import BASE_FONTSIZE, LETTER_FONTSIZE, TEXT_WIDTH_IN  # noqa: E402
from figure_supplementary_scores import (  # noqa: E402
    COUNT_YLABEL,
    FIGURES,
    Pair,
    compose_pair_figure,
    exposure_pairs,
    sample_cutoffs,
    save_figure,
    time_since_second_pairs,
)
from figure_time_horizon import bin_scores_around_cutoff  # noqa: E402
from tests.test_figure_score_delineation import synthetic_inputs  # noqa: E402
from tests.test_figure_style import (  # noqa: E402
    assert_stars_clear_of_the_top,
    assert_texts_do_not_overlap,
    assert_texts_inside,
)
from tests.test_figure_time_horizon import _long_scores  # noqa: E402

NOTE_ABX = "(Here -1.0 is last sample prior to abx since -3.0 months)"
NOTE_CUTOFF = "(Here -1.0 is last sample prior to cut-off since -3.0 months)"


def _cutoff_pairs():
    binned = bin_scores_around_cutoff(_long_scores(shift_after=2.0), max_samples=6.0)
    return [
        Pair(binned, "months_since_cutoff", "score", 0, f"title {n}", True, ("use_obs_until_day",))
        for n in (1, 2, 3)
    ]


class TestSelections(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exposed, cls.abx_df = synthetic_inputs()

    def test_cutoffs_are_drawn_per_month_with_the_notebook_seed(self):
        mapping = pd.DataFrame(
            {"host_id": [f"H{i}" for i in range(12)], "cutoff_month": [1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3]}
        )
        chosen = sample_cutoffs(mapping, {1: 2, 2: 3}, seed=8)
        self.assertEqual(chosen.groupby("cutoff_month").size().to_dict(), {1: 2, 2: 3})
        expected = pd.concat(
            [mapping[mapping["cutoff_month"] == 1].sample(n=2, random_state=8),
             mapping[mapping["cutoff_month"] == 2].sample(n=3, random_state=8)]
        )
        pd.testing.assert_frame_equal(chosen, expected)
        with self.assertRaises(ValueError):
            sample_cutoffs(mapping, {3: 4}, seed=8)

    def test_exposure_pairs_keep_the_pre_exposure_bins_when_ungrouped(self):
        grouped = exposure_pairs(self.exposed, self.abx_df, grouped=True)
        ungrouped = exposure_pairs(self.exposed, self.abx_df, grouped=False)
        self.assertEqual(
            [p.title for p in grouped],
            [f"Score_{n} before/after {n}{s} abx exposure" for n, s in zip((1, 2, 3), ("st", "nd", "rd"))],
        )
        self.assertEqual(min(grouped[2].data["diff_age_nth_abx"].astype(float)), -1.0)
        self.assertEqual(min(ungrouped[2].data["diff_age_nth_abx"].astype(float)), -3.0)
        self.assertEqual([p.grouped for p in ungrouped], [False] * 3)

    def test_third_exposure_split_by_time_since_second_inclusive_four_months(self):
        pairs = time_since_second_pairs(self.exposed, self.abx_df)
        self.assertEqual(
            [p.title for p in pairs],
            [
                "Score_3 before/after 3rd abx exposure\ntime since 2nd abx exposure <= 4 months",
                "Score_3 before/after 3rd abx exposure\ntime since 2nd abx exposure > 4 months",
            ],
        )
        short, long_ = (set(p.data["host_id"]) for p in pairs)
        # gaps 2nd -> 3rd: H3 4.0 (<= 4), H4 3.0, H5 3.0, H6 4.0, i.e. none above 4
        self.assertEqual(short, {"H3", "H4", "H5", "H6"} & set(self.exposed["host_id"]))
        self.assertEqual(long_, set())
        self.assertEqual([p.grouped for p in pairs], [False, False])


class TestComposition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        exposed, abx_df = synthetic_inputs()
        cls.figs = {
            "baseline": compose_pair_figure([_cutoff_pairs()], FIGURES["baseline"]),
            "priorexposure": compose_pair_figure(
                [exposure_pairs(exposed, abx_df, False), exposure_pairs(exposed, abx_df, False)[:2]],
                FIGURES["priorexposure"],
            ),
            "supp_shannon_scores": compose_pair_figure(
                [exposure_pairs(exposed, abx_df, True)], FIGURES["supp_shannon_scores"]
            ),
        }
        for fig in cls.figs.values():
            fig.canvas.draw()

    @classmethod
    def tearDownClass(cls):
        plt.close("all")

    def test_figures_are_drawn_at_the_text_width_into_their_former_files(self):
        self.assertEqual(
            {name: spec.file_name for name, spec in FIGURES.items()},
            {
                "baseline": "baseline.pdf",
                "priorexposure": "priorexposure.pdf",
                "supp_shannon_scores": "supp_shannon_scores.pdf",
            },
        )
        for name, fig in self.figs.items():
            self.assertAlmostEqual(fig.get_size_inches()[0], TEXT_WIDTH_IN)
            self.assertLessEqual(fig.get_size_inches()[1], 6.0)

    def test_count_panels_x_labels_and_titles(self):
        for name, fig in self.figs.items():
            counts = fig.axes[1::2]
            self.assertEqual({ax.get_ylabel() for ax in counts}, {COUNT_YLABEL}, name)
            self.assertEqual(COUNT_YLABEL, "# samples")
        baseline = self.figs["baseline"]
        self.assertEqual([ax.get_xlabel() for ax in baseline.axes[1::2]], ["", "", "Months since cut-off"])
        self.assertEqual([ax.get_title() for ax in baseline.axes[0::2]], ["title 1", "title 2", "title 3"])
        prior = self.figs["priorexposure"]
        self.assertEqual(
            [ax.get_xlabel() for ax in prior.axes[1::2]],
            ["", "", "Months since abx exposure", "", "Months since 3rd abx exposure"],
        )

    def test_one_legend_at_the_bottom_above_the_footnote_where_there_is_one(self):
        footnotes = {"baseline": NOTE_CUTOFF, "priorexposure": None, "supp_shannon_scores": NOTE_ABX}
        for name, fig in self.figs.items():
            self.assertTrue(all(ax.get_legend() is None for ax in fig.axes), name)
            (legend,) = fig.legends
            box = legend.get_window_extent()
            centre = (min(ax.get_window_extent().x0 for ax in fig.axes) + max(ax.get_window_extent().x1 for ax in fig.axes)) / 2
            self.assertAlmostEqual((box.x0 + box.x1) / 2, centre, delta=1.0, msg=name)
            lowest = min(ax.xaxis.label.get_window_extent().y0 for ax in fig.axes[1::2] if ax.get_xlabel())
            self.assertLessEqual(box.y1, lowest, name)
            notes = [t for t in fig.texts if t.get_text().startswith("(Here")]
            if footnotes[name] is None:
                self.assertEqual(notes, [], name)
            else:
                (note,) = notes
                self.assertEqual(note.get_text(), footnotes[name])
                self.assertGreaterEqual(box.y0, note.get_window_extent().y1, name)

    def test_letters_per_column_or_per_pair(self):
        letters = {name: sorted(t.get_text() for t in fig.texts if t.get_fontweight() == "bold") for name, fig in self.figs.items()}
        self.assertEqual(letters, {"baseline": [], "priorexposure": ["a", "b"], "supp_shannon_scores": ["a", "b", "c"]})
        for fig in self.figs.values():
            for text in fig.texts:
                if text.get_fontweight() == "bold":
                    self.assertEqual(text.get_fontsize(), LETTER_FONTSIZE)

    def test_letters_of_one_row_stand_at_one_height(self):
        fig = compose_pair_figure(
            [exposure_pairs(*synthetic_inputs(), False), time_since_second_pairs(*synthetic_inputs())[:1]],
            FIGURES["priorexposure"],
        )
        a, b = sorted((t for t in fig.texts if t.get_fontweight() == "bold"), key=lambda t: t.get_text())
        self.assertAlmostEqual(a.get_position()[1], b.get_position()[1])
        plt.close(fig)

    def test_layout_checks(self):
        for name, fig in self.figs.items():
            assert_texts_inside(fig)
            assert_texts_do_not_overlap(fig)
            assert_stars_clear_of_the_top(fig, fig.axes[0::2])
            sizes = {t.get_fontsize() for ax in fig.axes for t in [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels() if t.get_text()}
            self.assertEqual(sizes, {BASE_FONTSIZE}, name)

    def test_y_labels_share_one_distance(self):
        for name, fig in self.figs.items():
            distances = [ax.get_window_extent().x0 - ax.yaxis.label.get_window_extent().x1 for ax in fig.axes]
            self.assertLess(max(distances) - min(distances), 0.5, name)

    def test_save_figure_writes_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "baseline.pdf")
            save_figure(self.figs["baseline"], path)
            with open(path, "rb") as f:
                self.assertEqual(f.read(4), b"%PDF")


if __name__ == "__main__":
    unittest.main()
