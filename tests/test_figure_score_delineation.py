import os
import sys
import tempfile
import unittest
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(src_dir)

from figure_score_delineation import (  # noqa: E402
    BASE_FONTSIZE,
    FIGURE_SIZE_IN,
    LETTER_FONTSIZE,
    STAR_FONTSIZE,
    compose_score_delineation,
    save_figure,
    scores_folder_name,
    trajectory_legend_kws,
)
from utils_eval_score import (  # noqa: E402
    MATCHED_DIFF_COL,
    REL_MATCHED_DIFF_COL,
    PanelStyle,
    _create_subplot,
    _plot_score_after_nth_abx_exposure,
    calculate_matched_metric_n_diff,
    calculate_relative_matched_diff,
    plot_trajectory,
)

DAYS_PER_MONTH = 30.437
EVENTS = {
    "H1": [6.0],
    "H2": [5.0, 9.0],
    "H3": [4.0, 8.0, 12.0],
    "H4": [7.0, 10.0, 13.0],
    "H5": [3.0, 6.0, 9.0],
    "H6": [5.5, 11.0, 15.0],
}
HOSTS = ["H3", "H4", "H6"]
MATCHING_VARS = ["month_bin", "delivery_mode", "diet_milk", "diet_weaning"]
METRIC = "div_alpha_faith_pd"


def _samples(host_id, events, rng, exposed):
    rows = []
    for month in np.arange(1.0, 20.0, 0.5):
        cum = sum(1 for e in events if e <= month)
        # scores rise after each exposure so that some bins become significant
        bump = 1.5 * cum
        rows.append(
            dict(
                host_id=host_id,
                day=int(round(month * DAYS_PER_MONTH)),
                month5_bin=month,
                month_bin=int(round(month)),
                abx=exposed,
                abx_any_cumcount=cum,
                score_1=rng.gamma(2.0, 0.4) + bump,
                score_2=rng.gamma(2.0, 0.4) + bump,
                score_3=rng.gamma(2.0, 0.4) + bump,
                delivery_mode="vaginal",
                diet_milk="bd",
                diet_weaning="no" if month < 6 else "yes",
                **{METRIC: 5.0 + 0.5 * month - 2.0 * cum + rng.normal(0.0, 0.5)},
            )
        )
    return rows


def synthetic_inputs(seed=0):
    """Exposed and unexposed samples plus exposure events, as the notebooks build them."""
    rng = np.random.default_rng(seed)
    exposed = pd.DataFrame(
        [row for h, ev in EVENTS.items() for row in _samples(h, ev, rng, True)]
    )
    unexposed = pd.DataFrame(
        [row for h in ["N1", "N2", "N3"] for row in _samples(h, [], rng, False)]
    )
    abx_df = pd.DataFrame(
        [
            dict(host_id=h, abx_start_age_months=e)
            for h, ev in EVENTS.items()
            for e in ev
        ]
    )
    exposed = calculate_matched_metric_n_diff(METRIC, exposed, unexposed, MATCHING_VARS)
    exposed = calculate_relative_matched_diff(exposed, METRIC)
    return exposed, abx_df


def _texts_of(ax):
    texts = [ax.title, ax.xaxis.label, ax.yaxis.label]
    texts += ax.get_xticklabels() + ax.get_yticklabels()
    if ax.get_legend() is not None:
        texts += ax.get_legend().get_texts()
    return [t for t in texts if t.get_text() != ""]


class TestHelpers(unittest.TestCase):
    def test_relative_matched_diff_is_percent_of_matched_mean(self):
        df = pd.DataFrame(
            {f"mean_{METRIC}": [20.0, 10.0], MATCHED_DIFF_COL: [5.0, -1.0]}
        )
        out = calculate_relative_matched_diff(df, METRIC)
        np.testing.assert_allclose(out[REL_MATCHED_DIFF_COL], [25.0, -10.0])

    def test_trajectory_legend_corner_is_pinned_for_published_hosts_only(self):
        self.assertEqual(trajectory_legend_kws("E021822")["loc"], "upper right")
        self.assertEqual(trajectory_legend_kws("P015091")["loc"], "upper left")
        self.assertEqual(trajectory_legend_kws("H3")["loc"], "best")
        self.assertLess(trajectory_legend_kws("H3")["labelspacing"], 0.5)

    def test_scores_folder_name_matches_notebook_convention(self):
        self.assertEqual(
            scores_folder_name("nc_std", 10, "--RD-True", ""),
            "using-SF_nc_std_z_scores--moving_avg-10-cummax--RD-True",
        )


class TestPanelStyle(unittest.TestCase):
    def _box_inputs(self):
        data = pd.DataFrame(
            {
                "host_id": list("abcdef") * 2,
                "diff_age_nth_abx": [-1.0] * 6 + [0.0] * 6,
                "score_1": [
                    1.0,
                    2.0,
                    1.5,
                    1.2,
                    0.8,
                    1.1,
                    4.0,
                    5.0,
                    3.5,
                    4.2,
                    3.8,
                    10.0,
                ],
            }
        )
        result_df = pd.DataFrame(
            {
                "P-value unpaired": [0.01],
                "# samples unpaired": [6],
                "P-value paired": [0.5],
                "# samples paired": [6],
            },
            index=pd.Index([0.0], name="t1"),
        )
        return data, result_df

    def tearDown(self):
        plt.close("all")

    def test_default_style_keeps_former_headroom_and_star_size(self):
        data, result_df = self._box_inputs()
        fig, axs = _create_subplot(
            "diff_age_nth_abx",
            "score_1",
            data,
            "t",
            "# samples",
            "x",
            1.0,
            1,
            result_df,
        )
        self.assertEqual(len(fig.axes), 2)
        self.assertAlmostEqual(axs[0].get_ylim()[1], 1.7 * 10.0)
        stars = [t for t in axs[0].texts if "*" in t.get_text()]
        self.assertEqual(len(stars), 1)
        self.assertAlmostEqual(stars[0].get_position()[1], 1.2 * 10.0)
        self.assertEqual(stars[0].get_fontsize(), 24)
        self.assertEqual(stars[0].get_va(), "baseline")
        self.assertEqual(axs[0].get_legend().legend_handles[0].get_markersize(), 12)
        self.assertEqual(axs[1].get_legend().legend_handles[0].get_linewidth(), 3)

    def test_style_sets_headroom_star_rows_and_font_sizes(self):
        data, result_df = self._box_inputs()
        style = PanelStyle(
            fontsize=6,
            ylim_factor=1.2,
            star_rows=(1.0, 0.9),
            star_fontsize=10,
            legend_markersize=5,
            legend_linewidth=1.2,
            star_va="center",
        )
        fig, axs = plt.subplots(2, 1)
        _create_subplot(
            "diff_age_nth_abx",
            "score_1",
            data,
            "t",
            "# samples",
            "x",
            1.0,
            1,
            result_df,
            axs=axs,
            style=style,
        )
        self.assertAlmostEqual(axs[0].get_ylim()[1], 1.2 * 10.0)
        stars = [t for t in axs[0].texts if "*" in t.get_text()]
        self.assertAlmostEqual(stars[0].get_position()[1], 1.0 * 10.0)
        self.assertEqual(stars[0].get_fontsize(), 10)
        self.assertEqual(stars[0].get_va(), "center")
        self.assertEqual(axs[0].get_legend().legend_handles[0].get_markersize(), 5)
        self.assertEqual(axs[1].get_legend().legend_handles[0].get_linewidth(), 1.2)
        for ax in axs:
            for text in _texts_of(ax):
                self.assertEqual(text.get_fontsize(), 6, text.get_text())
        self.assertEqual(len(fig.axes), 2)


class TestLegacyPanels(unittest.TestCase):
    def setUp(self):
        self.exposed, self.abx_df = synthetic_inputs()

    def tearDown(self):
        plt.close("all")

    def test_score_panel_is_saved_with_former_file_name(self):
        from utils_eval_score import _select_samples_around_nth_abx_exposure

        selected = _select_samples_around_nth_abx_exposure(
            self.exposed,
            self.abx_df,
            n=1,
            min_samples=-3.0,
            max_samples=6.0,
            group_samples=True,
            score_var="score_1",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = _plot_score_after_nth_abx_exposure(
                selected,
                "diff_age_nth_abx",
                "score_1",
                1,
                path_to_save=tmp + "/",
                flag="score_1",
                min_samples=-3.0,
                max_samples=6.0,
                grouped_samples=True,
            )
            self.assertEqual(os.path.basename(path), "score_after_abx1st_score_1.pdf")
            self.assertTrue(os.path.exists(path))
        self.assertIn("(Here -1.0 is last sample prior", plt.gcf().axes[1].get_xlabel())

    def test_trajectory_panel_is_saved_with_former_file_name(self):
        with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # plt.show() on Agg
            plot_trajectory(
                self.exposed,
                self.abx_df,
                "H3",
                ["score_1", "score_2", "score_3"],
                path_to_output=tmp,
                flag="all_scores",
            )
            self.assertTrue(
                os.path.exists(os.path.join(tmp, "indv_trajectory_H3_all_scores.pdf"))
            )

    def test_trajectory_legend_takes_keyword_arguments(self):
        fig, ax = plt.subplots()
        plot_trajectory(
            self.exposed,
            self.abx_df,
            "H3",
            ["score_1"],
            ax=ax,
            fontsize=6,
            legend_kws={"labelspacing": 0.2},
        )
        self.assertEqual(ax.get_legend().labelspacing, 0.2)

    def test_trajectory_draws_on_given_axis(self):
        fig, ax = plt.subplots()
        plot_trajectory(
            self.exposed,
            self.abx_df,
            "H3",
            ["score_1", "score_2", "score_3"],
            ax=ax,
            fontsize=6,
        )
        self.assertEqual(ax.get_title(), "Trajectory Over Time for Host ID: H3")
        event_lines = [line for line in ax.lines if line.get_color() == "darkred"]
        self.assertEqual(len(event_lines), len(EVENTS["H3"]))
        self.assertEqual(
            [t.get_text() for t in ax.get_legend().get_texts()],
            ["score_1", "score_2", "score_3", "abx event"],
        )
        self.assertEqual(len(fig.axes), 1)


class TestComposedFigure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        exposed, abx_df = synthetic_inputs()
        cls.fig = compose_score_delineation(exposed, abx_df, HOSTS)
        cls.trajectory_axes = cls.fig.axes[:3]
        cls.pair_axes = cls.fig.axes[3:]
        cls.box_axes = cls.pair_axes[0::2]
        cls.count_axes = cls.pair_axes[1::2]

    @classmethod
    def tearDownClass(cls):
        plt.close("all")

    def test_page_size_equals_former_figure(self):
        np.testing.assert_allclose(self.fig.get_size_inches(), FIGURE_SIZE_IN)
        np.testing.assert_allclose(
            FIGURE_SIZE_IN, (547.52 / 72, 621.682 / 72), rtol=1e-6
        )

    def test_three_trajectory_panels_then_six_box_count_pairs(self):
        self.assertEqual(len(self.fig.axes), 15)
        self.assertEqual(
            [ax.get_title() for ax in self.trajectory_axes],
            [f"Trajectory Over Time for Host ID: {h}" for h in HOSTS],
        )
        self.assertEqual(
            [ax.get_title() for ax in self.box_axes],
            [
                "Score_1 before/after 1st abx exposure",
                "Score_2 before/after 2nd abx exposure",
                "Score_3 before/after 3rd abx exposure",
                "Δ matched alpha diversity [%] before/after 1st abx exposure",
                "Δ matched alpha diversity [%] before/after 2nd abx exposure",
                "Δ matched alpha diversity [%] before/after 3rd abx exposure",
            ],
        )
        self.assertEqual({ax.get_ylabel() for ax in self.count_axes}, {"# samples"})

    def test_box_panels_use_former_colours(self):
        def box_colour(ax):
            return [p.get_facecolor() for p in ax.patches][0]

        # seaborn draws the boxes with its default saturation of 0.75
        skyblue = matplotlib.colors.to_rgba(sns.desaturate("skyblue", 0.75))
        purple = matplotlib.colors.to_rgba(sns.desaturate("purple", 0.75))
        for ax in self.box_axes[:3]:
            np.testing.assert_allclose(box_colour(ax), skyblue)
        for ax in self.box_axes[3:]:
            np.testing.assert_allclose(box_colour(ax), purple)

    def test_x_axis_note_only_under_bottom_pairs(self):
        note = "(Here -1.0 is last sample prior to abx since -3.0 months)"
        with_note = [ax for ax in self.fig.axes if note in ax.get_xlabel()]
        self.assertEqual(with_note, [self.count_axes[2], self.count_axes[5]])
        self.assertTrue(
            all(ax.get_xlabel().startswith("Months since") for ax in self.count_axes)
        )

    def test_panel_letters_are_bold_and_large(self):
        letters = {t.get_text(): t for t in self.fig.texts if t.get_text() in "abc"}
        self.assertEqual(sorted(letters), ["a", "b", "c"])
        for text in letters.values():
            self.assertEqual(text.get_fontweight(), "bold")
            self.assertEqual(text.get_fontsize(), LETTER_FONTSIZE)

    def test_all_panel_text_shares_one_font_size(self):
        sizes = {text.get_fontsize() for ax in self.fig.axes for text in _texts_of(ax)}
        self.assertEqual(sizes, {BASE_FONTSIZE})
        stars = [t for ax in self.box_axes for t in ax.texts if "*" in t.get_text()]
        self.assertGreater(len(stars), 0)
        self.assertEqual({t.get_fontsize() for t in stars}, {STAR_FONTSIZE})

    def test_y_labels_are_aligned_within_each_column(self):
        self.fig.canvas.draw()
        for column in (self.pair_axes[:6], self.pair_axes[6:]):
            # labels are anchored at their edge facing the axis (x1 for left labels)
            x1 = [ax.yaxis.label.get_window_extent().x1 for ax in column]
            self.assertLess(max(x1) - min(x1), 0.5, x1)

    def test_matched_y_label_is_split_in_two_lines(self):
        self.assertEqual(
            {ax.get_ylabel() for ax in self.box_axes[3:]},
            {"Δ matched alpha\ndiversity [%]"},
        )
        self.assertEqual(
            {ax.get_ylabel() for ax in self.box_axes[:3]},
            {"score_1", "score_2", "score_3"},
        )

    def test_trajectory_legends_are_compact(self):
        for ax in self.trajectory_axes:
            self.assertLess(ax.get_legend().labelspacing, 0.5)

    def test_matched_stars_are_centred_below_the_axis_top(self):
        for ax in self.box_axes[3:]:
            top = ax.get_ylim()[1]
            for star in [t for t in ax.texts if "*" in t.get_text()]:
                self.assertEqual(star.get_va(), "center")
                self.assertLess(star.get_position()[1], 0.85 * top)

    def test_save_figure_writes_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "score_delineation.pdf")
            save_figure(self.fig, path)
            with open(path, "rb") as f:
                self.assertEqual(f.read(4), b"%PDF")


if __name__ == "__main__":
    unittest.main()
