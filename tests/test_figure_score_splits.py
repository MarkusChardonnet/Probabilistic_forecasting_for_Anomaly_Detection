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

from figure_score_splits import (  # noqa: E402
    COUNT_YLABEL,
    FIGURES,
    FOOTNOTE,
    PAIRED_STAR_GAP_PT,
    RC_PARAMS,
    STAR_TOP_GAP_PT,
    UNPAIRED_STAR_GAP_PT,
    compose_split_figure,
    save_figures,
    split_by_abx_type,
    split_by_age,
    split_by_duration,
    split_by_milk_diet_in_first_year,
    star_layout,
    two_line_title,
)
from figure_style import (  # noqa: E402
    BASE_FONTSIZE,
    LEGEND_FRAME_ALPHA,
    LETTER_FONTSIZE,
    SIGNIFICANCE_LABELS,
    STAR_FONTSIZE,
    TEXT_WIDTH_IN,
    rc_params,
)
from tests.test_figure_style import (  # noqa: E402
    assert_stars_clear_of_the_top,
    assert_texts_do_not_overlap,
    assert_texts_inside,
)
from utils_eval_score import (  # noqa: E402
    PanelStyle,
    _create_subplot,
    _select_samples_around_nth_abx_exposure,
    box_ylim,
)

DAYS_PER_MONTH = 30.437
# exposure ages in months, durations in days and antibiotic types per host
EVENTS = {
    "H1": ([6.0], [5], ["Penicillin"]),
    "H2": ([5.0, 9.0], [10, 3], ["Penicillin", "Cephalosporine"]),
    "H3": ([4.0, 8.0, 12.0], [7, 7, 7], ["Other, Penicillin", "Penicillin", "Other"]),
    "H4": ([7.0, 10.0, 13.0], [3, 14, 5], ["Macrolide", "Penicillin", "Penicillin"]),
    "H5": ([3.0, 6.0, 9.0], [5, 5, 5], ["Penicillin", "Other", "Other"]),
    "H6": ([5.5, 11.0, 15.0], [10, 10, 10], ["Cotrimoxazole", "Cephalosporine", "Other"]),
    "H7": ([13.0, 16.0], [5, 10], ["Penicillin", "Other"]),
    "H8": ([14.0, 17.5], [10, 10], ["Other", "Penicillin"]),
    "H9": ([12.0], [14], ["Penicillin"]),
    "H10": ([8.0, 14.0], [5, 5], ["Cephalosporine", "Other"]),
    "H11": ([9.5, 12.5], [10, 3], ["Other", "Penicillin"]),
    "H12": ([16.0, 19.0], [3, 10], ["Penicillin", "Cephalosporine"]),
}
FORMULA_HOSTS = {"H2", "H5", "H10", "H12"}
NOTE = "(Here -1.0 is last sample prior to abx since -3.0 months)"


def _milk_diet(host_id, month):
    if host_id in FORMULA_HOSTS:
        return "fd"
    return "bd" if month < 4 else "mixed"


def _samples(host_id, ages, durations, rng, exposed):
    rows = []
    for month in np.arange(1.0, 20.0, 0.5):
        before = [i for i, age in enumerate(ages) if age <= month]
        cum = len(before)
        rows.append(
            dict(
                host_id=host_id,
                day=int(round(month * DAYS_PER_MONTH)),
                month5_bin=month,
                month_bin=int(round(month)),
                abx=exposed,
                abx_any_cumcount=cum,
                abx_any_last_dur_days=durations[before[-1]] if before else np.nan,
                score_1=rng.gamma(2.0, 0.4) + 1.5 * cum,
                score_2=rng.gamma(2.0, 0.4) + 1.5 * cum,
                score_3=rng.gamma(2.0, 0.4) + 1.5 * cum,
                delivery_mode="vaginal",
                diet_milk=_milk_diet(host_id, month),
                diet_weaning="no" if month < 6 else "yes",
            )
        )
    return rows


def synthetic_inputs(seed=0):
    """Samples of exposed infants and their exposures, as the notebooks build them."""
    rng = np.random.default_rng(seed)
    exposed = pd.DataFrame(
        [
            row
            for host, (ages, durations, _) in EVENTS.items()
            for row in _samples(host, ages, durations, rng, True)
        ]
    )
    abx_df = pd.DataFrame(
        [
            dict(host_id=host, abx_start_age_months=age, abx_type=kind, abx_reason="r")
            for host, (ages, _, kinds) in EVENTS.items()
            for age, kind in zip(ages, kinds)
        ]
    )
    return exposed, abx_df


def _selected(exposed, abx_df, n):
    return _select_samples_around_nth_abx_exposure(
        exposed,
        abx_df,
        n=n,
        min_samples=-3.0,
        max_samples=6.0,
        group_samples=True,
        score_var=f"score_{n}",
    )


def _texts_of(ax):
    texts = [ax.title, ax.xaxis.label, ax.yaxis.label]
    texts += ax.get_xticklabels() + ax.get_yticklabels()
    return [t for t in texts if t.get_text() != "" and t.get_visible()]


def _stars_of(ax):
    return [t for t in ax.texts if "*" in t.get_text()]


def _points(fig, pixels):
    return pixels * 72 / fig.dpi


class TestStyle(unittest.TestCase):
    def test_shared_print_style(self):
        self.assertEqual(RC_PARAMS, rc_params())
        self.assertEqual(RC_PARAMS["font.size"], BASE_FONTSIZE)


class TestBoxYlim(unittest.TestCase):
    def test_limits_match_the_former_panels(self):
        self.assertEqual(box_ylim(pd.Series([0.5, 10.0]), 1.2), (-1, 12.0))
        self.assertEqual(box_ylim(pd.Series([-5.0, 10.0]), 1.7), (-5.5, 17.0))


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


class TestFixedYlimStyle(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_fixed_limits_apply_and_stars_follow_the_shared_maximum(self):
        data = pd.DataFrame(
            {
                "host_id": list("abcdef") * 2,
                "diff_age_nth_abx": [-1.0] * 6 + [0.0] * 6,
                "score_1": [1.0, 2.0, 1.5, 1.2, 0.8, 1.1, 4.0, 5.0, 3.5, 4.2, 3.8, 10.0],
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
        fig, axs = plt.subplots(2, 1)
        style = PanelStyle(ylim_factor=1.2, star_rows=(1.0, 0.9), ylim=(-1, 24.0), legend="none")
        _create_subplot(
            "diff_age_nth_abx", "score_1", data, "t", "# samples", "x", 1.0, 1, result_df,
            axs=axs, style=style,
        )
        self.assertEqual(axs[0].get_ylim(), (-1.0, 24.0))
        (star,) = _stars_of(axs[0])
        # 24 / 1.2 = 20 is the maximum the shared limits were derived from
        self.assertAlmostEqual(star.get_position()[1], 20.0)
        self.assertIsNone(axs[0].get_legend())
        self.assertIsNone(axs[1].get_legend())


class TestTwoLineTitle(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_tag_moves_to_the_second_line(self):
        fig, ax = plt.subplots()
        ax.set_title("Score_1 before/after 1st abx exposure duration: < 7 days")
        two_line_title(ax, "duration: < 7 days")
        self.assertEqual(ax.get_title(), "Score_1 before/after 1st abx exposure\nduration: < 7 days")
        self.assertEqual(ax.title.get_fontsize(), BASE_FONTSIZE)


class TestSplits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        exposed, abx_df = synthetic_inputs()
        cls.first = _selected(exposed, abx_df, 1)
        cls.second = _selected(exposed, abx_df, 2)

    def _hosts(self, groups):
        return {tag: set(frame["host_id"]) for tag, frame in groups}

    def test_duration_split_keeps_hosts_with_their_pre_exposure_samples(self):
        groups = split_by_duration(self.first)
        hosts = self._hosts(groups)
        self.assertEqual(list(hosts), ["duration: < 7 days", "duration: >= 7 days"])
        self.assertIn("H1", hosts["duration: < 7 days"])
        # 7 days counts as '>= 7 days'
        self.assertIn("H3", hosts["duration: >= 7 days"])
        self.assertFalse(hosts["duration: < 7 days"] & hosts["duration: >= 7 days"])
        short = dict(groups)["duration: < 7 days"]
        self.assertIn(-1.0, set(short.loc[short["host_id"] == "H1", "diff_age_nth_abx"]))

    def test_duration_split_is_host_level_as_in_the_notebook(self):
        # before the 2nd exposure the last exposure is the 1st one, whose
        # duration can differ: H4 (3 then 14 days) enters both groups
        hosts = self._hosts(split_by_duration(self.second))
        self.assertIn("H4", hosts["duration: < 7 days"] & hosts["duration: >= 7 days"])

    def test_age_split_boundary_is_twelve_months_inclusive(self):
        hosts = self._hosts(split_by_age(self.first))
        self.assertEqual(list(hosts), ["age: <= 12 months", "age: 12 - 24 months"])
        self.assertIn("H9", hosts["age: <= 12 months"])
        self.assertIn("H7", hosts["age: 12 - 24 months"])
        self.assertNotIn("H7", hosts["age: <= 12 months"])

    def test_abx_type_split_is_exact_penicillin_versus_others(self):
        hosts = self._hosts(split_by_abx_type(self.second))
        self.assertEqual(list(hosts), ["Abx type: Penicillin", "Abx type: Others"])
        self.assertIn("H3", hosts["Abx type: Penicillin"])
        self.assertIn("H2", hosts["Abx type: Others"])
        first_hosts = self._hosts(split_by_abx_type(self.first))
        self.assertIn("H3", first_hosts["Abx type: Others"])  # 'Other, Penicillin'

    def test_milk_diet_split_takes_first_year_exposures_per_sample(self):
        groups = split_by_milk_diet_in_first_year(self.first)
        self.assertEqual(
            [tag for tag, _ in groups],
            ["age: <= 12 months, diet: ['mixed', 'bd']", "age: <= 12 months, diet: ['fd']"],
        )
        hosts = self._hosts(groups)
        self.assertEqual(hosts["age: <= 12 months, diet: ['fd']"], {"H2", "H5", "H10"})
        self.assertNotIn("H7", hosts["age: <= 12 months, diet: ['mixed', 'bd']"])
        for _, frame in groups:
            self.assertTrue((frame["age_nth_abx"] <= 12).all())


class TestComposedFigures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        exposed, abx_df = synthetic_inputs()
        cls.exposed, cls.abx_df = exposed, abx_df
        cls.figs = {
            name: compose_split_figure(exposed, abx_df, spec) for name, spec in FIGURES.items()
        }
        for fig in cls.figs.values():
            fig.canvas.draw()

    @classmethod
    def tearDownClass(cls):
        plt.close("all")

    @staticmethod
    def _groups(fig, n_groups):
        """Axes per group: [(box, count), (box, count)] in drawing order."""
        axes = fig.axes
        per_group = len(axes) // n_groups
        return [
            [tuple(axes[g * per_group + 2 * i : g * per_group + 2 * i + 2]) for i in range(2)]
            for g in range(n_groups)
        ]

    def _figure_maximum(self, spec):
        return max(
            frame[f"score_{group.n}"].max()
            for group in spec.groups
            for _, frame in group.split(_selected(self.exposed, self.abx_df, group.n))
        )

    def test_figures_are_printed_at_the_text_width(self):
        heights = {"scoresplit": 5.7, "penicillin_2nd": 4.4, "scoresplit_age_diet": 4.4}
        for name, fig in self.figs.items():
            np.testing.assert_allclose(fig.get_size_inches(), (TEXT_WIDTH_IN, heights[name]))
            self.assertLessEqual(fig.get_size_inches()[1], 6.0)

    def test_upper_pair_has_the_full_title_and_the_lower_pair_only_its_split(self):
        self.assertEqual(len(self.figs["scoresplit"].axes), 16)
        self.assertEqual(
            [ax.get_title() for ax in self.figs["scoresplit"].axes[0::2]],
            [
                "Score_1 before/after 1st abx exposure\nduration: < 7 days",
                "duration: >= 7 days",
                "Score_2 before/after 2nd abx exposure\nduration: < 7 days",
                "duration: >= 7 days",
                "Score_1 before/after 1st abx exposure\nage: <= 12 months",
                "age: 12 - 24 months",
                "Score_2 before/after 2nd abx exposure\nage: <= 12 months",
                "age: 12 - 24 months",
            ],
        )
        self.assertEqual(
            [ax.get_title() for ax in self.figs["penicillin_2nd"].axes[0::2]],
            ["Score_2 before/after 2nd abx exposure\nAbx type: Penicillin", "Abx type: Others"],
        )
        self.assertEqual(
            [ax.get_title() for ax in self.figs["scoresplit_age_diet"].axes[0::2]],
            [
                "Score_1 before/after 1st abx exposure\nage: <= 12 months, diet: ['mixed', 'bd']",
                "age: <= 12 months, diet: ['fd']",
            ],
        )

    def test_all_panels_of_a_figure_share_their_y_axis_ranges(self):
        for name, fig in self.figs.items():
            boxes, counts = fig.axes[0::2], fig.axes[1::2]
            self.assertEqual(len({ax.get_ylim() for ax in boxes}), 1, name)
            self.assertEqual(len({ax.get_ylim() for ax in counts}), 1, name)
            self.assertEqual(counts[0].get_ylim()[0], 0)
            self.assertEqual(boxes[0].get_ylim()[0], -1)

    def test_stars_stand_above_the_data_and_below_the_axes_top(self):
        stars = []
        for name, spec in FIGURES.items():
            fig = self.figs[name]
            maximum = self._figure_maximum(spec)
            for ax in fig.axes[0::2]:
                top_px = ax.get_window_extent().y1
                for star in _stars_of(ax):
                    x, y = star.get_position()
                    self.assertGreater(y, maximum)
                    gap_pt = _points(fig, top_px - ax.transData.transform((x, y))[1])
                    self.assertGreaterEqual(gap_pt, STAR_TOP_GAP_PT - 1e-6, name)
                    stars.append(star)
        self.assertGreater(len(stars), 0)
        self.assertEqual({s.get_fontsize() for s in stars}, {STAR_FONTSIZE})
        self.assertEqual({s.get_va() for s in stars}, {"center"})
        for fig in self.figs.values():
            assert_stars_clear_of_the_top(fig, fig.axes[0::2])

    def test_all_y_labels_of_a_figure_share_one_distance_to_their_axes(self):
        for name, fig in self.figs.items():
            distances = [
                ax.get_window_extent().x0 - ax.yaxis.label.get_window_extent().x1
                for ax in fig.axes
            ]
            self.assertLess(max(distances) - min(distances), 0.5, name)

    @staticmethod
    def _legend_of(fig):
        legends = list(fig.legends) + [ax.get_legend() for ax in fig.axes if ax.get_legend()]
        (legend,) = legends
        return legend

    @staticmethod
    def _centre_of_axes(fig):
        return (min(ax.get_window_extent().x0 for ax in fig.axes) + max(ax.get_window_extent().x1 for ax in fig.axes)) / 2

    def test_one_legend_centred_above_the_centred_footnote_below_the_last_row(self):
        for name, fig in self.figs.items():
            self.assertTrue(all(ax.get_legend() is None for ax in fig.axes), name)
            (legend,) = fig.legends
            footnote = [t for t in fig.texts if t.get_text() == NOTE][0]
            self.assertEqual(footnote.get_ha(), "center", name)
            box, note = legend.get_window_extent(), footnote.get_window_extent()
            centre = self._centre_of_axes(fig)
            self.assertAlmostEqual((box.x0 + box.x1) / 2, centre, delta=1.0, msg=name)
            self.assertAlmostEqual((note.x0 + note.x1) / 2, centre, delta=1.0, msg=name)
            self.assertGreaterEqual(box.y0, note.y1, name)
            lowest_labels = min(
                ax.xaxis.label.get_window_extent().y0 for ax in fig.axes[1::2] if ax.get_xlabel()
            )
            self.assertLessEqual(box.y1, lowest_labels, name)

    def test_legends_list_the_two_tests_and_show_stars_only_if_stars_are_drawn(self):
        for name, fig in self.figs.items():
            legend = self._legend_of(fig)
            self.assertEqual([t.get_text() for t in legend.get_texts()], SIGNIFICANCE_LABELS)
            self.assertEqual({t.get_fontsize() for t in legend.get_texts()}, {BASE_FONTSIZE})
            # the legend keeps the first artist of a star-and-line handle
            has_star_markers = any(h.get_marker() == "*" for h in legend.legend_handles)
            stars = [t for ax in fig.axes[0::2] for t in _stars_of(ax)]
            self.assertEqual(has_star_markers, len(stars) > 0, name)

    def test_legend_without_stars_when_a_figure_draws_none(self):
        exposed = self.exposed.copy()
        for n in (1, 2, 3):
            exposed[f"score_{n}"] = 1.0
        fig = compose_split_figure(exposed, self.abx_df, FIGURES["scoresplit_age_diet"])
        self.assertEqual([t for ax in fig.axes for t in _stars_of(ax)], [])
        (legend,) = fig.legends
        self.assertEqual([h.get_marker() for h in legend.legend_handles], ["None", "None"])

    def test_footnote_once_per_figure_and_x_labels_under_the_lowest_pairs(self):
        for name, spec in FIGURES.items():
            fig = self.figs[name]
            self.assertEqual(FOOTNOTE, NOTE)
            self.assertEqual([t.get_text() for t in fig.texts].count(NOTE), 1)
            for (_, count_top), (_, count_bottom) in self._groups(fig, len(spec.groups)):
                self.assertEqual(count_top.get_xlabel(), "")
                self.assertTrue(count_bottom.get_xlabel().startswith("Months since"))
                self.assertNotIn("Here", count_bottom.get_xlabel())

    def test_panel_letters_only_on_the_four_group_figure(self):
        letters = [t for t in self.figs["scoresplit"].texts if t.get_text() in "abcd"]
        self.assertEqual(sorted(t.get_text() for t in letters), ["a", "b", "c", "d"])
        for text in letters:
            self.assertEqual(text.get_fontweight(), "bold")
            self.assertEqual(text.get_fontsize(), LETTER_FONTSIZE)
        a, b, c, d = sorted(letters, key=lambda t: t.get_text())
        self.assertLess(a.get_position()[0], b.get_position()[0])
        self.assertGreater(a.get_position()[1], c.get_position()[1])
        self.assertAlmostEqual(a.get_position()[1], b.get_position()[1])
        for name in ["penicillin_2nd", "scoresplit_age_diet"]:
            self.assertEqual([t.get_text() for t in self.figs[name].texts], [NOTE])

    def test_all_panel_text_shares_one_font_size(self):
        for fig in self.figs.values():
            sizes = {text.get_fontsize() for ax in fig.axes for text in _texts_of(ax)}
            self.assertEqual(sizes, {BASE_FONTSIZE})

    def test_texts_do_not_overlap_and_stay_on_the_page(self):
        for name, spec in FIGURES.items():
            fig = self.figs[name]
            assert_texts_inside(fig)
            assert_texts_do_not_overlap(fig)
            page = fig.bbox
            outside = [t.get_text() for t in fig.texts if not page.contains(*t.get_window_extent().p0) or not page.contains(*t.get_window_extent().p1)]
            self.assertEqual(outside, [], name)
            for group in self._groups(fig, len(spec.groups)):
                for ax_box, ax_count in group:
                    self.assertFalse(
                        ax_box.yaxis.label.get_window_extent().overlaps(
                            ax_count.yaxis.label.get_window_extent()
                        ),
                        (name, ax_box.get_title()),
                    )
            titles = [ax.title.get_window_extent() for ax in fig.axes[0::2]]
            below = [
                t.get_window_extent()
                for ax in fig.axes[1::2]
                for t in [ax.xaxis.label] + ax.get_xticklabels()
                if t.get_text()
            ]
            for title in titles:
                self.assertFalse(any(title.overlaps(box) for box in below), name)
                self.assertTrue(page.contains(title.x0, title.y1), name)

    def test_count_plots_are_labelled_samples_and_box_plots_by_score(self):
        for name, spec in FIGURES.items():
            fig = self.figs[name]
            self.assertEqual({ax.get_ylabel() for ax in fig.axes[1::2]}, {COUNT_YLABEL})
            self.assertEqual(COUNT_YLABEL, "# samples")
            self.assertEqual(
                {ax.get_ylabel() for ax in fig.axes[0::2]},
                {f"score_{group.n}" for group in spec.groups},
            )

    def test_titles_fit_their_axes(self):
        for name, fig in self.figs.items():
            for ax in fig.axes[0::2]:
                self.assertLessEqual(
                    ax.title.get_window_extent().width, ax.get_window_extent().width, name
                )

    def test_y_labels_are_aligned_within_each_column(self):
        groups = self._groups(self.figs["scoresplit"], 4)
        for column in ([groups[0], groups[2]], [groups[1], groups[3]]):
            axes = [ax for group in column for pair in group for ax in pair]
            x1 = [ax.yaxis.label.get_window_extent().x1 for ax in axes]
            self.assertLess(max(x1) - min(x1), 0.5, x1)

    def test_save_figures_writes_the_former_file_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = save_figures(self.exposed, self.abx_df, tmp, ["penicillin_2nd", "scoresplit_age_diet"])
            self.assertEqual(
                [os.path.basename(p) for p in paths],
                ["penicillin_2nd.pdf", "scoresplit_age_diet.pdf"],
            )
            for path in paths:
                with open(path, "rb") as f:
                    self.assertEqual(f.read(4), b"%PDF")


if __name__ == "__main__":
    unittest.main()
