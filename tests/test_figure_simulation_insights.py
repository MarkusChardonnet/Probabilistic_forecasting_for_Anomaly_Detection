import os
import sys
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")

import json  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(src_dir)

from configs import config  # noqa: E402
from figure_simulation_insights import (  # noqa: E402
    ANOMALY_TYPES,
    FIGURE_SIZE_IN,
    LEGEND_LABELS,
    NEIGHBOURING_SCORES,
    PATH_INDEX,
    STEPS_AHEAD,
    AnomalyPanel,
    compose_simulation_insights,
    extend_labels,
    load_anomaly_panel,
    require_dataset,
)
from figure_style import BASE_FONTSIZE, LETTER_FONTSIZE  # noqa: E402
from tests.test_figure_style import assert_texts_inside  # noqa: E402


def _panel(name, rng, nb_steps=401):
    t = np.linspace(0.0, 1.0, nb_steps)
    labels = np.zeros(nb_steps, dtype=int)
    labels[120:220] = 1
    scores = np.full(nb_steps, np.nan)
    scores[25:-5] = rng.random(nb_steps - 30)
    weights = rng.random((len(STEPS_AHEAD), 2 * NEIGHBOURING_SCORES + 1))
    return AnomalyPanel(
        name=name,
        weights=weights,
        steps_ahead=STEPS_AHEAD,
        neighbors=list(range(-NEIGHBOURING_SCORES, NEIGHBOURING_SCORES + 1)),
        t=t,
        true_path=np.sin(6 * t) * 0.3 + 0.5,
        anomaly_labels=labels,
        scores=scores,
        predicted_labels=(np.nan_to_num(scores) > 0.5).astype(int),
        threshold=0.5,
    )


def _touch(*parts):
    path = os.path.join(*parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"")


def _write_overview(folder):
    """model_overview.csv with the forecast model id 1 of the configuration."""
    pd.DataFrame(
        [{"id": 1, "description": json.dumps({"data_dict": "AD_OrnsteinUhlenbeckWithSeason_3_dict"})}]
    ).to_csv(os.path.join(folder, "model_overview.csv"))


def _texts_of(ax):
    texts = [ax.title, ax.xaxis.label, ax.yaxis.label]
    texts += ax.get_xticklabels() + ax.get_yticklabels()
    return [t for t in texts if t.get_text() != ""]


class TestHelpers(unittest.TestCase):
    def test_constants_follow_the_published_evaluation(self):
        self.assertEqual(
            ANOMALY_TYPES,
            {"cutoff": "Drift", "diffusion": "Diffusion", "noise": "Noise", "spike": "Spike"},
        )
        self.assertEqual(PATH_INDEX, {"cutoff": 0, "diffusion": 4, "noise": 1, "spike": 3})
        self.assertEqual(STEPS_AHEAD, [1, 3, 5, 7, 10, 12, 15, 18, 20])
        self.assertEqual(NEIGHBOURING_SCORES, 5)
        np.testing.assert_allclose(FIGURE_SIZE_IN, (5.17, 5.35))

    def test_extend_labels_adds_one_step_on_each_side_of_an_anomaly(self):
        np.testing.assert_array_equal(extend_labels(np.array([0, 1, 1, 0, 0])), [1, 1, 1, 1, 0])
        np.testing.assert_array_equal(extend_labels(np.array([0, 0, 0])), [0, 0, 0])
        np.testing.assert_array_equal(extend_labels(np.array([1, 0, 0, 1])), [1, 1, 1, 1])

    def test_loader_names_the_missing_model_overview(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as ctx:
                load_anomaly_panel("spike", tmp + "/", 1, "last", 3)
        self.assertIn("model_overview.csv", str(ctx.exception))

    def test_loader_names_the_missing_forecast_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_overview(tmp)
            with self.assertRaises(FileNotFoundError) as ctx:
                load_anomaly_panel("spike", tmp + "/", 1, "last", 3)
        self.assertIn("best_checkpoint", str(ctx.exception))

    def test_loader_names_the_missing_ad_module_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_overview(tmp)
            _touch(tmp, "id-1", "best_checkpoint", "checkpt.tar")
            with self.assertRaises(FileNotFoundError) as ctx:
                load_anomaly_panel("spike", tmp + "/", 1, "last", 3)
        self.assertIn("saved_AD_module_spike", str(ctx.exception))

    def test_require_dataset_names_a_dataset_missing_from_the_overview(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "dataset_overview.csv"), "w") as f:
                f.write(",name,id,description\n")
            with self.assertRaises(FileNotFoundError) as ctx:
                require_dataset("AD_OrnsteinUhlenbeckWithSeason_spike_dict", tmp + "/")
        self.assertIn("AD_OrnsteinUhlenbeckWithSeason_spike_dict", str(ctx.exception))

    def test_require_dataset_names_missing_files_of_a_listed_dataset(self):
        data_dict = "AD_OrnsteinUhlenbeckWithSeason_spike_dict"
        with tempfile.TemporaryDirectory() as tmp:
            desc = json.dumps(eval("config." + data_dict), sort_keys=True)
            pd.DataFrame(
                [{"name": "AD_OrnsteinUhlenbeckWithSeason", "id": 7, "description": desc}]
            ).to_csv(os.path.join(tmp, "dataset_overview.csv"))
            with self.assertRaises(FileNotFoundError) as ctx:
                require_dataset(data_dict, tmp + "/")
        self.assertIn("AD_OrnsteinUhlenbeckWithSeason-7", str(ctx.exception))


class TestComposedFigure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.panels = [_panel(name, rng) for name in ANOMALY_TYPES.values()]
        cls.fig = compose_simulation_insights(cls.panels)
        cls.heat_axes = [ax for ax in cls.fig.axes if ax.get_label() == "weights"]
        cls.path_axes = [ax for ax in cls.fig.axes if ax.get_label() == "scores"]
        cls.twin_axes = [ax for ax in cls.fig.axes if ax.get_label() == "process"]
        cls.colorbars = [ax for ax in cls.fig.axes if ax.get_label() == "<colorbar>"]

    @classmethod
    def tearDownClass(cls):
        plt.close("all")

    def test_four_rows_of_heat_map_and_path_panel(self):
        np.testing.assert_allclose(self.fig.get_size_inches(), FIGURE_SIZE_IN)
        self.assertEqual(len(self.heat_axes), 4)
        self.assertEqual(len(self.path_axes), 4)
        self.assertEqual(len(self.twin_axes), 4)
        self.assertEqual(len(self.colorbars), 4)
        self.assertEqual(
            [ax.get_title() for ax in self.heat_axes], ["Drift", "Diffusion", "Noise", "Spike"]
        )
        self.assertEqual(
            [ax.get_title() for ax in self.path_axes],
            ["Drift anomaly", "Diffusion anomaly", "Noise anomaly", "Spike anomaly"],
        )

    def test_heat_map_axes_show_neighbors_and_steps_ahead(self):
        for ax in self.heat_axes:
            self.assertEqual(list(ax.get_xticks()), list(range(11)))
            self.assertEqual(
                [t.get_text() for t in ax.get_yticklabels()], [str(k) for k in STEPS_AHEAD]
            )
            self.assertEqual(ax.get_ylabel(), "steps-ahead ($k$)")
            self.assertEqual(len(ax.images), 1)
        # the x axis is shared along the column: tick labels (every second
        # neighbour, 7 pt labels) and the x label on the bottom row only
        self.assertEqual(
            [ax.get_xlabel() for ax in self.heat_axes],
            ["", "", "", "neighboring timestamps ($l$)"],
        )
        self.assertEqual([ax.get_xticklabels() for ax in self.heat_axes[:3]], [[]] * 3)
        self.assertEqual(
            [t.get_text() for t in self.heat_axes[3].get_xticklabels()],
            ["", "-4", "", "-2", "", "0", "", "2", "", "4", ""],
        )

    def test_path_panels_share_one_legend_and_one_x_label(self):
        self.assertEqual(len(self.fig.legends), 1)
        self.assertEqual([t.get_text() for t in self.fig.legends[0].get_texts()], LEGEND_LABELS)
        self.assertEqual(
            LEGEND_LABELS,
            [
                "predicted scores",
                "score threshold",
                "predicted anomaly",
                "true path, no anomaly",
                "true path, anomaly",
            ],
        )
        self.assertEqual([ax.get_xlabel() for ax in self.path_axes], ["", "", "", "$t$"])
        self.assertEqual(
            [len(ax.get_xticklabels()) > 0 for ax in self.path_axes],
            [False, False, False, True],
        )
        self.assertEqual({ax.get_ylabel() for ax in self.path_axes}, {"score"})
        self.assertEqual({ax.get_ylabel() for ax in self.twin_axes}, {"process values"})
        self.assertEqual({ax.get_ylim() for ax in self.path_axes}, {(0.0, 1.0)})
        for ax in self.fig.axes:
            self.assertIsNone(ax.get_legend())

    def test_legend_below_panel_b_grouped_into_score_and_path_entries(self):
        (legend,) = self.fig.legends
        self.fig.canvas.draw()
        box = legend.get_window_extent()
        bottom = self.path_axes[-1]
        axes = bottom.get_window_extent()
        self.assertAlmostEqual((box.x0 + box.x1) / 2, (axes.x0 + axes.x1) / 2, delta=1.0)
        self.assertLess(box.y1, bottom.xaxis.label.get_window_extent().y0)
        # one column per group: the score entries, then the path entries
        x0 = [round(t.get_window_extent().x0, 1) for t in legend.get_texts()]
        self.assertEqual(len(set(x0[:3])), 1)
        self.assertEqual(len(set(x0[3:])), 1)
        self.assertLess(x0[0], x0[3])

    def test_predicted_anomaly_is_shaded_where_the_scores_exceed_the_threshold(self):
        ax = self.path_axes[0]
        shaded = [c for c in ax.collections if c.get_label() == "predicted anomaly"]
        self.assertEqual(len(shaded), 1)
        threshold = [l for l in ax.lines if l.get_label() == "score threshold"]
        self.assertEqual(threshold[0].get_ydata()[0], 0.5)

    def test_all_texts_and_legends_stay_on_the_page(self):
        assert_texts_inside(self.fig)

    def test_letters_and_font_sizes(self):
        letters = {t.get_text(): t for t in self.fig.texts if t.get_text() in "ab"}
        self.assertEqual(sorted(letters), ["a", "b"])
        self.assertEqual({t.get_fontsize() for t in letters.values()}, {LETTER_FONTSIZE})
        axes = self.heat_axes + self.path_axes + self.twin_axes + self.colorbars
        sizes = {t.get_fontsize() for ax in axes for t in _texts_of(ax)}
        self.assertEqual(sizes, {BASE_FONTSIZE})
        self.assertEqual({t.get_fontsize() for t in self.fig.legends[0].get_texts()}, {BASE_FONTSIZE})


if __name__ == "__main__":
    unittest.main()
