import os
import sys
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(src_dir)

from figure_simulation_insights import (  # noqa: E402
    ANOMALY_TYPES,
    NEIGHBOURING_SCORES,
    RC_PARAMS,
    STEPS_AHEAD,
    AnomalyPanel,
    compose_simulation_insights,
    extend_labels,
)
from figure_style import save_figure  # noqa: E402
from simulation_insights_pdf import (  # noqa: E402
    GRID_STEPS,
    grid_index,
    linear_map,
    panels_from_pdf,
    parse_number,
    shrink_runs,
)

T = np.linspace(0.0, 1.0, GRID_STEPS + 1)


def _runs(n, ranges):
    labels = np.zeros(n, dtype=int)
    for start, end in ranges:
        labels[start : end + 1] = 1
    return labels


def _panel(name, seed):
    """A panel as in the submitted figure: scores 0 outside the feasible grid points."""
    rng = np.random.default_rng(seed)
    n = GRID_STEPS + 1
    true_path = 0.6 + 0.2 * np.sin(5 * T + seed) + 0.03 * rng.standard_normal(n)
    scores = np.zeros(n)
    scores[25:396] = np.clip(0.5 + 0.4 * np.sin(9 * T[25:396] + seed) + 0.05 * rng.standard_normal(371), 0, 1)
    anomaly = _runs(n, [(110 + 10 * seed, 160 + 10 * seed), (300, 300)])
    predicted = _runs(n, [(105 + 10 * seed, 170 + 10 * seed), (250, 252), (330, 330)])
    weights = rng.uniform(0.001, 0.05, size=(len(STEPS_AHEAD), 2 * NEIGHBOURING_SCORES + 1))
    return AnomalyPanel(
        name=name,
        weights=weights,
        steps_ahead=STEPS_AHEAD,
        neighbors=list(range(-NEIGHBOURING_SCORES, NEIGHBOURING_SCORES + 1)),
        t=T,
        true_path=true_path,
        anomaly_labels=anomaly,
        scores=scores,
        predicted_labels=predicted,
        threshold=0.5,
    )


class TestHelpers(unittest.TestCase):
    def test_grid_has_four_hundred_steps_of_the_unit_interval(self):
        self.assertEqual(GRID_STEPS, 400)

    def test_grid_index_rounds_grid_times_and_rejects_others(self):
        np.testing.assert_array_equal(grid_index(np.array([0.0, 0.0025, 0.50001, 1.0])), [0, 1, 200, 400])
        with self.assertRaises(ValueError):
            grid_index(np.array([0.00125]))

    def test_shrinking_inverts_the_one_point_extension_of_the_plotted_regions(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            labels = (rng.random(60) < 0.15).astype(int)
            plotted = extend_labels(labels)
            np.testing.assert_array_equal(extend_labels(shrink_runs(plotted)), plotted)

    def test_shrinking_keeps_regions_at_the_path_ends(self):
        labels = np.array([1, 0, 0, 0, 0, 0, 1])
        plotted = extend_labels(labels)
        np.testing.assert_array_equal(shrink_runs(plotted), labels)

    def test_linear_map_fits_tick_positions_to_values(self):
        slope, intercept = linear_map([(10.0, 0.2), (30.0, 0.6), (20.0, 0.4)])
        self.assertAlmostEqual(slope, 0.02)
        self.assertAlmostEqual(intercept, 0.0)

    def test_parse_number_reads_unicode_minus(self):
        self.assertEqual(parse_number("−0.5"), -0.5)
        self.assertEqual(parse_number("0.025"), 0.025)


class TestRoundTrip(unittest.TestCase):
    """Panels drawn by the Fig. 2 composer are read back from its PDF."""

    @classmethod
    def setUpClass(cls):
        cls.panels = [_panel(name, seed) for seed, name in enumerate(ANOMALY_TYPES.values())]
        cls.fig = compose_simulation_insights(cls.panels)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fig2.pdf")
            with plt.rc_context(RC_PARAMS):
                save_figure(cls.fig, path)
            cls.recovered = panels_from_pdf(path)

    @classmethod
    def tearDownClass(cls):
        plt.close("all")

    def _points(self, ax, t, values):
        """Positions in points of (t, values) drawn on `ax`."""
        return ax.transData.transform(np.c_[t, values]) * 72 / self.fig.dpi

    @staticmethod
    def _distance_to_curve(points, curve):
        """Distance of each point to the neighbouring segments of `curve` (same grid)."""

        def to_segments(p, a, b):
            ab = b - a
            u = np.clip(((p - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
            return np.linalg.norm(p - (a + u[:, None] * ab), axis=-1)

        before = np.r_[np.inf, to_segments(points[1:], curve[:-1], curve[1:])]
        after = np.r_[to_segments(points[:-1], curve[:-1], curve[1:]), np.inf]
        return np.minimum(before, after)

    def test_four_panels_in_row_order(self):
        self.assertEqual([p.name for p in self.recovered], list(ANOMALY_TYPES.values()))
        for panel in self.recovered:
            np.testing.assert_allclose(panel.t, T)
            self.assertEqual(panel.steps_ahead, STEPS_AHEAD)
            self.assertEqual(panel.neighbors, list(range(-5, 6)))

    def test_weights_are_recovered_to_one_colour_level(self):
        for original, recovered in zip(self.panels, self.recovered):
            level = (original.weights.max() - original.weights.min()) / 256
            np.testing.assert_allclose(recovered.weights, original.weights, atol=level + 1e-12)

    def test_redrawn_paths_and_scores_follow_the_drawn_curves(self):
        process = [ax for ax in self.fig.axes if ax.get_label() == "process"]
        scores = [ax for ax in self.fig.axes if ax.get_label() == "scores"]
        for original, recovered, ax_path, ax_score in zip(
            self.panels, self.recovered, process, scores
        ):
            drawn = self._points(ax_path, T, original.true_path)
            redrawn = self._points(ax_path, recovered.true_path * 0 + T, recovered.true_path)
            self.assertLess(self._distance_to_curve(drawn, redrawn).max(), 0.2)
            drawn = self._points(ax_score, T, original.scores)
            redrawn = self._points(ax_score, T, recovered.scores)
            self.assertLess(self._distance_to_curve(drawn, redrawn).max(), 0.2)
            self.assertAlmostEqual(recovered.threshold, 0.5, places=3)

    def test_values_are_exact_on_most_of_the_grid(self):
        for original, recovered in zip(self.panels, self.recovered):
            exact = np.isclose(recovered.true_path, original.true_path, atol=1e-6)
            self.assertGreater(exact.mean(), 0.9)

    def test_plotted_anomaly_regions_are_recovered(self):
        for original, recovered in zip(self.panels, self.recovered):
            np.testing.assert_array_equal(
                extend_labels(recovered.anomaly_labels), extend_labels(original.anomaly_labels)
            )
            np.testing.assert_array_equal(
                extend_labels(recovered.predicted_labels), extend_labels(original.predicted_labels)
            )


if __name__ == "__main__":
    unittest.main()
