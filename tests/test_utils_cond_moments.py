import os
import sys
import unittest

import numpy as np
from scipy.stats import norm

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(src_dir)

import AD_modules  # noqa: E402
from utils_cond_moments import cond_moments_long_table  # noqa: E402


def _toy_inputs():
    """4 time steps, 2 infants, 1 coordinate, moments ['id', 'var']."""
    cond_moments = np.zeros((4, 2, 1, 2))
    cond_moments[:, :, 0, 0] = [[10.0, 20.0], [11.0, 21.0], [12.0, 22.0], [13.0, 23.0]]
    cond_moments[:, :, 0, 1] = [[4.0, 1.0], [4.0, 1.0], [9.0, 1.0], [9.0, 4.0]]
    observed_dates = np.array(
        [[False, False], [True, False], [False, True], [True, True]])
    obs = np.zeros((4, 2, 1))
    obs[:, :, 0] = [[0.0, 0.0], [9.0, 0.0], [0.0, 25.0], [7.0, 23.0]]
    cutoff_adj_sf = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.5], [1.2, 1.5]])
    abx_exposure = np.array([[0, 0, 1, 1], [0, 0, 0, 2]])  # [infant, time]
    return dict(
        host_id=np.array(["A", "B"]), abx_labels=np.array([1, 1]),
        cond_moments=cond_moments, observed_dates=observed_dates, obs=obs,
        cutoff_adj_sf=cutoff_adj_sf, abx_exposure=abx_exposure,
        output_vars=["id", "var"])


def _left_tail_scores(inputs, epsilon=1e-4):
    moments = inputs["cond_moments"]
    return AD_modules.gaussian_scoring(
        obs=inputs["obs"], cond_exp=np.expand_dims(moments[:, :, :, 0], 3),
        cond_var=np.expand_dims(moments[:, :, :, 1], 3),
        observed_dates=inputs["observed_dates"], scoring_metric="left-tail",
        min_var_val=epsilon, std_sf=inputs["cutoff_adj_sf"])


class TestCondMomentsLongTable(unittest.TestCase):
    def test_one_row_per_observed_day_with_moments_and_exposure(self):
        table = cond_moments_long_table(**_toy_inputs())
        self.assertEqual(
            list(table.columns),
            ["host_id", "abx", "day", "obs", "cond_mean", "cond_std", "sf",
             "abx_exposure"])
        self.assertEqual(table[["host_id", "day"]].values.tolist(),
                         [["A", 1], ["A", 3], ["B", 2], ["B", 3]])
        np.testing.assert_allclose(table["obs"], [9.0, 7.0, 25.0, 23.0])
        np.testing.assert_allclose(table["cond_mean"], [11.0, 13.0, 22.0, 23.0])
        np.testing.assert_allclose(table["cond_std"], [2.0, 3.0, 1.0, 2.0])
        np.testing.assert_allclose(table["sf"], [1.0, 1.2, 1.5, 1.5])
        self.assertEqual(table["abx_exposure"].tolist(), [0, 1, 0, 2])
        self.assertEqual(table["abx"].tolist(), [1, 1, 1, 1])

    def test_table_reproduces_left_tail_normal_score(self):
        inputs = _toy_inputs()
        table = cond_moments_long_table(**inputs)
        z = (table["obs"] - table["cond_mean"]) / (table["cond_std"] * table["sf"])
        score_from_table = -np.log(norm.cdf(z) + 1e-10)
        scores = _left_tail_scores(inputs)
        expected = [scores[t, s, 0, 0] for s, t in [(0, 1), (0, 3), (1, 2), (1, 3)]]
        np.testing.assert_allclose(score_from_table, expected)

    def test_negative_variance_corrected_like_scoring(self):
        inputs = _toy_inputs()
        inputs["cond_moments"][1, 0, 0, 1] = -1.0
        table = cond_moments_long_table(**inputs, epsilon=1e-4)
        self.assertAlmostEqual(table["cond_std"].iloc[0], 0.01)
        z = (table["obs"] - table["cond_mean"]) / (table["cond_std"] * table["sf"])
        scores = _left_tail_scores(inputs)
        expected = [scores[t, s, 0, 0] for s, t in [(0, 1), (0, 3), (1, 2), (1, 3)]]
        np.testing.assert_allclose(-np.log(norm.cdf(z) + 1e-10), expected)

    def test_starting_date_offsets_day(self):
        table = cond_moments_long_table(**_toy_inputs(), starting_date=10)
        self.assertEqual(table["day"].tolist(), [11, 13, 12, 13])

    def test_power2_moments_give_same_std(self):
        inputs = _toy_inputs()
        moments = inputs["cond_moments"]
        power2 = moments.copy()
        power2[:, :, :, 1] = moments[:, :, :, 1] + moments[:, :, :, 0] ** 2
        inputs["cond_moments"], inputs["output_vars"] = power2, ["id", "power-2"]
        table = cond_moments_long_table(**inputs)
        np.testing.assert_allclose(table["cond_std"], [2.0, 3.0, 1.0, 2.0])


if __name__ == "__main__":
    unittest.main()


class TestScoresColumn(unittest.TestCase):
    def test_scores_array_adds_matching_score_column(self):
        inputs = _toy_inputs()
        scores = _left_tail_scores(inputs)[:, :, 0, 0].T  # [infant, time]
        table = cond_moments_long_table(**inputs, scores=scores)
        expected = [scores[s, t] for s, t in [(0, 1), (0, 3), (1, 2), (1, 3)]]
        np.testing.assert_allclose(table["score"], expected)
        self.assertEqual(list(table.columns)[-1], "score")

    def test_no_scores_means_no_score_column(self):
        table = cond_moments_long_table(**_toy_inputs())
        self.assertNotIn("score", table.columns)
