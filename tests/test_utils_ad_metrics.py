import os
import sys
import unittest

import numpy as np
import pandas as pd

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(src_dir)

from utils_ad_metrics import (  # noqa: E402
    METRIC_COLUMNS,
    add_macro_f1,
    classification_metrics,
)

# Column order written by evaluate_AD.quant_eval before the macro-averaged F1
# score was added; existing quant_eval CSVs follow it.
LEGACY_COLUMNS = [
    "precision_0", "precision_1", "recall_0", "recall_1", "fscore_0", "fscore_1",
    "support_0", "support_1", "TN", "FP", "FN", "TP", "f1score_micro",
]


class TestClassificationMetrics(unittest.TestCase):
    def setUp(self):
        # TN=6, FP=2, FN=1, TP=1
        self.y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        self.y_pred = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0])
        self.row = classification_metrics(self.y_true, self.y_pred)

    def test_columns_extend_legacy_order(self):
        self.assertEqual(METRIC_COLUMNS, LEGACY_COLUMNS + ["f1score_macro"])
        self.assertEqual(list(self.row.index), METRIC_COLUMNS)

    def test_confusion_counts_and_supports(self):
        self.assertEqual(self.row[["TN", "FP", "FN", "TP"]].tolist(), [6, 2, 1, 1])
        self.assertEqual(self.row[["support_0", "support_1"]].tolist(), [8, 2])

    def test_per_label_metrics(self):
        np.testing.assert_allclose(self.row["precision_0"], 6 / 7)
        np.testing.assert_allclose(self.row["recall_0"], 6 / 8)
        np.testing.assert_allclose(self.row["fscore_0"], 0.8)
        np.testing.assert_allclose(self.row["precision_1"], 1 / 3)
        np.testing.assert_allclose(self.row["recall_1"], 1 / 2)
        np.testing.assert_allclose(self.row["fscore_1"], 0.4)

    def test_micro_f1_is_accuracy_and_macro_is_unweighted_mean(self):
        np.testing.assert_allclose(self.row["f1score_micro"], 0.7)
        np.testing.assert_allclose(self.row["f1score_macro"], 0.6)

    def test_rare_label_pulls_macro_below_micro(self):
        # 1000 normal and 5 anomalous grid points: all anomalies recalled, 4%
        # of the normal points falsely flagged -> micro F1 (accuracy) stays
        # high while precision of label 1 and the macro F1 are low.
        y_true = np.array([0] * 1000 + [1] * 5)
        y_pred = y_true.copy()
        y_pred[:40] = 1
        row = classification_metrics(y_true, y_pred)
        np.testing.assert_allclose(row["recall_1"], 1.0)
        np.testing.assert_allclose(row["precision_1"], 5 / 45)
        np.testing.assert_allclose(row["f1score_micro"], 965 / 1005)
        self.assertLess(row["f1score_macro"], 0.65)
        np.testing.assert_allclose(
            row["f1score_macro"], (row["fscore_0"] + row["fscore_1"]) / 2)

    def test_single_class_without_positive_predictions(self):
        # anomaly-free evaluation set: every metric of label 1 is zero and no
        # exception is raised
        y_true = np.zeros(20, dtype=int)
        row = classification_metrics(y_true, y_true)
        self.assertEqual(row[["support_1", "TP", "FP", "FN"]].tolist(), [0, 0, 0, 0])
        self.assertEqual(row[["precision_1", "recall_1", "fscore_1"]].tolist(), [0, 0, 0])
        np.testing.assert_allclose(row["f1score_micro"], 1.0)
        np.testing.assert_allclose(row["f1score_macro"], 0.5)


class TestAddMacroF1(unittest.TestCase):
    def test_adds_mean_of_per_label_f1_to_legacy_table(self):
        legacy = pd.DataFrame(
            [[0.0, 0.379, 0.0, 1.0, 0.0, 0.549, 34579.0, 21071.0,
              0.0, 34579.0, 0.0, 21071.0, 0.379],
             [0.0] * 6 + [55650.0, 0.0, 0.0, 55650.0, 0.0, 0.0, 0.0]],
            columns=LEGACY_COLUMNS, index=["anomalous", "anomaly_free"])
        result = add_macro_f1(legacy)
        self.assertEqual(list(result.columns), METRIC_COLUMNS)
        np.testing.assert_allclose(
            result["f1score_macro"].to_numpy(), [0.549 / 2, 0.0])
        self.assertNotIn("f1score_macro", legacy.columns)


if __name__ == "__main__":
    unittest.main()
