"""Classification metrics of the anomaly detection (AD) modules on labelled
grid points of the synthetic evaluation sets (evaluate_AD.quant_eval)."""

import pandas as pd
from sklearn import metrics

LABELS = [0, 1]

# One row per evaluation set; the first 13 columns match the quant_eval CSVs
# written before the macro-averaged F1 score was added.
METRIC_COLUMNS = (
    [f"{name}_{label}" for name in ["precision", "recall", "fscore", "support"]
     for label in LABELS]
    + ["TN", "FP", "FN", "TP", "f1score_micro", "f1score_macro"]
)


def classification_metrics(ad_labels, pred_labels):
    """Per-label precision, recall, F1 score and support, the confusion
    counts and the micro- and macro-averaged F1 scores of binary anomaly
    labels, as a Series indexed by METRIC_COLUMNS.

    The micro average weights each label by its support and equals the
    accuracy; the macro average is the unweighted mean of the two labels' F1
    scores and hence reflects the detection of a rare anomalous label.
    """
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(
        ad_labels, pred_labels, labels=LABELS, zero_division=0)
    confusion = metrics.confusion_matrix(ad_labels, pred_labels, labels=LABELS)
    f1_micro = metrics.f1_score(
        ad_labels, pred_labels, labels=LABELS, average="micro", zero_division=0)
    f1_macro = metrics.f1_score(
        ad_labels, pred_labels, labels=LABELS, average="macro", zero_division=0)
    values = [
        *precision, *recall, *fscore, *support, *confusion.ravel(),
        f1_micro, f1_macro,
    ]
    return pd.Series(values, index=METRIC_COLUMNS, dtype=float)


def add_macro_f1(df_quant_res):
    """Return a copy of a quant_eval table (one row per evaluation set) with
    the macro-averaged F1 score appended, computed from the per-label F1
    scores; for tables written before this column existed."""
    df_quant_res = df_quant_res.copy()
    df_quant_res["f1score_macro"] = (
        df_quant_res[["fscore_0", "fscore_1"]].mean(axis=1))
    return df_quant_res[METRIC_COLUMNS]
