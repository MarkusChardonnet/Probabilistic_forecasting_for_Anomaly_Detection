import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    print(classification_report(y_true, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 2), dpi=400)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    return report


def plot_distribution(columns, dataframes, figsize=(10, 6), kde=True):
    """
    Plot distribution(s) of specified column(s) from one or multiple DataFrames.
    """
    if isinstance(columns, str):
        columns = [columns]

    num_cols = len(columns)
    sharex = num_cols > 1  # Share x-axis if multiple columns

    fig, axes = plt.subplots(
        num_cols, 1, figsize=figsize, squeeze=False, sharex=sharex, dpi=400
    )

    for ax, column in zip(axes.flatten(), columns):
        for label, df in dataframes.items():
            sns.histplot(df[column], kde=kde, bins=30, label=label, alpha=0.6, ax=ax)
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column if num_cols == 1 else "")
        ax.set_ylabel("Frequency")
        ax.legend()

    plt.tight_layout()
    plt.show()
