import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def calculate_metrics(y_true, y_pred):
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


def report_metrics(y_true, y_pred, df_metrics, idx_id):
    report = calculate_metrics(y_true, y_pred)

    df_metrics.loc[idx_id, "weighted_avg_f1"] = report["weighted avg"]["f1-score"]
    df_metrics.loc[idx_id, "macro_avg_f1"] = report["macro avg"]["f1-score"]
    df_metrics.loc[idx_id, "true_f1_score"] = report["1.0"]["f1-score"]
    df_metrics.loc[idx_id, "accuracy"] = report["accuracy"]
    df_metrics.loc[idx_id, "MCC"] = matthews_corrcoef(y_true, y_pred)

    return df_metrics


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


def train_n_evaluate_rf_model(target, features_ls, data, stratify_split=True):
    seed = 42
    X = data[features_ls]
    y = data[target]

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=["object"]).columns

    # Preprocessing for categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ],
        remainder="passthrough",  # keep numerical features as is
    )

    # Create a pipeline with preprocessing and classifier
    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=seed)),
        ]
    )

    # Split the data
    train_size = 0.7
    if stratify_split:
        gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
        split = gss.split(data, groups=data["host_id"])
        train_idx, test_idx = next(split)

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1 - train_size, random_state=seed
        )

    # Train the classifier
    clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)

    # evaluate classification
    df_results = report_metrics(
        y_test,
        y_pred,
        pd.DataFrame(),
        "RF",
    )
    return df_results
