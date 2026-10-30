"""
Composes the figure of the anomalies detected in the simulated time series
(Fig. 2 of the manuscript): the learned score aggregation weights of the
anomaly detection (AD) modules (a) and example test paths with their predicted
scores and anomaly regions (b) for the drift, diffusion, noise and spike
anomalies. The panels are computed as in evaluate_AD.py (plot-only mode) from
the trained forecast model and the four trained AD modules, and drawn at the
printed size of the figure (figure_style). Run from src/:

    uv run python figure_simulation_insights.py \\
        --ad_models_path=../data/saved_models_AD_OrnsteinUhlenbeckWithSeason/ \\
        --output=simulation_insights.pdf

Required inputs below `ad_models_path`: model_overview.csv, id-<forecast_model_id>/
best_checkpoint/checkpt.tar (forecast model), id-<forecast_model_id>/
saved_AD_module_<type>/id-0/<ad_module_version>_checkpoint/checkpt.tar (AD
modules, type in cutoff, diffusion, noise, spike), and in the training-data
folder the anomalous datasets AD_OrnsteinUhlenbeckWithSeason_<type>_dict of the
configuration and the training set of the forecast model.
"""

import json
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from absl import app, flags
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import AD_modules
import data_utils
import evaluate_AD
import models_synthetic as models
from configs import config
from figure_score_delineation import FLAGS  # noqa: F401  (output flag)
from figure_style import (
    BASE_FONTSIZE,
    LETTER_FONTSIZE,
    TEXT_WIDTH_IN,
    add_panel_letters,
    rc_params,
    save_figure,
)

# anomaly types (names of the configuration) in the row order of the figure,
# with the index of the test path shown in column b (as published)
ANOMALY_TYPES = {
    "cutoff": "Drift",
    "diffusion": "Diffusion",
    "noise": "Noise",
    "spike": "Spike",
}
PATH_INDEX = {"cutoff": 0, "diffusion": 4, "noise": 1, "spike": 3}
# AD module configuration (configs/synthetic/config_AD_model.py)
STEPS_AHEAD = [1, 3, 5, 7, 10, 12, 15, 18, 20]
NEIGHBOURING_SCORES = 5
CLASS_THRES = 0.5
TEST_SIZE, SEED = 0.2, 398
# the plotted batch of the evaluation: the first max(paths_to_plot) + 1 test
# paths with paths_to_plot = (0, 1, 2, 3, 4)
PLOT_BATCH_SIZE = 5
DATASET_DICT = "AD_OrnsteinUhlenbeckWithSeason_{}_dict"
AD_MODULE_FOLDER = "saved_AD_module_{}"

# height such that figure and caption fit one page
FIGURE_SIZE_IN = (TEXT_WIDTH_IN, 5.35)
RC_PARAMS = rc_params()
# colours of the published panels: first two of the tableau-colorblind10 cycle
# for the true path, green scores, red threshold and anomaly shading
TRUE_PATH_COLORS = ("#006BA4", "#FF800E")
SCORE_COLOR, ANOMALY_COLOR = "green", "red"
LEGEND_LABELS = [
    "predicted scores",
    "score threshold",
    "predicted anomaly",
    "true path, no anomaly",
    "true path, anomaly",
]
# the x axes are shared along the columns: tick labels and x labels on the
# bottom row only
GRID = dict(
    left=0.09,
    right=0.91,
    top=0.965,
    bottom=0.17,
    wspace=0.4,
    hspace=0.35,
    width_ratios=[1.25, 2],
)
COLORBAR = dict(shrink=0.6, pad=0.04, aspect=12)
# every neighbour has a tick, every second one a label
NEIGHBOR_LABEL_STEP = 2
LETTER_Y = 0.99
LETTER_OFFSET = 0.075
# legend of the path panels centred below them, one column per group of
# entries: the scores (left axis), then the paths (right axis)
LEGEND = dict(
    loc="upper center",
    ncol=2,
    frameon=False,
    borderaxespad=0.0,
    handlelength=1.4,
    columnspacing=1.2,
    handletextpad=0.5,
)
LEGEND_GAP_PT = 1.5

flags.DEFINE_string(
    "ad_models_path",
    "../data/saved_models_AD_OrnsteinUhlenbeckWithSeason/",
    "folder of the forecast model and its AD modules",
)
flags.DEFINE_integer("forecast_model_id", 1, "id of the forecast model")
flags.DEFINE_string(
    "ad_module_version", "last", "AD module checkpoint: 'last' (epoch 50) or 'best'"
)


@dataclass(frozen=True)
class AnomalyPanel:
    """Weights of one AD module and the scored test path of one anomaly type."""

    name: str
    weights: np.ndarray
    steps_ahead: list
    neighbors: list
    t: np.ndarray
    true_path: np.ndarray
    anomaly_labels: np.ndarray
    scores: np.ndarray
    predicted_labels: np.ndarray
    threshold: float


def extend_labels(labels):
    """Labels extended by one grid point on each side of an anomaly (as plotted)."""
    labels = np.asarray(labels)
    extended = labels.copy()
    extended[1:][np.logical_and(labels[1:] == 0, labels[:-1] == 1)] = 1
    extended[:-1][np.logical_and(labels[1:] == 1, labels[:-1] == 0)] = 1
    return extended


def _require(path, what):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{what} not found: {path}")


def require_dataset(data_dict, training_data_path=data_utils.training_data_path):
    """
    Raises FileNotFoundError naming `data_dict` (a dataset dictionary of the
    configuration) if it is not listed in dataset_overview.csv of
    `training_data_path` or its data.npy / metadata.txt are missing.
    """
    df_overview, overview_file = data_utils.get_dataset_overview(training_data_path)
    description = json.dumps(eval("config." + data_dict), sort_keys=True)
    listed = df_overview.loc[df_overview["description"] == description]
    if len(listed) == 0:
        raise FileNotFoundError(
            f"dataset {data_dict} not found in {overview_file}; generate it with "
            "data_utils.py (see the README)"
        )
    name, time_id = listed.iloc[-1][["name", "id"]]
    folder = os.path.join(training_data_path, f"{name}-{int(time_id)}", "")
    for filename in ("data.npy", "metadata.txt"):
        _require(os.path.join(folder, filename), f"file of dataset {data_dict}")


def forecast_checkpoint_path(ad_models_path, forecast_model_id, load_which="best"):
    return os.path.join(
        ad_models_path, f"id-{forecast_model_id}", f"{load_which}_checkpoint", ""
    )


def ad_module_checkpoint_path(ad_models_path, forecast_model_id, anomaly_type, version):
    return os.path.join(
        ad_models_path,
        f"id-{forecast_model_id}",
        AD_MODULE_FOLDER.format(anomaly_type),
        "id-0",
        f"{version}_checkpoint",
        "",
    )


def require_inputs(ad_models_path, forecast_model_id, anomaly_type, ad_module_version):
    """Checks every input of `load_anomaly_panel` and returns the forecast parameters."""
    forecast_param = forecast_param_from_overview(ad_models_path, forecast_model_id)
    _require(
        os.path.join(forecast_checkpoint_path(ad_models_path, forecast_model_id), "checkpt.tar"),
        "forecast model checkpoint",
    )
    _require(
        os.path.join(
            ad_module_checkpoint_path(
                ad_models_path, forecast_model_id, anomaly_type, ad_module_version
            ),
            "checkpt.tar",
        ),
        f"AD module checkpoint ({anomaly_type})",
    )
    require_dataset(forecast_param["data_dict"])
    require_dataset(DATASET_DICT.format(anomaly_type))
    return forecast_param


def forecast_param_from_overview(ad_models_path, forecast_model_id):
    """Parameters of the forecast model from model_overview.csv, as evaluate_AD."""
    overview = os.path.join(ad_models_path, "model_overview.csv")
    _require(overview, "model overview of the forecast models (model_overview.csv)")
    df_overview = pd.read_csv(overview, index_col=0)
    rows = df_overview.loc[df_overview["id"] == forecast_model_id, "description"]
    if len(rows) == 0:
        raise FileNotFoundError(f"model id {forecast_model_id} not in {overview}")
    forecast_param = json.loads(rows.values[0])
    forecast_param["model_id"] = forecast_model_id
    if "dataset" not in forecast_param:
        data_dict = forecast_param["data_dict"]
        if isinstance(data_dict, str):
            data_dict = eval("config." + data_dict)
        forecast_param["dataset"] = data_dict["model_name"]
    return forecast_param


def load_forecast_model(ad_models_path, forecast_model_id, load_which="best", device="cpu"):
    """The trained forecast model with its parameters and collate function."""
    forecast_param = forecast_param_from_overview(ad_models_path, forecast_model_id)
    params_dict, collate_fn = evaluate_AD.get_forecast_model_param_dict(**forecast_param)
    model = models.NJODE(**params_dict)
    model.to(device)
    load_path = forecast_checkpoint_path(ad_models_path, forecast_model_id, load_which)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
    models.get_ckpt_model(load_path, model, optimizer, device)
    model.eval()
    return model, params_dict, collate_fn, forecast_param


def load_ad_module(
    ad_models_path, forecast_model_id, anomaly_type, version, output_vars, replace_values, device="cpu"
):
    """The trained AD module of `anomaly_type` (checkpoint `version`)."""
    ad_module = AD_modules.AD_module(
        output_vars=output_vars,
        steps_ahead=STEPS_AHEAD,
        smoothing=NEIGHBOURING_SCORES,
        replace_values=replace_values,
        class_thres=CLASS_THRES,
    )
    load_path = ad_module_checkpoint_path(
        ad_models_path, forecast_model_id, anomaly_type, version
    )
    optimizer = torch.optim.Adam(ad_module.parameters(), lr=0.01, weight_decay=0.0005)
    AD_modules.get_ckpt_model(load_path, ad_module, optimizer, device)
    ad_module.eval()
    return ad_module


def test_batch(anomaly_type, collate_fn, batch_size=PLOT_BATCH_SIZE):
    """
    The first `batch_size` test paths of the anomalous dataset (the split and
    the plotted batch of evaluate_AD; the forecasts of a path depend on the
    observation times of its batch).
    """
    dataset, dataset_id = data_utils._get_dataset_name_id_from_dict(
        data_dict=DATASET_DICT.format(anomaly_type)
    )
    dataset_id = int(dataset_id)
    metadata = data_utils.load_metadata(stock_model_name=dataset, time_id=dataset_id)
    _, test_idx = train_test_split(
        np.arange(metadata["nb_paths"]), test_size=TEST_SIZE, random_state=SEED
    )
    data_test = evaluate_AD.IrregularADDataset(model_name=dataset, time_id=dataset_id, idx=test_idx)
    dl_plot = DataLoader(
        dataset=data_test, collate_fn=collate_fn, shuffle=False, batch_size=batch_size
    )
    return next(iter(dl_plot)), metadata


def score_batch(forecast_model, ad_module, batch, delta_t, T, output_vars, device="cpu"):
    """
    Grid times, true paths, anomaly labels, aggregated scores and predicted
    labels of `batch` (arrays [paths, dimension, grid points]), as computed in
    evaluate_AD.plot_one_path_with_pred.
    """
    M = batch["M"].to(device) if batch["M"] is not None else None
    start_M = batch["start_M"].to(device) if batch["start_M"] is not None else None
    path_t_true_X = np.linspace(0.0, T, int(np.round(T / delta_t)) + 1)
    true_X = batch["true_paths"]
    with torch.no_grad():
        path_t, path_y = forecast_model.custom_forward(
            batch["times"],
            batch["time_ptr"],
            batch["X"].to(device),
            batch["obs_idx"],
            delta_t,
            T,
            batch["start_X"].to(device),
            batch["n_obs_ot"].to(device),
            steps=STEPS_AHEAD,
            get_loss=False,
            M=M,
            start_M=start_M,
        )
    nb_steps, nb_vars = path_t_true_X.shape[0], len(output_vars)
    batch_size, dimension = batch["ad_labels"].shape[:2]
    cond_moments = np.full((nb_steps, batch_size, dimension, nb_vars, len(STEPS_AHEAD)), np.nan)
    for i in range(len(STEPS_AHEAD)):
        index_times = np.array(
            [np.any(np.abs(t - path_t[i]) < 1e-10) for t in path_t_true_X]
        )
        index_times = np.argwhere(index_times).reshape(-1)
        for m in range(nb_vars):
            cond_moments[index_times, :, :, m, i] = path_y[i].detach().cpu().numpy()[
                :, :, m * dimension : (m + 1) * dimension
            ]
    obs = torch.tensor(true_X, dtype=torch.float32).permute(2, 0, 1)
    cond_moments = torch.tensor(cond_moments, dtype=torch.float32)
    predicted_labels, scores = ad_module.get_predicted_label(obs, cond_moments)
    scores = scores.detach().cpu().numpy().transpose((1, 2, 0))
    predicted_labels = predicted_labels.detach().cpu().numpy().transpose((1, 2, 0))
    return path_t_true_X, true_X, batch["ad_labels"], scores, predicted_labels


def load_anomaly_panel(
    anomaly_type,
    ad_models_path,
    forecast_model_id=1,
    ad_module_version="last",
    path_index=0,
    device="cpu",
):
    """
    The AnomalyPanel of `anomaly_type` from the saved models and datasets
    (every input is checked first). The replacement values of the AD module
    are computed from a random training batch as in evaluate_AD; the seed makes
    the script deterministic.
    """
    require_inputs(ad_models_path, forecast_model_id, anomaly_type, ad_module_version)
    forecast_model, params_dict, collate_fn, forecast_param = load_forecast_model(
        ad_models_path, forecast_model_id, device=device
    )
    output_vars = params_dict["output_vars"]
    torch.manual_seed(SEED)
    replace_values = evaluate_AD.get_replace_forecast_values(
        model=forecast_model,
        data_dict=forecast_param["data_dict"],
        collate_fn=collate_fn,
        steps_ahead=STEPS_AHEAD,
        device=device,
    )
    ad_module = load_ad_module(
        ad_models_path,
        forecast_model_id,
        anomaly_type,
        ad_module_version,
        output_vars,
        replace_values,
        device,
    )
    batch, metadata = test_batch(anomaly_type, collate_fn)
    t, true_X, ad_labels, scores, predicted = score_batch(
        forecast_model, ad_module, batch, metadata["dt"], metadata["maturity"], output_vars, device
    )
    return AnomalyPanel(
        name=ANOMALY_TYPES[anomaly_type],
        weights=ad_module.get_weights().squeeze().clone().detach().numpy(),
        steps_ahead=STEPS_AHEAD,
        neighbors=list(range(-NEIGHBOURING_SCORES, NEIGHBOURING_SCORES + 1)),
        t=t,
        true_path=true_X[path_index, 0, :],
        anomaly_labels=ad_labels[path_index, 0, :],
        scores=scores[path_index, 0, :],
        predicted_labels=predicted[path_index, 0, :],
        threshold=ad_module.threshold,
    )


def _draw_weights(fig, ax, panel, bottom_row):
    im = ax.imshow(panel.weights)
    ax.set_xticks(np.arange(panel.weights.shape[1]))
    ax.set_yticks(np.arange(panel.weights.shape[0]))
    ax.set_xticklabels(
        [str(i) if i % NEIGHBOR_LABEL_STEP == 0 else "" for i in panel.neighbors]
    )
    ax.set_yticklabels([str(k) for k in panel.steps_ahead])
    ax.set_title(panel.name, fontsize=BASE_FONTSIZE)
    ax.set_xlabel("neighboring timestamps ($l$)" if bottom_row else "", fontsize=BASE_FONTSIZE)
    ax.set_ylabel("steps-ahead ($k$)", fontsize=BASE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=BASE_FONTSIZE, labelbottom=bottom_row)
    colorbar = fig.colorbar(im, ax=ax, **COLORBAR)
    colorbar.ax.tick_params(labelsize=BASE_FONTSIZE)


def _draw_path(ax, panel, bottom_row):
    anomaly = extend_labels(panel.anomaly_labels)
    predicted = extend_labels(panel.predicted_labels)
    path_anomaly = np.ma.masked_where(anomaly == 0, panel.true_path)
    scores = np.ma.masked_where(np.isnan(panel.scores), panel.scores)
    ax2 = ax.twinx()
    ax2.set_label("process")
    ax.set_title(f"{panel.name} anomaly", fontsize=BASE_FONTSIZE)
    ax2.plot(panel.t, panel.true_path, label=LEGEND_LABELS[3], color=TRUE_PATH_COLORS[0])
    ax2.plot(panel.t, path_anomaly, label=LEGEND_LABELS[4], color=TRUE_PATH_COLORS[1])
    ax.plot(panel.t, scores, label=LEGEND_LABELS[0], color=SCORE_COLOR, alpha=0.75)
    ax.fill_between(
        panel.t,
        0,
        1,
        where=predicted == 1,
        facecolor=ANOMALY_COLOR,
        alpha=0.25,
        label=LEGEND_LABELS[2],
    )
    ax.axhline(y=panel.threshold, color="r", linestyle="-", alpha=0.5, label=LEGEND_LABELS[1])
    ax.set_ylim(0, 1)
    ax.set_ylabel("score", fontsize=BASE_FONTSIZE)
    ax2.set_ylabel("process values", fontsize=BASE_FONTSIZE)
    ax.set_xlabel("$t$" if bottom_row else "", fontsize=BASE_FONTSIZE)
    for axis in (ax, ax2):
        axis.tick_params(axis="both", labelsize=BASE_FONTSIZE)
    ax.tick_params(axis="x", labelbottom=bottom_row)


def legend_handles():
    """Handles of the shared legend of the path panels, in LEGEND_LABELS order."""
    return [
        Line2D([0], [0], color=SCORE_COLOR, alpha=0.75),
        Line2D([0], [0], color="r", alpha=0.5),
        Patch(facecolor=ANOMALY_COLOR, alpha=0.25),
        Line2D([0], [0], color=TRUE_PATH_COLORS[0]),
        Line2D([0], [0], color=TRUE_PATH_COLORS[1]),
    ]


def compose_simulation_insights(panels):
    """
    Composes the figure: one row per AnomalyPanel of `panels` with the
    aggregation weights (column a) and the scored test path (column b).
    """
    with plt.rc_context(RC_PARAMS):
        fig = plt.figure(figsize=FIGURE_SIZE_IN)
        grid = fig.add_gridspec(len(panels), 2, **GRID)
        first_row = []
        for row, panel in enumerate(panels):
            bottom_row = row == len(panels) - 1
            ax_weights = fig.add_subplot(grid[row, 0], label="weights")
            _draw_weights(fig, ax_weights, panel, bottom_row)
            ax_scores = fig.add_subplot(grid[row, 1], label="scores")
            _draw_path(ax_scores, panel, bottom_row)
            if row == 0:
                first_row = [ax_weights, ax_scores]
        fig.draw_without_rendering()
        bottom = [ax for ax in fig.axes if ax.get_label() == "scores"][-1]
        axes = bottom.get_window_extent()
        label_bottom = bottom.xaxis.label.get_window_extent().y0
        fig.legend(
            legend_handles(),
            LEGEND_LABELS,
            fontsize=BASE_FONTSIZE,
            bbox_to_anchor=(
                (axes.x0 + axes.x1) / 2 / fig.bbox.width,
                (label_bottom - LEGEND_GAP_PT * fig.dpi / 72) / fig.bbox.height,
            ),
            **LEGEND,
        )
        positions = {
            letter: (ax.get_position().x0 - LETTER_OFFSET, LETTER_Y)
            for letter, ax in zip("ab", first_row)
        }
        add_panel_letters(fig, positions, LETTER_FONTSIZE)
    return fig


def main(argv):
    del argv
    panels = [
        load_anomaly_panel(
            anomaly_type,
            FLAGS.ad_models_path,
            FLAGS.forecast_model_id,
            FLAGS.ad_module_version,
            PATH_INDEX[anomaly_type],
        )
        for anomaly_type in ANOMALY_TYPES
    ]
    fig = compose_simulation_insights(panels)
    output = FLAGS.output or os.path.join(
        FLAGS.ad_models_path, f"id-{FLAGS.forecast_model_id}", "simulation_insights.pdf"
    )
    with plt.rc_context(RC_PARAMS):
        save_figure(fig, output)
    print("Saved figure to", output)


if __name__ == "__main__":
    app.run(main)
