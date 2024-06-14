# =====================================================================================================================
import pandas as pd
import json
from absl import app
from absl import flags
import torch  # machine learning
import torch.nn as nn
import tqdm  # process bar for iterations
import numpy as np  # large arrays and matrices, functions
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import os, sys
import pandas as pd  # data analysis and manipulation
import json  # storing and exchanging data
import time
import socket
import matplotlib  # plots
import matplotlib.colors
import matplotlib.pyplot as plt
from torch.backends import cudnn
import gc
import math
from sklearn import metrics
import seaborn as sns

from configs import config
import models
import data_utils
from AD_modules import AD_module, Simple_AD_module
import AD_modules

try:
    from telegram_notifications import send_bot_message as SBM
except Exception:
    from configs.config import SendBotMessage as SBM


# =====================================================================================================================
# FLAGS
FLAGS = flags.FLAGS

flags.DEFINE_string("forecast_params", None, "name of the params list (in config.py) to "
                                    "use for parallel run")
flags.DEFINE_string("forecast_model_ids", None,
                    "name of list of model ids (in config.py) to run or list "
                    "of model ids")
flags.DEFINE_string("forecast_saved_models_path", None,
                    "path where the models are saved")
flags.DEFINE_string("anomaly_data_dict", None,
                    "blabla")
flags.DEFINE_string("ad_params", None,
                    "blabla")
flags.DEFINE_bool("evaluate", False, "whether to evaluate the model")
flags.DEFINE_bool("compute_scores", False, "whether to compute the scores")
flags.DEFINE_bool("evaluate_scores", False, "whether to evaluate the scores")

flags.DEFINE_bool("USE_GPU", False, "whether to use GPU for training")
flags.DEFINE_bool("ANOMALY_DETECTION", False,
                  "whether to run in torch debug mode")
flags.DEFINE_integer("N_DATASET_WORKERS", 0,
                     "number of processes that generate batches in parallel")

# check whether running on computer or server
if 'ada-' not in socket.gethostname():
    SERVER = False
    flags.DEFINE_integer("NB_JOBS", 1,
                         "nb of parallel jobs to run  with joblib")
    flags.DEFINE_integer("NB_CPUS", 1, "nb of CPUs used by each training")
    flags.DEFINE_bool("SEND", False, "whether to send with telegram bot")
else:
    SERVER = True
    flags.DEFINE_integer("NB_JOBS", 24,
                         "nb of parallel jobs to run  with joblib")
    flags.DEFINE_integer("NB_CPUS", 2, "nb of CPUs used by each training")
    flags.DEFINE_bool("SEND", True, "whether to send with telegram bot")
    matplotlib.use('Agg')

print(socket.gethostname())
print('SERVER={}'.format(SERVER))


# ==============================================================================
# Global variables

data_path = config.data_path
# saved_models_path = config.saved_models_path
train_data_path = config.training_data_path
flagfile = config.flagfile

default_ode_nn = ((50, 'tanh'), (50, 'tanh'))
default_readout_nn = ((50, 'tanh'), (50, 'tanh'))
default_enc_nn = ((50, 'tanh'), (50, 'tanh'))

N_DATASET_WORKERS = 0
USE_GPU = False

training_data_path = config.training_data_path

# =====================================================================================================================
# Functions
makedirs = config.makedirs

def load_AD_dataset(stock_model_name="BlackScholes", time_id=None):
    """
    load a saved dataset by its name and id
    :param stock_model_name: str, name
    :param time_id: int, id
    :return: np.arrays of stock_paths, observed_dates, number_observations
                dict of hyperparams of the dataset
    """
    time_id = data_utils._get_time_id(stock_model_name=stock_model_name, time_id=time_id)
    path = '{}{}-{}/'.format(training_data_path, stock_model_name, int(time_id))

    with open('{}data.npy'.format(path), 'rb') as f:
        stock_paths = np.load(f)
        observed_dates = np.load(f)
        nb_obs = np.load(f)
        ad_labels = np.load(f)
    with open('{}metadata.txt'.format(path), 'r') as f:
        hyperparam_dict = json.load(f)

    return stock_paths, observed_dates, nb_obs, ad_labels, hyperparam_dict


def get_model_predictions(
        dl, device, forecast_model, output_vars, T, delta_t, dimension):
    """
    Get the NJODE model predictions for the given dataset
    """
    b = next(iter(dl))

    times = b["times"]
    time_ptr = b["time_ptr"]
    X = b["X"].to(device)
    Z = b["Z"].to(device)
    S = b["S"].to(device)
    start_X = b["start_X"].to(device)
    start_Z = b["start_Z"].to(device)
    start_S = b["start_S"].to(device)
    obs_idx = b["obs_idx"]
    n_obs_ot = b["n_obs_ot"].to(device)
    observed_dates = np.transpose(b['observed_dates'], (1, 0)).astype(np.bool)
    path_t_true_X = np.linspace(0., T, int(np.round(T / delta_t)) + 1)
    true_X = b["true_paths"]
    abx_labels = b["abx_observed"]
    host_id = b["host_id"]

    with torch.no_grad():
        res = forecast_model.get_pred(
            times=times, time_ptr=time_ptr, X=torch.cat((X, Z), dim=1),
            obs_idx=obs_idx, delta_t=None, S=S, start_S=start_S,
            T=T, start_X=torch.cat((start_X, start_Z), dim=1))
        path_y_pred = res['pred'].detach().cpu().numpy()
        path_t_pred = res['pred_t']
        torch.cuda.empty_cache()

    indices = []
    for t in path_t_true_X:
        indices.append(np.argmin(np.abs(path_t_pred - t)))
    y_preds = path_y_pred[indices]

    nb_steps = path_t_true_X.shape[0]
    nb_moments = len(output_vars)

    cond_moments = y_preds.reshape(
        (nb_steps, dl.batch_size, dimension, nb_moments))
    # cond_moments[0] = np.nan
    observed_dates[0] = False

    return cond_moments, observed_dates, true_X, abx_labels, host_id


def compute_scores(
        forecast_saved_models_path,
        forecast_model_id=None,
        dataset='microbial_genus',
        forecast_param=None,
        load_best=True,
        use_gpu=None,
        nb_cpus=None,
        n_dataset_workers=None,
        nb_MC_samples=10**5,
        verbose=False,
        epsilon=1e-6,
        seed=333,
        send=False,
        use_replace_values=False,
        dirichlet_use_coord=None,
        **options
):
    """
    Compute the anomaly detection scores

    args:
        forecast_saved_models_path: str, path to the saved models
        forecast_model_id: int, id of the model
        dataset: str, name of the dataset
        forecast_param: dict, parameters of the forecast model (NJODE)
        load_best: bool, whether to load the best model instance or the last one
            (during training)
        use_gpu: bool, whether to use GPU
        nb_cpus: int, number of CPUs to use
        n_dataset_workers: int, number of dataset workers
        nb_MC_samples: int, number of MC samples for approximating the pvalue
            when using the dirichlet distribution
        verbose: bool, whether to print the progress
        epsilon: float, small value to avoid division by zero etc.
        seed: int, seed for reproducibility
        use_replace_values: bool, whether to use replace values for variance
        dirichlet_use_coord: int, which coordinate to use for the dirichlet
            distribution as factor, if None: median of all coordinates is used
    """
    global USE_GPU, N_CPUS, N_DATASET_WORKERS
    if use_gpu is not None:
        USE_GPU = use_gpu
    if nb_cpus is not None:
        N_CPUS = nb_cpus
    if n_dataset_workers is not None:
        N_DATASET_WORKERS = n_dataset_workers

    if USE_GPU and torch.cuda.is_available():
        gpu_num = 0
        device = torch.device("cuda:{}".format(gpu_num))
        torch.cuda.set_device(gpu_num)
    else:
        device = torch.device("cpu")

    scores_dict = {
        "model_id": forecast_model_id, "dataset": dataset,
        "load_best": load_best,
        "nb_MC_samples": nb_MC_samples, "seed": seed, "epsilon": epsilon,
        "use_replace_values": use_replace_values,
        "dirichlet_use_coord": dirichlet_use_coord
    }

    # load dataset-metadata

    train_idx = np.load(os.path.join(
        train_data_path, dataset, "all", 'train_idx.npy'
    ), allow_pickle=True)
    val_idx = np.load(os.path.join(
        train_data_path, dataset, "all", 'val_idx.npy'
    ), allow_pickle=True)

    data_train = data_utils.MicrobialDataset(
        dataset_name=dataset, idx=train_idx)
    data_val = data_utils.MicrobialDataset(
        dataset_name=dataset, idx=val_idx)

    dataset_metadata = data_train.get_metadata()
    dimension = dataset_metadata['dimension']
    T = dataset_metadata['maturity']
    delta_t = dataset_metadata['dt']  # copy metadata
    starting_date = dataset_metadata['starting_date']

    # get additional plotting information
    plot_forecast_predictions = False
    if 'plot_forecast_predictions' in options:
        plot_forecast_predictions = options['plot_forecast_predictions']
    std_factor = 1  # factor with which the std is multiplied
    if 'plot_variance' in options:
        plot_variance = options['plot_variance']
    if 'std_factor' in options:
        std_factor = options['std_factor']
    class_thres = 0.5
    autom_thres = None
    if 'class_thres' in options:
        class_thres = options['class_thres']
        if 'autom_thres' in options:
            autom_thres = options['autom_thres']

    # get all needed paths
    forecast_model_path = '{}id-{}/'.format(
        forecast_saved_models_path, forecast_model_id)
    forecast_model_path_save_best = '{}best_checkpoint/'.format(
        forecast_model_path)
    forecast_model_path_save_last = '{}last_checkpoint/'.format(
        forecast_model_path)
    ad_path = '{}anomaly_detection/'.format(forecast_model_path)
    which = 'best' if load_best else 'last'
    scores_path = '{}scores_{}/'.format(ad_path, which)
    makedirs(scores_path)

    # get params_dict
    forecast_params_dict, collate_fn = get_forecast_model_param_dict(
        **forecast_param
    )
    output_vars = forecast_params_dict['output_vars']

    forecast_model = models.NJODE(
        **forecast_params_dict)  # get NJODE model class from
    forecast_model.to(device)

    forecast_optimizer = torch.optim.Adam(forecast_model.parameters(), lr=0.001,
                                          weight_decay=0.0005)
    path = forecast_model_path_save_best if load_best else \
        forecast_model_path_save_last
    models.get_ckpt_model(path, forecast_model,
                          forecast_optimizer, device)
    forecast_model.eval()
    del forecast_optimizer

    dl_train = DataLoader(
        dataset=data_train, collate_fn=collate_fn, shuffle=False,
        batch_size=len(train_idx))
    dl_val = DataLoader(
        dataset=data_val, collate_fn=collate_fn, shuffle=False,
        batch_size=len(val_idx))

    cond_moments, observed_dates, true_X, abx_labels, host_id = \
        get_model_predictions(
            dl_train, device, forecast_model, output_vars, T, delta_t,
            dimension)
    if use_replace_values:
        replace_values = get_replace_forecast_values(
            cond_moments=cond_moments, output_vars=output_vars, device=device)
        replace_values = replace_values['var']
    else:
        replace_values = None

    ad_module = Simple_AD_module(
        output_vars=output_vars,
        nb_MC_samples=nb_MC_samples,
        distribution_class="beta",
        replace_values=replace_values,
        class_thres=class_thres,
        seed=seed,
        epsilon=epsilon,
        dirichlet_use_coord=dirichlet_use_coord,
        verbose=verbose)

    # train data
    obs = true_X.transpose(2, 0, 1)
    ad_scores = ad_module(obs, cond_moments, observed_dates)
    with open('{}train_ad_scores.npy'.format(scores_path), 'wb') as f:
        np.save(f, ad_scores)
        np.save(f, abx_labels)
        np.save(f, host_id)
    data = np.concatenate(
        [host_id.reshape(-1,1), abx_labels.reshape(-1, 1), ad_scores], axis=1)
    cols = ['host_id', 'abx'] + ['ad_score_day-{}'.format(i+starting_date)
                                 for i in range(ad_scores.shape[1])]
    df = pd.DataFrame(data, columns=cols)
    csvpath = '{}train_ad_scores.csv'.format(scores_path)
    df.to_csv(csvpath, index=False)

    # test data
    cond_moments, observed_dates, true_X, abx_labels, host_id = \
        get_model_predictions(
            dl_val, device, forecast_model, output_vars, T, delta_t, dimension)
    obs = true_X.transpose(2, 0, 1)
    ad_scores = ad_module(obs, cond_moments, observed_dates)
    with open('{}val_ad_scores.npy'.format(scores_path), 'wb') as f:
        np.save(f, ad_scores)
        np.save(f, abx_labels)
        np.save(f, host_id)
    data = np.concatenate(
        [host_id.reshape(-1,1), abx_labels.reshape(-1, 1), ad_scores], axis=1)
    cols = ['host_id', 'abx'] + ['ad_score_day-{}'.format(i+starting_date)
                                 for i in range(ad_scores.shape[1])]
    df = pd.DataFrame(data, columns=cols)
    csvpath_val = '{}val_ad_scores.csv'.format(scores_path)
    df.to_csv(csvpath_val, index=False)

    if send:
        files_to_send = [csvpath, csvpath_val]
        caption = "scores - {} - id={}".format(which, forecast_model_id)
        SBM.send_notification(
            text="description: {}".format(scores_dict),
            chat_id=config.CHAT_ID,
            files=files_to_send,
            text_for_files=caption
        )


def _plot_n_save_histograms(abx_samples, non_abx_samples, split, path_to_save):
        fig, ax = plt.subplots(4, 2, figsize=(6*2, 4*4))
        ax[0, 0].hist(np.nanmin(abx_samples, axis=1), label='abx min', bins=50)
        ax[0, 1].hist(np.nanmin(non_abx_samples, axis=1), label='non-abx min', bins=50)
        ax[1, 0].hist(np.nanmax(abx_samples, axis=1), label='abx max', bins=50)
        ax[1, 1].hist(np.nanmax(non_abx_samples, axis=1), label='non-abx max', bins=50)
        ax[2, 0].hist(np.nanmean(abx_samples, axis=1), label='abx mean', bins=50)
        ax[2, 1].hist(np.nanmean(non_abx_samples, axis=1), label='non-abx mean', bins=50)
        ax[3, 0].hist(np.nanmedian(abx_samples, axis=1), label='abx median', bins=50)
        ax[3, 1].hist(np.nanmedian(non_abx_samples, axis=1), label='non-abx median', bins=50)
        ax[0, 0].set_title("abx")
        ax[0, 1].set_title("non-abx")
        ax[0, 0].set_ylabel("min")
        ax[1, 0].set_ylabel("max")
        ax[2, 0].set_ylabel("mean")
        ax[3, 0].set_ylabel("median")
        plt.tight_layout()

        impath = f"{path_to_save}hist_{split}.pdf"
        plt.savefig(impath, format='pdf')
        return impath


def _transform_scores(scores_wide: pd.DataFrame, days_per_month=30.437) -> pd.DataFrame:
    # transform df from wide to long
    scores = scores_wide.melt(
        id_vars=["host_id", "abx"], var_name="day", value_name="score"
    )
    scores["day"] = scores["day"].str.extract(r"ad_score_day-(\d+)")[0].astype(int)
    scores.sort_values(["abx", "host_id", "day"], inplace=True)

    # bin by month
    scores["month_bin"] = (scores["day"] / days_per_month).round().astype(int)
    scores["month5_bin"] = (scores["day"] / days_per_month * 2).round() / 2
    return scores


def _create_subplot(x_axis, y_axis, data, title, ylabel, xlabel, n=None):
    gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[1, 0.5])
    fig = plt.figure(figsize=(12, 8), dpi=400)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)
    axs = [ax1, ax2]

    sns.boxplot(x=x_axis, y=y_axis, data=data, ax=axs[0], color="skyblue")
    axs[0].set_title(title)
    axs[0].set_xlabel("")

    if n is not None:
        axs[0].axvline(3, color="darkred")

    grouped_counts = data.groupby(x_axis)[y_axis].count().reset_index(name="counts")
    sns.barplot(x=x_axis, y="counts", data=grouped_counts, color="peachpuff", ax=axs[1])

    if n is not None:
        axs[1].axvline(3, color="darkred")

    axs[1].set_ylabel(ylabel)
    axs[1].set_xlabel(xlabel)

    plt.tight_layout()
    return fig, axs


def _plot_score_over_age(df: pd.DataFrame, flag: str, path_to_save: str) -> str:
    x_axis = "month_bin"
    y_axis = "score"

    title = flag
    ylabel = f"# samples w {y_axis}"
    xlabel = f"age in {x_axis}"

    fig, _ = _create_subplot(x_axis, y_axis, df, title, ylabel, xlabel)

    path_to_plot = f"{path_to_save}score_over_age_{flag}.pdf"
    plt.savefig(path_to_plot)
    return path_to_plot


def _get_abx_info(path_to_abx_ts: str) -> pd.DataFrame:
    abx_df = pd.read_csv(path_to_abx_ts, sep="\t", index_col=0)
    abx_df = abx_df["abx_start_age_months"].reset_index()
    abx_df.sort_values(["host_id", "abx_start_age_months"], inplace=True)
    abx_df["abx_any_cumcount"] = abx_df.groupby("host_id").cumcount() + 1
    return abx_df


def _select_samples_around_nth_abx_exposure(md_df, abx_df, n=1):
    """
    Get observed samples around n-th abx exposure (n=1 is first abx exposure,
    n=2 is second etc.)

    Args:
        md_df (pd.DataFrame): Contains relevant metadata per host.
        abx_df (pd.DataFrame): Contains start month of abx exposure per host.
        n (int, optional): n-th antibiotics exposure to evaluate. Defaults to 1.

    Returns:
        pd.DataFrame: Dataframe with observed samples around n-th abx exposure.
    """
    # indexing starts at zero
    n = n - 1
    # calculate age at n-th abx exposure for all hosts
    abx_nth_age = abx_df.groupby("host_id").nth(n)
    abx_nth_age = abx_nth_age.rename(columns={"abx_start_age_months": "age_nth_abx"})

    # add this column to all_samples
    all_samples = pd.merge(md_df, abx_nth_age, on="host_id", how="left")

    # calculate time of samples since n-th abx exposure
    all_samples = all_samples.assign(
        diff_age_nth_abx=all_samples["month5_bin"] - all_samples["age_nth_abx"]
    )
    # round to full months for simplicity. note: added 0.01 since lots of 0.5
    # would otw be rounded down leading to uneven sample distribution
    all_samples["diff_age_nth_abx"] = all_samples["diff_age_nth_abx"] + 0.01
    all_samples["diff_age_nth_abx"] = all_samples["diff_age_nth_abx"].round(0)

    # select only samples before and after nth abx exposure
    abx_nth_samples = all_samples.loc[
        np.logical_and(
            ~all_samples["diff_age_nth_abx"].isna(),
            all_samples["abx_any_cumcount"] <= (n + 1),
        ),
        :,
    ]

    # only select samples that are up to 3 months prior to n-th abx exposure and
    # 12 months after
    abx_nth_samples = abx_nth_samples.loc[
        np.logical_and(
            abx_nth_samples["diff_age_nth_abx"] >= -3.0,
            abx_nth_samples["diff_age_nth_abx"] <= 12.0,
        ),
        :,
    ]
    # fix -0.0 artifact
    abx_nth_samples["diff_age_nth_abx"] = abx_nth_samples["diff_age_nth_abx"].replace(
        {-0.0: 0.0}
    )
    # remove samples with no observed features
    abx_nth_samples = abx_nth_samples.dropna(subset=["score"])

    return abx_nth_samples


def _get_ordinal_suffix(n):
    return (
        "st"
        if n % 10 == 1 and n != 11
        else "nd"
        if n % 10 == 2 and n != 12
        else "rd"
        if n % 10 == 3 and n != 13
        else "th"
    )


def _plot_score_after_nth_abx_exposure(
    data: pd.DataFrame, x_axis: str, y_axis: str, n: int, path_to_save: str, flag: str
) -> str:
    suff = _get_ordinal_suffix(n)

    title = f"Score before/after {n}{suff} abx exposure"
    ylabel = f"# samples w {y_axis}"
    xlabel = f"Months since {n}{suff} abx exposure"

    fig, _ = _create_subplot(x_axis, y_axis, data, title, ylabel, xlabel, n)

    path_to_plot = f"{path_to_save}score_after_abx{n}{suff}_{flag}.pdf"
    plt.savefig(path_to_plot)
    return path_to_plot


def evaluate_scores(
        forecast_saved_models_path, forecast_model_id=None, load_best=True,
        validation=False, send=False, dataset=None, **options):
    """
    Evaluate the anomaly detection scores

    args:
        forecast_saved_models_path: str, path to the saved models
        forecast_model_id: int, id of the model
        load_best: bool, whether to load the best model instance or the last one
            (during training)
        validation: bool, whether to evaluate the validation (if True) set or
            the training set
    """
    # get dataset metadata
    train_idx = np.load(os.path.join(
        train_data_path, dataset, "all", 'train_idx.npy'
    ), allow_pickle=True)
    data_train = data_utils.MicrobialDataset(
        dataset_name=dataset, idx=train_idx)
    dataset_metadata = data_train.get_metadata()

    forecast_model_path = '{}id-{}/'.format(
        forecast_saved_models_path, forecast_model_id)
    ad_path = '{}anomaly_detection/'.format(forecast_model_path)
    which = 'best' if load_best else 'last'
    scores_path = '{}scores_{}/'.format(ad_path, which)
    evaluation_path = '{}evaluation_{}/'.format(ad_path, which)
    makedirs(evaluation_path)

    raw_dataset_name = dataset_metadata["dataset"]
    version = raw_dataset_name.split("_")[3]
    path_to_abx_ts = f"data/original_data/ts_vat19_abx_{version}.tsv"
    abx_df = _get_abx_info(path_to_abx_ts)

    for split in ['train', 'val']:
        # load scores
        ad_scores = pd.read_csv(f"{scores_path}{split}_ad_scores.csv")
        score_cols = [
            x for x in ad_scores.columns
            if x.startswith("ad_score_day-")
        ]

        # plot histograms
        impath_hist = _plot_n_save_histograms(
            abx_samples=ad_scores.loc[ad_scores["abx"], score_cols].values,
            non_abx_samples=ad_scores.loc[~ad_scores["abx"], score_cols].values,
            split=split,
            path_to_save=evaluation_path,
        )

        # plot scores over age
        ad_scores_flat = _transform_scores(ad_scores)
        # abx
        abx_scores_flat = ad_scores_flat[ad_scores_flat["abx"]].copy()
        impath_age_abx = _plot_score_over_age(
            abx_scores_flat, f"{split}_abx", evaluation_path
        )
        # no abx
        impath_age_noabx = _plot_score_over_age(
            ad_scores_flat[~ad_scores_flat["abx"]], f"{split}_noabx", evaluation_path
        )

        # plot scores after n-th abx exposure (performed only for abx samples)
        impath_nth_abx_score = {}
        for n in [1, 2]:
            scores_abx_nth_samples = _select_samples_around_nth_abx_exposure(
                abx_scores_flat, abx_df, n=n
            )
            n_path = _plot_score_after_nth_abx_exposure(
                scores_abx_nth_samples,
                x_axis="diff_age_nth_abx",
                y_axis="score",
                n=n,
                path_to_save=evaluation_path,
                flag=split,
            )
            impath_nth_abx_score[n] = n_path

        # send to telegram
        if send:
            # histograms
            caption = "scores-histogram - {} - id={}".format(which, forecast_model_id)
            SBM.send_notification(
                text=None,
                chat_id=config.CHAT_ID,
                files=[impath_hist],
                text_for_files=caption
            )
            # scores over age
            dic_img_age = {"abx": impath_age_abx, "noabx":impath_age_noabx}
            for k, v in dic_img_age.items():
                caption = "scores-over-age {} - {} - id={}".format(k, which, forecast_model_id)
                SBM.send_notification(
                    text=None,
                    chat_id=config.CHAT_ID,
                    files=[v],
                    text_for_files=caption
                )
            # scores after n-th abx exposure
            for k, v in impath_nth_abx_score.items():
                caption = "scores-after-{}th-abx-exposure - {} - id={}".format(
                    k, which, forecast_model_id
                )
                SBM.send_notification(
                    text=None, chat_id=config.CHAT_ID, files=[v], text_for_files=caption
                )

        # print(np.all(np.isnan(abx_samples), axis=1).sum())
        # print(np.all(np.isnan(non_abx_samples), axis=1).sum())

    # train_score = metrics.roc_auc_score(train_abx_labels, train_ad_scores)
    # val_score = metrics.roc_auc_score(val_abx_labels, val_ad_scores)


def get_replace_forecast_values(cond_moments, output_vars, replace_with='mean',
                                variables = ['var'], device = 'cpu'):
    
    dimension = cond_moments.shape[2]

    replace_values = {}
    for variable in variables:
        which = np.argmax(np.array(output_vars) == variable)
        values = cond_moments[:,:,:,which].reshape(-1, dimension)

        condition = ~np.isnan(values)
        if variable == 'var':
            condition = np.logical_and(condition, values >= 0)

        rv = np.empty((dimension))
        rv[:] = np.nan
        for j in range(dimension):
            vals = values[:,j]
            cond = condition[:,j]
            if replace_with == 'mean':
                rv[j] = np.mean(vals[cond])

        replace_values[variable] = rv.copy()
    
    return replace_values





def compute_metrics(ad_labels, ad_scores, mask, autom_thres = None, filename = '',
                    threshold = 0.5, metric = 'f1_score', path=None):
    
    ad_labels = ad_labels[mask].astype(int).reshape(-1)
    ad_scores = ad_scores[mask].reshape(-1)

    if path is not None:
        fpr, tpr, thresholds  = metrics.roc_curve(ad_labels, ad_scores)
        #create ROC curve
        plt.plot(fpr,tpr)
        plt.plot(fpr, thresholds, color='r')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(path + filename + '-roc_curve')
        plt.close()

    if autom_thres is not None:
        autom_thres = autom_thres.split('-')
        if autom_thres[0] == 'FPR_limit':
            limit = float(autom_thres[1])
            indices = np.arange(len(fpr))
            indices[fpr > limit] = 0
            idx = np.argmax(indices)
        if autom_thres[0] == 'TPR_limit':
            limit = float(autom_thres[1])
            indices = np.arange(len(tpr))
            indices[tpr < limit] = np.inf
            idx = np.argmin(indices)
        threshold = thresholds[idx]

    predicted_ad_labels = np.zeros_like(ad_scores).astype(int).reshape(-1)
    predicted_ad_labels[ad_scores > threshold] = 1

    score = None
    if metric == 'recall':
        score = metrics.recall_score(ad_labels, predicted_ad_labels)
    elif metric == 'precision':
        score = metrics.precision_score(ad_labels, predicted_ad_labels)
    elif metric == 'f1_score':
        score = metrics.f1_score(ad_labels, predicted_ad_labels)
    elif metric == 'accuracy':
        score = metrics.accuracy_score(ad_labels, predicted_ad_labels)
    elif metric == 'balanced_accuracy':
        score = metrics.balanced_accuracy_score(ad_labels, predicted_ad_labels)
    elif metric == 'roc_auc':
        score = metrics.roc_auc_score(ad_labels, predicted_ad_labels)

    if autom_thres is not None:
        print("Threshold found : {} for FPR of {} and TPR of : {}".format(threshold, fpr[idx], tpr[idx]))


    # print(score)
    
    return score, threshold

def get_forecast_model_param_dict(
        epochs=100,
        seed=398,
        batch_size=200,
        hidden_size=10, 
        bias=True, 
        dataset='microbial_genus',
        dataset_id=0,
        dropout_rate=0.1,
        ode_nn=default_ode_nn, 
        readout_nn=default_readout_nn,
        enc_nn=default_enc_nn, 
        use_rnn=False,
        solver="euler",
        weight=0.5, 
        weight_decay=1.,
        **options):
    
    data_train = data_utils.MicrobialDataset(dataset_name=dataset)
    dataset_metadata = data_train.get_metadata()
    input_size = dataset_metadata['dimension']
    dimension = dataset_metadata['dimension']
    dimension_dyn_feat = dataset_metadata['dimension_dyn_feat']
    dimension_sig_feat = dataset_metadata['dimension_sig_feat']
    output_size = input_size
    T = dataset_metadata['maturity']
    delta_t = dataset_metadata['dt']  # copy metadata
    if 'period' in dataset_metadata:
        t_period = dataset_metadata['period']
    else:
        t_period = T
    if 'solver_delta_t_factor' in options:
        model_delta_t = delta_t / options['solver_delta_t_factor']
    else:
        model_delta_t = delta_t
    if 'scale_dt' in options:
        if options['scale_dt'] == 'automatic':
            options['scale_dt'] = 1. / delta_t
            if 'solver_delta_t_factor' in options:
                options['scale_dt'] *= options['solver_delta_t_factor']
    weight_evolve = None
    if 'weight_evolve' in options:
        weight_evolve = options['weight_evolve']
        if weight_evolve is not None:
            weight_evolve_type = options['weight_evolve']['type']
            if weight_evolve_type == 'linear':
                if options['weight_evolve']['reach'] == None:
                    options['weight_evolve']['reach'] = epochs
    zero_weight_init = False
    if 'zero_weight_init' in options:
        zero_weight_init = options['zero_weight_init']

    # specify the input and output variables of the model, as function of X
    input_vars = ['id']
    output_vars = ['id']
    if 'func_appl_X' in options:  # list of functions to apply to the paths in X
        functions = options['func_appl_X']
        collate_fn, mult = data_utils.MicrobialCollateFnGen(functions) #, scaling_factor=data_scaling_factor)
        collate_fn_val, _ = data_utils.MicrobialCollateFnGen(functions)
        input_size = input_size * mult
        output_size = output_size * mult
        output_vars += functions
        input_vars += functions
    else:
        functions = None
        collate_fn, mult = data_utils.MicrobialCollateFnGen(None) #, scaling_factor=data_scaling_factor)
        collate_fn_val, _ = data_utils.MicrobialCollateFnGen(None)
        mult = 1
    # if we predict additional variables (the output size gets bigger than input size)
    if 'add_pred' in options:
        add_pred = options['add_pred']
        nb_pred_add = len(add_pred)
        #mult += nb_pred_add
        output_size += nb_pred_add * dimension
        output_vars += add_pred
    input_size += dimension_dyn_feat
    
    opt_eval_loss = np.nan
    params_dict = {  # create a dictionary of the wanted parameters
        'input_size': input_size, 'epochs': epochs,
        'hidden_size': hidden_size, 'output_size': output_size, 'bias': bias,
        'ode_nn': ode_nn, 'readout_nn': readout_nn, 'enc_nn': enc_nn,
        'use_rnn': use_rnn, 'zero_weight_init': zero_weight_init,
        'dropout_rate': dropout_rate, 'batch_size': batch_size,
        'solver': solver, 'dataset': dataset, 'seed': seed,
        'weight': weight, 'weight_evolve': weight_evolve,
        't_period': t_period, 'delta_t': model_delta_t,
        'output_vars': output_vars, 'input_vars': input_vars,
        'sigf_size': dimension_sig_feat,
        'options': options}

    return params_dict, collate_fn



    


def plot_AD_module_params(ad_module, path, filename, steps_ahead = None, dt=0.0025):

    if isinstance(ad_module, AD_module):

        weights = ad_module.get_weights().squeeze().clone().detach().numpy()
        steps_ahead = [str(dt * s) for s in steps_ahead]
        neighbors = [str(i) for i in range(-ad_module.smoothing, ad_module.smoothing+1)]

        fig, ax = plt.subplots(figsize=(10,8))
        im = ax.imshow(weights)
        ax.set_xticks(np.arange(weights.shape[1]))
        ax.set_yticks(np.arange(weights.shape[0]))
        ax.set_xticklabels(neighbors)
        ax.set_yticklabels(steps_ahead)
        fig.colorbar(im, ax=ax, shrink=0.25)
        plt.title('Weights of scores smoothing on forecasting horizon and neighbouring timestamps', fontsize=15)
        plt.xlabel('Neighbouring timestamps', fontsize=10)
        plt.ylabel('Forecasting horizons', fontsize=10)
        plt.savefig(path + filename + '_ad_module_weights.png')
        plt.close()


def plot_one_path_with_pred(
        device, forecast_model, ad_module, batch, delta_t, T,
        paths_to_plot=(0,), save_path='', filename='plot_{}.png',
        plot_variance=False, std_factor=1.96, model_name='PD-NJ-ODE',
        save_extras={'bbox_inches': 'tight', 'pad_inches': 0.01},
        output_vars=None, steps_ahead=[1], plot_forecast_predictions=False,
        forecast_horizons_to_plot=[1], anomaly_type='undefined',
):

    prop_cycle = plt.rcParams['axes.prop_cycle']  # change style of plot?
    colors = prop_cycle.by_key()['color']
    if plot_forecast_predictions:
        std_color = [list(matplotlib.colors.to_rgb(colors[c])) + [0.5] for c in range(len(forecast_horizons_to_plot) + 2)]
    makedirs(save_path)
    
    times = batch["times"]
    time_ptr = batch["time_ptr"]
    X = batch["X"].to(device)
    M = batch["M"]
    if M is not None:
        M = M.to(device)
    start_M = batch["start_M"]
    if start_M is not None:
        start_M = start_M.to(device)

    start_X = batch["start_X"].to(device)
    obs_idx = batch["obs_idx"]
    n_obs_ot = batch["n_obs_ot"].to(device)
    observed_dates = batch['observed_dates']
    path_t_true_X = np.linspace(0., T, int(np.round(T / delta_t)) + 1)
    true_M = batch["true_mask"]
    true_X = batch["true_paths"]
    ad_labels = batch["ad_labels"]

    with torch.no_grad():
        path_t, path_y = forecast_model.custom_forward(
        # hT, c_loss, path_t, path_h, path_y = model( 
            times, time_ptr, X, obs_idx, delta_t, T, start_X,
            n_obs_ot, steps=steps_ahead, get_loss=False, M=M,
            start_M=start_M)
        torch.cuda.empty_cache()
        
    nb_steps = path_t_true_X.shape[0]
    nb_steps_ahead = len(steps_ahead)
    batch_size = ad_labels.shape[0]
    dimension = ad_labels.shape[1]
    nb_vars = len(output_vars)

    cond_moments = np.empty((nb_steps, batch_size, dimension, nb_vars, nb_steps_ahead))
    cond_moments[:,:,:,:,:] = np.nan
    for i in range(nb_steps_ahead):
        index_times = np.empty_like(path_t_true_X)
        for j,t in enumerate(path_t_true_X):
            if np.any(np.abs(t*np.ones_like(path_t[i]) - path_t[i]) < 1e-10):
                index_times[j] = True
            else:
                index_times[j] = False
        index_times = np.argwhere(index_times).reshape(-1)
        for m in range(nb_vars):
            cond_moments[index_times,:,:,m,i] = path_y[i].detach().cpu().numpy()[:,:,m*dimension:(m+1)*dimension]

    obs = torch.tensor(true_X, dtype=torch.float32).permute(2,0,1)
    cond_moments = torch.tensor(cond_moments, dtype=torch.float32)
    ad_scores, mask = ad_module(obs, cond_moments)
    ad_scores = ad_scores.detach().cpu().numpy()
    ad_labels = ad_labels.transpose((2,0,1))
    predicted_ad_labels, scores = ad_module.get_predicted_label(obs, cond_moments)
    # predicted_ad_labels = predicted_ad_labels.transpose((1, 2, 0))

    scores = scores.detach().cpu().numpy().transpose((1, 2, 0))
    ad_labels = ad_labels.transpose((1, 2, 0))
    predicted_ad_labels = predicted_ad_labels.detach().cpu().numpy().transpose((1, 2, 0))

    ad_labels_plot = ad_labels.copy()
    ad_labels_plot[:,:,1:][np.logical_and(ad_labels[:,:,1:]==0,ad_labels[:,:,:-1]==1)] = 1
    ad_labels_plot[:,:,:-1][np.logical_and(ad_labels[:,:,1:]==1,ad_labels[:,:,:-1]==0)] = 1
    predicted_ad_labels_plot = predicted_ad_labels.copy()
    predicted_ad_labels_plot[:,:,1:][np.logical_and(predicted_ad_labels[:,:,1:]==0,predicted_ad_labels[:,:,:-1]==1)] = 1
    predicted_ad_labels_plot[:,:,:-1][np.logical_and(predicted_ad_labels[:,:,1:]==1,predicted_ad_labels[:,:,:-1]==0)] = 1

    mask_pred_path = np.ma.masked_where((predicted_ad_labels_plot == 1), true_X)
    mask_pred_path_w_anomaly = np.ma.masked_where((predicted_ad_labels_plot == 0), true_X)
    mask_data_path = np.ma.masked_where((ad_labels_plot == 1), true_X)
    mask_data_path_w_anomaly = np.ma.masked_where((ad_labels_plot == 0), true_X)
    mask_pred_scores = np.ma.masked_where((np.isnan(scores)), scores)


    for a in paths_to_plot:
        fig, axs = plt.subplots(dimension,2,figsize=(20, 7))
        if dimension == 1:
            axs = [axs]
        
        for j in range(dimension):
            # get the true_X at observed dates
            path_t_obs = []
            path_X_obs = []
            for k, od in enumerate(observed_dates[a]):
                if od == 1:
                    if true_M is None or (true_M is not None and
                                        true_M[a, j, k]==1):
                        path_t_obs.append(path_t_true_X[k])
                        path_X_obs.append(true_X[a, j, k])
            path_t_obs = np.array(path_t_obs)
            path_X_obs = np.array(path_X_obs)

            axs[j][0].set_title("Ground_truth, dimension {}".format(j+1))
            axs[j][0].plot(path_t_true_X, mask_data_path[a, j, :], label='true path, no anomaly',
                        color=colors[0])
            axs[j][0].plot(path_t_true_X, mask_data_path_w_anomaly[a, j, :], label='true path, anomaly',
                        color=colors[1])
            axs[j][1].set_title("Anomaly Detection, dimension {}".format(j+1))
            axs[j][1].plot(path_t_true_X, mask_pred_path[a, j, :], label='true path, no predicted anomaly',
                        color=colors[0])
            axs[j][1].plot(path_t_true_X, mask_pred_path_w_anomaly[a, j, :], label='true path, predicted anomaly',
                        color=colors[1])
            axs[j][1].plot(path_t_true_X, mask_pred_scores[a, j, :], label='Predicted scores',
                        color=colors[2])

            if plot_forecast_predictions:
                for h,s in enumerate(forecast_horizons_to_plot):
                    step_idx = steps_ahead.index(s)
                    exp_prediction = cond_moments[:,a,j,0,step_idx].detach().cpu().numpy()
                    axs[j][1].plot(path_t_true_X, exp_prediction, label=model_name + " expectation prediction {} steps ahead".format(s), color=colors[h+2])
                    if plot_variance:
                        assert(('var' in output_vars) or ('power-2' in output_vars))
                        if 'var' in output_vars:
                            which = np.argmax(np.array(output_vars) == 'var')
                            var_prediction = cond_moments[:,a,j,which,step_idx].detach().cpu().numpy()
                        elif 'power-2' in output_vars:
                            which = np.argmax(np.array(output_vars) == 'power-2')
                            exp_2_prediction = cond_moments[:,a,j,which,step_idx].detach().cpu().numpy()
                            var_prediction = exp_2_prediction - exp_prediction ** 2 
                        
                        if np.any(var_prediction < 0):
                            # print('WARNING: some true cond. variances below 0 -> clip')
                            var_prediction = np.maximum(0, var_prediction)
                        std_prediction = np.sqrt(var_prediction)
                        axs[j][1].fill_between(path_t_true_X,
                            exp_prediction - std_factor * std_prediction,
                            exp_prediction + std_factor * std_prediction,
                            label="standart deviation ({} factor) prediction {} steps ahead".format(std_factor, s), color=std_color[h+2])
            
            axs[j][0].legend()
            axs[j][1].legend()
            axs[j][0].set_ylim(0., 1.)
            axs[j][1].set_ylim(0., 1.)
            axs[j][1].axhline(y=ad_module.threshold, color='r', linestyle='-')

        plt.xlabel('$t$')
        plt.suptitle("Anomaly detection on {} anomalies".format(anomaly_type))
        save = os.path.join(save_path, filename.format(a))
        plt.savefig(save, **save_extras)
        plt.close()
    


def main(arg):

    del arg
    forecast_params_list = None
    forecast_model_ids = None
    if FLAGS.forecast_model_ids:
        try:
            forecast_model_ids = eval("config."+FLAGS.forecast_model_ids)
        except Exception:
            forecast_model_ids = eval(FLAGS.forecast_model_ids)
        print("evaluate forecast model ids: ", forecast_model_ids)
    if FLAGS.ad_params:
        ad_params = eval("config."+FLAGS.ad_params)
    if FLAGS.forecast_saved_models_path:
        try:
            forecast_saved_models_path = eval(
                "config."+FLAGS.forecast_saved_models_path)
        except Exception:
            forecast_saved_models_path = FLAGS.forecast_saved_models_path

    if forecast_model_ids is not None:
        forecast_model_overview_file_name = '{}model_overview.csv'.format(forecast_saved_models_path)
        
        df_overview = pd.read_csv(forecast_model_overview_file_name, index_col=0)
        max_id = np.max(df_overview['id'].values)

        forecast_params = []
        for model_id in forecast_model_ids:
            if model_id not in df_overview['id'].values:
                print("model_id={} does not exist yet -> skip".format(model_id))
            else:
                desc = (df_overview['description'].loc[
                    df_overview['id'] == model_id]).values[0]
                forecast_params_dict = json.loads(desc)
                forecast_params_dict['model_id'] = model_id
                forecast_params.append(forecast_params_dict)

        for i,forecast_param in enumerate(forecast_params):
            if 'dataset' not in forecast_param:
                if 'data_dict' not in forecast_param:
                    raise KeyError('the "dataset" needs to be specified')
                else:
                    data_dict = forecast_param["data_dict"]
                    if isinstance(data_dict, str):
                        data_dict = eval("config."+data_dict)
                    forecast_param["dataset"] = data_dict["model_name"]
            for ad_param in ad_params:
                print("AD param: ", ad_param)
                if 'dataset' in ad_param:
                    del ad_param["dataset"]
                if 'saved_models_path' in ad_param:
                    del ad_param["saved_models_path"]
                if FLAGS.compute_scores:
                    compute_scores(
                        send=FLAGS.SEND,
                        forecast_saved_models_path=forecast_saved_models_path,
                        forecast_model_id=forecast_model_ids[i],
                        forecast_param=forecast_param,
                        n_dataset_workers=FLAGS.N_DATASET_WORKERS,
                        use_gpu=FLAGS.USE_GPU, nb_cpus=FLAGS.NB_CPUS,
                        saved_models_path=forecast_saved_models_path,
                        dataset=forecast_param["dataset"],
                        **ad_param)
                if FLAGS.evaluate_scores:
                    evaluate_scores(
                        send=FLAGS.SEND,
                        forecast_saved_models_path=forecast_saved_models_path,
                        forecast_model_id=forecast_model_ids[i],
                        saved_models_path=forecast_saved_models_path,
                        dataset=forecast_param["dataset"],
                        **ad_param)


if __name__ == '__main__':
    app.run(main)