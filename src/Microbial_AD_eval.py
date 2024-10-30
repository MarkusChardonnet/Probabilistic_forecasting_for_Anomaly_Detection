# =====================================================================================================================
import json
import os
import sys
import socket

import data_utils
import matplotlib  # plots
import matplotlib.colors
import matplotlib.pyplot as plt
import models
import numpy as np  # large arrays and matrices, functions
import pandas as pd
import torch  # machine learning
from absl import app, flags
import AD_modules
from AD_modules import AD_module, DimAcc_AD_module, Simple_AD_module
from configs import config
from sklearn import metrics

import scipy.stats as stats
import seaborn as sns

from torch.utils.data import DataLoader


# since this is not a package - add src to path
src_dir = os.path.abspath("src")
sys.path.append(src_dir)
from utils_eval_score import _plot_score_over_age_legacy, _transform_scores

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
flags.DEFINE_bool("compute_zscore_scaling_factors", False, "whether to compute the z-score scaling factors")

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


def get_model_predictions(
        dl, device, forecast_model, output_vars, T, delta_t, dimension,
        only_jump_before_abx_exposure=False, use_only_dyn_ft_as_input=None,
        add_dynamic_cov=False, use_dyn_cov_after_abx=True,
        use_obs_until_t=None, sf=None):
    """
    Get the NJODE model predictions for the given dataset
    """
    b = next(iter(dl))
    true_abx_exposure = b["true_abx_exposure"]

    # compute days after cutoff (before only_jump_before_abx_exposure is
    #   changed)
    if only_jump_before_abx_exposure in [None, False, 0]:
        days_after_cutoff = np.zeros_like(true_abx_exposure) - 1
    else:
        days_after_cutoff = np.cumsum(
            true_abx_exposure >= only_jump_before_abx_exposure, axis=1) - 1
    if use_obs_until_t is not None:
        days_after_cutoff = np.arange(true_abx_exposure.shape[1]) - np.round(
            use_obs_until_t/delta_t) - 1
        days_after_cutoff = days_after_cutoff.reshape(1, -1).repeat(
            len(true_abx_exposure), axis=0)

    masked = False
    if use_only_dyn_ft_as_input == "after_nth_abx_exposure":
        masked = True
        if use_dyn_cov_after_abx:
            # set this to false, s.t. the dynamic covariates are actually used
            #   after the nth abx exposure
            only_jump_before_abx_exposure = False

    times = b["times"]
    time_ptr = b["time_ptr"]
    X = b["X"].to(device)
    Z = b["Z"].to(device)
    S = b["S"].to(device)
    start_X = b["start_X"].to(device)
    start_Z = b["start_Z"].to(device)
    start_S = b["start_S"].to(device)
    M_X = b["M_X"]
    M_Z = b["M_Z"]
    M_S = b["M_S"]
    start_M_X = b["start_M_X"]
    start_M_Z = b["start_M_Z"]
    start_M_S = b["start_M_S"]
    M = None
    start_M = None
    if masked:
        M_X = M_X.to(device)
        M_Z = M_Z.to(device)
        M_S = M_S.to(device)
        start_M_X = start_M_X.to(device)
        start_M_Z = start_M_Z.to(device)
        start_M_S = start_M_S.to(device)
        if add_dynamic_cov:
            M = torch.cat((M_X, M_Z), dim=1)
            start_M = torch.cat((start_M_X, start_M_Z), dim=1)
        else:
            M = M_X
            start_M = start_M_X
    obs_idx = b["obs_idx"]
    n_obs_ot = b["n_obs_ot"].to(device)
    observed_dates = np.transpose(b['observed_dates'], (1, 0)).astype(np.bool)
    path_t_true_X = np.linspace(0., T, int(np.round(T / delta_t)) + 1)
    true_X = b["true_paths"]
    abx_labels = b["abx_observed"]
    host_id = b["host_id"]
    abx_exposure = b["abx_exposure"]
    if add_dynamic_cov:
        X = torch.cat((X, Z), dim=1)
        start_X = torch.cat((start_X, start_Z), dim=1)

    with torch.no_grad():
        res = forecast_model.get_pred(
            times=times, time_ptr=time_ptr, X=X,
            obs_idx=obs_idx, delta_t=None, S=S, start_S=start_S,
            T=T, start_X=start_X,
            abx_exposure=abx_exposure,
            M=M, start_M=start_M, M_S=M_S, start_M_S=start_M_S,
            only_jump_before_abx_exposure=only_jump_before_abx_exposure,
            use_obs_until_t=use_obs_until_t)
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

    # get the scaling factors for the predicted stds based on the cut-off days
    cutoff_adj_sf = np.ones_like(days_after_cutoff)
    if sf is not None:
        for i in range(len(sf)):
            cutoff_adj_sf[days_after_cutoff==sf["days_since_cutoff"].iloc[i]]= \
                sf["std_z_scores"].iloc[i]
    cutoff_adj_sf = np.transpose(cutoff_adj_sf, (1, 0))

    # shape of cond_moments: [nb_steps, nb_samples, dimension, nb_moments]
    # shape of observed_dates: [nb_steps, nb_samples]
    # shape of true_X: [nb_samples, nb_steps, dimension]
    # shape of abx_labels: [nb_samples, nb_steps]
    # shape of host_id: [nb_samples]
    # shape of cutoff_adj_sf: [nb_steps, nb_samples]
    return (cond_moments, observed_dates, true_X, abx_labels, host_id,
            cutoff_adj_sf)


def _plot_conditionally_standardized_distribution(
        cond_moments, observed_dates, obs, output_vars, path_to_save,
        compare_to_dist="normal", replace_values=None, which_set='train',
        which_coord=0, eps=1e-4,
        **options):
    """
    Plot the conditionally standardized distribution

    Args:
        cond_moments: np.array, [nb_steps, nb_samples, dimension, nb_moments]
        observed_dates: np.array, [nb_steps, nb_samples]
        obs: np.array, [nb_steps, nb_samples, dimension]
        output_vars: list, list of output variables
        path_to_save: str, path to save the plot
        compare_to_dist: str, distribution to compare to, one of: 'normal',
            'lognormal'
        replace_values: np.array, [nb_steps, nb_samples, dimension], replace
            values for variance
        host_id: np.array, [nb_samples], host ids
        which_set: str, which set to plot, one of: 'train', 'val'
    """
    # cond_exp : [nb_steps, nb_samples, dimension]
    # cond_var : [nb_steps, nb_samples, dimension]
    cond_exp, cond_var = AD_modules.get_cond_exp_var(cond_moments, output_vars)
    cond_var = AD_modules.get_corrected_var(
        cond_var, min_var_val=eps, replace_var=replace_values)
    if compare_to_dist == "normal":
        cond_std = np.sqrt(cond_var)
        standardized_obs = (obs - cond_exp) / cond_std
    elif compare_to_dist == "lognormal":
        mu = np.log(cond_exp) - 0.5 * np.log(1 + cond_var / cond_exp ** 2)
        sigma = np.sqrt(np.log(1 + cond_var / cond_exp ** 2))
        standardized_obs = (np.log(obs) - mu)/ sigma
    elif compare_to_dist.startswith("t-"):
        nu = int(compare_to_dist.split("-")[1])
        mu = cond_exp
        sigma = np.sqrt(cond_var/(nu/(nu-2)))
        standardized_obs = (obs - mu) / sigma
    else:
        raise ValueError(f"compare_to_dist {compare_to_dist} not implemented")
    standardized_obs = standardized_obs[observed_dates]
    standardized_obs = np.clip(standardized_obs, -5, 5)
    standardized_obs = standardized_obs[:, which_coord]

    if compare_to_dist.startswith("t-"):
        pval = stats.kstest(standardized_obs.flatten(), "t", args=(nu,))[1]
    else:
        pval = stats.kstest(standardized_obs.flatten(), "norm")[1]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    sns.histplot(x=standardized_obs.flatten(), bins=50, kde=True,
                 ax=ax, stat="density", color="skyblue", label="observed")
    t = np.linspace(-5, 5, 1000)
    if compare_to_dist.startswith("t-"):
        ax.plot(t, stats.t.pdf(t, nu), color="darkred", linestyle="--",
                label=f"student-t (df={nu})")
        ax.set_title(
            f"Conditionally standardized distribution of no-abx {which_set}-set.\n" +
            f"Transformation: cond. {compare_to_dist} to stand. t (df={nu})\n" +
            f"p-value of KS test: {pval:.2e}")
    else:
        ax.plot(t, stats.norm.pdf(t, loc=0, scale=1),
                color="darkred", linestyle="--", label="standard normal")
        ax.set_title(
            f"Conditionally standardized distribution of no-abx {which_set}-set.\n"+
            f"Transformation: cond. {compare_to_dist} to stand. normal\n"+
            f"p-value of KS test: {pval:.2e}")
    plt.legend()
    plt.tight_layout()
    figpath = f"{path_to_save}cond_std_dist-{compare_to_dist}-{which_coord}.pdf"
    plt.savefig(figpath)
    plt.close()

    filepath = f"{path_to_save}cond_std_obs-{compare_to_dist}-{which_coord}.npy"
    with open(filepath, "wb") as f:
        np.save(f, standardized_obs)

    print(f"p-value of KS test: {pval}")

    return [figpath, filepath]


def compute_scores(
        forecast_saved_models_path,
        forecast_model_id=None,
        dataset='microbial_genus',
        scoring_distribution='dirichlet',
        forecast_param=None,
        load_best=True,
        use_gpu=None,
        nb_cpus=None,
        n_dataset_workers=None,
        nb_MC_samples=10**5,
        verbose=False,
        epsilon=1e-4,
        seed=333,
        send=False,
        use_replace_values=False,
        dirichlet_use_coord=None,
        aggregation_method="coord-0",
        scoring_metric='p-value',
        only_jump_before_abx_exposure=False,
        plot_cond_standardized_dist=[],
        use_dyn_cov_after_abx=True,
        reliability_eval_start_times=None,
        use_scaling_factors=False,
        preprocess_scaling_factors=True,
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
        aggregation_method: str, method to aggregate the scores (if needed);
            one of: 'mean', 'max', 'min', 'coord-n' for n >= 0 integer
        scoring_metric: str, metric to use for scoring. one of:
            - 'two-sided' or 'p-value': use the 2-sided p-value of the distribution
            - 'left-tail': use the left tail of the distribution
            - 'right-tail': use the right tail of the distribution
        plot_cond_std_dist: bool, whether to plot the conditionally standardized
            distribution for the no-abx validation samples under normal and
            lognormal assumption
        only_jump_before_abx_exposure: bool or int, whether to only update the
            input of the model before the first abx exposure (i.e., only use the
            jump part of the NJODE model before). This can be used to evaluate
            the scores based on the models state before the first abx exposure.
            if int, then this is the number of abx exposure until which
            observations are used as input (i.e., model is updated). Hence, True
            has the same effect as 1, False as infinity.
        plot_cond_standardized_dist: list of str, which distributions to plot
            for the conditionally standardized distribution.
            can include: 'normal', 'lognormal', 't-n' for n >= 3 integer
        use_dyn_cov_after_abx: bool, whether to use the dynamic covariates after
            the abx exposure as input to the model. only applies for those
            models, which were trained with use_only_dyn_ft_as_input!=None. all
            other models cannot use the dynamic covariates after the
            abx exposure.
        reliability_eval_start_times: None or list of int, start times to
            evaluate the reliability of the scores. In particular, for each
            start time, the model uses the inputs up to this time and afterwards
            predicts without further inputs. This evaluation is done on the
            no-abx validation samples. Start times are given in days.
        use_scaling_factors: bool, whether to use the scaling factors for the
            computation of scores. the scaling factors have to be computed with
            this function and scoring_distribution=z_score and the function
            compute_zscore_scaling_factors first.
        preprocess_scaling_factors: bool, whether to preprocess the scaling
            factors before using them. This includes:
            i) lower bounding them by 1
            ii) constraining them to be increasing

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
        "scoring_distribution": scoring_distribution,
        "load_best": load_best,
        "nb_MC_samples": nb_MC_samples, "seed": seed, "epsilon": epsilon,
        "use_replace_values": use_replace_values,
        "dirichlet_use_coord": dirichlet_use_coord,
        "aggregation_method": aggregation_method,
        "scoring_metric": scoring_metric,
        "only_jump_before_abx_exposure": only_jump_before_abx_exposure,
        "use_dyn_cov_after_abx": use_dyn_cov_after_abx,
        "reliability_eval_start_times": reliability_eval_start_times,
        "use_scaling_factors": use_scaling_factors,
        "preprocess_scaling_factors": preprocess_scaling_factors,
    }

    sf = None
    if use_scaling_factors:
        which = "best" if load_best else "last"
        sf_file = (f"{forecast_saved_models_path}anomaly_detection/zscore_"
                   f"scaling_factors_{which}/zscore_scaling_factors_"
                   f"{aggregation_method}.csv")
        sf = pd.read_csv(sf_file)
        if preprocess_scaling_factors:
            sf["std_z_scores"] = np.maximum(sf["std_z_scores"], 1)
            sf["std_z_scores"] = sf["std_z_scores"].cummax()

    # load dataset-metadata
    train_idx = np.load(os.path.join(
        train_data_path, dataset, "all", 'train_idx.npy'
    ), allow_pickle=True)
    val_idx = np.load(os.path.join(
        train_data_path, dataset, "all", 'val_idx.npy'
    ), allow_pickle=True)
    val_idx_noabx = np.load(os.path.join(
        train_data_path, dataset, "no_abx", 'val_idx.npy'
    ), allow_pickle=True)

    data_train = data_utils.MicrobialDataset(
        dataset_name=dataset, idx=train_idx)
    data_val = data_utils.MicrobialDataset(
        dataset_name=dataset, idx=val_idx)
    data_val_noabx = data_utils.MicrobialDataset(
        dataset_name=dataset, idx=val_idx_noabx)

    dataset_metadata = data_train.get_metadata()
    dimension = dataset_metadata['dimension']
    T = dataset_metadata['maturity']
    delta_t = dataset_metadata['dt']  # copy metadata
    starting_date = dataset_metadata['starting_date']
    class_thres = 0.5

    # get all needed paths
    forecast_model_path = '{}id-{}/'.format(
        forecast_saved_models_path, forecast_model_id)
    forecast_model_path_save_best = '{}best_checkpoint/'.format(
        forecast_model_path)
    forecast_model_path_save_last = '{}last_checkpoint/'.format(
        forecast_model_path)
    ad_path = '{}anomaly_detection/'.format(forecast_model_path)
    which = 'best' if load_best else 'last'
    scores_path = '{}scores_{}_{}/'.format(ad_path, which, scoring_distribution)
    makedirs(scores_path)

    # get params_dict
    (forecast_params_dict, collate_fn,
     use_only_dyn_ft_as_input, add_dynamic_cov) = get_forecast_model_param_dict(
        **forecast_param,
        only_jump_before_abx_exposure=only_jump_before_abx_exposure)
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
    dl_val_noabx = DataLoader(
        dataset=data_val_noabx, collate_fn=collate_fn, shuffle=False,
        batch_size=len(val_idx_noabx))

    cond_moments, observed_dates, true_X, abx_labels, host_id, cutoff_adj_sf = \
        get_model_predictions(
            dl_train, device, forecast_model, output_vars, T, delta_t,
            dimension,
            only_jump_before_abx_exposure=only_jump_before_abx_exposure,
            use_only_dyn_ft_as_input=use_only_dyn_ft_as_input,
            add_dynamic_cov=add_dynamic_cov,
            use_dyn_cov_after_abx=use_dyn_cov_after_abx, sf=sf)
    if use_replace_values:
        replace_values = get_replace_forecast_values(
            cond_moments=cond_moments[abx_labels == 0],
            output_vars=output_vars, device=device)
        replace_values = replace_values['var']
    else:
        replace_values = None

    if scoring_distribution == 'dirichlet':
        ad_module = Simple_AD_module(
            output_vars=output_vars,
            nb_MC_samples=nb_MC_samples,
            distribution_class="dirichlet",
            replace_values=replace_values,
            class_thres=class_thres,
            seed=seed,
            epsilon=epsilon,
            dirichlet_use_coord=dirichlet_use_coord,
            verbose=verbose)
    elif (scoring_distribution in ['normal', 'lognormal', 'z_score']
          or scoring_distribution.startswith("t-")):
        ad_module = Simple_AD_module(
            output_vars=output_vars,
            distribution_class=scoring_distribution,
            scoring_metric=scoring_metric,
            replace_values=replace_values,
            class_thres=class_thres,
            seed=seed,
            epsilon=epsilon,
            verbose=verbose,
            aggregation_method=aggregation_method)
    elif scoring_distribution == 'beta':
        ad_module = DimAcc_AD_module(
            output_vars=output_vars,
            dimension=dimension,
            distribution_class="beta",
            aggregation_method=aggregation_method,
            train_labels=abx_labels,
            replace_values=replace_values,
            class_thres=class_thres,
            epsilon=epsilon,)
    else:
        raise ValueError("scoring_distribution not implemented")

    # train data
    obs = true_X.transpose(2, 0, 1)
    ad_scores = ad_module(obs, cond_moments, observed_dates, cutoff_adj_sf)
    with open('{}train_ad_scores_{}_{}.npy'.format(
            scores_path, int(only_jump_before_abx_exposure),
            aggregation_method), 'wb') as f:
        np.save(f, ad_scores)
        np.save(f, abx_labels)
        np.save(f, host_id)
    data = np.concatenate(
        [host_id.reshape(-1,1), abx_labels.reshape(-1, 1), ad_scores], axis=1)
    cols = ['host_id', 'abx'] + ['ad_score_day-{}'.format(i+starting_date)
                                 for i in range(ad_scores.shape[1])]
    df = pd.DataFrame(data, columns=cols)
    csvpath = '{}train_ad_scores_{}_{}.csv'.format(
        scores_path, only_jump_before_abx_exposure, aggregation_method)
    df.to_csv(csvpath, index=False)

    filepaths = []
    if aggregation_method is not None and aggregation_method.startswith("coord-"):
        which_coord = int(aggregation_method.split("-")[1])
    else:
        which_coord = 0
    if plot_cond_standardized_dist is not None:
        dist_path = f'{ad_path}dist/train-noabx/'
        makedirs(dist_path)
        for dist in plot_cond_standardized_dist:
            filepaths += _plot_conditionally_standardized_distribution(
                cond_moments[:, abx_labels == 0],
                observed_dates[:, abx_labels == 0],
                obs[:, abx_labels == 0], output_vars, path_to_save=dist_path,
                compare_to_dist=dist, replace_values=replace_values,
                which_set='train', which_coord=which_coord, eps=epsilon)

    # test data
    cond_moments, observed_dates, true_X, abx_labels, host_id, cutoff_adj_sf = \
        get_model_predictions(
            dl_val, device, forecast_model, output_vars, T, delta_t, dimension,
            only_jump_before_abx_exposure=only_jump_before_abx_exposure,
            use_only_dyn_ft_as_input=use_only_dyn_ft_as_input,
            add_dynamic_cov=add_dynamic_cov,
            use_dyn_cov_after_abx=use_dyn_cov_after_abx, sf=sf)
    obs = true_X.transpose(2, 0, 1)
    ad_scores = ad_module(obs, cond_moments, observed_dates, cutoff_adj_sf)
    with open('{}val_ad_scores_{}_{}.npy'.format(
            scores_path, only_jump_before_abx_exposure,
            aggregation_method), 'wb') as f:
        np.save(f, ad_scores)
        np.save(f, abx_labels)
        np.save(f, host_id)
    data = np.concatenate(
        [host_id.reshape(-1,1), abx_labels.reshape(-1, 1), ad_scores], axis=1)
    cols = ['host_id', 'abx'] + ['ad_score_day-{}'.format(i+starting_date)
                                 for i in range(ad_scores.shape[1])]
    df = pd.DataFrame(data, columns=cols)
    csvpath_val = '{}val_ad_scores_{}_{}.csv'.format(
        scores_path, only_jump_before_abx_exposure, aggregation_method)
    df.to_csv(csvpath_val, index=False)
    
    if plot_cond_standardized_dist is not None:
        dist_path = f'{ad_path}dist/val-noabx/'
        makedirs(dist_path)
        for dist in plot_cond_standardized_dist:
            filepaths += _plot_conditionally_standardized_distribution(
                cond_moments[:, abx_labels==0], observed_dates[:, abx_labels==0],
                obs[:, abx_labels==0], output_vars, path_to_save=dist_path,
                compare_to_dist=dist, replace_values=replace_values,
                which_set='val', which_coord=which_coord, eps=epsilon)

    if reliability_eval_start_times is not None:
        reli_eval_path = f'{ad_path}reliability_eval-val-noabx_{which}_{scoring_distribution}/'
        makedirs(reli_eval_path)
        data_collect = []
        df = pd.DataFrame()
        for start_time in reliability_eval_start_times:
            print(f"compute reliability eval scores for start_time={start_time}")
            (cond_moments, observed_dates, true_X, abx_labels, host_id,
             cutoff_adj_sf) = get_model_predictions(
                dl_val_noabx, device, forecast_model, output_vars,
                T, delta_t, dimension,
                only_jump_before_abx_exposure=False,
                use_only_dyn_ft_as_input=use_only_dyn_ft_as_input,
                add_dynamic_cov=add_dynamic_cov,
                use_dyn_cov_after_abx=use_dyn_cov_after_abx,
                use_obs_until_t=start_time*delta_t, sf=sf)
            obs = true_X.transpose(2, 0, 1)
            ad_scores = ad_module(
                obs, cond_moments, observed_dates, cutoff_adj_sf)
            use_obs_until_day = ((start_time+starting_date) *
                                 np.ones((ad_scores.shape[0], 1)))
            if scoring_distribution == 'z_score':
                for d in range(ad_scores.shape[1]):
                    data = np.concatenate(
                        [host_id.reshape(-1, 1), abx_labels.reshape(-1, 1),
                         use_obs_until_day,
                         np.ones((len(host_id),1))*(d + starting_date),
                         ad_scores[:, d:d+1]],
                        axis=1)
                    cols = ['host_id', 'abx', 'use_obs_until_day', 'score_date',
                            'z_score']
                    df_ = pd.DataFrame(data, columns=cols)
                    df_ = df_.dropna(axis=0, how='any', inplace=False)
                    df_["days_since_cutoff"] = (
                            df_["score_date"] - df_["use_obs_until_day"])
                    df = pd.concat([df, df_])
            else:
                data = np.concatenate(
                    [host_id.reshape(-1, 1), abx_labels.reshape(-1, 1),
                     use_obs_until_day, ad_scores],
                    axis=1)
                cols = ['host_id', 'abx', 'use_obs_until_day'] + [
                    'ad_score_day-{}'.format(i + starting_date)
                    for i in range(ad_scores.shape[1])]
                # data_collect.append(data)
                df = pd.DataFrame(data, columns=cols)
                csvpath_val_releval = '{}val_noabx_ad_scores_{}_{}.csv'.format(
                    reli_eval_path, start_time, aggregation_method)
                df.to_csv(csvpath_val_releval, index=False)
        if scoring_distribution == 'z_score':
            csvpath_val_releval = '{}val_noabx_z_scores_{}.csv'.format(
                reli_eval_path, aggregation_method)
            df.to_csv(csvpath_val_releval, index=False)
            filepaths.append(csvpath_val_releval)

    if send:
        files_to_send = [csvpath, csvpath_val] + filepaths
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
    plt.close()
    return impath


def compute_zscore_scaling_factors(
        forecast_saved_models_path,
        forecast_model_id=None,
        load_best=True,
        aggregation_method="coord-0",
        scoring_distribution="normal",
        interval_length=30,
        shift_by=1,
        send=False,
        **kwargs):

    assert scoring_distribution == "z_score", \
        "scoring_distribution must be z_score"

    forecast_model_path = "{}id-{}/".format(
        forecast_saved_models_path, forecast_model_id
    )
    ad_path = "{}anomaly_detection/".format(forecast_model_path)
    which = "best" if load_best else "last"
    reli_eval_path = f'{ad_path}reliability_eval-val-noabx_{which}_{scoring_distribution}/'
    csvpath_val_releval = '{}val_noabx_z_scores_{}.csv'.format(
        reli_eval_path, aggregation_method)
    outpath = f'{ad_path}zscore_scaling_factors_{which}/'
    makedirs(outpath)
    filename = (f'{outpath}zscore_scaling_factors_{aggregation_method}.csv')

    df = pd.read_csv(csvpath_val_releval)
    max_dsc = df["days_since_cutoff"].max()
    data = []
    for dsc in range(0, max_dsc, shift_by):
        left = dsc - interval_length/2
        right = min(dsc + interval_length/2, max_dsc)
        data.append([dsc, left, right,
                     df.loc[(df["days_since_cutoff"] >= left) &
                            (df["days_since_cutoff"] <= right),
                     "z_score"].std()])

    cols = ["days_since_cutoff", "days_since_cutoff_std_int_left",
            "days_since_cutoff_std_int_right", "std_z_scores"]
    df_out = pd.DataFrame(data, cols)
    df_out.to_csv(filename, index=False)

    if send:
        caption = "z-scores scaling factors - {} - id={}".format(
            which, forecast_model_id)
        SBM.send_notification(
            text=None,
            chat_id=config.CHAT_ID,
            files=[filename],
            text_for_files=caption
        )



def evaluate_scores(
    forecast_saved_models_path,
    forecast_model_id=None,
    load_best=True,
    validation=False,
    send=False,
    dataset=None,
    only_jump_before_abx_exposure=False,
    aggregation_method="coord-0",
    scoring_distribution="normal",
    **options,
):
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
    forecast_model_path = "{}id-{}/".format(
        forecast_saved_models_path, forecast_model_id
    )
    ad_path = "{}anomaly_detection/".format(forecast_model_path)
    which = "best" if load_best else "last"
    scores_path = "{}scores_{}_{}/".format(ad_path, which, scoring_distribution)
    evaluation_path = "{}evaluation_{}_{}_{}/".format(
        ad_path, which, only_jump_before_abx_exposure, aggregation_method)
    makedirs(evaluation_path)


    all_scores = {}
    files_to_send = []
    for split in ["train", "val"]:
        # load scores
        ad_scores = pd.read_csv(
            f"{scores_path}{split}_ad_scores_{only_jump_before_abx_exposure}"
            f"_{aggregation_method}.csv")
        score_cols = [x for x in ad_scores.columns if x.startswith("ad_score_day-")]

        # plot histograms
        impath_hist = _plot_n_save_histograms(
            abx_samples=ad_scores.loc[ad_scores["abx"], score_cols].values,
            non_abx_samples=ad_scores.loc[~ad_scores["abx"], score_cols].values,
            split=split,
            path_to_save=evaluation_path,
        )
        files_to_send.append(impath_hist)

        # flatten and enrich scores
        all_scores[split] = _transform_scores(ad_scores)

    if send:
        # histograms
        caption = "scores-histogram - {} - id={}".format(
            which, forecast_model_id)
        SBM.send_notification(
            text=None,
            chat_id=config.CHAT_ID,
            files=[impath_hist],
            text_for_files=caption
        )

    noabx_train = all_scores["train"][~all_scores["train"]["abx"]].copy()
    noabx_val = all_scores["val"][~all_scores["val"]["abx"]].copy()

    abx_scores_train = all_scores["train"][all_scores["train"]["abx"]].copy()
    abx_scores_val = all_scores["val"][all_scores["val"]["abx"]].copy()
    abx_scores = pd.concat([abx_scores_train, abx_scores_val])
    abx_scores = abx_scores[abx_scores.score.notnull()].copy()

    all_scores_split = {
        "train_noabx_t": noabx_train,
        "val_noabx_t": noabx_val,
        "abx_t": abx_scores,
    }

    dic_img_age = {}
    for flag, scores in all_scores_split.items():
        # plot scores over age
        dic_img_age[flag] = _plot_score_over_age_legacy(scores, flag, evaluation_path)

    # send to telegram
    if send:
        # scores over age
        for k, v in dic_img_age.items():
            caption = "scores-over-age {} - {} - id={}".format(
                k, which, forecast_model_id)
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
        only_jump_before_abx_exposure=False,
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
    use_only_dyn_ft_as_input = None
    if 'use_only_dyn_ft_as_input' in options:
        use_only_dyn_ft_as_input = options['use_only_dyn_ft_as_input']
        if use_only_dyn_ft_as_input is not None:
            use_only_dyn_ft_as_input = "after_nth_abx_exposure"
            options['masked'] = True
    add_dynamic_cov = False
    if 'add_dynamic_cov' in options:
        add_dynamic_cov = options['add_dynamic_cov']

    # specify the input and output variables of the model, as function of X
    input_vars = ['id']
    output_vars = ['id']
    if 'func_appl_X' in options:  # list of functions to apply to the paths in X
        functions = options['func_appl_X']
        collate_fn, mult = data_utils.MicrobialCollateFnGen(
            functions, use_only_dyn_ft_as_input=use_only_dyn_ft_as_input,
            only_jump_before_abx_exposure=only_jump_before_abx_exposure)
        input_size = input_size * mult
        output_size = output_size * mult
        output_vars += functions
        input_vars += functions
    else:
        functions = None
        collate_fn, mult = data_utils.MicrobialCollateFnGen(
            None, use_only_dyn_ft_as_input=use_only_dyn_ft_as_input,
            only_jump_before_abx_exposure=only_jump_before_abx_exposure)
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
        'size_X': dimension*mult,
        'options': options}

    return params_dict, collate_fn, use_only_dyn_ft_as_input, add_dynamic_cov




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
                if FLAGS.compute_zscore_scaling_factors:
                    compute_zscore_scaling_factors(
                        send=FLAGS.SEND,
                        forecast_saved_models_path=forecast_saved_models_path,
                        forecast_model_id=forecast_model_ids[i],
                        **ad_param)


if __name__ == '__main__':
    app.run(main)