# =====================================================================================================================
import pandas as pd
import json

import sklearn.metrics
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

plt.rcParams.update({"font.family": "DejaVu Sans"})
plt.style.use("tableau-colorblind10")

from torch.backends import cudnn
import gc
import math
from sklearn import metrics

from configs import config
import models_synthetic as models
import data_utils
from AD_modules import AD_module
import AD_modules

try:
    from telegram_notifications import send_bot_message as SBM
except Exception:
    from configs.config import SendBotMessage as SBM



# =====================================================================================================================
# FLAGS
FLAGS = flags.FLAGS

config.del_all_flags(FLAGS)

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

flags.DEFINE_bool("plot_only", False, "whether to use the plot only mode")
flags.DEFINE_bool("quant_eval_in_plot_only", False, "whether to do the quantitative evaluation of the model during together with plot-only")
flags.DEFINE_string("which_AD_model_load", None, "one of: {'best', 'last', None}")


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

class IrregularADDataset(Dataset):
    """
    class for iterating over a dataset
    """
    def __init__(self, model_name, time_id=None, idx=None):
        stock_paths, observed_dates, nb_obs, ad_labels, hyperparam_dict = load_AD_dataset(
            stock_model_name=model_name, time_id=time_id)
        if idx is None:
            idx = np.arange(hyperparam_dict['nb_paths'])
        self.metadata = hyperparam_dict
        self.stock_paths = stock_paths[idx]
        self.observed_dates = observed_dates[idx]
        self.nb_obs = nb_obs[idx]
        self.ad_labels = ad_labels[idx]

    def __len__(self):
        return len(self.nb_obs)

    def __getitem__(self, idx):
        if type(idx) == int:
            idx = [idx]
        # stock_path dimension: [BATCH_SIZE, DIMENSION, TIME_STEPS]
        return {"idx": idx, "stock_path": self.stock_paths[idx], 
                "observed_dates": self.observed_dates[idx], 
                "nb_obs": self.nb_obs[idx], "ad_labels": self.ad_labels[idx],
                "dt": self.metadata['dt']}

'''
        hidden_size=10, 
        bias=True, 
        dropout_rate=0.1,
        ode_nn=default_ode_nn, 
        readout_nn=default_readout_nn,
        enc_nn=default_enc_nn, 
        use_rnn=False,
        solver="euler",
        weight=0.5, 
        weight_decay=1.,'''

def evaluate(
        saved_models_path,
        forecast_saved_models_path,
        forecast_model_id=None, 
        epochs=1,
        batch_size=100,
        seed=398,
        test_size=0.2,
        learning_rate=0.001,
        optim_method='adam',
        anomaly_data_dict=None, # data_dict from the data we aim to evaluate
        forecast_param = None,
        plot=True, 
        paths_to_plot=(0,),
        steps_ahead = [1],
        use_gpu=None,
        nb_cpus=None,
        n_dataset_workers=None,
        plot_only=False,
        quant_eval_in_plot_only=False,
        **options
):

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

    # load dataset-metadata
    
    anomaly_dataset, anomaly_dataset_id = data_utils._get_dataset_name_id_from_dict(
        data_dict=anomaly_data_dict)
    anomaly_dataset_id = int(anomaly_dataset_id)
    anomaly_dataset_metadata = data_utils.load_metadata(
        stock_model_name=anomaly_dataset, time_id=anomaly_dataset_id)

    dimension = anomaly_dataset_metadata['dimension']
    T = anomaly_dataset_metadata['maturity']
    delta_t = anomaly_dataset_metadata['dt']  # copy metadata
    t_period = anomaly_dataset_metadata['period']

    # load raw data
    # anomaly_idx = np.arange(anomaly_dataset_metadata["nb_paths"])
    train_idx, test_idx = train_test_split(
        np.arange(anomaly_dataset_metadata["nb_paths"]), test_size=test_size,
        random_state=seed)
    data_train = IrregularADDataset(
        model_name=anomaly_dataset, time_id=anomaly_dataset_id, idx=train_idx)
    data_test = IrregularADDataset(
        model_name=anomaly_dataset, time_id=anomaly_dataset_id, idx=test_idx)
        
    # get data-loader for training
    # get additional plotting information
    plot_forecast_predictions = False
    if 'plot_forecast_predictions' in options:
        plot_forecast_predictions = options['plot_forecast_predictions']
        if 'forecast_horizons_to_plot' in options:
            forecast_horizons_to_plot = options['forecast_horizons_to_plot']
        else:
            forecast_horizons_to_plot = steps_ahead[-1]
    std_factor = 1  # factor with which the std is multiplied
    plot_variance = True
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
        
    
    print(anomaly_dataset_metadata)
    #stockmodel = data_utils._STOCK_MODELS[
    #    dataset_metadata['model_name']](**dataset_metadata)

    # get all needed paths
    forecast_model_path = '{}id-{}/'.format(forecast_saved_models_path, forecast_model_id)
    forecast_model_path_save_best = '{}best_checkpoint/'.format(forecast_model_path)
    forecast_model_path_save_last = '{}last_checkpoint/'.format(forecast_model_path)
    load_which = 'last'
    if 'load_which' in options:
        load_which = options['load_which']
    if load_which == 'best':
        load_path = forecast_model_path_save_best
    else:
        load_path = forecast_model_path_save_last

    model_path = '{}{}id-{}/'.format(forecast_model_path,saved_models_path, 0) # to change
    model_path_save_last = '{}last_checkpoint/'.format(model_path)
    model_path_save_best = '{}best_checkpoint/'.format(model_path)
    # model_metric_file = '{}metric_id-{}.csv'.format(model_path, model_id)
    # plot_save_path = '{}plots/'.format(model_path)

    # get params_dict
    forecast_params_dict, collate_fn = get_forecast_model_param_dict(
        **forecast_param
    )
    desc = json.dumps(forecast_params_dict, sort_keys=True)  # serialize to a JSON formatted str
    output_vars = forecast_params_dict['output_vars']

    forecast_model = models.NJODE(**forecast_params_dict)  # get NJODE model class from
    model_name = 'PD-NJODE'
    forecast_model.to(device)

    forecast_optimizer = torch.optim.Adam(forecast_model.parameters(), lr=0.001, weight_decay=0.0005)
    models.get_ckpt_model(load_path, forecast_model, forecast_optimizer, device)
    forecast_model.eval()
    del forecast_optimizer

    dl = DataLoader(  # class to iterate over training data
        dataset=data_train, collate_fn=collate_fn, shuffle=True, 
        #batch_size=10)
        batch_size=batch_size)
    dl_test = DataLoader(  # class to iterate over training data
        dataset=data_test, collate_fn=collate_fn, shuffle=False,
        #batch_size=10)
        batch_size=len(test_idx))
    max_path_plotting = int(np.max(paths_to_plot))+1
    dl_plot = DataLoader(
        dataset=data_test, collate_fn=collate_fn, shuffle=False,
        batch_size=max_path_plotting)

    quantitative_eval = False
    if 'quantitative_AD_eval' in options:
        quantitative_eval = True
        quantitative_AD_eval = options['quantitative_AD_eval']
        data_an = quantitative_AD_eval['anomaly_data_dict']
        data_af = quantitative_AD_eval['anomaly_free_data_dict']
        ds, ds_id = data_utils._get_dataset_name_id_from_dict(
            data_dict=data_an)
        ds_id = int(ds_id)
        data_an = IrregularADDataset(model_name=ds, time_id=ds_id)
        ds, ds_id = data_utils._get_dataset_name_id_from_dict(
            data_dict=data_af)
        ds_id = int(ds_id)
        data_af = IrregularADDataset(model_name=ds, time_id=ds_id)
        dl_an_eval = DataLoader(  # class to iterate over training data
            dataset=data_an, collate_fn=collate_fn, shuffle=False,
            batch_size=len(data_an))
        dl_af_eval = DataLoader(  # class to iterate over training data
            dataset=data_af, collate_fn=collate_fn, shuffle=False,
            batch_size=len(data_af))

    eval_path = '{}/anomaly_detection/'.format(model_path) + anomaly_data_dict + '/'
    # if not os.path.exists(eval_model_path):
    #     os.mkdir(eval_model_path)
    anom_type = anomaly_dataset_metadata['anomaly_params']['type']
    # eval_path = eval_model_path + '/' + anom_type
    makedirs(eval_path)
    eval_plot_path = eval_path + 'plots/'
    eval_metrics_path = eval_path + 'metrics/'
    eval_weights_path = eval_path + 'ad_weights/'
    eval_quantitative_path = eval_path + 'quantitative_eval/'
    makedirs(eval_plot_path)
    makedirs(eval_metrics_path)
    makedirs(eval_weights_path)
    makedirs(eval_quantitative_path)

    replace_values = get_replace_forecast_values(
        model=forecast_model, data_dict=forecast_param['data_dict'],
        collate_fn=collate_fn, steps_ahead=steps_ahead, device=device)
    torch.cuda.empty_cache()

    prop_cycle = plt.rcParams['axes.prop_cycle']  # change style of plot?
    colors = prop_cycle.by_key()['color']
    std_color = list(matplotlib.colors.to_rgb(colors[1])) + [0.5]

    """
    times : observation times (for any batch element), [nb_obs], value in [0,nb_steps]
    time_ptr : cumulated number of observations until each time (counting across batch), [nb_obs+1] (because starts at 0)
    obs_idx : list of observed batch idx across time [time_ptr[-1]], value in [0,batch]
    """

    nb_steps_ahead = len(steps_ahead)
    max_steps_ahead = max(steps_ahead)
    neighbouring_scores = 5
    if 'neighbouring_scores' in options:
        neighbouring_scores = options['neighbouring_scores']
    ad_module = AD_module(
        output_vars=output_vars, steps_ahead=steps_ahead, smoothing=neighbouring_scores,
        replace_values=replace_values, class_thres=class_thres)
    if optim_method == 'adam':
        optimizer = torch.optim.Adam(ad_module.parameters(), lr=learning_rate, weight_decay=0.0005)
    elif optim_method == 'linear':
        epochs = 1
    else:
        raise ValueError('Unknown optimizer')
    best_score_val = -np.inf

    # load the AD module
    which_AD_model_load = None
    if optim_method == 'adam':
        if 'which_AD_model_load' in options:
            which_AD_model_load =options['which_AD_model_load']
        load_path_AD = None
        if which_AD_model_load == "last":
            load_path_AD = model_path_save_last
        elif which_AD_model_load == "best":
            load_path_AD = model_path_save_best
        if load_path_AD is not None:
            AD_modules.get_ckpt_model(
                load_path_AD, ad_module, optimizer, device)
            print("loaded AD module from {}".format(load_path_AD))


    # ------ convenience functions ------
    def get_score_and_pred(b):
        """
        Args:
            b: batch
        """
        if optim_method == 'adam':
            optimizer.zero_grad()  # reset the gradient
        times = b["times"]
        time_ptr = b["time_ptr"]
        X = b["X"].to(device)
        M = b["M"]
        if M is not None:
            M = M.to(device)
        start_M = b["start_M"]
        if start_M is not None:
            start_M = start_M.to(device)
        start_X = b["start_X"].to(device)
        obs_idx = b["obs_idx"]
        n_obs_ot = b["n_obs_ot"].to(device)
        observed_dates = b['observed_dates']
        path_t_true_X = np.linspace(0., T, int(np.round(T / delta_t)) + 1)
        true_M = b["true_mask"]
        true_X = b["true_paths"]
        ad_labels = b["ad_labels"]
        batch_size = start_X.shape[0]
        with torch.no_grad():
            path_t, path_y = forecast_model.custom_forward(
                # hT, c_loss, path_t, path_h, path_y = model(
                times, time_ptr, X, obs_idx, delta_t, T, start_X,
                n_obs_ot, steps=steps_ahead, get_loss=False, M=M,
                start_M=start_M)
            torch.cuda.empty_cache()
        nb_steps = path_t_true_X.shape[0]
        nb_moments = len(output_vars)
        cond_moments = np.empty(
            (nb_steps, batch_size, dimension, nb_moments, nb_steps_ahead))
        cond_moments[:, :, :, :, :] = np.nan
        for i in range(nb_steps_ahead):
            index_times = np.empty_like(path_t_true_X)
            for j, t in enumerate(path_t_true_X):
                if np.any(
                    np.abs(t * np.ones_like(path_t[i]) - path_t[i]) < 1e-10):
                    index_times[j] = True
                else:
                    index_times[j] = False
            index_times = np.argwhere(index_times).reshape(-1)
            for m in range(nb_moments):
                cond_moments[index_times, :, :, m, i] = path_y[i].detach(
                ).cpu().numpy()[:, :, m * dimension:(m + 1) * dimension]
        obs = torch.tensor(true_X, dtype=torch.float32).permute(2, 0, 1)
        cond_moments = torch.tensor(cond_moments, dtype=torch.float32)
        ad_scores, mask = ad_module(obs, cond_moments)
        ad_labels = ad_labels.transpose((2, 0, 1))
        return ad_scores, mask, cond_moments, obs, path_t, path_y, ad_labels

    def quant_eval(fname):
        ad_scores, mask, cond_moments, obs, path_t, path_y, \
            ad_labels_af = (get_score_and_pred(next(iter(dl_af_eval))))
        ad_scores_af = ad_scores.detach().cpu().numpy()
        ad_scores_af = ad_scores_af[mask].reshape(-1)
        pred_labels_af = (ad_scores_af > ad_module.threshold) * 1
        pred_labels_af = pred_labels_af.astype(int)
        ad_labels_af = ad_labels_af[mask].reshape(-1).astype(int)
        ad_scores, mask, cond_moments, obs, path_t, path_y, \
            ad_labels_an = (get_score_and_pred(next(iter(dl_an_eval))))
        ad_scores_an = ad_scores.detach().cpu().numpy()
        ad_scores_an = ad_scores_an[mask].reshape(-1)
        pred_labels_an = (ad_scores_an > ad_module.threshold) * 1
        pred_labels_an = pred_labels_an.astype(int)
        ad_labels_an = ad_labels_an[mask].reshape(-1).astype(int)

        precision_af, recall_af, fscore_af, support_af = (
            metrics.precision_recall_fscore_support(
                ad_labels_af, pred_labels_af, labels=[0, 1]))
        av_fscore_af = metrics.f1_score(
            ad_labels_af, pred_labels_af, labels=[0, 1], average='micro')
        cm_af = metrics.confusion_matrix(
            ad_labels_af, pred_labels_af, labels=[0, 1]).flatten()
        r_af = np.concatenate(
            [precision_af, recall_af, fscore_af, support_af, cm_af,
             np.array(av_fscore_af).reshape((1,))], axis=0)
        precision_an, recall_an, fscore_an, support_an = (
            metrics.precision_recall_fscore_support(
                ad_labels_an, pred_labels_an, labels=[0, 1]))
        av_fscore_an = metrics.f1_score(
            ad_labels_an, pred_labels_an, labels=[0, 1], average='micro')
        cm_an = metrics.confusion_matrix(
            ad_labels_an, pred_labels_an, labels=[0, 1]).flatten()
        r_an = np.concatenate(
            [precision_an, recall_an, fscore_an, support_an, cm_an,
             np.array(av_fscore_an).reshape((1,))], axis=0)
        label_names = []
        for l in ["precision", "recall", "fscore", "support"]:
            for c in [0, 1]:
                label_names.append("{}_{}".format(l, c))
        label_names += ["TN", "FP", "FN", "TP", "f1score_micro"]
        df_quant_res = pd.DataFrame(
            np.stack([r_an, r_af], axis=0),
            columns=label_names, index=["anomalous", "anomaly_free"])
        df_quant_res.to_csv(fname, index=True)
    # -----------------------------------

    # -------- plot only option ---------
    if plot_only:
        plot_filename = 'plot_from-{}'.format(which_AD_model_load)
        plot_AD_module_params(
            ad_module, steps_ahead=steps_ahead, path=eval_weights_path,
            filename=plot_filename, anomaly_type=anom_type)
        files_to_send = [eval_weights_path+plot_filename+'_ad_module_weights.pdf']

        batch = next(iter(dl_plot))
        plot_filename1 = plot_filename + '_path-{}.pdf'
        print("Plotting ..")
        plot_one_path_with_pred(device, forecast_model, ad_module, batch,
                                delta_t, T,
                                paths_to_plot=paths_to_plot,
                                save_path=eval_plot_path,
                                filename=plot_filename1,
                                plot_variance=plot_variance,
                                std_factor=std_factor,
                                output_vars=output_vars,
                                steps_ahead=steps_ahead,
                                plot_forecast_predictions=plot_forecast_predictions,
                                forecast_horizons_to_plot=forecast_horizons_to_plot,
                                anomaly_type=anom_type)
        torch.cuda.empty_cache()

        if quant_eval_in_plot_only:
            fname = "{}quant_eval_from-{}.csv".format(
                eval_quantitative_path, which_AD_model_load)
            quant_eval(fname=fname)
            files_to_send.append(fname)

        if FLAGS.SEND:
            caption = f"{anom_type}"
            for i in paths_to_plot:
                files_to_send.append(
                    os.path.join(eval_plot_path, plot_filename1.format(i)))
            SBM.send_notification(
                text=f'finished plot-only from AD module for anomaly: {anom_type}',
                chat_id=config.CHAT_ID,
                files=files_to_send,
                text_for_files=caption)

        return 0
    # -----------------------------------

    # --- training & eval of AD module --
    weights_filename = "epoch-{}".format(0)
    plot_AD_module_params(
        ad_module, steps_ahead=steps_ahead, path=eval_weights_path,
        filename=weights_filename, anomaly_type=anom_type)

    print("Starting training ... ")
    data_metrics = []
    for e in range(epochs): # change for learning parameters of AD module
        epoch = e+1
        print("Epoch {}".format(epoch))
        ad_module.train()
        for j, b in enumerate(dl):  # iterate over dataloader for validation set
            print("iteration {}".format(j + 1))
            ad_scores, mask, cond_moments, obs, path_t, path_y, ad_labels = (
                get_score_and_pred(b))
            ad_labels = torch.tensor(ad_labels).float()
            if optim_method == 'linear':
                ad_module.linear_solver(obs, ad_labels, cond_moments)
            elif optim_method == 'adam':
                ad_scores = ad_scores[mask]
                ad_labels = ad_labels[mask]
                loss = ad_module.loss(ad_scores, ad_labels)
                loss.backward()
                optimizer.step()
            del path_t, path_y, cond_moments

        with (torch.no_grad()):
            tot_score_val = 0
            ad_module.eval()
            print("Evaluation epoch {}".format(epoch))

            for _, b in enumerate(dl_test):
                ad_scores, mask, cond_moments, obs, path_t, path_y, \
                    ad_labels = (get_score_and_pred(b))
                ad_scores = ad_scores.detach().cpu().numpy()
                predicted_ad_labels, _ = ad_module.get_predicted_label(obs, cond_moments)
                predicted_ad_labels = predicted_ad_labels.detach().cpu().numpy()

                weights_filename = "epoch-{}".format(epoch)
                plot_AD_module_params(
                    ad_module, steps_ahead=steps_ahead, path=eval_weights_path,
                    filename=weights_filename, anomaly_type=anom_type)
                metrics_filename = "epoch-{}".format(epoch)
                score_val, threshold = compute_metrics(
                    ad_labels, ad_scores, mask, autom_thres = autom_thres,
                    threshold=ad_module.threshold,
                    path=eval_metrics_path, filename = metrics_filename)
                tot_score_val += score_val
                if threshold is not None:
                    ad_module.threshold = threshold
                del path_t, path_y, cond_moments
            print("current score:", tot_score_val)
            data_metrics.append([epoch, tot_score_val])

        if tot_score_val > best_score_val:
            AD_modules.save_checkpoint(ad_module, optimizer, model_path_save_best, epoch)
        AD_modules.save_checkpoint(ad_module, optimizer, model_path_save_last, epoch)

        if quantitative_eval:
            fname = "{}epoch-{}.csv".format(eval_quantitative_path, epoch)
            quant_eval(fname=fname)

        if plot:
            batch = next(iter(dl_plot))

            plot_filename = 'epoch-{}'.format(epoch)
            plot_filename = plot_filename + '_path-{}.png'
            print("Plotting ..")
            plot_one_path_with_pred(device, forecast_model, ad_module, batch, delta_t, T,
                paths_to_plot=paths_to_plot, save_path=eval_plot_path, filename=plot_filename,
                plot_variance=plot_variance, std_factor=std_factor,
                output_vars=output_vars, steps_ahead=steps_ahead, plot_forecast_predictions=plot_forecast_predictions,
                forecast_horizons_to_plot=forecast_horizons_to_plot, anomaly_type=anom_type)
            torch.cuda.empty_cache()

    # save metrics file
    df_metrics = pd.DataFrame(data=data_metrics, columns=["epoch", "f1_score"])
    df_filename = "{}metric.csv".format(eval_metrics_path)
    df_metrics.to_csv(df_filename, index=False)
    return 0


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


def get_replace_forecast_values(model, data_dict, collate_fn, steps_ahead, replace_with='mean',
                                variables = ['var'], batch_size = 10, device = 'cpu'):
    
    nb_steps_ahead = len(steps_ahead)
    output_vars = model.output_vars

    dataset, dataset_id = data_utils._get_dataset_name_id_from_dict(
        data_dict=data_dict)
    dataset_id = int(dataset_id)
    dataset_metadata = data_utils.load_metadata(stock_model_name=dataset,
                                                time_id=dataset_id)
    T = dataset_metadata['maturity']
    delta_t = dataset_metadata['dt']
    dimension = dataset_metadata['dimension']
    idx = np.arange(dataset_metadata["nb_paths"])
    data_train = IrregularADDataset(
        model_name=dataset, time_id=dataset_id, idx=idx)
    dl = DataLoader(  # class to iterate over training data
        dataset=data_train, collate_fn=collate_fn, shuffle=True, 
        batch_size=batch_size)
    
    b = next(iter(dl))

    times = b["times"]
    time_ptr = b["time_ptr"]
    X = b["X"].to(device)
    M = b["M"]
    if M is not None:
        M = M.to(device)
    start_M = b["start_M"]
    if start_M is not None:
        start_M = start_M.to(device)
    start_X = b["start_X"].to(device)
    obs_idx = b["obs_idx"]
    n_obs_ot = b["n_obs_ot"].to(device)
    observed_dates = b['observed_dates']
    path_t_true_X = np.linspace(0., T, int(np.round(T / delta_t)) + 1)
    true_M = b["true_mask"]
    true_X = b["true_paths"]
    ad_labels = b["ad_labels"]

    model.eval()

    with torch.no_grad():
        path_t, path_y = model.custom_forward(
        # hT, c_loss, path_t, path_h, path_y = model( 
            times, time_ptr, X, obs_idx, delta_t, T, start_X,
            n_obs_ot, steps=steps_ahead, get_loss=False, M=M,
            start_M=start_M)
        torch.cuda.empty_cache()
        
    nb_steps = path_t_true_X.shape[0]
    
    nb_vars = len(output_vars)
    cond_moments = np.empty((nb_steps, dl.batch_size, dimension, nb_vars, nb_steps_ahead))
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

    replace_values = {}
    for variable in variables:
        which = np.argmax(np.array(output_vars) == variable)
        values = cond_moments[:,:,:,which,:].reshape(-1, dimension, nb_steps_ahead)

        condition = ~np.isnan(values)
        if variable == 'var':
            condition = np.logical_and(condition, values >= 0)

        rv = np.empty((dimension, nb_steps_ahead))
        rv[:,:] = np.nan
        for j in range(dimension):
            for s in range(nb_steps_ahead):
                vals = values[:,j,s]
                cond = condition[:,j,s]
                if replace_with == 'mean':
                    rv[j,s] = np.mean(vals[cond])

        replace_values[variable] = rv.copy()

    del dl, data_train, model, b
    
    return replace_values

def get_forecast_model_param_dict(
        data_dict,
        epochs=100,
        seed=398,
        batch_size=200,
        hidden_size=10, 
        bias=True, 
        dropout_rate=0.1,
        ode_nn=default_ode_nn, 
        readout_nn=default_readout_nn,
        enc_nn=default_enc_nn, 
        use_rnn=False,
        solver="euler",
        weight=0.5, 
        weight_decay=1.,
        **options):
    
    dataset, dataset_id = data_utils._get_dataset_name_id_from_dict(
        data_dict=data_dict)
    dataset_id = int(dataset_id)
    dataset_metadata = data_utils.load_metadata(stock_model_name=dataset,
                                                time_id=dataset_id)
    
    input_size = dataset_metadata['dimension']
    dimension = dataset_metadata['dimension']
    output_size = input_size
    T = dataset_metadata['maturity']
    delta_t = dataset_metadata['dt']  # copy metadata
    t_period = dataset_metadata['period']
    if 'solver_delta_t_factor' in options:
        model_delta_t = delta_t / options['solver_delta_t_factor']
    else:
        model_delta_t = delta_t

    train_data_perc = None
    if 'train_data_perc' in options:
        train_data_perc = options['train_data_perc']
    if 'scale_dt' in options:
        if options['scale_dt'] == 'automatic':
            options['scale_dt'] = 1./delta_t
            if train_data_perc is not None:
                options['scale_dt'] *= train_data_perc
            if 'solver_delta_t_factor' in options:
                options['scale_dt'] *= options['solver_delta_t_factor']

    output_vars = ['id']
    if 'func_appl_X' in options:  # list of functions to apply to the paths in X
        functions = options['func_appl_X']
        collate_fn, mult = data_utils.CustomCollateFnGen(functions)
        input_size = input_size * mult
        output_size = output_size * mult
        output_vars += functions
    else:
        functions = None
        collate_fn, mult = data_utils.CustomCollateFnGen(None)
        mult = 1
    if 'add_pred' in options:
        add_pred = options['add_pred']
        nb_pred_add = len(add_pred)
        mult += nb_pred_add
        output_size += nb_pred_add * dimension
        output_vars += add_pred
    
    opt_eval_loss = np.nan
    params_dict = {  # create a dictionary of the wanted parameters
        'input_size': input_size,
        'hidden_size': hidden_size, 'output_size': output_size, 'bias': bias,
        'ode_nn': ode_nn, 'readout_nn': readout_nn, 'enc_nn': enc_nn,
        'use_rnn': use_rnn,
        'dropout_rate': dropout_rate, 'batch_size': batch_size,
        'solver': solver, 'dataset': dataset, 'dataset_id': dataset_id,
        'seed': seed,
        'weight': weight, 'weight_decay': weight_decay,
        'optimal_eval_loss': opt_eval_loss, 
        't_period': t_period, 'output_vars': output_vars, 'delta_t': delta_t,
        'options': options}

    return params_dict, collate_fn
    


def plot_AD_module_params(
    ad_module, path, filename, steps_ahead = None, dt=0.0025, anomaly_type="",
    save_extras={'bbox_inches': 'tight', 'pad_inches': 0.01}, ):
    if isinstance(ad_module, AD_module):
        if anomaly_type == 'cutoff':
            anomaly_type = 'drift'
        weights = ad_module.get_weights().squeeze().clone().detach().numpy()
        steps_ahead = [str(s) for s in steps_ahead]
        neighbors = [str(i) for i in range(-ad_module.smoothing, ad_module.smoothing+1)]

        fig, ax = plt.subplots(figsize=(5,5))
        im = ax.imshow(weights)
        ax.set_xticks(np.arange(weights.shape[1]))
        ax.set_yticks(np.arange(weights.shape[0]))
        ax.set_xticklabels(neighbors)
        ax.set_yticklabels(steps_ahead)
        fig.colorbar(im, ax=ax, shrink=0.25)
        # plt.title('Aggregation weights of scores', fontsize=12)
        plt.title(f'{anomaly_type}'.capitalize(), fontsize=12)
        plt.xlabel('neighbouring timestamps ($l$)', fontsize=9)
        plt.ylabel('steps-ahead ($k$)', fontsize=9)
        plt.savefig(path + filename + '_ad_module_weights.pdf', **save_extras, dpi=400)
        plt.close()


def plot_one_path_with_pred(
        device, forecast_model, ad_module, batch, delta_t, T,
        paths_to_plot=(0,), save_path='', filename='plot_{}.png',
        plot_variance=False, std_factor=1.96, model_name='PD-NJ-ODE',
        save_extras={'bbox_inches': 'tight', 'pad_inches': 0.01},
        output_vars=None, steps_ahead=[1], plot_forecast_predictions=False,
        forecast_horizons_to_plot=[1], anomaly_type='undefined',
):
    if anomaly_type == 'cutoff':
        anomaly_type = 'drift'

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
        fig, axs = plt.subplots(dimension, 1, figsize=(9,5))
        if dimension == 1:
            axs = [axs]

        for j in range(dimension):
            # get the true_X at observed dates
            path_t_obs = []
            path_X_obs = []
            for k, od in enumerate(observed_dates[a]):
                if od == 1:
                    if true_M is None or (true_M is not None and
                                          true_M[a, j, k] == 1):
                        path_t_obs.append(path_t_true_X[k])
                        path_X_obs.append(true_X[a, j, k])
            path_t_obs = np.array(path_t_obs)
            path_X_obs = np.array(path_X_obs)

            title = f"{anomaly_type} anomaly"
            if dimension > 1:
                title += ", dimension {}".format(j + 1)

            ax2 = axs[j].twinx()
            ax1 = axs[j]
            ax1.set_title(title.capitalize(), fontsize=12)
            ax2.plot(path_t_true_X, true_X[a, j, :],
                           label='true path, no anomaly',
                           color=colors[0])
            ax2.plot(path_t_true_X, mask_data_path_w_anomaly[a, j, :],
                           label='true path, anomaly',
                           color=colors[1])

            ax1.plot(path_t_true_X, mask_pred_scores[a, j, :],
                           label='predicted scores',
                           color="green", alpha=0.75)
            ax1.fill_between(
                path_t_true_X, 0, 1,
                where=predicted_ad_labels_plot[a, j, :]==1, facecolor='red', alpha=.25,
                label='predicted anomaly')
            ax1.axhline(
                y=ad_module.threshold, color='r', linestyle='-', alpha=0.5,
                label='score threshold')
            ax1.set_ylim(0,1)
            ax2.legend(loc='upper right', fontsize=9)
            ax1.set_ylabel('score', fontsize=9)
            ax1.legend(loc='upper left', fontsize=9)
            ax2.set_ylabel('process values', fontsize=9)
            ax1.set_xlabel('$t$', fontsize=9)
            # axs[j][0].set_ylim(0., 1.)

            if plot_forecast_predictions:
                for h, s in enumerate(forecast_horizons_to_plot):
                    step_idx = steps_ahead.index(s)
                    exp_prediction = cond_moments[:, a, j, 0,
                                     step_idx].detach().cpu().numpy()
                    axs[j].plot(path_t_true_X, exp_prediction,
                                   label=model_name + " expectation prediction {} steps ahead".format(
                                       s), color=colors[h + 2])
                    if plot_variance:
                        assert (('var' in output_vars) or (
                                'power-2' in output_vars))
                        if 'var' in output_vars:
                            which = np.argmax(np.array(output_vars) == 'var')
                            var_prediction = cond_moments[:, a, j, which,
                                             step_idx].detach().cpu().numpy()
                        elif 'power-2' in output_vars:
                            which = np.argmax(
                                np.array(output_vars) == 'power-2')
                            exp_2_prediction = cond_moments[:, a, j, which,
                                               step_idx].detach().cpu().numpy()
                            var_prediction = exp_2_prediction - exp_prediction ** 2

                        if np.any(var_prediction < 0):
                            # print('WARNING: some true cond. variances below 0 -> clip')
                            var_prediction = np.maximum(0, var_prediction)
                        std_prediction = np.sqrt(var_prediction)
                        axs[j].fill_between(path_t_true_X,
                                               exp_prediction - std_factor * std_prediction,
                                               exp_prediction + std_factor * std_prediction,
                                               label="standart deviation ({} factor) prediction {} steps ahead".format(
                                                   std_factor, s),
                                               color=std_color[h + 2])

        # plt.xlabel('$t$')
        # plt.suptitle("Anomaly detection on {} anomalies".format(anomaly_type))
        save = os.path.join(save_path, filename.format(a))
        plt.savefig(save, **save_extras, dpi=400)
        plt.close()

    ## ----- OLD VERSION ------
    # for a in paths_to_plot:
    #     fig, axs = plt.subplots(dimension,2,figsize=(20, 7))
    #     if dimension == 1:
    #         axs = [axs]
    #
    #     for j in range(dimension):
    #         # get the true_X at observed dates
    #         path_t_obs = []
    #         path_X_obs = []
    #         for k, od in enumerate(observed_dates[a]):
    #             if od == 1:
    #                 if true_M is None or (true_M is not None and
    #                                     true_M[a, j, k]==1):
    #                     path_t_obs.append(path_t_true_X[k])
    #                     path_X_obs.append(true_X[a, j, k])
    #         path_t_obs = np.array(path_t_obs)
    #         path_X_obs = np.array(path_X_obs)
    #
    #         axs[j][0].set_title("Ground_truth, dimension {}".format(j+1))
    #         axs[j][0].plot(path_t_true_X, mask_data_path[a, j, :], label='true path, no anomaly',
    #                     color=colors[0])
    #         axs[j][0].plot(path_t_true_X, mask_data_path_w_anomaly[a, j, :], label='true path, anomaly',
    #                     color=colors[1])
    #         axs[j][1].set_title("Anomaly Detection, dimension {}".format(j+1))
    #         axs[j][1].plot(path_t_true_X, mask_pred_path[a, j, :], label='true path, no predicted anomaly',
    #                     color=colors[0])
    #         axs[j][1].plot(path_t_true_X, mask_pred_path_w_anomaly[a, j, :], label='true path, predicted anomaly',
    #                     color=colors[1])
    #         axs[j][1].plot(path_t_true_X, mask_pred_scores[a, j, :], label='Predicted scores',
    #                     color=colors[2])
    #
    #         if plot_forecast_predictions:
    #             for h,s in enumerate(forecast_horizons_to_plot):
    #                 step_idx = steps_ahead.index(s)
    #                 exp_prediction = cond_moments[:,a,j,0,step_idx].detach().cpu().numpy()
    #                 axs[j][1].plot(path_t_true_X, exp_prediction, label=model_name + " expectation prediction {} steps ahead".format(s), color=colors[h+2])
    #                 if plot_variance:
    #                     assert(('var' in output_vars) or ('power-2' in output_vars))
    #                     if 'var' in output_vars:
    #                         which = np.argmax(np.array(output_vars) == 'var')
    #                         var_prediction = cond_moments[:,a,j,which,step_idx].detach().cpu().numpy()
    #                     elif 'power-2' in output_vars:
    #                         which = np.argmax(np.array(output_vars) == 'power-2')
    #                         exp_2_prediction = cond_moments[:,a,j,which,step_idx].detach().cpu().numpy()
    #                         var_prediction = exp_2_prediction - exp_prediction ** 2
    #
    #                     if np.any(var_prediction < 0):
    #                         # print('WARNING: some true cond. variances below 0 -> clip')
    #                         var_prediction = np.maximum(0, var_prediction)
    #                     std_prediction = np.sqrt(var_prediction)
    #                     axs[j][1].fill_between(path_t_true_X,
    #                         exp_prediction - std_factor * std_prediction,
    #                         exp_prediction + std_factor * std_prediction,
    #                         label="standart deviation ({} factor) prediction {} steps ahead".format(std_factor, s), color=std_color[h+2])
    #
    #         axs[j][0].legend()
    #         axs[j][1].legend()
    #         axs[j][0].set_ylim(0., 1.)
    #         axs[j][1].set_ylim(0., 1.)
    #         axs[j][1].axhline(y=ad_module.threshold, color='r', linestyle='-')
    #
    #     plt.xlabel('$t$')
    #     plt.suptitle("Anomaly detection on {} anomalies".format(anomaly_type))
    #     save = os.path.join(save_path, filename.format(a))
    #     plt.savefig(save, **save_extras)
    #     plt.close()
    


def main(arg):

    del arg
    forecast_params_list = None
    forecast_model_ids = None
    if FLAGS.forecast_model_ids:
        try:
            forecast_model_ids = eval("config."+FLAGS.forecast_model_ids)
        except Exception:
            forecast_model_ids = eval(FLAGS.forecast_model_ids)
    if FLAGS.ad_params:
        ad_params = eval("config."+FLAGS.ad_params)
    #get_training_overview_dict = None
    #if FLAGS.get_overview:
    #    get_training_overview_dict = eval("config."+FLAGS.get_overview)

    if forecast_model_ids is not None:
        forecast_params = forecast_params_list

        forecast_saved_models_path = FLAGS.forecast_saved_models_path
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
                        from configs import config
                        data_dict = eval("config."+data_dict)
                    forecast_param["dataset"] = data_dict["model_name"]
            # evaluate(anomaly_data_dict=anomaly_data_dict, **forecast_param)
            for ad_param in ad_params:
                if "which_AD_model_load" not in ad_param:
                    ad_param["which_AD_model_load"] = FLAGS.which_AD_model_load
                evaluate(
                    forecast_param=forecast_param,
                    forecast_model_id=forecast_model_ids[i],
                    forecast_saved_models_path=forecast_saved_models_path,
                    n_dataset_workers=FLAGS.N_DATASET_WORKERS,
                    use_gpu=FLAGS.USE_GPU, nb_cpus=FLAGS.NB_CPUS,
                    plot_only=FLAGS.plot_only,
                    quant_eval_in_plot_only=FLAGS.quant_eval_in_plot_only,
                    **ad_param)


if __name__ == '__main__':
    app.run(main)