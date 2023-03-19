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

from configs import config
import models
import data_utils
from AD_modules import AD_module


# =====================================================================================================================
# FLAGS
FLAGS = flags.FLAGS

flags.DEFINE_string("params", None, "name of the params list (in config.py) to "
                                    "use for parallel run")
flags.DEFINE_string("model_ids", None,
                    "name of list of model ids (in config.py) to run or list "
                    "of model ids")
#flags.DEFINE_string("get_overview", None,
#                    "name of the dict (in config.py) defining input for "
#                    "extras.get_training_overview")
flags.DEFINE_string("saved_models_path", config.saved_models_path,
                    "path where the models are saved")
flags.DEFINE_string("eval_data_dict", config.saved_models_path,
                    "path where the models are saved")

flags.DEFINE_bool("USE_GPU", False, "whether to use GPU for training")



# ==============================================================================
# Global variables

data_path = config.data_path
saved_models_path = config.saved_models_path
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

def evaluate(
        use_gpu=None,
        nb_cpus=None,
        model_id=None, 
        batch_size=100,
        seed=398,
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
        dataset='AD_TSAGen', 
        dataset_id=None, 
        data_dict=None, # data_dict from the data we aim to evaluate
        eval_data_dict=None, # data_dict from the data the model was trained on
        plot=True, paths_to_plot=(0,),
        saved_models_path=saved_models_path,
        steps_ahead = 10,
        **options
):
    
    use_cond_exp = False
    if 'use_cond_exp' in options:
        use_cond_exp = options['use_cond_exp']

    masked = False
    if 'masked' in options and 'other_model' not in options:
        masked = options['masked']

    initial_print = "model-id: {}\n".format(model_id)

    if USE_GPU and torch.cuda.is_available():
        gpu_num = 0
        device = torch.device("cuda:{}".format(gpu_num))
        torch.cuda.set_device(gpu_num)
        initial_print += '\nusing GPU'
    else:
        device = torch.device("cpu")
        initial_print += '\nusing CPU'

    # load dataset-metadata
    dataset, dataset_id = data_utils._get_dataset_name_id_from_dict(
        data_dict=data_dict)
    dataset_id = int(dataset_id)
    dataset_metadata = data_utils.load_metadata(stock_model_name=dataset,
                                                time_id=dataset_id)
    
    eval_dataset, eval_dataset_id = data_utils._get_dataset_name_id_from_dict(
        data_dict=eval_data_dict)
    eval_dataset_id = int(eval_dataset_id)
    eval_dataset_metadata = data_utils.load_metadata(stock_model_name=eval_dataset,
                                                time_id=eval_dataset_id)
    
    input_size = dataset_metadata['dimension']
    dimension = dataset_metadata['dimension']
    output_size = input_size
    T = dataset_metadata['maturity']
    delta_t = dataset_metadata['dt']  # copy metadata

    # load raw data
    eval_idx = np.arange(eval_dataset_metadata["nb_paths"])
    data_eval = IrregularADDataset(
        model_name=eval_dataset, time_id=eval_dataset_id, idx=eval_idx)
        
    # get data-loader for training
    if 'func_appl_X' in options:  # list of functions to apply to the paths in X
        functions = options['func_appl_X']
        collate_fn, mult = data_utils.CustomCollateFnGen(functions)
        input_size = input_size * mult
        output_size = output_size * mult
    else:
        functions = None
        collate_fn, mult = data_utils.CustomCollateFnGen(None)
        mult = 1

    dl = DataLoader(  # class to iterate over training data
        dataset=data_eval, collate_fn=collate_fn,
        shuffle=True, batch_size=batch_size)
    
    # get additional plotting information
    plot_variance = True
    std_factor = 1  # factor with which the std is multiplied
    if functions is not None and mult > 1:
        if 'plot_variance' in options:
            plot_variance = options['plot_variance']
        if 'std_factor' in options:
            std_factor = options['std_factor']
    ylabels = None
    if 'ylabels' in options:
        ylabels = options['ylabels']
    plot_same_yaxis = False
    if 'plot_same_yaxis' in options:
        plot_same_yaxis = options['plot_same_yaxis']
    plot_obs_prob = False
    if 'plot_obs_prob' in options:
        plot_obs_prob = options["plot_obs_prob"]
    
    print(eval_dataset_metadata)
    stockmodel = data_utils._STOCK_MODELS[
        dataset_metadata['model_name']](**dataset_metadata)
    
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
        'optimal_eval_loss': opt_eval_loss, 'options': options}
    desc = json.dumps(params_dict, sort_keys=True)

    # get all needed paths
    model_path = '{}id-{}/'.format(saved_models_path, model_id)
    model_path_save_best = '{}best_checkpoint/'.format(model_path)
    model_metric_file = '{}metric_id-{}.csv'.format(model_path, model_id)
    plot_save_path = '{}plots/'.format(model_path)
    if 'save_extras' in options:
        save_extras = options['save_extras']
    else:
        save_extras = {}

    eval_model_path = '../data/evaluation/' + data_dict
    if not os.path.exists(eval_model_path):
        os.mkdir(eval_model_path)
    anom_type = eval_dataset_metadata['anomaly_params']['type']
    eval_path = eval_model_path + '/' + anom_type
    if not os.path.exists(eval_path):
        os.mkdir(eval_path)
    eval_plot_path = eval_path + '/plots'
    if not os.path.exists(eval_plot_path):
        os.mkdir(eval_plot_path)

    if 'other_model' not in options:  # take NJODE model if not specified otherwise
        model = models.NJODE(**params_dict)  # get NJODE model class from
        model_name = 'NJODE'

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
    models.get_ckpt_model(model_path_save_best, model, optimizer, device)

    model.eval()

    prop_cycle = plt.rcParams['axes.prop_cycle']  # change style of plot?
    colors = prop_cycle.by_key()['color']
    std_color = list(matplotlib.colors.to_rgb(colors[1])) + [0.5]

    """
    times : observation times (for any batch element), [nb_obs], value in [0,nb_steps]
    time_ptr : cumulated number of observations until each time (counting across batch), [nb_obs+1] (because starts at 0)
    obs_idx : list of observed batch idx across time [time_ptr[-1]], value in [0,batch]
    """

    ad_module = AD_module(steps_ahead=steps_ahead, smoothing=5)
    optimizer = torch.optim.Adam(ad_module.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    ad_module.train()

    plot_AD_module_params(ad_module, eval_path)

    for e in range(1): # change for learning parameters of AD module
    # with torch.no_grad():  # no gradient needed for evaluation
        for i, b in enumerate(dl):  # iterate over dataloader for validation set
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

            mask_data_path = np.ma.masked_where((ad_labels == 1), true_X)
            mask_data_path_w_anomaly = np.ma.masked_where((ad_labels == 0), true_X)

            hT, c_loss, path_t, path_y = model.custom_forward(
            # hT, c_loss, path_t, path_h, path_y = model( 
                times, time_ptr, X, obs_idx, delta_t, T, start_X,
                n_obs_ot, steps=steps_ahead, get_loss=True, M=M,
                start_M=start_M, which_loss='standard')
                
            nb_steps = true_X.shape[2]
            nb_samples = true_X.shape[0]
            
            nb_moments = 1
            for m in range(2,10): 
                if 'power-{}'.format(str(m)) in functions:
                    nb_moments += 1
            cond_moments = np.empty((nb_steps, nb_samples, dimension, nb_moments, steps_ahead))
            cond_moments[:,:,:,:,:] = np.nan
            for i in range(steps_ahead):
                index_times = np.empty_like(path_t_true_X)
                for j,t in enumerate(path_t_true_X):
                    if np.any(np.abs(t*np.ones_like(path_t[i]) - path_t[i]) < 1e-10):
                        index_times[j] = True
                    else:
                        index_times[j] = False
                index_times = np.argwhere(index_times).reshape(-1)
                for m in range(nb_moments):
                    cond_moments[index_times,:,:,m,i] = path_y[i].detach().numpy()[:,:,m*dimension:(m+1)*dimension]

            obs = torch.tensor(true_X, dtype=torch.float32).permute(2,0,1)
            cond_moments = torch.tensor(cond_moments, dtype=torch.float32)
            scores, mask = ad_module(obs, cond_moments)

            ad_labels = torch.tensor(ad_labels).permute(2,0,1)
            scores = scores[mask]
            ad_labels = ad_labels[mask]

            loss = criterion(scores, ad_labels)
            loss.backward()
            optimizer.step()

            for name, param in ad_module.named_parameters():
                if param.requires_grad:
                    print( name, param.data )
 

            '''
            # path_to_plot
            # steps_ahead_to_plot
            for a in range(5):
                fig, axs = plt.subplots(dimension)
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

                    axs[j].plot(path_t_true_X, mask_data_path[a, j, :], label='true path, no anomaly',
                                color='b')
                    axs[j].plot(path_t_true_X, mask_data_path_w_anomaly[a, j, :], label='true path, anomaly',
                                color='r')
                    # axs[j].plot(path_t_true_X, true_X[a, j, :], label='true path',
                                # color=colors[0])
                    # for s in range(steps_ahead):
                        # axs[j].plot(times, cond_moments[:,a,j,0,s], label=model_name, color=colors[1])
                    axs[j].plot(times, cond_moments[:,a,j,0,-1], label=model_name, color=colors[1])

                plot_filename = 'path-{}.pdf'
                axs[-1].legend()
                plt.xlabel('$t$')
                save = os.path.join(eval_plot_path, plot_filename.format(a))
                plt.savefig(save, **save_extras)
                plt.close()'''

            break

    return 0

def plot_AD_module_params(ad_module, path, steps_ahead = None, dt=0.0025):
    steps_weights = ad_module.state_dict()['steps_weighting.weight'].squeeze().clone().detach().numpy()
    if steps_ahead is None:
        steps_ahead = np.arange(len(steps_weights))
        steps_ahead = [str(dt * s) for s in steps_ahead]
    if ad_module.smoothing > 0:
        smoothing_weights = ad_module.state_dict()['smoothing_weights.weight'].squeeze().clone().detach().numpy()

    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(steps_ahead, steps_weights, width = 0.1)
    plt.title('Importance of different forecasting horizons for Anomaly Detection', fontsize=15)
    plt.xlabel('Forecasting Horizons', fontsize=10)
    plt.ylabel('Horizon Weight', fontsize=10)
    ax.grid(False)
    ax.tick_params(bottom=False, left=True)
    plt.savefig(path + '/forecasting_horizon_weighting.png')

    neighbors = [str(i) for i in range(-ad_module.smoothing, ad_module.smoothing+1)]
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(neighbors, smoothing_weights, width = 0.1)
    plt.title('Importance of neighbouring anomaly scores', fontsize=15)
    plt.xlabel('Time neighbours', fontsize=10)
    plt.ylabel('Neighbour Weights', fontsize=10)
    ax.grid(False)
    ax.tick_params(bottom=False, left=True)
    plt.savefig(path + '/smoothing_weighting.png')


def plot_one_path_with_pred(
        device, model, batch, stockmodel, delta_t, T,
        path_to_plot=(0,), save_path='', filename='plot_{}.pdf',
        plot_variance=False, functions=None, std_factor=1,
        model_name=None, ylabels=None,
        save_extras={'bbox_inches': 'tight', 'pad_inches': 0.01},
        use_cond_exp=True, same_yaxis=False,
        plot_obs_prob=False, dataset_metadata=None,
):
    """
    plot one path of the stockmodel together with optimal cond. exp. and its
    prediction by the model
    :param device: torch.device, to use for computations
    :param model: models.NJODE instance
    :param batch: the batch from where to take the paths
    :param stockmodel: stock_model.StockModel instance, used to compute true
            cond. exp.
    :param delta_t: float
    :param T: float
    :param path_to_plot: list of ints, which paths to plot (i.e. which elements
            oof the batch)
    :param save_path: str, the path where to save the plot
    :param filename: str, the filename for the plot, should have one insertion
            possibility to put the path number
    :param plot_variance: bool, whether to plot the variance, if supported by
            functions (i.e. square has to be applied)
    :param functions: list of functions (as str), the functions applied to X
    :param std_factor: float, the factor by which std is multiplied
    :param model_name: str or None, name used for model in plots
    :param ylabels: None or list of str of same length as dimension of X
    :param save_extras: dict with extra options for saving plot
    :param use_cond_exp: bool, whether to plot the conditional expectation
    :param same_yaxis: bool, whether to plot all coordinates with same range on
        y-axis
    :param plot_obs_prob: bool, whether to plot the probability of an
        observation for all times
    :param dataset_metadata: needed if plot_obs_prob=true, the metadata of the
        used dataset to extract the observation probability
    :return: optimal loss
    """
    if model_name is None or model_name == "NJODE":
        model_name = 'our model'

    prop_cycle = plt.rcParams['axes.prop_cycle']  # change style of plot?
    colors = prop_cycle.by_key()['color']
    std_color = list(matplotlib.colors.to_rgb(colors[1])) + [0.5]

    makedirs(save_path)  # create a directory

    times = batch["times"]
    time_ptr = batch["time_ptr"]
    X = batch["X"].to(device)
    M = batch["M"]
    if M is not None:
        M = M.to(device)
    start_X = batch["start_X"].to(device)
    start_M = batch["start_M"]
    if start_M is not None:
        start_M = start_M.to(device)
    obs_idx = batch["obs_idx"]
    n_obs_ot = batch["n_obs_ot"].to(device)
    true_X = batch["true_paths"]
    bs, dim, time_steps = true_X.shape
    true_M = batch["true_mask"]
    observed_dates = batch['observed_dates']
    path_t_true_X = np.linspace(0., T, int(np.round(T / delta_t)) + 1)

    model.eval()  # put model in evaluation mode
    res = model.get_pred(
        times=times, time_ptr=time_ptr, X=X, obs_idx=obs_idx, delta_t=delta_t,
        T=T, start_X=start_X, M=M, start_M=start_M)
    path_y_pred = res['pred'].detach().numpy()
    path_t_pred = res['pred_t']

    # get variance path
    if plot_variance and (functions is not None) and ('power-2' in functions):
        which = np.argmax(np.array(functions) == 'power-2')+1
        y2 = path_y_pred[:, :, (dim * which):(dim * (which + 1))]
        path_var_pred = y2 - np.power(path_y_pred[:, :, 0:dim], 2)
        if np.any(path_var_pred < 0):
            print('WARNING: some predicted cond. variances below 0 -> clip')
            path_var_pred = np.maximum(0, path_var_pred)
        path_std_pred = np.sqrt(path_var_pred)
    else:
        plot_variance = False
    if use_cond_exp:
        if M is not None:
            M = M.detach().numpy()
        opt_loss, path_t_true, path_y_true = stockmodel.compute_cond_exp(
            times, time_ptr, X.detach().numpy(), obs_idx.detach().numpy(),
            delta_t, T, start_X.detach().numpy(), n_obs_ot.detach().numpy(),
            return_path=True, get_loss=True, weight=model.weight,
            M=M,)
    else:
        opt_loss = 0

    for i in path_to_plot:
        fig, axs = plt.subplots(dim)
        if dim == 1:
            axs = [axs]
        for j in range(dim):
            # get the true_X at observed dates
            path_t_obs = []
            path_X_obs = []
            for k, od in enumerate(observed_dates[i]):
                if od == 1:
                    if true_M is None or (true_M is not None and
                                          true_M[i, j, k]==1):
                        path_t_obs.append(path_t_true_X[k])
                        path_X_obs.append(true_X[i, j, k])
            path_t_obs = np.array(path_t_obs)
            path_X_obs = np.array(path_X_obs)

            axs[j].plot(path_t_true_X, true_X[i, j, :], label='true path',
                        color=colors[0])
            axs[j].scatter(path_t_obs, path_X_obs, label='observed',
                           color=colors[0])
            axs[j].plot(path_t_pred, path_y_pred[:, i, j],
                        label=model_name, color=colors[1])
            if plot_variance:
                # axs[j].plot(
                #     path_t_pred,
                #     path_y_pred[:, i, j] + std_factor*path_std_pred[:, i, j],
                #     color=colors[1])
                # axs[j].plot(
                #     path_t_pred,
                #     path_y_pred[:, i, j] - std_factor*path_std_pred[:, i, j],
                #     color=colors[1])
                axs[j].fill_between(
                    path_t_pred,
                    path_y_pred[:, i, j] - std_factor * path_std_pred[:, i, j],
                    path_y_pred[:, i, j] + std_factor * path_std_pred[:, i, j],
                    color=std_color)
            if use_cond_exp:
                axs[j].plot(path_t_true, path_y_true[:, i, j],
                            label='true conditional expectation',
                            linestyle=':', color=colors[2])
            if plot_obs_prob and dataset_metadata is not None:
                ax2 = axs[j].twinx()
                if "X_dependent_observation_prob" in dataset_metadata:
                    prob_f = eval(
                        dataset_metadata["X_dependent_observation_prob"])
                    obs_perc = prob_f(true_X[:, :, :])[i]
                else:
                    obs_perc = dataset_metadata['obs_perc']
                    obs_perc = np.ones_like(path_t_true_X) * obs_perc
                ax2.plot(path_t_true_X, obs_perc, color="red",
                         label="observation probability")
                ax2.set_ylim(-0.1, 1.1)
                ax2.set_ylabel("observation probability")
                axs[j].set_ylabel("X")
                ax2.legend()
            if ylabels:
                axs[j].set_ylabel(ylabels[j])
            if same_yaxis:
                low = np.min(true_X[i, :, :])
                high = np.max(true_X[i, :, :])
                eps = (high - low)*0.05
                axs[j].set_ylim([low-eps, high+eps])


        axs[-1].legend()
        plt.xlabel('$t$')
        save = os.path.join(save_path, filename.format(i))
        plt.savefig(save, **save_extras)
        plt.close()

    return opt_loss


def main(arg):
    """
    function to run parallel training with flags from command line
    """
    del arg
    params_list = None
    model_ids = None
    if FLAGS.params:
        params_list = eval("config."+FLAGS.params)
    elif FLAGS.model_ids:
        try:
            model_ids = eval("config."+FLAGS.model_ids)
        except Exception:
            model_ids = eval(FLAGS.model_ids)
    if FLAGS.eval_data_dict:
        eval_data_dict = eval("config."+FLAGS.eval_data_dict)
    #get_training_overview_dict = None
    #if FLAGS.get_overview:
    #    get_training_overview_dict = eval("config."+FLAGS.get_overview)

    if params_list is not None or model_ids is not None:
        params = params_list

        saved_models_path = FLAGS.saved_models_path
        if params is not None and 'saved_models_path' in params[0]:
            saved_models_path = params[0]['saved_models_path']
        model_overview_file_name = '{}model_overview.csv'.format(
            saved_models_path)
        
        df_overview = pd.read_csv(model_overview_file_name, index_col=0)
        max_id = np.max(df_overview['id'].values)

        params = []
        for model_id in model_ids:
            if model_id not in df_overview['id'].values:
                print("model_id={} does not exist yet -> skip".format(model_id))
            else:
                desc = (df_overview['description'].loc[
                    df_overview['id'] == model_id]).values[0]
                params_dict = json.loads(desc)
                params_dict['model_id'] = model_id
                params.append(params_dict)

        for param in params:
            if 'dataset' not in param:
                if 'data_dict' not in param:
                    raise KeyError('the "dataset" needs to be specified')
                else:
                    data_dict = param["data_dict"]
                    if isinstance(data_dict, str):
                        from configs import config
                        data_dict = eval("config."+data_dict)
                    param["dataset"] = data_dict["model_name"]
            evaluate(eval_data_dict=eval_data_dict, **param)


if __name__ == '__main__':
    app.run(main)