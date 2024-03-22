"""
author: Florian Krach & Calypso Herrera & Markus Chardonnet

implementation of the training (and evaluation) of NJ-ODE
"""

# =====================================================================================================================
from typing import List

import torch  # machine learning
import torch.nn as nn
import tqdm  # process bar for iterations
import numpy as np  # large arrays and matrices, functions
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os, sys
import pandas as pd  # data analysis and manipulation
import json  # storing and exchanging data
import time
import socket
import matplotlib  # plots
import matplotlib.colors
from torch.backends import cudnn
import copy
import gc
from torch.optim.lr_scheduler import CyclicLR
import glob

from configs import config
import models
import data_utils
sys.path.append("../")

try:
    from telegram_notifications import send_bot_message as SBM
except Exception:
    from configs.config import SendBotMessage as SBM


# =====================================================================================================================
# check whether running on computer or server
if 'ada-' not in socket.gethostname():
    SERVER = False
    N_CPUS = 1
    SEND = False
else:
    SERVER = True
    N_CPUS = 1
    SEND = True
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
print(socket.gethostname())
print('SERVER={}'.format(SERVER))


# ==============================================================================
# Global variables
CHAT_ID = config.CHAT_ID
ERROR_CHAT_ID = config.ERROR_CHAT_ID

data_path = config.data_path
train_data_path = config.training_data_path
saved_models_path = config.saved_models_path
flagfile = config.flagfile

METR_COLUMNS: List[str] = [
    'epoch', 'train_time', 'eval_time', 'train_loss', 'eval_loss', 'val_loss']
default_ode_nn = ((50, 'tanh'), (50, 'tanh'))
default_readout_nn = ((50, 'tanh'), (50, 'tanh'))
default_enc_nn = ((50, 'tanh'), (50, 'tanh'))

ANOMALY_DETECTION = False
N_DATASET_WORKERS = 0
USE_GPU = False

# =====================================================================================================================
# Functions
makedirs = config.makedirs


def train(
        anomaly_detection=None, n_dataset_workers=None, use_gpu=None,
        nb_cpus=None, send=None, gpu_num=0, use_wandb=False,
        model_id=None, epochs=100, batch_size=100, save_every=1,
        learning_rate=0.001, test_size=0.2, seed=398,
        hidden_size=10, bias=True, dropout_rate=0.1,
        ode_nn=default_ode_nn, readout_nn=default_readout_nn,
        enc_nn=default_enc_nn, use_rnn=False,
        solver="euler", weight=0.5, # weight_decay=1.,
        # dataset_id=None, data_dict=None,
        dataset='microbial_genus', dataset_split="all",
        plot=True, paths_to_plot=(0,),
        saved_models_path=saved_models_path,
        **options
):


    """
    training function for NJODE model (models.NJODE),
    the model is automatically saved in the model-save-path with the given
    model id, also all evaluations of the model are saved there

    :param anomaly_detection: used to pass on FLAG from parallel_train
    :param n_dataset_workers: used to pass on FLAG from parallel_train
    :param use_gpu: used to pass on FLAG from parallel_train
    :param nb_cpus: used to pass on FLAG from parallel_train
    :param send: used to pass on FLAG from parallel_train
    :param model_id: None or int, the id to save (or load if it already exists)
            the model, if None: next biggest unused id will be used
    :param epochs: int, number of epochs to train, each epoch is one cycle
            through all (random) batches of the training data
    :param batch_size: int
    :param save_every: int, defined number of epochs after each of which the
            model is saved and plotted if wanted. whenever the model has a new
            best eval-loss it is also saved, independent of this number (but not
            plotted)
    :param learning_rate: float
    :param test_size: float in (0,1), the percentage of samples to use for the
            test set (here there exists only a test set, since there is no extra
            evaluation)
    :param seed: int, seed for the random splitting of the dataset into train
            and test
    :param hidden_size: see models.NJODE
    :param bias: see models.NJODE
    :param dropout_rate: float
    :param ode_nn: see models.NJODE
    :param readout_nn: see models.NJODE
    :param enc_nn: see models.NJODE
    :param use_rnn: see models.NJODE
    :param solver: see models.NJODE
    :param weight: see models.NJODE
    :param weight_decay: see models.NJODE
    :param dataset: str, which dataset to use, supported: {'BlackScholes',
            'Heston', 'OrnsteinUhlenbeck'}. The corresponding dataset already
            needs to exist (create it first using data_utils.create_dataset)
    :param dataset_id: int or None, the id of the dataset to be used, if None,
            the latest generated dataset of the given name will be used
    :param data_dict: None, str or dict, if not None, the inputs dataset and
            dataset_id are overwritten. if str, the dict from config.py with the
            given name is loaded.
            from the dataset_overview.csv file, the
            dataset with a matching description is loaded.
    :param plot: bool, whether to plot
    :param paths_to_plot: list of ints, which paths of the test-set should be
            plotted
    :param saved_models_path: str, where to save the models
    :param options: kwargs, used keywords:
            'test_data_dict'    None, str or dict, if no None, this data_dict is
                            used to define the dataset for testing and
                            evaluation
            'func_appl_X'   list of functions (as str, see data_utils)
                            to apply to X
            'masked'        bool, whether the data is masked (i.e. has
                            incomplete observations)
            'save_extras'   bool, dict of options for saving the plots
            'plot_variance' bool, whether to plot also variance
            'std_factor'    float, the factor by which the std is multiplied
            'parallel'      bool, used by parallel_train.parallel_training
            'resume_training'   bool, used by parallel_train.parallel_training
            'plot_only'     bool, whether the model is used only to plot after
                            initiating or loading (i.e. no training) and exit
                            afterwards (used by demo)
            'ylabels'       list of str, see plot_one_path_with_pred()
            'plot_same_yaxis'   bool, whether to plot the same range on y axis
                            for all dimensions
            'plot_obs_prob' bool, whether to plot the observation probability
            'which_loss'    'standard' or 'easy', used by models.NJODE
            'residual_enc_dec'  bool, whether resNNs are used for encoder and
                            readout NN, used by models.NJODE, default True
            'use_y_for_ode' bool, whether to use y (after jump) or x_impute for
                            the ODE as input, only in masked case, default: True
            'coord_wise_tau'    bool, whether to use a coordinate wise tau
            'input_sig'     bool, whether to use the signature as input
            'level'         int, level of the signature that is used
            'input_current_t'   bool, whether to additionally input current time
                            to the ODE function f, default: False
            'enc_input_t'   bool, whether to use the time as input for the
                            encoder network. default: False
            'train_readout_only'    bool, whether to only train the readout
                            network
            'training_size' int, if given and smaller than
                            dataset_size*(1-test_size), then this is the umber
                            of samples used for the training set (randomly
                            selected out of original training set)
            'evaluate'      bool, whether to evaluate the model in the test set
                            (i.e. not only compute the eval_loss, but also
                            compute the mean difference between the true and the
                            predicted paths comparing at each time point)
            'load_best'     bool, whether to load the best checkpoint instead of
                            the last checkpoint when loading the model. Mainly
                            used for evaluating model at the best checkpoint.
            'gradient_clip' float, if provided, then gradient values are clipped
                            by the given value
            'clamp'         float, if provided, then output of model is clamped
                            to +/- the given value
            'other_model'   one of {'GRU_ODE_Bayes', randomizedNJODE};
                            the specifieed model is trained instead of the
                            controlled ODE-RNN model.
                            Other options/inputs might change or loose their
                            effect.
            'ode_input_scaling_func'    None or str in {'id', 'tanh'}, the
                            function used to scale inputs to the neuralODE.
                            default: tanh
                -> 'GRU_ODE_Bayes' has the following extra options with the
                    names 'GRU_ODE_Bayes'+<option_name>, for the following list
                    of possible choices for <options_name>:
                    '-mixing'   float, default: 0.0001, weight of the 2nd loss
                                term of GRU-ODE-Bayes
                    '-solver'   one of {"euler", "midpoint", "dopri5"}, default:
                                "euler"
                    '-impute'   bool, default: False,
                                whether to impute the last parameter
                                estimation of the p_model for the next ode_step
                                as input. the p_model maps (like the
                                readout_map) the hidden state to the
                                parameter estimation of the normal distribution.
                    '-logvar'   bool, default: True, wether to use logarithmic
                                (co)variace -> hardcodinng positivity constraint
                    '-full_gru_ode'     bool, default: True,
                                        whether to use the full GRU cell
                                        or a smaller version, see GRU-ODE-Bayes
                    '-p_hidden'         int, default: hidden_size, size of the
                                        inner hidden layer of the p_model
                    '-prep_hidden'      int, default: hidden_size, in the
                                        observational cell (i.e. jumps) a prior
                                        matrix multiplication transforms the
                                        input to have the size
                                        prep_hidden * input_size
                    '-cov_hidden'       int, default: hidden_size, size of the
                                        inner hidden layer of the covariate_map.
                                        the covariate_map is used as a mapping
                                        to get the initial h (for controlled
                                        ODE-RNN this is done by the encoder)
            'add_pred'      
            'train_data_perc'
            'fixed_data_perc'
            'scale_dt'
            'weight_evolve'
            'solver_delta_t_factor'
            'articial_train_encdec 
    """

    global ANOMALY_DETECTION, USE_GPU, SEND, N_CPUS, N_DATASET_WORKERS
    if anomaly_detection is not None:
        ANOMALY_DETECTION = anomaly_detection
    if use_gpu is not None:
        USE_GPU = use_gpu
    if send is not None:
        SEND = send
    if nb_cpus is not None:
        N_CPUS = nb_cpus
    if n_dataset_workers is not None:
        N_DATASET_WORKERS = n_dataset_workers

    if use_wandb:
        import wandb
        wandb.login()

    initial_print = "model-id: {}\n".format(model_id)

    masked = False
    if 'masked' in options and 'other_model' not in options:
        masked = options['masked']

    if ANOMALY_DETECTION:
        # allow backward pass to print the traceback of the forward operation
        #   if it fails, "nan" backward computation produces error
        torch.autograd.set_detect_anomaly(True)
        torch.manual_seed(0)
        np.random.seed(0)
        # set seed and deterministic to make reproducible
        cudnn.deterministic = True

    # set number of CPUs
    torch.set_num_threads(N_CPUS)

    # get the device for torch
    if USE_GPU and torch.cuda.is_available():
        # gpu_num = 1
        device = torch.device("cuda:{}".format(gpu_num))
        torch.cuda.set_device(gpu_num)
        initial_print += '\nusing GPU'
    else:
        device = torch.device("cpu")
        initial_print += '\nusing CPU'

    train_idx = np.load(os.path.join(
        train_data_path, dataset, dataset_split, 'train_idx.npy'
        ), allow_pickle=True)
    val_idx = np.load(os.path.join(
        train_data_path, dataset, dataset_split, 'val_idx.npy'
        ), allow_pickle=True)
    if 'test' in options and options['test'] == True:
        test_idx = np.load(os.path.join(
            train_data_path, dataset, dataset_split, 'test_idx.npy'
            ), allow_pickle=True)
        
    data_train = data_utils.MicrobialDataset(
        dataset_name=dataset, idx=train_idx)
    data_val = data_utils.MicrobialDataset(
        dataset_name=dataset, idx=val_idx)

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
            options['scale_dt'] = 1./delta_t
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
    
    # in case we want to evaluate the predictions, specifies the output variables
    eval_metrics = None
    if 'evaluate' in options and options['evaluate']:
        if 'eval_metrics' in options:
            eval_metrics = options['eval_metrics']
    if 'which_eval_loss' in options:
        which_eval_loss = options['which_eval_loss']
    else:
        which_eval_loss = 'standard'
    initial_print += "\neval loss: {}\n".format(which_eval_loss)

    # get data-loader for training
    dl = DataLoader(  # class to iterate over training data
        dataset=data_train, collate_fn=collate_fn,
        shuffle=True, batch_size=batch_size, num_workers=N_DATASET_WORKERS)
    dl_val = DataLoader(  # class to iterate over validation data
        dataset=data_val, collate_fn=collate_fn_val,
        shuffle=False, batch_size=batch_size, num_workers=N_DATASET_WORKERS)

    # get additional plotting information
    plot_variance = False
    std_factor = 1  # factor with which the std is multiplied
    if output_vars is not None and mult > 1:
        if 'plot_variance' in options:
            plot_variance = options['plot_variance']
        if 'std_factor' in options:
            std_factor = options['std_factor']
    plot_moments = False
    if output_vars is not None and mult > 1:
        if 'plot_moments' in options:
            plot_moments = options['plot_moments']
    ylabels = None
    if 'ylabels' in options:
        ylabels = options['ylabels']
    plot_same_yaxis = False
    if 'plot_same_yaxis' in options:
        plot_same_yaxis = options['plot_same_yaxis']
    plot_obs_prob = False
    if 'plot_obs_prob' in options:
        plot_obs_prob = options["plot_obs_prob"]
    plot_train = False
    if 'plot_train' in options:
        plot_train = True

    # get params_dict
    params_dict = {  # create a dictionary of the wanted parameters
        'input_size': input_size, 'epochs': epochs,
        'hidden_size': hidden_size, 'output_size': output_size, 'bias': bias,
        'ode_nn': ode_nn, 'readout_nn': readout_nn, 'enc_nn': enc_nn,
        'use_rnn': use_rnn, 'zero_weight_init': zero_weight_init,
        'dropout_rate': dropout_rate, 'batch_size': batch_size,
        'solver': solver, 'dataset': dataset, 'dataset_split': dataset_split,
        'learning_rate': learning_rate, 'test_size': test_size, 'seed': seed,
        'weight': weight, 'weight_evolve': weight_evolve,
        't_period': t_period, 'delta_t': model_delta_t,
        'output_vars': output_vars, 'input_vars': input_vars,
        'sigf_size': dimension_sig_feat,
        'options': options}
    desc = json.dumps(params_dict, sort_keys=True)  # serialize to a JSON formatted str

    # get overview file
    resume_training = False
    if ('parallel' in options and options['parallel'] is False) or \
            ('parallel' not in options):
        model_overview_file_name = '{}model_overview.csv'.format(
            saved_models_path
        )
        makedirs(saved_models_path)
        if not os.path.exists(model_overview_file_name):
            df_overview = pd.DataFrame(data=None, columns=['id', 'description'])
            max_id = 0
        else:
            df_overview = pd.read_csv(model_overview_file_name, index_col=0)  # read model overview csv file
            max_id = np.max(df_overview['id'].values)

        # get model_id, model params etc.
        if model_id is None:
            model_id = max_id + 1
        if model_id not in df_overview['id'].values:  # create new model ID
            initial_print += '\nnew model_id={}'.format(model_id)
            df_ov_app = pd.DataFrame([[model_id, desc]],
                                     columns=['id', 'description'])
            df_overview = pd.concat([df_overview, df_ov_app], ignore_index=True)
            df_overview.to_csv(model_overview_file_name)
        else:
            initial_print += '\nmodel_id already exists -> resume training'  # resume training if model already exists
            resume_training = True
            desc = (df_overview['description'].loc[
                df_overview['id'] == model_id]).values[0]
            params_dict = json.loads(desc)
    initial_print += '\nmodel params:\n{}'.format(desc)
    if 'resume_training' in options and options['resume_training'] is True:
        resume_training = True

    # get all needed paths
    model_path = '{}id-{}/'.format(saved_models_path, model_id)
    makedirs(model_path)
    model_path_save_last = '{}last_checkpoint/'.format(model_path)
    model_path_save_best = '{}best_checkpoint/'.format(model_path)
    makedirs(model_path_save_last)
    makedirs(model_path_save_best)
    model_metric_file = '{}metric_id-{}.csv'.format(model_path, model_id)
    plot_save_path = '{}plots/'.format(model_path)
    if 'save_extras' in options:
        save_extras = options['save_extras']
    else:
        save_extras = {}

    if use_wandb:
        wandb.init(
        # Set the project where this run will be logged
        project="pdnjode_forecasting", 
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"experiment_{model_id}", 
        # Track hyperparameters and run metadata
        config=params_dict)

    # get the model & optimizer
    if 'other_model' not in options:  # take NJODE model if not specified otherwise
        model = models.NJODE(**params_dict)  # get NJODE model class from
        model_name = 'NJODE'
    else:
        raise ValueError("Invalid argument for (option) parameter 'other_model'."
                         "Please check docstring for correct use.")
    model.to(device)  # pass model to CPU/GPU
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    gradient_clip = None
    if 'gradient_clip' in options:
        gradient_clip = options["gradient_clip"]

    # load saved model if wanted/possible
    best_val_loss = np.infty
    metr_columns = copy.deepcopy(METR_COLUMNS)
    val_loss_weights = 1.
    if which_eval_loss == 'val_variance':
        val_loss_names = ['val_loss_{}'.format(i+1) for i in range(4)]
        metr_columns += val_loss_names
        if 'val_loss_weights' in options:
            val_loss_weights = np.array(options['val_loss_weights'])
        else:
            val_loss_weights = np.ones(len(val_loss_names))
    else:
        val_loss_names = ['val_loss']
    if 'evaluate' in options and options['evaluate']:
        eval_metric_names = []
        for v in eval_metrics:
            eval_metric_names += [v + ' mean square error', v + ' std square error']
        metr_columns += eval_metric_names
    if resume_training:
        initial_print += '\nload saved model ...'
        try:
            if 'load_best' in options and options['load_best']:
                models.get_ckpt_model(model_path_save_best, model, optimizer,
                                      device)
            else:
                models.get_ckpt_model(model_path_save_last, model, optimizer,
                                      device)
            df_metric = pd.read_csv(model_metric_file, index_col=0)
            best_val_loss = np.min(df_metric['val_loss'].values)
            model.epoch += 1
            model.weight_step()
            initial_print += '\nepoch: {}, weight: {}'.format(
                model.epoch, model.weight)
        except Exception as e:
            initial_print += '\nloading model failed -> initiate new model'
            initial_print += '\nException:\n{}'.format(e)
            resume_training = False
    if not resume_training:
        initial_print += '\ninitiate new model ...'
        df_metric = pd.DataFrame(columns=metr_columns)

    # ---------- plot only option ------------
    if 'plot_only' in options and options['plot_only']:
        #for i, b in enumerate(dl_val):
        #    batch = b
        batch = next(iter(dl_val))
        model.epoch -= 1
        initial_print += '\nplotting ...'
        plot_filename = 'demo-plot_epoch-{}'.format(model.epoch)
        plot_filename = plot_filename + '_path-{}.pdf'
        plot_one_path_with_pred(
            device, model, batch, delta_t, T,
            path_to_plot=paths_to_plot, save_path=plot_save_path,
            filename=plot_filename, plot_variance=plot_variance,
            plot_moments=plot_moments, output_vars=output_vars,
            functions=input_vars, std_factor=std_factor,
            model_name=model_name, save_extras=save_extras, ylabels=ylabels,
            same_yaxis=plot_same_yaxis, plot_obs_prob=plot_obs_prob,
            dataset_metadata=dataset_metadata)
        if SEND:
            files_to_send = []
            caption = "{} - id={}".format(model_name, model_id)
            for i in paths_to_plot:
                files_to_send.append(sorted(glob.glob(
                    os.path.join(plot_save_path, plot_filename.format(
                        "{}*".format(i)))))[0])
            SBM.send_notification(
                text='finished plot-only: {}, id={}\n\n{}'.format(
                    model_name, model_id, desc),
                chat_id=config.CHAT_ID,
                files=files_to_send,
                text_for_files=caption
            )
        # initial_print += '\noptimal eval-loss (with current weight={:.5f}): ' \
        #                 '{:.5f}'.format(model.weight, curr_opt_loss)
        print(initial_print)
        return 0

    # ---------------- TRAINING ----------------
    skip_training = True
    if model.epoch <= epochs:  # check if it already trained the requested number of epochs
        skip_training = False

        # send notification
        if SEND:
            SBM.send_notification(
                text='start training - model id={}'.format(model_id),
                chat_id=config.CHAT_ID)
        initial_print += '\n\nmodel overview:'
        print(initial_print)
        print(model, '\n')

        # compute number of parameters
        nr_params = 0
        for name, param in model.named_parameters():
            skip = False
            for p_name in ['gru_debug', 'classification_model']:
                if p_name in name:
                    skip = True
            if not skip:
                nr_params += param.nelement()  # count number of parameters
        print('# parameters={}\n'.format(nr_params))

        # compute number of trainable params
        nr_trainable_params = 0
        for pg in optimizer.param_groups:
            for p in pg['params']:
                nr_trainable_params += p.nelement()
        print('# trainable parameters={}\n'.format(nr_trainable_params))
        print('start training ...')

    pre_train = False
    if 'pre-train' in options and isinstance(options['pre-train'], int):
        pre_train = True
        pre_train_eps = options['pre-train']
    if pre_train:
        model.train()
        for ep in range(pre_train_eps):
            optimizer.zero_grad()
            loss = model.forward_random_encoder_decoder(batch_size=batch_size, dimension=dimension)
            loss.backward()
            optimizer.step()

    t = time.time()  # return the time in seconds since the epoch
    metric_app = []
    while model.epoch <= epochs:
        model.train()  # set model in train mode (e.g. BatchNorm)
        for i, b in tqdm.tqdm(enumerate(dl)):  # iterate over the dataloader
            optimizer.zero_grad()  # reset the gradient
            times = b["times"]  # Produce instance of byte type instead of str type
            time_ptr = b["time_ptr"]  # pointer
            X = b["X"].to(device)
            Z = b["Z"].to(device)
            S = b["S"].to(device)

            start_X = b["start_X"].to(device)
            start_Z = b["start_Z"].to(device)
            start_S = b["start_S"].to(device)
            obs_idx = b["obs_idx"]
            n_obs_ot = b["n_obs_ot"].to(device)

            if 'other_model' not in options:
                if 'add_dynamic_cov' in options and options['add_dynamic_cov']:
                    hT, loss = model(
                        times=times, time_ptr=time_ptr, X=torch.cat((X,Z),dim=1), obs_idx=obs_idx,
                        delta_t=None, T=T, start_X=torch.cat((start_X,start_Z),dim=1), n_obs_ot=n_obs_ot,
                        S=S, start_S=start_S, return_path=False, get_loss=True)
                else:
                    hT, loss = model(
                        times=times, time_ptr=time_ptr, X=X, obs_idx=obs_idx,
                        delta_t=None, T=T, start_X=start_X, n_obs_ot=n_obs_ot,
                        S=S, start_S=start_S, return_path=False, get_loss=True)
            else:
                raise ValueError
            loss.backward()  # compute gradient of each weight regarding loss function
            if gradient_clip is not None:
                nn.utils.clip_grad_value_(
                    model.parameters(), clip_value=gradient_clip)
            optimizer.step()  # update weights by ADAM optimizer
            for param in model.parameters():
                param.grad = None
            if ANOMALY_DETECTION:
                print(r"current loss: {}".format(loss.detach().cpu().numpy()))
            del hT, times, time_ptr, X, Z, start_X, start_Z, obs_idx, n_obs_ot

        """if pre_train:
            optimizer.zero_grad()
            loss = model.forward_random_encoder_decoder(batch_size=batch_size, dimension=dimension)
            loss.backward()
            optimizer.step()"""

        # -------- evaluation --------
        if model.epoch % save_every == 0:
            train_time = time.time() - t  # difference between current time and start time
            t = time.time()
            batch = None
            with torch.no_grad():  # no gradient needed for evaluation
                loss_vals = np.zeros(len(val_loss_names))
                eval_loss = 0
                num_obs = 0
                if 'evaluate' in options and options['evaluate']:
                    eval_msd = np.zeros(len(eval_metric_names))
                model.eval()  # set model in evaluation mode
                for i, b in enumerate(dl_val):  # iterate over dataloader for validation set
                    #if plot:
                    #    batch = b
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
                    true_paths = b["true_paths"]
                    true_mask = b["true_mask"]

                    if 'other_model' not in options:
                        if 'add_dynamic_cov' in options and options['add_dynamic_cov']:
                            hT, c_loss = model(
                                times=times, time_ptr=time_ptr, X=torch.cat((X,Z),dim=1), obs_idx=obs_idx, delta_t=None, T=T, 
                                start_X=torch.cat((start_X,start_Z),dim=1), n_obs_ot=n_obs_ot, return_path=False, get_loss=True,
                                S=S, start_S=start_S, which_loss=which_eval_loss) # which_loss='standard'
                        else:
                            hT, c_loss = model(
                                times=times, time_ptr=time_ptr, X=X, obs_idx=obs_idx, delta_t=None, T=T, start_X=start_X,
                                n_obs_ot=n_obs_ot, return_path=False, get_loss=True,
                                S=S, start_S=start_S, which_loss=which_eval_loss) # which_loss='standard'
                    else:
                        raise ValueError
                    loss_vals += c_loss.detach().cpu().numpy()
                    num_obs += 1  # count number of observations

                    if 'other_model' not in options:
                        if 'add_dynamic_cov' in options and options['add_dynamic_cov']:
                            hT, c2_loss = model(
                                times=times, time_ptr=time_ptr, X=torch.cat((X,Z),dim=1), obs_idx=obs_idx, delta_t=None, T=T, 
                                S=S, start_S=start_S, start_X=torch.cat((start_X,start_Z),dim=1),
                                n_obs_ot=n_obs_ot, return_path=False, get_loss=True,)
                        else:
                            hT, c2_loss = model(
                                times=times, time_ptr=time_ptr, X=X, obs_idx=obs_idx, delta_t=None, T=T, start_X=start_X,
                                S=S, start_S=start_S, n_obs_ot=n_obs_ot, return_path=False,
                                get_loss=True,)
                    else:
                        raise ValueError
                    eval_loss += c2_loss.detach().cpu().numpy()

                    # mean squared difference evaluation
                    if 'evaluate' in options and options['evaluate']:
                        if 'add_dynamic_cov' in options and options['add_dynamic_cov']:
                            _eval_msd = model.evaluate(
                                times=times, time_ptr=time_ptr, X=torch.cat((X,Z),dim=1),
                                obs_idx=obs_idx, delta_t=delta_t, T=T, S=S, start_S=start_S,
                                start_X=torch.cat((start_X,start_Z),dim=1), n_obs_ot=n_obs_ot,
                                return_paths=False, true_paths=true_paths,
                                true_mask=true_mask, eval_vars=eval_metrics,) # mult=mult)
                        else:
                            _eval_msd = model.evaluate(
                                times=times, time_ptr=time_ptr, X=X,
                                obs_idx=obs_idx, delta_t=delta_t, T=T,
                                start_X=start_X, n_obs_ot=n_obs_ot, S=S, start_S=start_S,
                                return_paths=False, true_paths=true_paths,
                                true_mask=true_mask, eval_vars=eval_metrics,) # mult=mult)
                        eval_msd += _eval_msd

                eval_time = time.time() - t
                loss_vals = loss_vals / num_obs
                eval_loss = eval_loss / num_obs
                if 'evaluate' in options and options['evaluate']:
                    eval_msd = eval_msd / num_obs
                train_loss = loss.detach().cpu().numpy()
                print_str = "epoch {}, weight={:.5f}, train-loss={:.5f}, " \
                            "".format(
                    model.epoch, model.weight, train_loss)
                print_str += "eval-loss={:.5f}, ".format(eval_loss)
                if len(loss_vals) > 1:
                    for v,value in enumerate(loss_vals):
                        print_str += "val-loss-{}={:.5f}, ".format(v,value)
                    loss_val = np.sum(loss_vals * val_loss_weights)
                    loss_vals = np.concatenate([np.array([loss_val]), loss_vals],axis=0)
                else:
                    loss_val = loss_vals[0]
                print_str += "val-loss={:.5f}, ".format(loss_val)
                print(print_str)
            if 'evaluate' in options and options['evaluate']:
                metric_app.append([model.epoch, train_time, eval_time, train_loss, eval_loss] + list(loss_vals) + list(eval_msd))
                string = "evaluation : \n"
                for v,value in enumerate(eval_msd):
                    string += eval_metric_names[v] + " : {:.5f}\n".format(value)
                print(string)
            else:
                metric_app.append([model.epoch, train_time, eval_time, train_loss, eval_loss] + list(loss_vals))

            if use_wandb:
                wandb.log({"epoch": model.epoch, "train_time": train_time, "eval_time": eval_time, "train_loss": train_loss, \
                    "eval_loss": eval_loss})

            # save model
            if plot:
                batch = next(iter(dl_val))
                print('plotting ...')
                plot_filename = 'epoch-{}'.format(model.epoch)
                plot_filename = plot_filename + '_path-{}.pdf'
                plot_one_path_with_pred(
                    device=device, model=model, batch=batch,
                    delta_t=delta_t, T=T,
                    path_to_plot=paths_to_plot, save_path=plot_save_path,
                    filename=plot_filename, plot_variance=plot_variance,
                    plot_moments=plot_moments, output_vars=output_vars,
                    functions=input_vars, std_factor=std_factor,
                    model_name=model_name, save_extras=save_extras,
                    ylabels=ylabels,
                    same_yaxis=plot_same_yaxis, plot_obs_prob=plot_obs_prob,
                    dataset_metadata=dataset_metadata,
                    )
                # plot_weights_filename = 'epoch-{}'.format(model.epoch) + '_input_ode_nn_weights'
                # model.ode_f.plot_input_weights(os.path.join(plot_save_path, plot_weights_filename)) # plot weights
                # plot_features_filename = 'epoch-{}'.format(model.epoch) + '_input_ode_nn_features'
                # model.ode_f.input_features_save_and_reset(os.path.join(plot_save_path, plot_features_filename))
                if plot_train:
                    batch = next(iter(dl))
                    plot_filename1 = 'epoch-{}_train'.format(model.epoch)
                    plot_filename1 = plot_filename1 + '_path-{}.pdf'
                    plot_one_path_with_pred(
                        device=device, model=model, batch=batch,
                        delta_t=delta_t, T=T,
                        path_to_plot=paths_to_plot, save_path=plot_save_path,
                        filename=plot_filename1, plot_variance=plot_variance,
                        plot_moments=plot_moments, output_vars=output_vars,
                        functions=input_vars, std_factor=std_factor,
                        model_name=model_name, save_extras=save_extras,
                        ylabels=ylabels,
                        same_yaxis=plot_same_yaxis, plot_obs_prob=plot_obs_prob,
                        dataset_metadata=dataset_metadata,
                        )
                del batch
            print('save model ...', end="")
            print("mode id:", model_id)
            df_m_app = pd.DataFrame(data=metric_app, columns=metr_columns)
            df_metric = pd.concat([df_metric, df_m_app], ignore_index=True)
            df_metric.to_csv(model_metric_file)
            models.save_checkpoint(model, optimizer, model_path_save_last,
                                   model.epoch)
            metric_app = []
            print('saved!')
            if loss_val < best_val_loss:
                print('save new best model: last-best-loss: {:.5f}, '
                    'new-best-loss: {:.5f}, epoch: {}'.format(
                    best_val_loss, loss_val, model.epoch))
                df_m_app = pd.DataFrame(data=metric_app, columns=metr_columns)
                df_metric = pd.concat([df_metric, df_m_app], ignore_index=True)
                df_metric.to_csv(model_metric_file)
                # models.save_checkpoint(model, optimizer, model_path_save_last,
                #                     model.epoch)
                models.save_checkpoint(model, optimizer, model_path_save_best,
                                    model.epoch)
                metric_app = []
                best_val_loss = loss_val
                print('saved!')
            print("-"*100)
            t = time.time()

        model.epoch += 1
        model.weight_step()

    # send notification
    if SEND and not skip_training:
        files_to_send = [model_metric_file]
        caption = "{} - id={}".format(model_name, model_id)
        if plot:
            for i in paths_to_plot:
                files_to_send.append(sorted(glob.glob(
                    os.path.join(plot_save_path, plot_filename.format(
                        "{}*".format(i)))))[0])
        if plot_train:
            for i in paths_to_plot:
                files_to_send.append(sorted(glob.glob(
                    os.path.join(plot_save_path, plot_filename1.format(
                        "{}*".format(i)))))[0])
        SBM.send_notification(
            text='finished training: {}, id={}\n\n{}'.format(
                model_name, model_id, desc),
            chat_id=config.CHAT_ID,
            files=files_to_send,
            text_for_files=caption)

    # delete model & free memory
    del model, dl, dl_val, data_train, data_val
    if use_wandb:
        wandb.finish()
    # gc.collect()

    return 0


def plot_one_path_with_pred(
        device, model, batch, delta_t, T,
        path_to_plot=(0,), save_path='', filename='plot_{}.png',
        plot_variance=False, plot_moments=False, functions=None, std_factor=1,
        model_name=None, ylabels=None,
        save_extras={'bbox_inches': 'tight', 'pad_inches': 0.01},
        same_yaxis=False,
        plot_obs_prob=False, dataset_metadata=None,
        output_vars=None, max_dim_per_plot=5,
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

    :param output_vars: list of str, the output variables of the model that depend on X
    """
    if model_name is None or model_name == "NJODE":
        model_name = 'our model'

    prop_cycle = plt.rcParams['axes.prop_cycle']  # change style of plot?
    colors = prop_cycle.by_key()['color']
    std_color = list(matplotlib.colors.to_rgb(colors[1])) + [0.5]
    true_std_color = list(matplotlib.colors.to_rgb(colors[2])) + [0.5]

    makedirs(save_path)  # create a directory

    times = batch["times"]
    time_ptr = batch["time_ptr"]
    X = batch["X"].to(device)
    Z = batch["Z"].to(device)
    S = batch["S"].to(device)
    M = batch["M"]
    if M is not None:
        M = M.to(device)
    start_X = batch["start_X"].to(device)
    start_Z = batch["start_Z"].to(device)
    start_S = batch["start_S"].to(device)
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
    """res = model.get_pred(
        times=times, time_ptr=time_ptr, X=X, obs_idx=obs_idx, delta_t=None,
        T=T, start_X=start_X, M=M, start_M=start_M)"""
    res = model.get_pred(
        times=times, time_ptr=time_ptr, X=torch.cat((X,Z),dim=1), obs_idx=obs_idx, delta_t=None,
        T=T, start_X=torch.cat((start_X,start_Z),dim=1), S=S, start_S=start_S)
    path_y_pred = res['pred'].detach().cpu().numpy()
    path_t_pred = res['pred_t']

    # get variance path
    if plot_variance and (output_vars is not None) and ('var' in output_vars):
        which = np.argmax(np.array(output_vars) == 'var')
        path_var_pred = path_y_pred[:, :, (dim * which):(dim * (which + 1))]
        if np.any(path_var_pred < 0):
            print('WARNING: some predicted cond. variances below 0 -> clip')
            print(np.sum(path_var_pred < 0), " out of ", path_var_pred.reshape(-1).shape[0], " values")
            path_var_pred = np.maximum(0, path_var_pred)
        path_std_pred = np.sqrt(path_var_pred)
    elif plot_variance and (functions is not None) and ('power-2' in functions):
        which = np.argmax(np.array(functions) == 'power-2')
        y2 = path_y_pred[:, :, (dim * which):(dim * (which + 1))]
        path_var_pred = y2 - np.power(path_y_pred[:, :, 0:dim], 2)
        if np.any(path_var_pred < 0):
            print('WARNING: some predicted cond. variances below 0 -> clip')
            print(np.sum(path_var_pred < 0), " out of ", path_var_pred.reshape(-1).shape[0], " values")
            path_var_pred = np.maximum(0, path_var_pred)
        path_std_pred = np.sqrt(path_var_pred)
    else:
        path_std_pred = None
    if ((output_vars is None) or ('var' not in output_vars)) and ((functions is None) or ('power-2' not in functions)):
        plot_variance = False

    for i in path_to_plot:
        if dim < max_dim_per_plot:
            fig, axs = plt.subplots(dim)
        if dim == 1:
            axs = [axs]
        for j in range(dim):
            if dim > max_dim_per_plot and j % max_dim_per_plot == 0:
                fig, axs = plt.subplots(min(max_dim_per_plot, dim-j))
                counter = 1
            else:
                counter += 1
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

            # axs[j % max_dim_per_plot].plot(path_t_true_X, true_X[i, j, :], label='true path',
            #            color=colors[0])
            axs[j % max_dim_per_plot].scatter(path_t_obs, path_X_obs, label='observed',
                           color=colors[0])
            axs[j % max_dim_per_plot].plot(path_t_pred, path_y_pred[:, i, j],
                        label=model_name + " predicted \n conditional expectation", color=colors[1])
            if plot_variance and path_std_pred is not None:
                axs[j % max_dim_per_plot].fill_between(
                    path_t_pred,
                    path_y_pred[:, i, j] - std_factor * path_std_pred[:, i, j],
                    path_y_pred[:, i, j] + std_factor * path_std_pred[:, i, j],
                    label=model_name + " predicted \n conditional standard \n deviation (*{})".format(std_factor), color=std_color)
        
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
                axs[j % max_dim_per_plot].set_ylabel("X")
                ax2.legend()
            if ylabels:
                axs[j % max_dim_per_plot].set_ylabel(ylabels[j])
            if same_yaxis:
                low = np.min(true_X[i, :, :])
                high = np.max(true_X[i, :, :])
                eps = (high - low)*0.05
                axs[j % max_dim_per_plot].set_ylim([low-eps, high+eps])

            if j == dim-1 or counter == max_dim_per_plot:
                plt.subplots_adjust(right=0.7)
                plt.xlabel('$t$')
                if dim < max_dim_per_plot:
                    plt.legend(bbox_to_anchor=(1.02, dim/2.+0.1), loc="center left")
                    save = os.path.join(save_path, filename.format(i))
                else:
                    plt.legend(bbox_to_anchor=(1.02, counter/2.+0.1), loc="center left")
                    save = os.path.join(save_path, filename.format("{}_dim-{}-{}".format(i,j-counter+1,j)))
                plt.savefig(save, **save_extras)
                plt.close()
    
        if plot_moments:
            for m in range(2,10):
                if 'power-{}'.format(m) in functions:
                    which = np.argmax(np.array(functions) == 'power-{}'.format(m))
                    fig, axs = plt.subplots(dim)
                    if dim == 1:
                        axs = [axs]
                    for j in range(dim):
                        if dim > max_dim_per_plot and j % max_dim_per_plot == 0:
                            fig, axs = plt.subplots(min(max_dim_per_plot, dim-j))
                            counter = 1
                        else:
                            counter += 1
                        path_t_obs = []
                        path_X_m_obs = []
                        for k, od in enumerate(observed_dates[i]):
                            if od == 1:
                                if true_M is None or (true_M is not None and
                                                    true_M[i, j, k]==1):
                                    path_t_obs.append(path_t_true_X[k])
                                    path_X_m_obs.append(np.power(true_X, m)[i, j, k])
                        path_t_obs = np.array(path_t_obs)
                        path_X_m_obs = np.array(path_X_m_obs)

                        # axs[j % max_dim_per_plot].plot(path_t_true_X, np.power(true_X, m)[i, j, :], label='true path power {}'.format(m),
                        #            color=colors[0])
                        axs[j % max_dim_per_plot].scatter(path_t_obs, path_X_m_obs, label='observed power {}'.format(m),
                                    color=colors[0])
                        axs[j % max_dim_per_plot].plot(path_t_pred, path_y_pred[:, i, dim * which + j],
                                    label=model_name + " predicted \n conditional moment {}".format(m), color=colors[1])

                        if plot_obs_prob and dataset_metadata is not None:
                            ax2 = axs[j % max_dim_per_plot].twinx()
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
                            axs[j % max_dim_per_plot].set_ylabel("X")
                            ax2.legend()
                        if ylabels:
                            axs[j % max_dim_per_plot].set_ylabel(ylabels[j])
                        if same_yaxis:
                            low = np.min(true_X[i, :, :])
                            high = np.max(true_X[i, :, :])
                            eps = (high - low)*0.05
                            axs[j % max_dim_per_plot].set_ylim([low-eps, high+eps])

                        if j == dim-1 or counter == max_dim_per_plot:
                            plt.subplots_adjust(right=0.7)
                            plt.xlabel('$t$')
                            if dim < max_dim_per_plot:
                                plt.legend(bbox_to_anchor=(1.02, dim/2.+0.1), loc="center left")
                                save = os.path.join(save_path, filename.format('{}_moment-{}'.format(i,m)))
                            else:
                                plt.legend(bbox_to_anchor=(1.02, counter/2.+0.1), loc="center left")
                                save = os.path.join(save_path, filename.format("{}_moment-{}_dim-{}-{}".format(i,m,j-counter+1,j)))
                            plt.savefig(save, **save_extras)
                            plt.close()