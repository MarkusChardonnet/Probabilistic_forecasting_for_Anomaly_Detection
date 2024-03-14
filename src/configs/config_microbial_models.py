import numpy as np

from configs.config_utils import get_parameter_array, get_dataset_overview, \
    makedirs, data_path, training_data_path


epochs = 3000
save_every = 100
batch_size = 30
learning_rate = 0.001
seed = 398
bias = True
dropout_rate = 0.1
use_rnn = True
input_sig = True
func_appl_X = ["power-2"]
add_pred = ["var"]
solver = "euler"
solver_delta_t_factor = 1./7.
weight = 0
eval_metrics = ['exp','std']
std_factor = 1.96
plot_variance = True
plot_moments = False
paths_to_plot = (0,1,2,)
input_current_t = True
enc_input_t = False
scale_dt = 1.
test = False
pre_train = 10000
add_readout_activation = ('sum2one',['id'])
datasets_genus = ["microbial_genus", "microbial_genus_sig_div", "microbial_genus_sig_highab", "microbial_genus_sig_nonzero"]
datasets_otu = ["microbial_otu", "microbial_otu_sig_div", "microbial_otu_sig_highab", "microbial_otu_sig_nonzero"]
dataset_splits = ["all", "no_abx"]

hidden_size = 300
ode_nn = ((300, 'tanh'), (300, 'relu'))
ode_nn1 = ((300, 'tanh'), (300, 'tanh'))
enc_nn = ((200, 'tanh'), (300, 'tanh'))
readout_nn = ((300, 'tanh'), (200, 'tanh'))

# base setup
microbial_genus_models_path = "{}saved_models_microbial_genus/".format(data_path)
param_list_microbial_genus = []

# with rnn and signature
param_dict_microbial_genus_sig_rnn = {
        'dataset': datasets_genus,
        'dataset_split': dataset_splits,
        'epochs': [epochs],
        'batch_size': [batch_size],
        'save_every': [save_every],
        'learning_rate': [learning_rate],
        'seed': [seed],
        'hidden_size': [hidden_size],
        'bias': [bias],
        'dropout_rate': [dropout_rate],
        'ode_nn': [ode_nn],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [True],
        'input_sig': [True],
        'func_appl_X': [[]],              # [["power-2", "power-3", "power-4"]]
        'add_pred': [[]],
        'test': [test],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor],
        'weight': [0.5],
        'plot': [True],
        'which_loss': ['easy'],
        'which_val_loss': ['standard'],
        'evaluate': [False],
        'eval_metrics': [eval_metrics],
        'paths_to_plot': [paths_to_plot],
        'plot_variance': [False],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        'saved_models_path': [microbial_genus_models_path],
        'use_cond_exp': [True],
        'input_current_t': [input_current_t],
        'periodic_current_t': [True],
        'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
        'add_readout_activation': [add_readout_activation], # ('softmax',['id']) ('sum2one',['id'])
        'add_dynamic_cov': [True],
        'pre-train': [10000],
        'zero_weight_init': [False],
    }
param_list_microbial_genus += get_parameter_array(param_dict=param_dict_microbial_genus_sig_rnn)
# with rnn
param_dict_microbial_genus_rnn = {
        'dataset': datasets_genus,
        'dataset_id': dataset_splits,
        'epochs': [epochs],
        'batch_size': [batch_size],
        'save_every': [save_every],
        'learning_rate': [learning_rate],
        'seed': [seed],
        'hidden_size': [hidden_size],
        'bias': [bias],
        'dropout_rate': [dropout_rate],
        'ode_nn': [ode_nn],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [True],
        'input_sig': [False],
        'func_appl_X': [[]],              # [["power-2", "power-3", "power-4"]]
        'add_pred': [[]],
        'test': [test],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor],
        'weight': [0.5],
        'plot': [True],
        'which_loss': ['easy'],
        'which_val_loss': ['standard'],
        'evaluate': [False],
        'eval_metrics': [eval_metrics],
        'paths_to_plot': [paths_to_plot],
        'plot_variance': [False],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        'saved_models_path': [microbial_genus_models_path],
        'use_cond_exp': [True],
        'input_current_t': [input_current_t],
        'periodic_current_t': [True],
        'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
        'add_readout_activation': [add_readout_activation], # ('softmax',['id']) ('sum2one',['id'])
        'add_dynamic_cov': [True],
        'pre-train': [10000],
        'zero_weight_init': [False],
    }
param_list_microbial_genus += get_parameter_array(param_dict=param_dict_microbial_genus_rnn)
# with signature and residual connection
param_dict_microbial_genus_sig_res = {
        'dataset': datasets_genus,
        'dataset_id': dataset_splits,
        'epochs': [epochs],
        'batch_size': [batch_size],
        'save_every': [save_every],
        'learning_rate': [learning_rate],
        'seed': [seed],
        'hidden_size': [hidden_size],
        'bias': [bias],
        'dropout_rate': [dropout_rate],
        'ode_nn': [ode_nn],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [False],
        'input_sig': [True],
        'residual_enc_dec': [True],
        'func_appl_X': [[]],              # [["power-2", "power-3", "power-4"]]
        'add_pred': [[]],
        'test': [test],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor],
        'weight': [0.5],
        'plot': [True],
        'which_loss': ['easy'],
        'which_val_loss': ['standard'],
        'evaluate': [False],
        'eval_metrics': [eval_metrics],
        'paths_to_plot': [paths_to_plot],
        'plot_variance': [False],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        'saved_models_path': [microbial_genus_models_path],
        'use_cond_exp': [True],
        'input_current_t': [input_current_t],
        'periodic_current_t': [True],
        'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
        'add_readout_activation': [add_readout_activation], # ('softmax',['id']) ('sum2one',['id'])
        'add_dynamic_cov': [True],
        'pre-train': [0],
        'zero_weight_init': [True]
    }
param_list_microbial_genus += get_parameter_array(param_dict=param_dict_microbial_genus_sig_res)
# with residual connection
param_dict_microbial_genus_res = {
        'dataset': datasets_genus,
        'dataset_id': dataset_splits,
        'epochs': [epochs],
        'batch_size': [batch_size],
        'save_every': [save_every],
        'learning_rate': [learning_rate],
        'seed': [seed],
        'hidden_size': [hidden_size],
        'bias': [bias],
        'dropout_rate': [dropout_rate],
        'ode_nn': [ode_nn],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [False],
        'input_sig': [False],
        'residual_enc_dec': [True],
        'func_appl_X': [[]],              # [["power-2", "power-3", "power-4"]]
        'add_pred': [[]],
        'test': [test],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor],
        'weight': [0.5],
        'plot': [True],
        'which_loss': ['easy'],
        'which_val_loss': ['standard'],
        'evaluate': [False],
        'eval_metrics': [eval_metrics],
        'paths_to_plot': [paths_to_plot],
        'plot_variance': [False],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        'saved_models_path': [microbial_genus_models_path],
        'use_cond_exp': [True],
        'input_current_t': [input_current_t],
        'periodic_current_t': [True],
        'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
        'add_readout_activation': [add_readout_activation], # ('softmax',['id']) ('sum2one',['id'])
        'add_dynamic_cov': [True],
        'pre-train': [0],
        'zero_weight_init': [True],
    }
param_list_microbial_genus += get_parameter_array(param_dict=param_dict_microbial_genus_res)



microbial_otu_models_path = "{}saved_models_microbial_otu/".format(data_path)
param_list_microbial_otu = []
# with rnn and signature
param_dict_microbial_otu_sig_rnn = {
        'dataset': datasets_otu,
        'dataset_split': dataset_splits,
        'epochs': [epochs],
        'batch_size': [batch_size],
        'save_every': [save_every],
        'learning_rate': [learning_rate],
        'seed': [seed],
        'hidden_size': [hidden_size],
        'bias': [bias],
        'dropout_rate': [dropout_rate],
        'ode_nn': [ode_nn],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [True],
        'input_sig': [True],
        'func_appl_X': [[]],              # [["power-2", "power-3", "power-4"]]
        'add_pred': [[]],
        'test': [test],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor],
        'weight': [0.5],
        'plot': [True],
        'which_loss': ['easy'],
        'which_val_loss': ['standard'],
        'evaluate': [False],
        'eval_metrics': [eval_metrics],
        'paths_to_plot': [paths_to_plot],
        'plot_variance': [False],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        'saved_models_path': [microbial_otu_models_path],
        'use_cond_exp': [True],
        'input_current_t': [input_current_t],
        'periodic_current_t': [True],
        'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
        'add_readout_activation': [add_readout_activation], # ('softmax',['id']) ('sum2one',['id'])
        'add_dynamic_cov': [True],
        'pre-train': [10000],
        'zero_weight_init': [False],
    }
param_list_microbial_otu += get_parameter_array(param_dict=param_dict_microbial_otu_sig_rnn)
# with rnn
param_dict_microbial_otu_rnn = {
        'dataset': datasets_otu,
        'dataset_id': dataset_splits,
        'epochs': [epochs],
        'batch_size': [batch_size],
        'save_every': [save_every],
        'learning_rate': [learning_rate],
        'seed': [seed],
        'hidden_size': [hidden_size],
        'bias': [bias],
        'dropout_rate': [dropout_rate],
        'ode_nn': [ode_nn],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [True],
        'input_sig': [False],
        'func_appl_X': [[]],              # [["power-2", "power-3", "power-4"]]
        'add_pred': [[]],
        'test': [test],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor],
        'weight': [0.5],
        'plot': [True],
        'which_loss': ['easy'],
        'which_val_loss': ['standard'],
        'evaluate': [False],
        'eval_metrics': [eval_metrics],
        'paths_to_plot': [paths_to_plot],
        'plot_variance': [False],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        'saved_models_path': [microbial_otu_models_path],
        'use_cond_exp': [True],
        'input_current_t': [input_current_t],
        'periodic_current_t': [True],
        'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
        'add_readout_activation': [add_readout_activation], # ('softmax',['id']) ('sum2one',['id'])
        'add_dynamic_cov': [True],
        'pre-train': [10000],
        'zero_weight_init': [False],
    }
param_list_microbial_otu += get_parameter_array(param_dict=param_dict_microbial_otu_rnn)
# with signature and residual connection
param_dict_microbial_otu_sig_res = {
        'dataset': datasets_otu,
        'dataset_id': dataset_splits,
        'epochs': [epochs],
        'batch_size': [batch_size],
        'save_every': [save_every],
        'learning_rate': [learning_rate],
        'seed': [seed],
        'hidden_size': [hidden_size],
        'bias': [bias],
        'dropout_rate': [dropout_rate],
        'ode_nn': [ode_nn],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [False],
        'input_sig': [True],
        'residual_enc_dec': [True],
        'func_appl_X': [[]],              # [["power-2", "power-3", "power-4"]]
        'add_pred': [[]],
        'test': [test],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor],
        'weight': [0.5],
        'plot': [True],
        'which_loss': ['easy'],
        'which_val_loss': ['standard'],
        'evaluate': [False],
        'eval_metrics': [eval_metrics],
        'paths_to_plot': [paths_to_plot],
        'plot_variance': [False],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        'saved_models_path': [microbial_otu_models_path],
        'use_cond_exp': [True],
        'input_current_t': [input_current_t],
        'periodic_current_t': [True],
        'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
        'add_readout_activation': [add_readout_activation], # ('softmax',['id']) ('sum2one',['id'])
        'add_dynamic_cov': [True],
        'pre-train': [0],
        'zero_weight_init': [True]
    }
param_list_microbial_otu += get_parameter_array(param_dict=param_dict_microbial_otu_sig_res)
# with residual connection
param_dict_microbial_otu_res = {
        'dataset': datasets_otu,
        'dataset_id': dataset_splits,
        'epochs': [epochs],
        'batch_size': [batch_size],
        'save_every': [save_every],
        'learning_rate': [learning_rate],
        'seed': [seed],
        'hidden_size': [hidden_size],
        'bias': [bias],
        'dropout_rate': [dropout_rate],
        'ode_nn': [ode_nn],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [False],
        'input_sig': [False],
        'residual_enc_dec': [True],
        'func_appl_X': [[]],              # [["power-2", "power-3", "power-4"]]
        'add_pred': [[]],
        'test': [test],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor],
        'weight': [0.5],
        'plot': [True],
        'which_loss': ['easy'],
        'which_val_loss': ['standard'],
        'evaluate': [False],
        'eval_metrics': [eval_metrics],
        'paths_to_plot': [paths_to_plot],
        'plot_variance': [False],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        'saved_models_path': [microbial_otu_models_path],
        'use_cond_exp': [True],
        'input_current_t': [input_current_t],
        'periodic_current_t': [True],
        'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
        'add_readout_activation': [add_readout_activation], # ('softmax',['id']) ('sum2one',['id'])
        'add_dynamic_cov': [True],
        'pre-train': [0],
        'zero_weight_init': [True],
    }
param_list_microbial_otu += get_parameter_array(param_dict=param_dict_microbial_otu_res)





# ------------------------------------------------------------------------------
# testing on otu:
#   - different loss and eval loss function
#   - use RNN with residual connection to see whether there is really no
#       path-dependency
#   - only train on no_abx and only with highabundance signature features

microbial_otu_models_path2 = "{}saved_models_microbial_otu2/".format(data_path)
param_list_microbial_otu2 = []

param_dict_microbial_otu_sig_rnn = {
        'dataset': ["microbial_otu_sig_highab"],
        'dataset_split': ["no_abx"],
        'epochs': [epochs],
        'batch_size': [batch_size],
        'save_every': [save_every],
        'learning_rate': [learning_rate],
        'seed': [seed],
        'hidden_size': [hidden_size],
        'bias': [bias],
        'dropout_rate': [dropout_rate],
        'ode_nn': [ode_nn, ode_nn1],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [True, False],
        'input_sig': [True, False],
        'residual_enc_dec': [True, False],
        'func_appl_X': [[]],              # [["power-2", "power-3", "power-4"]]
        'add_pred': [[]],
        'test': [test],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor, 1.],
        'weight': [0.5],
        'plot': [True],
        'which_loss': ['easy', 'noisy_obs'],
        'which_eval_loss': ['noisy_obs'],
        'evaluate': [False],
        'eval_metrics': [eval_metrics],
        'paths_to_plot': [paths_to_plot],
        'plot_variance': [False],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        'saved_models_path': [microbial_otu_models_path2],
        'use_cond_exp': [True],
        'input_current_t': [input_current_t],
        'periodic_current_t': [True],
        'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
        'add_readout_activation': [add_readout_activation], # ('softmax',['id']) ('sum2one',['id'])
        'add_dynamic_cov': [True],
        'pre-train': [10000],
        'zero_weight_init': [False],
    }
param_list_microbial_otu2 += get_parameter_array(param_dict=param_dict_microbial_otu_sig_rnn)

overview_dict_microbial_otu2 = dict(
    ids_from=1, ids_to=len(param_list_microbial_otu2),
    path=microbial_otu_models_path2,
    params_extract_desc=('dataset', 'dataset_split',
                         'ode_nn', 'enc_nn', 'readout_nn',
                         'dropout_rate', 'hidden_size', 'batch_size',
                         'which_loss', 'which_eval_loss',
                         'solver_delta_t_factor',
                         'residual_enc_dec', 'use_rnn',
                         'input_sig', 'level', ),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "eval_loss", "eval_loss", "eval_loss_min"),
    ),
    sortby=["eval_loss_min"],
)

plot_paths_microbial_otu2 = {
    'model_ids': [27, 25, 36], 'saved_models_path': microbial_otu_models_path2,
    'which': 'best', 'paths_to_plot': [0,1,2,3,4,5], 'wait_time': 5,
    'save_extras': {'bbox_inches': 'tight', 'pad_inches': 0.01},}


# ------------------------------------------------------------------------------
# also train variance estimator

microbial_otu_models_path3 = "{}saved_models_microbial_otu3/".format(data_path)
param_list_microbial_otu3 = []

param_dict_microbial_otu_sig_rnn = {
        'dataset': ["microbial_otu_sig_highab"],
        'dataset_split': ["no_abx"],
        'epochs': [epochs],
        'batch_size': [batch_size],
        'save_every': [save_every],
        'learning_rate': [learning_rate],
        'seed': [seed],
        'hidden_size': [hidden_size],
        'bias': [bias],
        'dropout_rate': [dropout_rate],
        'ode_nn': [ode_nn, ode_nn1],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [True, False],
        'input_sig': [True, False],
        'residual_enc_dec': [True, False],
        'func_appl_X': [["power-2"]],              # [["power-2", "power-3", "power-4"]]
        'add_pred': [["var"]],
        'test': [test],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor],
        'weight': [0.],
        'weight_evolve': [{'type': 'linear', 'target': 1, 'reach': None}],
        'plot': [True],
        'which_loss': ['variance'],
        'which_eval_loss': ['eval_variance'],
        'evaluate': [True],
        'eval_metrics': [eval_metrics],
        'paths_to_plot': [(0,)],
        'plot_variance': [True],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        'saved_models_path': [microbial_otu_models_path3],
        'use_cond_exp': [True],
        'input_current_t': [input_current_t],
        'periodic_current_t': [True],
        'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
        'add_readout_activation': [add_readout_activation], # ('softmax',['id']) ('sum2one',['id'])
        'add_dynamic_cov': [True],
        'pre-train': [10000],
        'zero_weight_init': [False],
    }
param_list_microbial_otu3 += get_parameter_array(param_dict=param_dict_microbial_otu_sig_rnn)

overview_dict_microbial_otu3 = dict(
    ids_from=1, ids_to=len(param_list_microbial_otu3),
    path=microbial_otu_models_path3,
    params_extract_desc=('dataset', 'dataset_split',
                         'ode_nn', 'enc_nn', 'readout_nn',
                         'dropout_rate', 'hidden_size', 'batch_size',
                         'which_loss', 'which_eval_loss',
                         'solver_delta_t_factor',
                         'residual_enc_dec', 'use_rnn',
                         'input_sig', 'level', ),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "eval_loss", "eval_loss", "eval_loss_min"),
    ),
    sortby=["eval_loss_min"],
)