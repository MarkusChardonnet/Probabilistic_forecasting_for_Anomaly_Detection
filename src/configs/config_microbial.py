import numpy as np

from configs.config_utils import get_parameter_array, get_dataset_overview, \
    makedirs, data_path, training_data_path


batch_size = 20
learning_rate = 0.001
seed = 398
hidden_size = 100
bias = True
dropout_rate = 0.1
ode_nn = ((200, 'tanh'), (200, 'relu'))
enc_nn = ((100, 'tanh'), (100, 'tanh'))
readout_nn = ((100, 'tanh'), (100, 'tanh'))
use_rnn = True
input_sig = True
func_appl_X = ["power-2"]
add_pred = ["var"]
solver = "euler"
solver_delta_t_factor = 1
weight = 0
weight_evolve = {'type':'linear', 'target': 1, 'reach': None}
eval_metrics = ['exp','std']
std_factor = 1.96
plot_variance = True
plot_moments = True
paths_to_plot = (0,1,2,3,4,)
input_current_t = True
enc_input_t = False
scale_dt = 1.
test = False

# base setup
microbial_genus_models_path_base = "{}saved_models_microbial_genus_base/".format(data_path)
param_list_microbial_genus_base = []
param_dict_microbial_genus_1_base = {
        'dataset': ["microbial_genus"],
        'dataset_id': ["no_abx"],
        'epochs': [50],
        'batch_size': [batch_size],
        'save_every': [1],
        'learning_rate': [learning_rate],
        'seed': [seed],
        'hidden_size': [hidden_size],
        'bias': [bias],
        'dropout_rate': [dropout_rate],
        'ode_nn': [ode_nn],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [use_rnn],
        'input_sig': [input_sig],
        'func_appl_X': [func_appl_X],              # [["power-2", "power-3", "power-4"]]
        'add_pred': [add_pred],
        'test': [test],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor],
        'weight': [weight],
        'weight_evolve': [weight_evolve],
        'plot': [False],
        'which_loss': ['variance'],
        'which_eval_loss': ['eval_variance'],
        'evaluate': [False],
        'eval_metrics': [eval_metrics],
        'paths_to_plot': [paths_to_plot],
        'plot_variance': [plot_variance],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        'saved_models_path': [microbial_genus_models_path_base],
        'use_cond_exp': [True],
        'input_current_t': [input_current_t],
        'periodic_current_t': [True],
        'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
    }
param_list_microbial_genus_base += get_parameter_array(param_dict=param_dict_microbial_genus_1_base)