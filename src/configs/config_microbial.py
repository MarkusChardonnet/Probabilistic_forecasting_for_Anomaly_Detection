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
plot_moments = False
paths_to_plot = (0,1,2,)
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
        'epochs': [10000],
        'batch_size': [batch_size],
        'save_every': [100],
        'learning_rate': [learning_rate],
        'seed': [seed],
        'hidden_size': [300, 500],
        'bias': [bias],
        'dropout_rate': [dropout_rate],
        'ode_nn': [((200, 'tanh'), (200, 'relu'))],
        'readout_nn': [((100, 'tanh'), (100, 'tanh'))],
        'enc_nn': [((100, 'tanh'), (100, 'tanh'))],
        'use_rnn': [use_rnn],
        'input_sig': [input_sig],
        'func_appl_X': [func_appl_X],              # [["power-2", "power-3", "power-4"]]
        'add_pred': [add_pred],
        'test': [test],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor],
        'weight': [weight],
        # 'weight_evolve': [weight_evolve],
        'plot': [True],
        'which_loss': ['variance'],
        'which_val_loss': ['val_variance'],
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
        'add_readout_activation': [('sum2one',['id'])], # ('softmax',['id']) ('sum2one',['id'])
        'add_dynamic_cov': [True],
        'pre-train': [10000],
    }
# param_list_microbial_genus_base += get_parameter_array(param_dict=param_dict_microbial_genus_1_base)

param_dict_microbial_genus_2_base = {
        'dataset': ["microbial_genus"],
        'dataset_id': ["no_abx"],
        'epochs': [10000],
        'batch_size': [batch_size],
        'save_every': [1],
        'learning_rate': [learning_rate],
        'seed': [seed],
        'hidden_size': [500],
        'bias': [bias],
        'dropout_rate': [dropout_rate],
        'ode_nn': [((400, 'tanh'), (400, 'relu'))],
        'readout_nn': [((200, 'tanh'), (200, 'tanh'))],
        'enc_nn': [((200, 'tanh'), (200, 'tanh'))],
        'use_rnn': [use_rnn],
        'input_sig': [input_sig],
        'func_appl_X': [func_appl_X],              # [["power-2", "power-3", "power-4"]]
        'add_pred': [add_pred],
        'test': [test],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor],
        'weight': [weight],
        # 'weight_evolve': [weight_evolve],
        'plot': [True],
        'which_loss': ['variance'],
        'which_val_loss': ['val_variance'],
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
        'add_readout_activation': [('sum2one',['id'])], # ('softmax',['id']) ('sum2one',['id'])
        'add_dynamic_cov': [True],
        'pre-train': [1],
    }
param_list_microbial_genus_base += get_parameter_array(param_dict=param_dict_microbial_genus_2_base)

# test config
microbial_genus_models_path_test = "{}saved_models_microbial_genus_test/".format(data_path)
param_list_microbial_genus_test = []
param_dict_microbial_genus_1_test = {
        'dataset': ["microbial_genus"],
        'dataset_id': ["no_abx"],
        'epochs': [10],
        'batch_size': [batch_size],
        'save_every': [1],
        'learning_rate': [learning_rate],
        'seed': [seed],
        'hidden_size': [100],
        'bias': [bias],
        'dropout_rate': [dropout_rate],
        'ode_nn': [((100, 'tanh'), (100, 'relu'))],
        'readout_nn': [((100, 'tanh'), (100, 'tanh'))],
        'enc_nn': [((100, 'tanh'), (100, 'tanh'))],
        'use_rnn': [use_rnn],
        'input_sig': [input_sig],
        'func_appl_X': [func_appl_X],              # [["power-2", "power-3", "power-4"]]
        'add_pred': [add_pred],
        'test': [test],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor],
        'weight': [weight],
        # 'weight_evolve': [weight_evolve],
        'plot': [True],
        'plot_train': [True],
        'which_loss': ['variance'],
        'which_val_loss': ['val_variance'],
        'val_loss_weights':[[1.,1.,1.,1.]],
        'evaluate': [False],
        'eval_metrics': [eval_metrics],
        'paths_to_plot': [paths_to_plot],
        'plot_variance': [plot_variance],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        'saved_models_path': [microbial_genus_models_path_test],
        'use_cond_exp': [True],
        'input_current_t': [input_current_t],
        'periodic_current_t': [True],
        'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
        'add_readout_activation': [('sum2one',['id'])], # ('softmax',['id']) ('sum2one',['id'])
        'add_dynamic_cov': [True],
        'pre-train': [1],
    }
param_list_microbial_genus_test += get_parameter_array(param_dict=param_dict_microbial_genus_1_test)


AD_microbial_genus = "{}saved_AD_microbial_genus/".format(data_path)
param_dict_AD_microbial_genus = []
param_dict_AD_microbial_genus_1 = {
    "dataset": ['microbial_genus'],
    "dataset_id": ['no_abx'],
    'anomaly_data_dict': ['AD_OrnsteinUhlenbeckWithSeason_cutoff_dict'],
    'class_thres': [0.5], # 'automatic'
    'autom_thres': [None], #['FPR_limit-0.05'],
    'saved_models_path': [AD_microbial_genus],
    'paths_to_plot': [(0,1,2,3,4,)],
    'plot_forecast_predictions': [False],
    'plot_variance': [True],
    'std_factor': [1.96],
}
param_dict_AD_microbial_genus += get_parameter_array(param_dict=param_dict_AD_microbial_genus_1)