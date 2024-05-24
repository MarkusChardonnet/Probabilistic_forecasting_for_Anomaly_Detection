import numpy as np

from configs.config_utils import get_parameter_array, get_dataset_overview, \
    makedirs, data_path, training_data_path


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
pre_train = 10000
add_readout_activation = ('sum2one',['id'])
dataset = "microbial_genus"
dataset_id = "all" #  "no_abx"

hidden_size_0 = 200
ode_nn_0 = ((500, 'tanh'), (300, 'relu'))
enc_nn_0 = ((500, 'tanh'), (300, 'tanh'))
readout_nn_0 = ((200, 'tanh'), (150, 'tanh'))

hidden_size_1 = 400
ode_nn_1 = ((200, 'tanh'), (200, 'relu'))
enc_nn_1 = ((200, 'tanh'), (300, 'tanh'))
readout_nn_1 = ((300, 'tanh'), (200, 'tanh'))

"""
ode_nn_1 = ((400, 'tanh'), (400, 'relu'))
enc_nn_1 = ((150, 'tanh'), (150, 'tanh'))
readout_nn_1 = ((150, 'tanh'), (150, 'tanh'))
"""

hidden_size_0 = 200
hidden_size_1 = 500
hidden_size_2 = 1000

nn_0 = ((100, 'tanh'), (100, 'tanh'))
nn_1 = ((300, 'tanh'), (300, 'tanh'))
nn_2 = ((500, 'tanh'), (500, 'tanh'))
nn_0b = ((200, 'tanh'), (100, 'tanh'))
nn_1b = ((500, 'tanh'), (250, 'tanh'))
nn_2b = ((1000, 'tanh'), (500, 'tanh'))


# base setup
microbial_genus_models_path_base = "{}saved_models_microbial_genus_base/".format(data_path)
param_list_microbial_genus_base = []

for _nn in [nn_0, nn_1, nn_2, nn_0b, nn_1b, nn_2b]:
    param_dict_microbial_genus_mult_base = {
            'dataset': [dataset],
            'dataset_id': [dataset_id],
            'epochs': [3000],
            'batch_size': [batch_size],
            'save_every': [1],
            'learning_rate': [learning_rate],
            'seed': [seed],
            'hidden_size': [hidden_size_0, hidden_size_1],
            'bias': [bias],
            'dropout_rate': [dropout_rate],
            'ode_nn': [_nn],
            'readout_nn': [_nn],
            'enc_nn': [_nn],
            'use_rnn': [False],
            'input_sig': [True],
            'func_appl_X': [[]],              # [["power-2", "power-3", "power-4"]]
            'add_pred': [[]],
            'test': [test],
            'solver': [solver],
            'solver_delta_t_factor': [solver_delta_t_factor],
            'weight': [0.5],
            # 'weight_evolve': [weight_evolve],
            'plot': [True],
            'which_loss': ['easy'],
            'which_val_loss': ['standard'],
            'evaluate': [False],
            'eval_metrics': [eval_metrics],
            'paths_to_plot': [paths_to_plot],
            'plot_variance': [False],
            'std_factor': [std_factor],
            'plot_moments': [plot_moments],
            'saved_models_path': [microbial_genus_models_path_base],
            'use_cond_exp': [True],
            'input_current_t': [input_current_t],
            'periodic_current_t': [True],
            'scale_dt': [scale_dt],
            'enc_input_t': [enc_input_t],
            'add_readout_activation': [add_readout_activation], # ('softmax',['id']) ('sum2one',['id'])
            'add_dynamic_cov': [True],
            'pre-train': [1],
            'zero_weight_init': [True],
        }
    param_list_microbial_genus_base += get_parameter_array(param_dict=param_dict_microbial_genus_mult_base)

param_dict_microbial_genus_1_base = {
        'dataset': [dataset],
        'dataset_id': [dataset_id],
        'epochs': [4000],
        'batch_size': [batch_size],
        'save_every': [1],
        'learning_rate': [learning_rate],
        'seed': [seed],
        'hidden_size': [hidden_size_0],
        'bias': [bias],
        'dropout_rate': [dropout_rate],
        'ode_nn': [ode_nn_0],
        'readout_nn': [readout_nn_0],
        'enc_nn': [enc_nn_0],
        'use_rnn': [True],
        'input_sig': [True],
        'func_appl_X': [[]],              # [["power-2", "power-3", "power-4"]]
        'add_pred': [[]],
        'test': [test],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor],
        'weight': [0.5],
        # 'weight_evolve': [weight_evolve],
        'plot': [True],
        'which_loss': ['easy'],
        'which_val_loss': ['standard'],
        'evaluate': [False],
        'eval_metrics': [eval_metrics],
        'paths_to_plot': [paths_to_plot],
        'plot_variance': [False],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        'saved_models_path': [microbial_genus_models_path_base],
        'use_cond_exp': [True],
        'input_current_t': [input_current_t],
        'periodic_current_t': [True],
        'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
        'add_readout_activation': [add_readout_activation], # ('softmax',['id']) ('sum2one',['id'])
        'add_dynamic_cov': [True],
        'pre-train': [1],
        'zero_weight_init': [True],
    }
# param_list_microbial_genus_base += get_parameter_array(param_dict=param_dict_microbial_genus_1_base)
param_dict_microbial_genus_2_base = {
        'dataset': [dataset],
        'dataset_id': [dataset_id],
        'epochs': [10000],
        'batch_size': [batch_size],
        'save_every': [100],
        'learning_rate': [learning_rate],
        'seed': [seed],
        'hidden_size': [hidden_size_1],
        'bias': [bias],
        'dropout_rate': [dropout_rate],
        'ode_nn': [ode_nn_1],
        'readout_nn': [readout_nn_1],
        'enc_nn': [enc_nn_1],
        'use_rnn': [True],
        'input_sig': [False],
        'func_appl_X': [[]],              # [["power-2", "power-3", "power-4"]]
        'add_pred': [[]],
        'test': [test],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor],
        'weight': [0.5],
        # 'weight_evolve': [weight_evolve],
        'plot': [True],
        'which_loss': ['easy'],
        'which_val_loss': ['standard'],
        'evaluate': [False],
        'eval_metrics': [eval_metrics],
        'paths_to_plot': [paths_to_plot],
        'plot_variance': [False],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        'saved_models_path': [microbial_genus_models_path_base],
        'use_cond_exp': [True],
        'input_current_t': [input_current_t],
        'periodic_current_t': [True],
        'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
        'add_readout_activation': [add_readout_activation], # ('softmax',['id']) ('sum2one',['id'])
        'add_dynamic_cov': [True],
        'pre-train': [pre_train],
        'zero_weight_init': [False],
    }
# param_list_microbial_genus_base += get_parameter_array(param_dict=param_dict_microbial_genus_2_base)
param_dict_microbial_genus_3_base = {
        'dataset': [dataset],
        'dataset_id': [dataset_id],
        'epochs': [10000],
        'batch_size': [batch_size],
        'save_every': [100],
        'learning_rate': [learning_rate],
        'seed': [seed],
        'hidden_size': [hidden_size_0],
        'bias': [bias],
        'dropout_rate': [dropout_rate],
        'ode_nn': [ode_nn_0],
        'readout_nn': [readout_nn_0],
        'enc_nn': [enc_nn_0],
        'use_rnn': [False],
        'input_sig': [True],
        'residual_enc_dec': [True],
        'func_appl_X': [[]],              # [["power-2", "power-3", "power-4"]]
        'add_pred': [[]],
        'test': [test],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor],
        'weight': [0.5],
        # 'weight_evolve': [weight_evolve],
        'plot': [True],
        'which_loss': ['easy'],
        'which_val_loss': ['standard'],
        'evaluate': [False],
        'eval_metrics': [eval_metrics],
        'paths_to_plot': [paths_to_plot],
        'plot_variance': [False],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        'saved_models_path': [microbial_genus_models_path_base],
        'use_cond_exp': [True],
        'input_current_t': [input_current_t],
        'periodic_current_t': [True],
        'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
        'add_readout_activation': [add_readout_activation], # ('softmax',['id']) ('sum2one',['id'])
        'add_dynamic_cov': [True],
        # 'pre-train': [10000],
        'zero_weight_init': [True]
    }
# param_list_microbial_genus_base += get_parameter_array(param_dict=param_dict_microbial_genus_3_base)
param_dict_microbial_genus_4_base = {
        'dataset': [dataset],
        'dataset_id': [dataset_id],
        'epochs': [10000],
        'batch_size': [batch_size],
        'save_every': [100],
        'learning_rate': [learning_rate],
        'seed': [seed],
        'hidden_size': [hidden_size_0],
        'bias': [bias],
        'dropout_rate': [dropout_rate],
        'ode_nn': [ode_nn_0],
        'readout_nn': [readout_nn_0],
        'enc_nn': [enc_nn_0],
        'use_rnn': [False],
        'input_sig': [False],
        'residual_enc_dec': [True],
        'func_appl_X': [[]],              # [["power-2", "power-3", "power-4"]]
        'add_pred': [[]],
        'test': [test],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor],
        'weight': [0.5],
        # 'weight_evolve': [weight_evolve],
        'plot': [True],
        'which_loss': ['easy'],
        'which_val_loss': ['standard'],
        'evaluate': [False],
        'eval_metrics': [eval_metrics],
        'paths_to_plot': [paths_to_plot],
        'plot_variance': [False],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        'saved_models_path': [microbial_genus_models_path_base],
        'use_cond_exp': [True],
        'input_current_t': [input_current_t],
        'periodic_current_t': [True],
        'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
        'add_readout_activation': [add_readout_activation], # ('softmax',['id']) ('sum2one',['id'])
        'add_dynamic_cov': [True],
        # 'pre-train': [10000],
        'zero_weight_init': [True],
    }
# param_list_microbial_genus_base += get_parameter_array(param_dict=param_dict_microbial_genus_4_base)

param_dict_microbial_genus_5_base = {
        'dataset': [dataset],
        'dataset_id': [dataset_id],
        'epochs': [10000],
        'batch_size': [batch_size],
        'save_every': [100],
        'learning_rate': [learning_rate],
        'seed': [seed],
        'hidden_size': [400],
        'bias': [bias],
        'dropout_rate': [dropout_rate],
        'ode_nn': [((400, 'tanh'), (400, 'relu'))],
        'readout_nn': [((150, 'tanh'), (150, 'tanh'))],
        'enc_nn': [((150, 'tanh'), (150, 'tanh'))],
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
        'plot_variance': [False],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        'saved_models_path': [microbial_genus_models_path_base],
        'use_cond_exp': [True],
        'input_current_t': [input_current_t],
        'periodic_current_t': [True],
        'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
        'add_readout_activation': [add_readout_activation], # ('softmax',['id']) ('sum2one',['id'])
        'add_dynamic_cov': [True],
        'pre-train': [1],
    }
# param_list_microbial_genus_base += get_parameter_array(param_dict=param_dict_microbial_genus_5_base)

microbial_otu_models_path_base = "{}saved_models_microbial_otu_base/".format(data_path)
param_list_microbial_otu_base = []

for _nn in [nn_0, nn_1, nn_2, nn_0b, nn_1b, nn_2b]:
    param_dict_microbial_otu_mult_base = {
            'dataset': ["microbial_otu"],
            'dataset_id': ["all"],
            'epochs': [3000],
            'batch_size': [batch_size],
            'save_every': [1],
            'learning_rate': [learning_rate],
            'seed': [seed],
            'hidden_size': [hidden_size_0, hidden_size_1],
            'bias': [bias],
            'dropout_rate': [dropout_rate],
            'ode_nn': [_nn],
            'readout_nn': [_nn],
            'enc_nn': [_nn],
            'use_rnn': [False],
            'input_sig': [True],
            'func_appl_X': [[]],              # [["power-2", "power-3", "power-4"]]
            'add_pred': [[]],
            'test': [test],
            'solver': [solver],
            'solver_delta_t_factor': [solver_delta_t_factor],
            'weight': [0.5],
            # 'weight_evolve': [weight_evolve],
            'plot': [True],
            'which_loss': ['easy'],
            'which_val_loss': ['standard'],
            'evaluate': [False],
            'eval_metrics': [eval_metrics],
            'paths_to_plot': [paths_to_plot],
            'plot_variance': [False],
            'std_factor': [std_factor],
            'plot_moments': [plot_moments],
            'saved_models_path': [microbial_otu_models_path_base],
            'use_cond_exp': [True],
            'input_current_t': [input_current_t],
            'periodic_current_t': [True],
            'scale_dt': [scale_dt],
            'enc_input_t': [enc_input_t],
            'add_readout_activation': [add_readout_activation], # ('softmax',['id']) ('sum2one',['id'])
            'add_dynamic_cov': [True],
            'pre-train': [1],
            'zero_weight_init': [True],
        }
    param_list_microbial_otu_base += get_parameter_array(param_dict=param_dict_microbial_otu_mult_base)

plot_paths_microbial_genus_base_dict = {
    'model_ids': [12],
    'saved_models_path': "{}saved_models_microbial_genus_base/".format(data_path),
    'which': 'last',
    'paths_to_plot': [0,1,2,3,4],
    'save_extras': {'bbox_inches': 'tight', 'pad_inches': 0.01},
}

# test config
microbial_genus_models_path_test = "{}saved_models_microbial_genus_test/".format(data_path)
param_list_microbial_genus_test = []
param_dict_microbial_genus_1_test = {
        'dataset': [dataset],
        'dataset_id': [dataset_id],
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



# ==============================================================================
# FLORIAN
# ==============================================================================

# ------------------------------------------------------------------------------
# AD for otu
AD_microbial_otu3 = "{}saved_models_microbial_otu3/".format(data_path)
AD_microbial_otu3_ids = [68]
param_list_AD_microbial_otu = []
param_dict_AD_microbial_otu = {
        "dataset": ['microbial_otu_sig_highab'],
        'saved_models_path': [AD_microbial_otu3],
        'load_best': [True,],
        'nb_MC_samples': [10**4],
        'epsilon': [1e-8, 1e-6],
        'verbose': [True],
        'seed': [seed],
}
param_list_AD_microbial_otu += get_parameter_array(
        param_dict=param_dict_AD_microbial_otu)

param_dict_AD_microbial_otu = {
        "dataset": ['microbial_otu_sig_highab'],
        'saved_models_path': [AD_microbial_otu3],
        'load_best': [True,],
        'nb_MC_samples': [10**4],
        'epsilon': [1e-6],
        'verbose': [True],
        'seed': [seed],
        'use_replace_values': [True],
        'dirichlet_use_coord': [1, 4, 11, 65],
}
param_list_AD_microbial_otu1 = get_parameter_array(
        param_dict=param_dict_AD_microbial_otu)


# ------------------------------------------------------------------------------
# AD for genus
AD_microbial_genus3 = "{}saved_models_microbial_genus3/".format(data_path)
AD_microbial_genus3_ids = []
param_list_AD_microbial_genus = []
param_dict_AD_microbial_genus = {
        "dataset": ['microbial_genus_sig_highab'],
        'saved_models_path': [AD_microbial_genus3],
        'load_best': [True,],
        'nb_MC_samples': [10**4],
        'epsilon': [1e-8, 1e-6],
        'verbose': [True],
        'seed': [seed],
}
param_list_AD_microbial_genus += get_parameter_array(
        param_dict=param_dict_AD_microbial_genus)




