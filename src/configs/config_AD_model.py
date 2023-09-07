import numpy as np

from configs.config_utils import get_parameter_array, get_dataset_overview, \
    makedirs, data_path, training_data_path

AD_OrnsteinUhlenbeckWithSeason_models_path = "{}saved_models_AD_OrnsteinUhlenbeckWithSeason/".format(data_path)
'''

param_list_AD_OrnsteinUhlenbeckWithSeason_3 = []
for size in [50,100]:
    nn = ((size, 'tanh'), (size, 'relu'))
    param_dict_AD_OrnsteinUhlenbeckWithSeason_3_1 = {
        'epochs': [25],
        'batch_size': [200],
        'save_every': [1],
        'learning_rate': [0.001, 0.005],
        'test_size': [0.2],
        'seed': [398],
        'hidden_size': [50, 100],
        'bias': [True],
        'dropout_rate': [0.1],
        'ode_nn': [nn],
        'readout_nn': [nn],
        'enc_nn': [nn],
        'use_rnn': [False, True],
        'input_sig': [True],
        'func_appl_X': [["power-2"]],              # [["power-2", "power-3", "power-4"]]
        'solver': ["euler"],
        'weight': [0.5],
        'weight_decay': [1.],
        'data_dict': ['AD_OrnsteinUhlenbeckWithSeason_3_dict'],
        'plot': [True],
        'which_loss': ['easy'],
        'evaluate': [False],
        'paths_to_plot': [(0,1,2,3,4,)],
        'plot_variance': [True],
        'std_factor': [1.96],
        'plot_moments': [True],
        'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_3_models_path],
        'use_cond_exp': [True],
        'input_current_t': [True],
        'current_period': [True],
        'enc_input_t': [False],
    }
    param_list_AD_OrnsteinUhlenbeckWithSeason_3 += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_3_1)

'''

ode_nn_1 = ((300, 'tanh'), (300, 'relu'))
ode_nn_2 = ((64, 'tanh'), (128, 'tanh'), (64, 'relu'))
enc_nn = ((200, 'tanh'), (200, 'tanh'))
readout_nn = ((200, 'tanh'), (200, 'tanh'))
weight = 1.
weight_decay = 0.99
hidden_size = 150
batch_size = 200
param_list_AD_OrnsteinUhlenbeckWithSeason = []
param_dict_AD_OrnsteinUhlenbeckWithSeason_1 = {
        'epochs': [100],
        'batch_size': [200],
        'save_every': [1],
        'learning_rate': [0.001],
        'test_size': [0.2],
        'seed': [398],
        'hidden_size': [150],
        'bias': [True],
        'dropout_rate': [0.1],
        'ode_nn': [ode_nn_1],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [True],
        'input_sig': [True],
        'func_appl_X': [["power-2"]],              # [["power-2", "power-3", "power-4"]]
        'add_pred': [[]],
        'solver': ["euler"],
        'solver_delta_t_factor': [1],
        'weight': [1.],
        'weight_decay': [0.99],
        'data_dict': ['AD_OrnsteinUhlenbeckWithSeason_3_dict'],
        'plot': [True],
        'which_loss': ['ad'],
        'evaluate': [False],
        'paths_to_plot': [(0,1,2,3,4,)],
        'plot_variance': [True],
        'std_factor': [1.96],
        'plot_moments': [True],
        'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_models_path],
        'use_cond_exp': [True],
        'input_current_t': [True],
        'periodic_current_t': [True],
        'train_data_perc': [0.1],
        'scale_dt': ['automatic'],            # (obs_perc / dt) 
        'enc_input_t': [False],
        # 'data_scaling_factor': [10],
        # 'add_readout_activation': [('elu',['var'])]
    }
# param_list_AD_OrnsteinUhlenbeckWithSeason += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_1)
param_dict_AD_OrnsteinUhlenbeckWithSeason_2 = {
        'epochs': [50],
        'batch_size': [200],
        'save_every': [1],
        'learning_rate': [0.001],
        'test_size': [0.2],
        'seed': [398],
        'hidden_size': [150],
        'bias': [True],
        'dropout_rate': [0.1],
        'ode_nn': [ode_nn_1],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [True],
        'input_sig': [True],
        'func_appl_X': [["power-2"]],              # [["power-2", "power-3", "power-4"]]
        'add_pred': [['var']],
        'solver': ["euler"],
        'solver_delta_t_factor': [1],
        'weight': [1.],
        'weight_decay': [0.99],
        'data_dict': ['AD_OrnsteinUhlenbeckWithSeason_3_dict'],
        'plot': [True],
        'which_loss': ['ad_var2'],
        'evaluate': [False],
        'paths_to_plot': [(0,1,2,3,4,)],
        'plot_variance': [True],
        'std_factor': [1.96],
        'plot_moments': [True],
        'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_models_path],
        'use_cond_exp': [True],
        'input_current_t': [True],
        'periodic_current_t': [True],
        'train_data_perc': [0.1],
        'scale_dt': ['automatic'],            # (obs_perc / dt) 
        'enc_input_t': [False],
        'add_readout_activation': [('identity',['var'])]
    }
# param_list_AD_OrnsteinUhlenbeckWithSeason += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_2)
param_dict_AD_OrnsteinUhlenbeckWithSeason_3 = {
        'epochs': [50],
        'batch_size': [200],
        'save_every': [1],
        'learning_rate': [0.001],
        'test_size': [0.2],
        'seed': [398],
        'hidden_size': [150],
        'bias': [True],
        'dropout_rate': [0.1],
        'ode_nn': [ode_nn_1],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [True],
        'input_sig': [True],
        'func_appl_X': [["power-2"]],              # [["power-2", "power-3", "power-4"]]
        'add_pred': [['var']],
        'solver': ["euler"],
        'solver_delta_t_factor': [1],
        'weight': [0],
        'weight_evolve': [{'type':'linear', 'target': 1, 'reach': None}], # [{'type':'decay', 'decay': 0.99}],
        #'weight_decay': [0.99],
        'data_dict': ['AD_OrnsteinUhlenbeckWithSeason_3_dict'],
        'plot': [True],
        'which_loss': ['ad_var'],
        'evaluate': [True],
        'evaluate_vars': [['id','var']],
        'paths_to_plot': [(0,1,2,3,4,)],
        'plot_variance': [True],
        'std_factor': [1.96],
        'plot_moments': [True],
        'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_models_path],
        'use_cond_exp': [True],
        'input_current_t': [True],
        'periodic_current_t': [True],
        # 'training_size': [200],
        'validation_size': [200],
        'train_data_perc': [0.1],
        'scale_dt': ['automatic'],            # (obs_perc / dt) 
        'enc_input_t': [False],
        # 'data_scaling_factor': [10],
        'add_readout_activation': [('identity',['var'])]
    }
param_list_AD_OrnsteinUhlenbeckWithSeason += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_3)
param_dict_AD_OrnsteinUhlenbeckWithSeason_4 = {
        'epochs': [50],
        'batch_size': [200],
        'save_every': [1],
        'learning_rate': [0.001],
        'test_size': [0.2],
        'seed': [398],
        'hidden_size': [150],
        'bias': [True],
        'dropout_rate': [0.1],
        'ode_nn': [ode_nn_1],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [True],
        'input_sig': [True],
        'func_appl_X': [["power-2"]],              # [["power-2", "power-3", "power-4"]]
        'add_pred': [[]],
        'solver': ["euler"],
        'solver_delta_t_factor': [1],
        'weight': [0.5],
        'weight_decay': [1.],
        'data_dict': ['AD_OrnsteinUhlenbeckWithSeason_3_dict'],
        'plot': [True],
        'which_loss': ['easy'],
        'evaluate': [False],
        'paths_to_plot': [(0,1,2,3,4,)],
        'plot_variance': [True],
        'std_factor': [1.96],
        'plot_moments': [True],
        'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_models_path],
        'use_cond_exp': [True],
        'input_current_t': [True],
        'periodic_current_t': [True],
        'train_data_perc': [0.1],
        'scale_dt': ['automatic'],            # (obs_perc / dt) 
        'enc_input_t': [False],
        # 'add_readout_activation': [('elu',['var'])]
    }
# param_list_AD_OrnsteinUhlenbeckWithSeason += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_4)
param_dict_AD_OrnsteinUhlenbeckWithSeason_5 = {
        'epochs': [50],
        'batch_size': [200],
        'save_every': [1],
        'learning_rate': [0.001],
        'test_size': [0.2],
        'seed': [398],
        'hidden_size': [150],
        'bias': [True],
        'dropout_rate': [0.1],
        'ode_nn': [ode_nn_1],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [True],
        'input_sig': [True],
        'func_appl_X': [["power-2"]],              # [["power-2", "power-3", "power-4"]]
        'add_pred': [[]],
        'solver': ["euler"],
        'solver_delta_t_factor': [1],
        'weight': [0.5],
        'weight_decay': [1.],
        'data_dict': ['AD_OrnsteinUhlenbeckWithSeason_3_dict'],
        'plot': [True],
        'which_loss': ['easy_bis'],
        'evaluate': [False],
        'paths_to_plot': [(0,1,2,3,4,)],
        'plot_variance': [True],
        'std_factor': [1.96],
        'plot_moments': [True],
        'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_models_path],
        'use_cond_exp': [True],
        'input_current_t': [True],
        'periodic_current_t': [True],
        'train_data_perc': [0.1],
        'scale_dt': ['automatic'],            # (obs_perc / dt) 
        'enc_input_t': [False],
        # 'plot_only': [True]
        # 'add_readout_activation': [('elu',['var'])]
    }
# param_list_AD_OrnsteinUhlenbeckWithSeason += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_5)


plot_paths_AD_OrnsteinUhlenbeckWithSeason_dict = {
    'model_ids': [1],
    'saved_models_path': "{}saved_models_AD_OrnsteinUhlenbeckWithSeason_dim2/".format(data_path),
    'which': 'last', 
    'paths_to_plot': [0,1,2,3,4],
    'save_extras': {'bbox_inches': 'tight', 'pad_inches': 0.01},
}


AD_module_path_noise = "{}saved_AD_module_noise/".format(data_path)
param_dict_AD_modules_noise = []
param_dict_AD_modules_noise_1 = {
    'steps_ahead': [[1,3,5,7,10,12,15,18,20]],
    'test_size': [0.2],
    'batch_size': [100],
    'seed': [398],
    'learning_rate': [0.01],
    'optim_method': ['adam'],
    'anomaly_data_dict': ['AD_OrnsteinUhlenbeckWithSeason_noise_dict'],
    'epochs': [50],
    'class_thres': [0.5], # 'automatic'
    'autom_thres': [None], #['FPR_limit-0.05'],
    'saved_models_path': [AD_module_path_noise],
    'paths_to_plot': [(0,1,2,3,4,)],
    'plot_forecast_predictions': [False],
    'forecast_horizons_to_plot': [(10,)],
    'plot_variance': [True],
    'std_factor': [1.96],
}
param_dict_AD_modules_noise += get_parameter_array(param_dict=param_dict_AD_modules_noise_1)


AD_module_path_cutoff = "{}saved_AD_module_cutoff/".format(data_path)
param_dict_AD_modules_cutoff = []
param_dict_AD_modules_cutoff_1 = {
    'steps_ahead': [[1,3,5,7,10,12,15,18,20]],
    'test_size': [0.2],
    'batch_size': [100],
    'seed': [398],
    'learning_rate': [0.01],
    'optim_method': ['adam'],
    'anomaly_data_dict': ['AD_OrnsteinUhlenbeckWithSeason_cutoff_dict'],
    'epochs': [50],
    'class_thres': [0.5], # 'automatic'
    'autom_thres': [None], #['FPR_limit-0.05'],
    'saved_models_path': [AD_module_path_cutoff],
    'paths_to_plot': [(0,1,2,3,4,)],
    'plot_forecast_predictions': [False],
    'forecast_horizons_to_plot': [(10,)],
    'plot_variance': [True],
    'std_factor': [1.96],
}
param_dict_AD_modules_cutoff += get_parameter_array(param_dict=param_dict_AD_modules_cutoff_1)

AD_module_path_diffusion = "{}saved_AD_module_cutoff/".format(data_path)
param_dict_AD_modules_diffusion = []
param_dict_AD_modules_diffusion_1 = {
    'steps_ahead': [[1,3,5,7,10,12,15,18,20]],
    'test_size': [0.2],
    'batch_size': [100],
    'seed': [398],
    'learning_rate': [0.01],
    'optim_method': ['adam'],
    'anomaly_data_dict': ['AD_OrnsteinUhlenbeckWithSeason_diffusion_dict'],
    'epochs': [50],
    'class_thres': [0.5], # 'automatic'
    'autom_thres': [None], #['FPR_limit-0.05'],
    'saved_models_path': [AD_module_path_diffusion],
    'paths_to_plot': [(0,1,2,3,4,)],
    'plot_forecast_predictions': [False],
    'forecast_horizons_to_plot': [(10,)],
    'plot_variance': [True],
    'std_factor': [1.96],
}
param_dict_AD_modules_diffusion += get_parameter_array(param_dict=param_dict_AD_modules_diffusion_1)

AD_module_path_spike = "{}saved_AD_module_spike/".format(data_path)
param_dict_AD_modules_spike = []
param_dict_AD_modules_spike_1 = {
    'steps_ahead': [[1,3,5,7,10,12,15,18,20]],
    'test_size': [0.2],
    'batch_size': [100],
    'seed': [398],
    'learning_rate': [0.01],
    'optim_method': ['adam'],
    'anomaly_data_dict': ['AD_OrnsteinUhlenbeckWithSeason_spike_dict'],
    'epochs': [50],
    'class_thres': [0.5], # 'automatic'
    'autom_thres': [None], #['FPR_limit-0.05'],
    'saved_models_path': [AD_module_path_spike],
    'paths_to_plot': [(0,1,2,3,4,)],
    'plot_forecast_predictions': [False],
    'forecast_horizons_to_plot': [(10,)],
    'plot_variance': [True],
    'std_factor': [1.96],
}
param_dict_AD_modules_spike += get_parameter_array(param_dict=param_dict_AD_modules_spike_1)


overview_dict_AD_OrnsteinUhlenbeckWithSeason = dict(
    ids_from=1, ids_to=len(param_list_AD_OrnsteinUhlenbeckWithSeason),
    path=AD_OrnsteinUhlenbeckWithSeason_models_path,
    params_extract_desc=('dataset', 'network_size', 'nb_layers',
                         'activation_function_1', 'use_rnn',
                         'readout_nn', 'dropout_rate',
                         'hidden_size', 'batch_size', 'which_loss',
                         'input_sig', 'level'),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "evaluation_mean_diff",
         "evaluation_mean_diff", "evaluation_mean_diff_min"),
        ("min", "eval_loss", "eval_loss", "eval_loss_min"),
    ),
    sortby=["evaluation_mean_diff_min"],
)