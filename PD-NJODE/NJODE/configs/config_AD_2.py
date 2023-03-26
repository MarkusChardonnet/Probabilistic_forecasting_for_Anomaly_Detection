import numpy as np

from configs.config_utils import get_parameter_array, get_dataset_overview, \
    makedirs, data_path, training_data_path

AD_OrnsteinUhlenbeckWithSeason_1_dict = {
    'model_name': "AD_OrnsteinUhlenbeckWithSeason",
    'nb_paths': 1000, 'nb_steps': 200,
    'dimension': 1, 'obs_perc': 0.1, 'S0': None,
    'maturity': 1., 'return_vol': False,
    'speed': 15., 
    'noise': {'type': "gaussian", 'cov': 0.05},
    'volatility': {
        'vol_value': [[0.3,0.1],[0.1,0.3]],
        # 'vol_path': "configs/models/AD_OrnsteinUhlenBeck_volatility.pkl"
    },
    'initial_value': False,
    'season_params': {
        'seas_amp': 0.2,
        'seas_length': 200,
        'seas_nb': 2,
        'depth': 8,
        'forking_depth': 0,
        'seas_type': 'RMDF',
        'new_seas': True,
        'seas_path': "configs/models/AD_OrnsteinUhlenBeck_RMDF_season",
        'changing_seas': False,
    },
    'anomaly_params': {
        'type': None
    }
}

AD_OrnsteinUhlenbeckWithSeason_2_dict = {
    'model_name': "AD_OrnsteinUhlenbeckWithSeason",
    'nb_paths': 100000, 'nb_steps': 400,
    'dimension': 1, 'obs_perc': 0.1, 'S0': None,
    'maturity': 1., 'return_vol': False,
    'speed': 15., 
    'noise': {'type': "gaussian", 'cov': 0.0},   # 0.02
    'volatility': {
        'vol_value': [[0.3]], # [[0.3,0.15],[0.15,0.3]],
    },
    'initial_value': False,
    'season_params': {
        'seas_amp': 'automatic', # 0.1,
        'seas_range': (0.2,0.8),
        'seas_length': 200,
        'seas_nb': 2,
        'seas_type': 'NN',
        'nn_layers': [(16, 'tanh'),(16, 'tanh')], # [(32, 'tanh')], #
        'nn_bias': True,
        'nn_input': (['cos', 'sin'], 1),
        'new_seas': False,
        'seas_path': "configs/models/AD_OrnsteinUhlenBeck_NN_season_2",
        'changing_seas': False,
    },
    'anomaly_params': {
        'type': None
    }
}

AD_OrnsteinUhlenbeckWithSeason_3_dict = {
    'model_name': "AD_OrnsteinUhlenbeckWithSeason",
    'nb_paths': 100000, 'nb_steps': 400,
    'dimension': 1, 'obs_perc': 1., 'S0': None,
    'maturity': 1., 'return_vol': False,
    'speed': 15., 
    'noise': {'type': "gaussian", 'cov': 0.0},   # 0.02
    'volatility': {
        'vol_value': 0.3, # [[0.3,0.15],[0.15,0.3]],
    },
    'initial_value': False,
    'season_params': {
        'seas_amp': 'automatic', # 0.1,
        'seas_range': (0.2,0.8),
        'seas_length': 200,
        'seas_nb': 2,
        'seas_type': 'NN',
        'nn_layers': [(16, 'tanh'),(16, 'tanh')], # [(32, 'tanh')], #
        'nn_bias': True,
        'nn_input': (['cos', 'sin'], 1),
        'new_seas': False,
        'seas_path': "configs/models/AD_OrnsteinUhlenBeck_NN_season_2",
        'changing_seas': False,
    },
    'anomaly_params': {
        'type': None
    }
}

AD_OrnsteinUhlenbeckWithSeason_test_dict = {
    'model_name': "AD_OrnsteinUhlenbeckWithSeason",
    'nb_paths': 500, 'nb_steps': 400,
    'dimension': 1, 'obs_perc': 0.1, 'S0': None,
    'maturity': 1., 'return_vol': False,
    'speed': 15., 
    'noise': {'type': "gaussian", 'cov': 0.0},   # 0.02
    'volatility': {
        'vol_value': 0.15, # [[0.3,0.15],[0.15,0.3]],
    },
    'initial_value': False,
    'season_params': {
        'seas_amp': 'automatic', # 0.1,
        'seas_range': (0.2,0.8),
        'seas_length': 200,
        'seas_nb': 2,
        'seas_type': 'NN',
        'nn_layers': [(16, 'tanh'),(16, 'tanh')], # [(32, 'tanh')],
        'nn_bias': True,
        'nn_input': (['cos', 'sin'], 1),
        'new_seas': True,
        'seas_path': "configs/models/AD_OrnsteinUhlenBeck_NN_season_test",
        'changing_seas': False,
    },
    'anomaly_params': {
        'type': None
    }
}


AD_OrnsteinUhlenbeckWithSeason_3_models_path = "{}saved_models_AD_OrnsteinUhlenbeckWithSeason_3/".format(data_path)
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

nn = ((150, 'tanh'), (150, 'relu'))
param_list_AD_OrnsteinUhlenbeckWithSeason_3 = []
param_dict_AD_OrnsteinUhlenbeckWithSeason_3_1 = {
        'epochs': [100],
        'batch_size': [200],
        'save_every': [1],
        'learning_rate': [0.001],
        'test_size': [0.2],
        'seed': [398],
        'hidden_size': [100],
        'bias': [True],
        'dropout_rate': [0.1],
        'ode_nn': [nn],
        'readout_nn': [nn],
        'enc_nn': [nn],
        'use_rnn': [False],
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



overview_dict_AD_OrnsteinUhlenbeckWithSeason_3 = dict(
    ids_from=1, ids_to=len(param_list_AD_OrnsteinUhlenbeckWithSeason_3),
    path=AD_OrnsteinUhlenbeckWithSeason_3_models_path,
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
plot_paths_AD_OrnsteinUhlenbeckWithSeason_3_dict = {
    'model_ids': [33, 34, 35, 41, 43, 50],
    'saved_models_path': AD_OrnsteinUhlenbeckWithSeason_3_models_path,
    'which': 'best', 'paths_to_plot': [0,1,2,3,4,5],
    'save_extras': {'bbox_inches': 'tight', 'pad_inches': 0.01},}