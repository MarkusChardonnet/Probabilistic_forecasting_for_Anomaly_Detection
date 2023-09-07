import numpy as np

from configs.config_utils import get_parameter_array, get_dataset_overview, \
    makedirs, data_path, training_data_path


AD_GLISSM_1_dict = {
    'model_name': "AD_GLISSM",
    'nb_paths': 10000, 'nb_steps': 1000,
    'dimension': 1, 'obs_perc': 0.1,
    'd_state': 5, 'd_noise': 1,
    'model_file': 'AD_GLISSM_1.pt',
    'maturity': 1000, 'return_vol': False
}

ode_nn = ((50, 'tanh'), (50, 'tanh'))
readout_nn = ((50, 'tanh'), (50, 'tanh'))
enc_nn = ((50, 'tanh'), (50, 'tanh'))

AD_GLISSM_models_path = "{}saved_models_AD_GLISSM/".format(data_path)

param_list_AD_GLISSM2 = []
param_dict_AD_GLISSM2_1 = {
    'epochs': [1],
    'batch_size': [200],
    'save_every': [1],                    # wrt to epochs -> to save last model (not best) + metric file 
    'learning_rate': [0.001],
    'test_size': [0.2],
    'seed': [398],
    'hidden_size': [10, 50],
    'bias': [True],
    'dropout_rate': [0.1],               # all networks + layers
    'ode_nn': [ode_nn],
    'readout_nn': [readout_nn],
    'enc_nn': [enc_nn],
    'use_rnn': [True],
    'input_sig': [True],
    'func_appl_X': [[]],              # [["power-2", "power-3", "power-4"]]
    'solver': ["euler"],                # ode solver, euler only one implemented
    'weight': [0.5],
    'weight_decay': [1.],                  # balance between loss components, 
    'data_dict': ['AD_GLISSM_1_dict'],
    'plot': [True],
    'which_loss': ['easy'],
    'evaluate': [True],
    'paths_to_plot': [(0,1,2,3,4,)],
    'saved_models_path': [AD_GLISSM_models_path],
    'use_cond_exp': [False]
}
param_list_AD_GLISSM2 += get_parameter_array(param_dict=param_dict_AD_GLISSM2_1)

# ???
overview_dict_AD_GLISSM2 = dict(
    ids_from=1, ids_to=len(param_list_AD_GLISSM2),
    path=AD_GLISSM_models_path,
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

plot_paths_AD_GLISSM_dict = {
    'model_ids': [33, 34, 35, 41, 43, 50],
    'saved_models_path': AD_GLISSM_models_path,
    'which': 'best', 'paths_to_plot': [0,1,2,3,4,5],
    'save_extras': {'bbox_inches': 'tight', 'pad_inches': 0.01},}


AD_TSAGen_1_dict = {
    'model_name': "AD_TSAGen",
    'nb_paths': 1000, 'nb_steps': 200,
    'dimension': 1, 'obs_perc': 0.1,
    'maturity': 1., 'return_vol': False,
    'noise_type': "gaussian", 
    'shape_path': "configs/models/TSAGen_season_generator.pkl", 'new_shape': True,
    'parameters': {
        'trend_level': 0.5,
        'trend_slope': 0.,
        'seas_amp': 0.2,
        'seas_length': 200,
        'seas_nb': 1,
        'noise_mu': 0.,
        'noise_std': 0.01,
        'noise_skew': 0.,
        'noise_kurt': 1.,
        'depth': 10,
        'forking_depth': 6,
        'drift_amp': 0.2,
        'drift_freq': 0.2,
    },
    'anomaly_params': {
        'type': None
    }
}

AD_TSAGen_1_variance_dict = {
    'model_name': "AD_TSAGen",
    'nb_paths': 20, 'nb_steps': 200,
    'dimension': 1, 'obs_perc': 1,
    'maturity': 1., 'return_vol': False,
    'noise_type': "gaussian", 
    'shape_path': "configs/models/TSAGen_season_generator.pkl", 'new_shape': False,
    'parameters': {
        'trend_level': 0.5,
        'trend_slope': 0.,
        'seas_amp': 0.2,
        'seas_length': 200,
        'seas_nb': 10,
        'noise_mu': 0.,
        'noise_std': 0.01,
        'noise_skew': 0.,
        'noise_kurt': 1.,
        'depth': 10,
        'forking_depth': 6,
        'drift_amp': 0.2,
        'drift_freq': 0.2,
    },
    'anomaly_params': {
        'type': 'variance',
        'occurence_law': 'single',
        'occurence_prob': 0.5,
        'occurence_range': (0.1,0.9),
        'occurence_pos_law': 'uniform',
        'length_law': 'uniform',
        'length_range': (20,80),
        'variance_factor_law': 'uniform',
        'variance_factor_range': (2,5)
    }
}

AD_TSAGen_1_deformation_dict = {
    'model_name': "AD_TSAGen",
    'nb_paths': 20, 'nb_steps': 200,
    'dimension': 1, 'obs_perc': 1,
    'maturity': 1., 'return_vol': False,
    'noise_type': "gaussian", 
    'shape_path': "configs/models/TSAGen_season_generator.pkl", 'new_shape': False,
    'parameters': {
        'trend_level': 0.5,
        'trend_slope': 0.,
        'seas_amp': 0.2,
        'seas_length': 200,
        'seas_nb': 10,
        'noise_mu': 0.,
        'noise_std': 0.01,
        'noise_skew': 0.,
        'noise_kurt': 1.,
        'depth': 10,
        'forking_depth': 6,
        'drift_amp': 0.2,
        'drift_freq': 0.2,
    },
    'anomaly_params': {
        'type': 'deformation',
        'occurence_law': 'single',
        'occurence_prob': 0.5,
        'occurence_range': (1,9),
        'occurence_pos_law': 'uniform',
        'forking_depth_law': 'delta',
        'forking_depth': 9
    }
}

AD_TSAGen_2_dict = {
    'model_name': "AD_TSAGen",
    'nb_paths': 10000, 'nb_steps': 1000,
    'dimension': 1, 'obs_perc': 0.1,
    'maturity': 1000, 'return_vol': False,
    'noise_type': "gaussian", 
    'shape_path': "configs/models/TSAGen_season_generator.pkl", 'new_shape': True,
    'parameters': {
        'trend_level': 0.5,
        'trend_slope': 0.,
        'seas_amp': 0.2,
        'seas_length': 200,
        'seas_nb': 10,
        'noise_mu': 0.,
        'noise_std': 0.01,
        'noise_skew': 0.,
        'noise_kurt': 1.,
        'depth': 10,
        'forking_depth': 6,
        'drift_amp': 0.2,
        'drift_freq': 0.2,
    },
    'anomaly_params': {
        'type': None
    }
}

AD_TSAGen_2_variance_dict = {
    'model_name': "AD_TSAGen",
    'nb_paths': 20, 'nb_steps': 1000,
    'dimension': 1, 'obs_perc': 1,
    'maturity': 1000, 'return_vol': False,
    'noise_type': "gaussian", 
    'shape_path': "configs/models/TSAGen_season_generator.pkl", 'new_shape': False,
    'parameters': {
        'trend_level': 0.5,
        'trend_slope': 0.,
        'seas_amp': 0.2,
        'seas_length': 1000,
        'seas_nb': 10,
        'noise_mu': 0.,
        'noise_std': 0.01,
        'noise_skew': 0.,
        'noise_kurt': 1.,
        'depth': 10,
        'forking_depth': 6,
        'drift_amp': 0.2,
        'drift_freq': 0.2,
    },
    'anomaly_params': {
        'type': 'variance',
        'occurence_law': 'single',
        'occurence_prob': 0.5,
        'occurence_range': (0.1,0.9),
        'occurence_pos_law': 'uniform',
        'length_law': 'uniform',
        'length_range': (20,80),
        'variance_factor_law': 'uniform',
        'variance_factor_range': (2,5)
    }
}

AD_TSAGen_2_deformation_dict = {
    'model_name': "AD_TSAGen",
    'nb_paths': 20, 'nb_steps': 1000,
    'dimension': 1, 'obs_perc': 1,
    'maturity': 1000, 'return_vol': False,
    'noise_type': "gaussian", 
    'shape_path': "configs/models/TSAGen_season_generator.pkl", 'new_shape': False,
    'parameters': {
        'trend_level': 0.5,
        'trend_slope': 0.,
        'seas_amp': 0.2,
        'seas_length': 1000,
        'seas_nb': 10,
        'noise_mu': 0.,
        'noise_std': 0.01,
        'noise_skew': 0.,
        'noise_kurt': 1.,
        'depth': 10,
        'forking_depth': 6,
        'drift_amp': 0.2,
        'drift_freq': 0.2,
    },
    'anomaly_params': {
        'type': 'deformation',
        'occurence_law': 'single',
        'occurence_prob': 0.5,
        'occurence_range': (1,9),
        'occurence_pos_law': 'uniform',
        'forking_depth_law': 'delta',
        'forking_depth': 9
    }
}

AD_TSAGen_models_path = "{}saved_models_AD_TSAGen/".format(data_path)

param_list_AD_TSAGen1 = []
param_dict_AD_TSAGen1_1 = {
    'epochs': [200],
    'batch_size': [50],
    'save_every': [1],                    # wrt to epochs -> to save last model (not best) + metric file 
    'learning_rate': [0.005],
    'test_size': [0.2],
    'seed': [398],
    'hidden_size': [50],
    'bias': [True],
    'dropout_rate': [0.1],               # all networks + layers
    'ode_nn': [ode_nn],
    'readout_nn': [readout_nn],
    'enc_nn': [enc_nn],
    'use_rnn': [True],
    'input_sig': [True],
    'func_appl_X': [["power-2"]],              # [["power-2", "power-3", "power-4"]]
    'solver': ["euler"],                # ode solver, euler only one implemented
    'weight': [0.5],
    'weight_decay': [1.],                  # balance between loss components, 
    'data_dict': ['AD_TSAGen_1_dict'],
    'plot': [True],
    'which_loss': ['easy'],
    'evaluate': [True],
    'paths_to_plot': [(0,1,2,3,4,)],
    'saved_models_path': [AD_TSAGen_models_path],
    'use_cond_exp': [False]
}
param_list_AD_TSAGen1 += get_parameter_array(param_dict=param_dict_AD_TSAGen1_1)

AD_TSAGen2_models_path = "{}saved_models_AD_TSAGen2/".format(data_path)

nn = ((100, 'tanh'), (100, 'relu'))

param_list_AD_TSAGen2 = []
param_dict_AD_TSAGen2_1 = {
    'epochs': [200],
    'batch_size': [200],
    'save_every': [1],                    # wrt to epochs -> to save last model (not best) + metric file 
    'learning_rate': [0.001],
    'test_size': [0.2],
    'seed': [398],
    'hidden_size': [100],
    'bias': [True],
    'dropout_rate': [0.1],               # all networks + layers
    'ode_nn': [nn],
    'readout_nn': [nn],
    'enc_nn': [nn],
    'use_rnn': [True],
    'input_sig': [True],
    'func_appl_X': [["power-2"]],              # [["power-2", "power-3", "power-4"]]
    'solver': ["euler"],                # ode solver, euler only one implemented
    'weight': [0.5],
    'weight_decay': [1.],                  # balance between loss components, 
    'data_dict': ['AD_TSAGen_2_dict'],
    'plot': [True],
    'which_loss': ['easy'],
    'evaluate': [True],
    'paths_to_plot': [(0,1,2,3,4,)],
    'saved_models_path': [AD_TSAGen2_models_path],
    'use_cond_exp': [False]
}
param_list_AD_TSAGen2 += get_parameter_array(param_dict=param_dict_AD_TSAGen2_1)

overview_dict_AD_TSAGen1 = dict(
    ids_from=1, ids_to=len(param_list_AD_TSAGen1),
    path=AD_TSAGen_models_path,
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

overview_dict_AD_TSAGen2 = dict(
    ids_from=1, ids_to=len(param_list_AD_TSAGen2),
    path=AD_TSAGen2_models_path,
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

plot_paths_AD_TSAGen_dict = {
    'model_ids': [33, 34, 35, 41, 43, 50],
    'saved_models_path': AD_TSAGen_models_path,
    'which': 'best', 'paths_to_plot': [0,1,2,3,4,5],
    'save_extras': {'bbox_inches': 'tight', 'pad_inches': 0.01},}

plot_paths_AD_TSAGen2_dict = {
    'model_ids': [33, 34, 35, 41, 43, 50],
    'saved_models_path': AD_TSAGen2_models_path,
    'which': 'best', 'paths_to_plot': [0,1,2,3,4,5],
    'save_extras': {'bbox_inches': 'tight', 'pad_inches': 0.01},}

