import numpy as np
import copy

from configs.config_utils import get_parameter_array, get_dataset_overview, \
    makedirs, data_path, training_data_path


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
        'nn_layers': [(16, 'relu'),(16, 'relu')], # [(32, 'tanh')], #
        'nn_bias': True,
        'nn_input': (['cos', 'sin'], 1),
        'new_seas': False,
        'seas_path': "configs/models/AD_OrnsteinUhlenBeck_NN_season",
        'changing_seas': False,
    },
    'anomaly_params': {
        'type': None
    }
}
AD_OrnsteinUhlenbeckWithSeason_eval = copy.deepcopy(
    AD_OrnsteinUhlenbeckWithSeason_3_dict)
AD_OrnsteinUhlenbeckWithSeason_eval['nb_paths'] = 1500
AD_OrnsteinUhlenbeckWithSeason_plot = copy.deepcopy(
    AD_OrnsteinUhlenbeckWithSeason_3_dict)
AD_OrnsteinUhlenbeckWithSeason_plot['nb_paths'] = 1

AD_OrnsteinUhlenbeckWithSeason_dim2_dict = {
    'model_name': "AD_OrnsteinUhlenbeckWithSeason",
    'nb_paths': 10000, 'nb_steps': 400,
    'dimension': 2, 'obs_perc': 1., 'S0': None,
    'maturity': 1., 'return_vol': False,
    'speed': [[10.,5.],[0.,15.]],
    'noise': {'type': "gaussian", 'cov': 0.0},   # 0.02
    'volatility': {
        'vol_value': [[0.3,0.15],[0.15,0.3]],
    },
    'initial_value': False,
    'season_params': {
        'seas_amp': 'automatic', # 0.1,
        'seas_range': (0.2,0.8),
        'seas_length': 200,
        'seas_nb': 2,
        'seas_type': 'NN',
        'nn_layers': [(16, 'relu'),(16, 'relu')], # [(32, 'tanh')], #
        'nn_bias': True,
        'nn_input': (['cos', 'sin'], 1),
        'new_seas': False,
        'seas_path': "configs/models/AD_OrnsteinUhlenBeck_NN_season_dim2",
        'changing_seas': False,
    },
    'anomaly_params': {
        'type': None
    }
}

AD_OrnsteinUhlenbeckWithSeason_5seas_dict = {
    'model_name': "AD_OrnsteinUhlenbeckWithSeason",
    'nb_paths': 50000, 'nb_steps': 1000,
    'dimension': 1, 'obs_perc': 1., 'S0': None,
    'maturity': 1., 'return_vol': False,
    'speed': 15.,
    'noise': {'type': "gaussian", 'cov': 0.0},   # 0.02
    'volatility': {
        'vol_value': 0.3,
    },
    'initial_value': False,
    'season_params': {
        'seas_amp': 'automatic', # 0.1,
        'seas_range': (0.2,0.8),
        'seas_length': 200,
        'seas_nb': 5,
        'seas_type': 'NN',
        'nn_layers': [(16, 'relu'),(16, 'relu')], # [(32, 'tanh')], #
        'nn_bias': True,
        'nn_input': (['cos', 'sin'], 1),
        'new_seas': False,
        'seas_path': "configs/models/AD_OrnsteinUhlenBeck_NN_season_5seas",
        'changing_seas': False,
    },
    'anomaly_params': {
        'type': None
    }
}

AD_OrnsteinUhlenbeckWithSeason_test_dict = {
    'model_name': "AD_OrnsteinUhlenbeckWithSeason",
    'nb_paths': 500, 'nb_steps': 400,
    'dimension': 3, 'obs_perc': 0.1, 'S0': None,
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

Cloud_KPI_daily_dict = {
    'model_name': "Cloud_KPI",
    'nb_paths': 401, 'nb_steps': 287,
    'dimension': 3, 'obs_perc': 1., 'S0': None,
    'maturity': 1.,
}

Cloud_KPI_daily_transformed_dict = {
    'model_name': "Cloud_KPI",
    'nb_paths': 401, 'nb_steps': 287,
    'dimension': 3, 'obs_perc': 1., 'S0': None,
    'maturity': 1., "input_transformation": "log,0.5"
}

Cloud_KPI_daily_transformed2_dict = {
    'model_name': "Cloud_KPI",
    'nb_paths': 401, 'nb_steps': 287,
    'dimension': 3, 'obs_perc': 1., 'S0': None,
    'maturity': 1., "input_transformation": "log_transformed"
}