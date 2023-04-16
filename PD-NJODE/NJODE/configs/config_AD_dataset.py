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
    'nb_paths': 500, 'nb_steps': 400,
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

AD_OrnsteinUhlenbeckWithSeason_dim2_dict = {
    'model_name': "AD_OrnsteinUhlenbeckWithSeason",
    'nb_paths': 100000, 'nb_steps': 400,
    'dimension': 2, 'obs_perc': 1., 'S0': None,
    'maturity': 1., 'return_vol': False,
    'speed': [[15.,2.],[0.,15.]],
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
        'new_seas': True,
        'seas_path': "configs/models/AD_OrnsteinUhlenBeck_NN_season_dim2",
        'changing_seas': False,
    },
    'anomaly_params': {
        'type': None
    }
}

AD_OrnsteinUhlenbeckWithSeason_5seas_dict = {
    'model_name': "AD_OrnsteinUhlenbeckWithSeason",
    'nb_paths': 10000, 'nb_steps': 1000,
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
        'new_seas': True,
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
    'dimension': 2, 'obs_perc': 0.1, 'S0': None,
    'maturity': 1., 'return_vol': False,
    'speed': [[15.,2.],[2.,15.]], 
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