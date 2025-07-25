
# AD_OrnsteinUhlenbeckWithSeason_deformation_dict = AD_OrnsteinUhlenbeckWithSeason_3_dict

occurence_law = 'single'
occurence_prob = 1
occurence_pos_law = 'uniform'
occurence_pos_range = (0.1,0.9)
occurence_len_law = 'uniform'
occurence_len_range =  (0.1,0.6)
dim_occurence_law = 'indep'
dim_occurence_prob = 1.
dim_occurence_pos = 'indep'

AD_OrnsteinUhlenbeckWithSeason_deformation_dict = {
    'model_name': "AD_OrnsteinUhlenbeckWithSeason",
    'nb_paths': 1000, 'nb_steps': 400,
    'dimension': 1, 'obs_perc': 1., 'S0': None,
    'maturity': 1., 'return_vol': False,
    'speed': 15., 
    'noise': {'type': "gaussian", 'cov': 0.0},
    'volatility': {
        'vol_value': 0.3,
    },
    'initial_value': False,
    'season_params': {
        'seas_amp': 'automatic',
        'seas_range': (0.2,0.8),
        'seas_length': 200,
        'seas_nb': 2,
        'seas_type': 'NN',
        'nn_layers': [(16, 'relu'),(16, 'relu')],
        'nn_bias': True,
        'nn_input': (['cos', 'sin'], 1),
        'new_seas': False,
        'seas_path': "configs/models/AD_OrnsteinUhlenBeck_NN_season",
        'changing_seas': False,
    },
    'anomaly_params': {
        'type': 'deformation',
        'occurence_law': occurence_law,
        'occurence_prob': occurence_prob,
        'occurence_pos_law': occurence_pos_law,
        'occurence_pos_range': occurence_pos_range,
        'occurence_len_law': occurence_len_law,
        'occurence_len_range': occurence_len_range,
        'dim_occurence_law': dim_occurence_law,
        'dim_occurence_prob': dim_occurence_prob,
        'dim_occurence_pos': dim_occurence_pos,
    }
}

AD_OrnsteinUhlenbeckWithSeason_diffusion_dict = {
    'model_name': "AD_OrnsteinUhlenbeckWithSeason",
    'nb_paths': 1000, 'nb_steps': 400,
    'dimension': 1, 'obs_perc': 1., 'S0': None,
    'maturity': 1., 'return_vol': False,
    'speed': 15., 
    'noise': {'type': "gaussian", 'cov': 0.0},
    'volatility': {
        'vol_value': 0.3,
    },
    'initial_value': False,
    'season_params': {
        'seas_amp': 'automatic',
        'seas_range': (0.2,0.8),
        'seas_length': 200,
        'seas_nb': 2,
        'seas_type': 'NN',
        'nn_layers': [(16, 'relu'),(16, 'relu')],
        'nn_bias': True,
        'nn_input': (['cos', 'sin'], 1),
        'new_seas': False,
        'seas_path': "configs/models/AD_OrnsteinUhlenBeck_NN_season",
        'changing_seas': False,
    },
    'anomaly_params': {
        'type': 'diffusion',
        'occurence_law': occurence_law,
        'occurence_prob': occurence_prob,
        'occurence_pos_law': occurence_pos_law,
        'occurence_pos_range': occurence_pos_range,
        'occurence_len_law': occurence_len_law,
        'occurence_len_range': occurence_len_range,
        'dim_occurence_law': dim_occurence_law,
        'dim_occurence_prob': dim_occurence_prob,
        'dim_occurence_pos': dim_occurence_pos,

        'diffusion_change': 'multiplicative', # or 'additive'
        # 'dim_diffusion_change': 'individual', # or 'common'
        'diffusion_deviation': 5.
    }
}

AD_OrnsteinUhlenbeckWithSeason_noise_dict = {
    'model_name': "AD_OrnsteinUhlenbeckWithSeason",
    'nb_paths': 1000, 'nb_steps': 400,
    'dimension': 1, 'obs_perc': 1., 'S0': None,
    'maturity': 1., 'return_vol': False,
    'speed': 15., 
    'noise': {'type': "gaussian", 'cov': 0.0},
    'volatility': {
        'vol_value': 0.3,
    },
    'initial_value': False,
    'season_params': {
        'seas_amp': 'automatic',
        'seas_range': (0.2,0.8),
        'seas_length': 200,
        'seas_nb': 2,
        'seas_type': 'NN',
        'nn_layers': [(16, 'relu'),(16, 'relu')],
        'nn_bias': True,
        'nn_input': (['cos', 'sin'], 1),
        'new_seas': False,
        'seas_path': "configs/models/AD_OrnsteinUhlenBeck_NN_season",
        'changing_seas': False,
    },
    'anomaly_params': {
        'type': 'noise',
        'occurence_law': occurence_law,
        'occurence_prob': occurence_prob,
        'occurence_pos_law': occurence_pos_law,
        'occurence_pos_range': occurence_pos_range,
        'occurence_len_law': occurence_len_law,
        'occurence_len_range': occurence_len_range,
        'dim_occurence_law': dim_occurence_law,
        'dim_occurence_prob': dim_occurence_prob,
        'dim_occurence_pos': dim_occurence_pos,

        'noise_change': 'additive', # or 'additive'
        # 'dim_diffusion_change': 'individual', # or 'common'
        'noise_deviation': 0.05
    }
}

AD_OrnsteinUhlenbeckWithSeason_cutoff_dict = {
    'model_name': "AD_OrnsteinUhlenbeckWithSeason",
    'nb_paths': 1000, 'nb_steps': 400,
    'dimension': 1, 'obs_perc': 1., 'S0': None,
    'maturity': 1., 'return_vol': False,
    'speed': 15., 
    'noise': {'type': "gaussian", 'cov': 0.0},
    'volatility': {
        'vol_value': 0.3,
    },
    'initial_value': False,
    'season_params': {
        'seas_amp': 'automatic',
        'seas_range': (0.2,0.8),
        'seas_length': 200,
        'seas_nb': 2,
        'seas_type': 'NN',
        'nn_layers': [(16, 'relu'),(16, 'relu')],
        'nn_bias': True,
        'nn_input': (['cos', 'sin'], 1),
        'new_seas': False,
        'seas_path': "configs/models/AD_OrnsteinUhlenBeck_NN_season",
        'changing_seas': False,
    },
    'anomaly_params': {
        'type': 'cutoff',
        'occurence_law': occurence_law,
        'occurence_prob': occurence_prob,
        'occurence_pos_law': occurence_pos_law,
        'occurence_pos_range': occurence_pos_range,
        'occurence_len_law': occurence_len_law,
        'occurence_len_range': occurence_len_range,
        'dim_occurence_law': dim_occurence_law,
        'dim_occurence_prob': dim_occurence_prob,
        'dim_occurence_pos': dim_occurence_pos,

        'cutoff_level_law': 'current_level',
        'cutoff_level_range': (0.,1.),
    }
}

AD_OrnsteinUhlenbeckWithSeason_scale_dict = {
    'model_name': "AD_OrnsteinUhlenbeckWithSeason",
    'nb_paths': 1000, 'nb_steps': 400,
    'dimension': 1, 'obs_perc': 1., 'S0': None,
    'maturity': 1., 'return_vol': False,
    'speed': 15., 
    'noise': {'type': "gaussian", 'cov': 0.0},
    'volatility': {
        'vol_value': 0.3,
    },
    'initial_value': False,
    'season_params': {
        'seas_amp': 'automatic',
        'seas_range': (0.2,0.8),
        'seas_length': 200,
        'seas_nb': 2,
        'seas_type': 'NN',
        'nn_layers': [(16, 'relu'),(16, 'relu')],
        'nn_bias': True,
        'nn_input': (['cos', 'sin'], 1),
        'new_seas': False,
        'seas_path': "configs/models/AD_OrnsteinUhlenBeck_NN_season",
        'changing_seas': False,
    },
    'anomaly_params': {
        'type': 'scale',
        'occurence_law': occurence_law,
        'occurence_prob': occurence_prob,
        'occurence_pos_law': occurence_pos_law,
        'occurence_pos_range': occurence_pos_range,
        'occurence_len_law': occurence_len_law,
        'occurence_len_range': occurence_len_range,
        'dim_occurence_law': dim_occurence_law,
        'dim_occurence_prob': dim_occurence_prob,
        'dim_occurence_pos': dim_occurence_pos,

        'scale_level_law': 'uniform',
        'scale_level_range': (1.5,3),
    }
}

AD_OrnsteinUhlenbeckWithSeason_trend_dict = {
    'model_name': "AD_OrnsteinUhlenbeckWithSeason",
    'nb_paths': 1000, 'nb_steps': 400,
    'dimension': 1, 'obs_perc': 1., 'S0': None,
    'maturity': 1., 'return_vol': False,
    'speed': 15., 
    'noise': {'type': "gaussian", 'cov': 0.0},
    'volatility': {
        'vol_value': 0.3,
    },
    'initial_value': False,
    'season_params': {
        'seas_amp': 'automatic',
        'seas_range': (0.2,0.8),
        'seas_length': 200,
        'seas_nb': 2,
        'seas_type': 'NN',
        'nn_layers': [(16, 'relu'),(16, 'relu')],
        'nn_bias': True,
        'nn_input': (['cos', 'sin'], 1),
        'new_seas': False,
        'seas_path': "configs/models/AD_OrnsteinUhlenBeck_NN_season",
        'changing_seas': False,
    },
    'anomaly_params': {
        'type': 'trend',
        'occurence_law': occurence_law,
        'occurence_prob': occurence_prob,
        'occurence_pos_law': occurence_pos_law,
        'occurence_pos_range': occurence_pos_range,
        'occurence_len_law': occurence_len_law,
        'occurence_len_range': occurence_len_range,
        'dim_occurence_law': dim_occurence_law,
        'dim_occurence_prob': dim_occurence_prob,
        'dim_occurence_pos': dim_occurence_pos,

        'trend_level_law': 'uniform',
        'trend_level_range': (0.5,1.),
        'trend_level_sign': 'both', # or 'plus' or 'minus'
    }
}

AD_OrnsteinUhlenbeckWithSeason_spike_dict = {
    'model_name': "AD_OrnsteinUhlenbeckWithSeason",
    'nb_paths': 1000, 'nb_steps': 400,
    'dimension': 1, 'obs_perc': 1., 'S0': None,
    'maturity': 1., 'return_vol': False,
    'speed': 15., 
    'noise': {'type': "gaussian", 'cov': 0.0},
    'volatility': {
        'vol_value': 0.3,
    },
    'initial_value': False,
    'season_params': {
        'seas_amp': 'automatic',
        'seas_range': (0.2,0.8),
        'seas_length': 200,
        'seas_nb': 2,
        'seas_type': 'NN',
        'nn_layers': [(16, 'relu'),(16, 'relu')],
        'nn_bias': True,
        'nn_input': (['cos', 'sin'], 1),
        'new_seas': False,
        'seas_path': "configs/models/AD_OrnsteinUhlenBeck_NN_season",
        'changing_seas': False,
    },
    'anomaly_params': {
        'type': 'spike',
        # 'occurence': 'per_time',
        'occurence_prob': 0.005,
        'occurence_pos_range': (0.1,0.9),
        # 'spike_type': 'additive',
        'spike_amp_law': 'uniform',
        'spike_amp_range': (0.2,0.5),
        # 'dim_occurence_law': 'indep',
    }
}