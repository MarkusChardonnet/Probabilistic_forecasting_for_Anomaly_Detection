import numpy as np

from configs.config_utils import get_parameter_array, get_dataset_overview, \
    makedirs, data_path, training_data_path

AD_OrnsteinUhlenbeckWithSeason_models_path_dataset_size = "{}saved_models_AD_OrnsteinUhlenbeckWithSeason_dataset_size/".format(data_path)
param_list_AD_OrnsteinUhlenbeckWithSeason_dataset_size = []
ode_nn_1 = ((300, 'tanh'), (300, 'relu'))
enc_nn = ((200, 'tanh'), (200, 'tanh'))
readout_nn = ((200, 'tanh'), (200, 'tanh'))
param_dict_AD_OrnsteinUhlenbeckWithSeason_1_dataset_size = {
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
        #'weight': [1],
        #'weight_decay': [0.99],
        'weight': [0],
        'weight_evolve': [{'type':'linear', 'target': 1, 'reach': None}],
        'data_dict': ['AD_OrnsteinUhlenbeckWithSeason_3_dict'],
        'plot': [True],
        'which_loss': ['ad_var'],
        'evaluate': [True],
        'evaluate_vars': [['exp','std']],
        'paths_to_plot': [(0,1,2,3,4,)],
        'plot_variance': [True],
        'std_factor': [1.96],
        'plot_moments': [True],
        'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_models_path_dataset_size],
        'use_cond_exp': [True],
        'input_current_t': [True],
        'periodic_current_t': [True],
        'validation_size': [200],
        'train_data_perc': [0.1],
        'scale_dt': ['automatic'],
        'enc_input_t': [False],
    }
param_list_AD_OrnsteinUhlenbeckWithSeason_dataset_size += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_1_dataset_size)
param_dict_AD_OrnsteinUhlenbeckWithSeason_2_dataset_size = {
        'epochs': [4000],
        'batch_size': [200],
        'save_every': [100],
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
        'weight_evolve': [{'type':'linear', 'target': 1, 'reach': None}],
        'data_dict': ['AD_OrnsteinUhlenbeckWithSeason_3_dict'],
        'plot': [True],
        'which_loss': ['ad_var'],
        'evaluate': [True],
        'evaluate_vars': [['exp','std']],
        'paths_to_plot': [(0,1,2,3,4,)],
        'plot_variance': [True],
        'std_factor': [1.96],
        'plot_moments': [True],
        'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_models_path_dataset_size],
        'use_cond_exp': [True],
        'input_current_t': [True],
        'periodic_current_t': [True],
        'training_size': [1000],
        'validation_size': [200],
        'train_data_perc': [0.1],
        'scale_dt': ['automatic'],
        'enc_input_t': [False],
    }
param_list_AD_OrnsteinUhlenbeckWithSeason_dataset_size += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_2_dataset_size)
param_dict_AD_OrnsteinUhlenbeckWithSeason_3_dataset_size = {
        'epochs': [20000],
        'batch_size': [200],
        'save_every': [500],
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
        'weight_evolve': [{'type':'linear', 'target': 1, 'reach': None}],
        'data_dict': ['AD_OrnsteinUhlenbeckWithSeason_3_dict'],
        'plot': [True],
        'which_loss': ['ad_var'],
        'evaluate': [True],
        'evaluate_vars': [['exp','std']],
        'paths_to_plot': [(0,1,2,3,4,)],
        'plot_variance': [True],
        'std_factor': [1.96],
        'plot_moments': [True],
        'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_models_path_dataset_size],
        'use_cond_exp': [True],
        'input_current_t': [True],
        'periodic_current_t': [True],
        'training_size': [200],
        'validation_size': [200],
        'train_data_perc': [0.1],
        'scale_dt': ['automatic'],
        'enc_input_t': [False],
    }
param_list_AD_OrnsteinUhlenbeckWithSeason_dataset_size += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_3_dataset_size)


AD_OrnsteinUhlenbeckWithSeason_models_path_var_pos = "{}saved_models_AD_OrnsteinUhlenbeckWithSeason_var_pos/".format(data_path)
param_list_AD_OrnsteinUhlenbeckWithSeason_var_pos = []
ode_nn_1 = ((300, 'tanh'), (300, 'relu'))
enc_nn = ((200, 'tanh'), (200, 'tanh'))
readout_nn = ((200, 'tanh'), (200, 'tanh'))
param_dict_AD_OrnsteinUhlenbeckWithSeason_1_var_pos = {
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
        'weight_evolve': [{'type':'linear', 'target': 1, 'reach': None}],
        'data_dict': ['AD_OrnsteinUhlenbeckWithSeason_3_dict'],
        'plot': [True],
        'which_loss': ['ad_var_pos'],
        'evaluate': [True],
        'evaluate_vars': [['exp','std']],
        'paths_to_plot': [(0,1,2,3,4,)],
        'plot_variance': [True],
        'std_factor': [1.96],
        'plot_moments': [True],
        'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_models_path_var_pos],
        'use_cond_exp': [True],
        'input_current_t': [True],
        'periodic_current_t': [True],
        'validation_size': [200],
        'train_data_perc': [0.1],
        'scale_dt': ['automatic'],
        'enc_input_t': [False],
    }
param_list_AD_OrnsteinUhlenbeckWithSeason_var_pos += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_1_var_pos)


# warning, it was lauched with add_pred = ['var']
AD_OrnsteinUhlenbeckWithSeason_models_path_wo_var = "{}saved_models_AD_OrnsteinUhlenbeckWithSeason_wo_var/".format(data_path)
param_list_AD_OrnsteinUhlenbeckWithSeason_wo_var = []
ode_nn_1 = ((300, 'tanh'), (300, 'relu'))
enc_nn = ((200, 'tanh'), (200, 'tanh'))
readout_nn = ((200, 'tanh'), (200, 'tanh'))
param_dict_AD_OrnsteinUhlenbeckWithSeason_1_wo_var = {
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
        'weight': [0],
        'weight_evolve': [{'type':'linear', 'target': 1, 'reach': None}],
        'data_dict': ['AD_OrnsteinUhlenbeckWithSeason_3_dict'],
        'plot': [True],
        'which_loss': ['ad'],
        'evaluate': [True],
        'evaluate_vars': [['exp','std']],
        'paths_to_plot': [(0,1,2,3,4,)],
        'plot_variance': [True],
        'std_factor': [1.96],
        'plot_moments': [True],
        'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_models_path_wo_var],
        'use_cond_exp': [True],
        'input_current_t': [True],
        'periodic_current_t': [True],
        'validation_size': [200],
        'train_data_perc': [0.1],
        'scale_dt': ['automatic'],
        'enc_input_t': [False],
    }
param_list_AD_OrnsteinUhlenbeckWithSeason_wo_var += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_1_wo_var)

AD_OrnsteinUhlenbeckWithSeason_models_path_modif_loss = "{}saved_models_AD_OrnsteinUhlenbeckWithSeason_modif_loss/".format(data_path)
param_list_AD_OrnsteinUhlenbeckWithSeason_modif_loss = []
ode_nn_1 = ((300, 'tanh'), (300, 'relu'))
enc_nn = ((200, 'tanh'), (200, 'tanh'))
readout_nn = ((200, 'tanh'), (200, 'tanh'))
param_dict_AD_OrnsteinUhlenbeckWithSeason_1_modif_loss = {
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
        'func_appl_X': [[]],              # [["power-2", "power-3", "power-4"]]
        'add_pred': [[]],
        'solver': ["euler"],
        'solver_delta_t_factor': [1],
        'weight': [0],
        'weight_evolve': [{'type':'linear', 'target': 1, 'reach': None}],
        'data_dict': ['AD_OrnsteinUhlenbeckWithSeason_3_dict'],
        'plot': [True],
        'which_loss': ['easy', 'easy_bis'],
        'evaluate': [True],
        'evaluate_vars': [['exp']],
        'paths_to_plot': [(0,1,2,3,4,)],
        'plot_variance': [True],
        'std_factor': [1.96],
        'plot_moments': [True],
        'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_models_path_modif_loss],
        'use_cond_exp': [True],
        'input_current_t': [True],
        'periodic_current_t': [True],
        'validation_size': [200],
        'train_data_perc': [0.1],
        'scale_dt': ['automatic'],
        'enc_input_t': [False],
    }
param_list_AD_OrnsteinUhlenbeckWithSeason_modif_loss += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_1_modif_loss)

AD_OrnsteinUhlenbeckWithSeason_models_path_fixed_obs_perc = "{}saved_models_AD_OrnsteinUhlenbeckWithSeason_fixed_obs_perc/".format(data_path)
param_list_AD_OrnsteinUhlenbeckWithSeason_fixed_obs_perc = []
ode_nn_1 = ((300, 'tanh'), (300, 'relu'))
enc_nn = ((200, 'tanh'), (200, 'tanh'))
readout_nn = ((200, 'tanh'), (200, 'tanh'))
param_dict_AD_OrnsteinUhlenbeckWithSeason_1_fixed_obs_perc = {
        'epochs': [50000],
        'batch_size': [200],
        'save_every': [500],
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
        'weight_evolve': [{'type':'linear', 'target': 1, 'reach': None}],
        'data_dict': ['AD_OrnsteinUhlenbeckWithSeason_3_dict'],
        'plot': [True],
        'which_loss': ['ad_var'],
        'evaluate': [True],
        'evaluate_vars': [['exp','std']],
        'paths_to_plot': [(0,1,2,3,4,)],
        'plot_variance': [True],
        'std_factor': [1.96],
        'plot_moments': [True],
        'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_models_path_fixed_obs_perc],
        'use_cond_exp': [True],
        'input_current_t': [True],
        'periodic_current_t': [True],
        'training_size': [200],
        'validation_size': [200],
        'fixed_data_perc': [0.1],
        'scale_dt': ['automatic'],
        'enc_input_t': [False],
    }
param_list_AD_OrnsteinUhlenbeckWithSeason_fixed_obs_perc += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_1_fixed_obs_perc)

AD_OrnsteinUhlenbeckWithSeason_models_path_dim2 = "{}saved_models_AD_OrnsteinUhlenbeckWithSeason_dim2/".format(data_path)
param_list_AD_OrnsteinUhlenbeckWithSeason_dim2 = []
ode_nn_1 = ((400, 'tanh'), (400, 'relu'))
enc_nn = ((200, 'tanh'), (200, 'tanh'))
readout_nn = ((200, 'tanh'), (200, 'tanh'))
param_dict_AD_OrnsteinUhlenbeckWithSeason_1_dim2 = {
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
        'weight_evolve': [{'type':'linear', 'target': 1, 'reach': None}],
        'data_dict': ['AD_OrnsteinUhlenbeckWithSeason_dim2_dict'],
        'plot': [True],
        'which_loss': ['ad_var'],
        'evaluate': [True],
        'evaluate_vars': [['exp','std']],
        'paths_to_plot': [(0,1,2,3,4,)],
        'plot_variance': [True],
        'std_factor': [1.96],
        'plot_moments': [True],
        'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_models_path_dim2],
        'use_cond_exp': [True],
        'input_current_t': [True],
        'periodic_current_t': [True],
        'validation_size': [200],
        'train_data_perc': [0.1],
        'scale_dt': ['automatic'],
        'enc_input_t': [False],
    }
param_list_AD_OrnsteinUhlenbeckWithSeason_dim2 += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_1_dim2)

AD_OrnsteinUhlenbeckWithSeason_models_path_5seas = "{}saved_models_AD_OrnsteinUhlenbeckWithSeason_5seas/".format(data_path)
param_list_AD_OrnsteinUhlenbeckWithSeason_5seas = []
ode_nn_1 = ((300, 'tanh'), (300, 'relu'))
enc_nn = ((200, 'tanh'), (200, 'tanh'))
readout_nn = ((200, 'tanh'), (200, 'tanh'))
param_dict_AD_OrnsteinUhlenbeckWithSeason_1_5seas = {
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
        'weight_evolve': [{'type':'linear', 'target': 1, 'reach': None}],
        'data_dict': ['AD_OrnsteinUhlenbeckWithSeason_5seas_dict'],
        'plot': [True],
        'which_loss': ['ad_var'],
        'evaluate': [True],
        'evaluate_vars': [['exp','std']],
        'paths_to_plot': [(0,1,2,3,4,)],
        'plot_variance': [True],
        'std_factor': [1.96],
        'plot_moments': [True],
        'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_models_path_5seas],
        'use_cond_exp': [True],
        'input_current_t': [True],
        'periodic_current_t': [True],
        'validation_size': [200],
        'train_data_perc': [0.1],
        'scale_dt': ['automatic'],
        'enc_input_t': [False],
    }
param_list_AD_OrnsteinUhlenbeckWithSeason_5seas += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_1_5seas)


AD_OrnsteinUhlenbeckWithSeason_models_path_pre_train = "{}saved_models_AD_OrnsteinUhlenbeckWithSeason_pre_train/".format(data_path)
param_list_AD_OrnsteinUhlenbeckWithSeason_pre_train = []
ode_nn_1 = ((300, 'tanh'), (300, 'relu'))
enc_nn = ((200, 'tanh'), (200, 'tanh'))
readout_nn = ((200, 'tanh'), (200, 'tanh'))
param_dict_AD_OrnsteinUhlenbeckWithSeason_1_pre_train = {
        'epochs': [1000],
        'batch_size': [200],
        'save_every': [5],
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
        'weight_evolve': [{'type':'linear', 'target': 1, 'reach': None}],
        'data_dict': ['AD_OrnsteinUhlenbeckWithSeason_3_dict'],
        'plot': [True],
        'which_loss': ['ad_var'],
        'evaluate': [True],
        'evaluate_vars': [['exp','std']],
        'paths_to_plot': [(0,1,2,3,4,)],
        'plot_variance': [True],
        'std_factor': [1.96],
        'plot_moments': [True],
        'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_models_path_pre_train],
        'use_cond_exp': [True],
        'input_current_t': [True],
        'periodic_current_t': [True],
        'training_size': [200],
        'validation_size': [200],
        'train_data_perc': [0.1],
        'scale_dt': ['automatic'],
        'enc_input_t': [False],
    }
# param_list_AD_OrnsteinUhlenbeckWithSeason_pre_train += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_1_pre_train)
param_dict_AD_OrnsteinUhlenbeckWithSeason_2_pre_train = {
        'epochs': [1000],
        'batch_size': [200],
        'save_every': [5],
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
        'weight_evolve': [{'type':'linear', 'target': 1, 'reach': None}],
        'data_dict': ['AD_OrnsteinUhlenbeckWithSeason_3_dict'],
        'plot': [True],
        'which_loss': ['ad_var'],
        'evaluate': [True],
        'evaluate_vars': [['exp','std']],
        'paths_to_plot': [(0,1,2,3,4,)],
        'plot_variance': [True],
        'std_factor': [1.96],
        'plot_moments': [True],
        'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_models_path_pre_train],
        'use_cond_exp': [True],
        'input_current_t': [True],
        'periodic_current_t': [True],
        'training_size': [200],
        'validation_size': [200],
        'train_data_perc': [0.1],
        'scale_dt': ['automatic'],
        'enc_input_t': [False],
        'pre-train': [True]
    }
param_list_AD_OrnsteinUhlenbeckWithSeason_pre_train += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_2_pre_train)