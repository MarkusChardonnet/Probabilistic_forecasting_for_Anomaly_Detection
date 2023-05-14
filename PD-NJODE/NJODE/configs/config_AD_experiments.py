import numpy as np

from configs.config_utils import get_parameter_array, get_dataset_overview, \
    makedirs, data_path, training_data_path


batch_size = 200
learning_rate = 0.001
seed = 398
hidden_size = 150
bias = True
dropout_rate = 0.1
test_size = 0.2
ode_nn_1 = ((300, 'tanh'), (300, 'relu'))
ode_nn_2 = ((400, 'tanh'), (400, 'relu'))
enc_nn = ((200, 'tanh'), (200, 'tanh'))
readout_nn = ((200, 'tanh'), (200, 'tanh'))
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
validation_size = 200
scale_dt = 'automatic'
train_data_perc = 0.1


###### MAIN DATASET
AD_OrnsteinUhlenbeckWithSeason_models_path_experiments = "{}saved_models_AD_OrnsteinUhlenbeckWithSeason_EXPERIMENTS/".format(data_path)
param_list_AD_OrnsteinUhlenbeckWithSeason_experiments = []

# base setup
AD_OrnsteinUhlenbeckWithSeason_models_path_base = "{}saved_models_AD_OrnsteinUhlenbeckWithSeason_base/".format(data_path)
param_list_AD_OrnsteinUhlenbeckWithSeason_base = []
param_dict_AD_OrnsteinUhlenbeckWithSeason_1_base = {
        'epochs': [50],
        'batch_size': [batch_size],
        'save_every': [1],
        'learning_rate': [learning_rate],
        'test_size': [test_size],
        'seed': [seed],
        'hidden_size': [hidden_size],
        'bias': [bias],
        'dropout_rate': [dropout_rate],
        'ode_nn': [ode_nn_1],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [use_rnn],
        'input_sig': [input_sig],
        'func_appl_X': [func_appl_X],              # [["power-2", "power-3", "power-4"]]
        'add_pred': [add_pred],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor],
        'weight': [weight],
        'weight_evolve': [weight_evolve],
        'data_dict': ['AD_OrnsteinUhlenbeckWithSeason_3_dict'],
        'plot': [True],
        'which_loss': ['variance'],
        'which_eval_loss': ['eval_variance'],
        'evaluate': [True],
        'eval_metrics': [eval_metrics],
        'paths_to_plot': [paths_to_plot],
        'plot_variance': [plot_variance],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        # 'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_models_path_base],
        'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_models_path_experiments],
        'use_cond_exp': [True],
        'input_current_t': [input_current_t],
        'periodic_current_t': [True],
        'validation_size': [validation_size],
        'train_data_perc': [train_data_perc],
        'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
    }
param_list_AD_OrnsteinUhlenbeckWithSeason_base += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_1_base)
# param_list_AD_OrnsteinUhlenbeckWithSeason_experiments += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_1_base)

# with smaller dataset sizes (200 and 1000), but equal number of forward passes
AD_OrnsteinUhlenbeckWithSeason_models_path_dataset_size = "{}saved_models_AD_OrnsteinUhlenbeckWithSeason_dataset_size/".format(data_path)
param_list_AD_OrnsteinUhlenbeckWithSeason_dataset_size = []
param_dict_AD_OrnsteinUhlenbeckWithSeason_1_dataset_size = {
        'epochs': [4000],
        'batch_size': [batch_size],
        'save_every': [100],
        'learning_rate': [learning_rate],
        'test_size': [test_size],
        'seed': [seed],
        'hidden_size': [hidden_size],
        'bias': [bias],
        'dropout_rate': [dropout_rate],
        'ode_nn': [ode_nn_1],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [use_rnn],
        'input_sig': [input_sig],
        'func_appl_X': [func_appl_X],
        'add_pred': [add_pred],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor],
        'weight': [weight],
        'weight_evolve': [weight_evolve],
        'data_dict': ['AD_OrnsteinUhlenbeckWithSeason_3_dict'],
        'plot': [True],
        'which_loss': ['variance'],
        'which_eval_loss': ['eval_variance'],
        'evaluate': [True],
        'eval_metrics': [['exp','std']],
        'paths_to_plot': [paths_to_plot],
        'plot_variance': [plot_variance],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        # 'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_models_path_dataset_size],
        'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_models_path_experiments],
        'use_cond_exp': [True],
        'input_current_t': [input_current_t],
        'periodic_current_t': [True],
        'training_size': [1000],
        'validation_size': [validation_size],
        'train_data_perc': [train_data_perc],
        'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
    }
param_list_AD_OrnsteinUhlenbeckWithSeason_dataset_size += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_1_dataset_size)
param_list_AD_OrnsteinUhlenbeckWithSeason_experiments += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_1_dataset_size)
param_dict_AD_OrnsteinUhlenbeckWithSeason_2_dataset_size = {
        'epochs': [20000],
        'batch_size': [batch_size],
        'save_every': [500],
        'learning_rate': [learning_rate],
        'test_size': [test_size],
        'seed': [seed],
        'hidden_size': [hidden_size],
        'bias': [bias],
        'dropout_rate': [dropout_rate],
        'ode_nn': [ode_nn_1],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [use_rnn],
        'input_sig': [input_sig],
        'func_appl_X': [func_appl_X],
        'add_pred': [add_pred],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor],
        'weight': [weight],
        'weight_evolve': [weight_evolve],
        'data_dict': ['AD_OrnsteinUhlenbeckWithSeason_3_dict'],
        'plot': [True],
        'which_loss': ['variance'],
        'which_eval_loss': ['eval_variance'],
        'evaluate': [True],
        'eval_metrics': [['exp','std']],
        'paths_to_plot': [paths_to_plot],
        'plot_variance': [plot_variance],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        # 'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_models_path_dataset_size],
        'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_models_path_experiments],
        'use_cond_exp': [True],
        'input_current_t': [input_current_t],
        'periodic_current_t': [True],
        'training_size': [200],
        'validation_size': [validation_size],
        'train_data_perc': [train_data_perc],
        'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
    }
param_list_AD_OrnsteinUhlenbeckWithSeason_dataset_size += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_2_dataset_size)
param_list_AD_OrnsteinUhlenbeckWithSeason_experiments += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_2_dataset_size)

# without the term enforcing variance positiveness
AD_OrnsteinUhlenbeckWithSeason_models_path_var_pos = "{}saved_models_AD_OrnsteinUhlenbeckWithSeason_var_pos/".format(data_path)
param_list_AD_OrnsteinUhlenbeckWithSeason_var_pos = []
param_dict_AD_OrnsteinUhlenbeckWithSeason_1_var_pos = {
        'epochs': [50],
        'batch_size': [batch_size],
        'save_every': [1],
        'learning_rate': [learning_rate],
        'test_size': [test_size],
        'seed': [seed],
        'hidden_size': [hidden_size],
        'bias': [True],
        'dropout_rate': [dropout_rate],
        'ode_nn': [ode_nn_1],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [use_rnn],
        'input_sig': [input_sig],
        'func_appl_X': [func_appl_X],
        'add_pred': [add_pred],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor],
        'weight': [weight],
        'weight_evolve': [weight_evolve],
        'data_dict': ['AD_OrnsteinUhlenbeckWithSeason_3_dict'],
        'plot': [True],
        'which_loss': ['variance_bis'],
        'which_eval_loss': ['eval_variance'],
        'evaluate': [True],
        'eval_metrics': [eval_metrics],
        'paths_to_plot': [paths_to_plot],
        'plot_variance': [plot_variance],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        # 'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_models_path_var_pos],
        'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_models_path_experiments],
        'use_cond_exp': [True],
        'input_current_t': [input_current_t],
        'periodic_current_t': [True],
        'validation_size': [validation_size],
        'train_data_perc': [train_data_perc],
        'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
    }
param_list_AD_OrnsteinUhlenbeckWithSeason_var_pos += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_1_var_pos)
# param_list_AD_OrnsteinUhlenbeckWithSeason_experiments += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_1_var_pos)

# without the prediction of the variance (it is computed from the difference between the cond exp squared and the squared cond exp)
AD_OrnsteinUhlenbeckWithSeason_models_path_wo_var = "{}saved_models_AD_OrnsteinUhlenbeckWithSeason_wo_var/".format(data_path)
param_list_AD_OrnsteinUhlenbeckWithSeason_wo_var = []
param_dict_AD_OrnsteinUhlenbeckWithSeason_1_wo_var = {
        'epochs': [50],
        'batch_size': [batch_size],
        'save_every': [1],
        'learning_rate': [learning_rate],
        'test_size': [test_size],
        'seed': [seed],
        'hidden_size': [hidden_size],
        'bias': [True],
        'dropout_rate': [dropout_rate],
        'ode_nn': [ode_nn_1],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [use_rnn],
        'input_sig': [input_sig],
        'func_appl_X': [func_appl_X],
        'add_pred': [[]],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor],
        'weight': [weight],
        'weight_evolve': [weight_evolve],
        'data_dict': ['AD_OrnsteinUhlenbeckWithSeason_3_dict'],
        'plot': [True],
        'which_loss': ['moment_2'],
        'which_eval_loss': ['eval_variance'],
        'evaluate': [True],
        'eval_metrics': [['exp','std']],
        'paths_to_plot': [paths_to_plot],
        'plot_variance': [plot_variance],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        # 'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_models_path_wo_var],
        'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_models_path_experiments],
        'use_cond_exp': [True],
        'input_current_t': [input_current_t],
        'periodic_current_t': [True],
        'validation_size': [validation_size],
        'train_data_perc': [train_data_perc],
        'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
    }
param_list_AD_OrnsteinUhlenbeckWithSeason_wo_var += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_1_wo_var)
# param_list_AD_OrnsteinUhlenbeckWithSeason_experiments += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_1_wo_var)

# with per path fixed observations (over epochs) for training on small dataset (200 paths)
AD_OrnsteinUhlenbeckWithSeason_models_path_fixed_obs_perc = "{}saved_models_AD_OrnsteinUhlenbeckWithSeason_fixed_obs_perc/".format(data_path)
param_list_AD_OrnsteinUhlenbeckWithSeason_fixed_obs_perc = []
param_dict_AD_OrnsteinUhlenbeckWithSeason_1_fixed_obs_perc = {
        'epochs': [20000],
        'batch_size': [batch_size],
        'save_every': [500],
        'learning_rate': [learning_rate],
        'test_size': [test_size],
        'seed': [seed],
        'hidden_size': [hidden_size],
        'bias': [True],
        'dropout_rate': [dropout_rate],
        'ode_nn': [ode_nn_1],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [use_rnn],
        'input_sig': [input_sig],
        'func_appl_X': [func_appl_X],
        'add_pred': [add_pred],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor],
        'weight': [weight],
        'weight_evolve': [weight_evolve],
        'data_dict': ['AD_OrnsteinUhlenbeckWithSeason_3_dict'],
        'plot': [True],
        'which_loss': ['variance'],
        'which_eval_loss': ['eval_variance'],
        'evaluate': [True],
        'eval_metrics': [eval_metrics],
        'paths_to_plot': [(0,1,2,3,4,)],
        'plot_variance': [plot_variance],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        # 'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_models_path_fixed_obs_perc],
        'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_models_path_experiments],
        'use_cond_exp': [True],
        'input_current_t': [input_current_t],
        'periodic_current_t': [True],
        'training_size': [200],
        'validation_size': [validation_size],
        'fixed_data_perc': [0.1],
        'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
    }
param_list_AD_OrnsteinUhlenbeckWithSeason_fixed_obs_perc += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_1_fixed_obs_perc)
param_list_AD_OrnsteinUhlenbeckWithSeason_experiments += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_1_fixed_obs_perc)

# without the adaptive weights
AD_OrnsteinUhlenbeckWithSeason_models_path_adaptive_loss_weights = "{}saved_models_AD_OrnsteinUhlenbeckWithSeason_adaptive_loss_weights/".format(data_path)
param_list_AD_OrnsteinUhlenbeckWithSeason_adaptive_loss_weights = []
param_dict_AD_OrnsteinUhlenbeckWithSeason_1_adaptive_loss_weights = {
        'epochs': [50],
        'batch_size': [batch_size],
        'save_every': [1],
        'learning_rate': [learning_rate],
        'test_size': [test_size],
        'seed': [seed],
        'hidden_size': [hidden_size],
        'bias': [bias],
        'dropout_rate': [dropout_rate],
        'ode_nn': [ode_nn_1],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [use_rnn],
        'input_sig': [input_sig],
        'func_appl_X': [func_appl_X],
        'add_pred': [add_pred],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor],
        'weight': [1],
        'weight_evolve': [None],
        'data_dict': ['AD_OrnsteinUhlenbeckWithSeason_3_dict'],
        'plot': [True],
        'which_loss': ['variance'],
        'which_eval_loss': ['eval_variance'],
        'evaluate': [True],
        'eval_metrics': [eval_metrics],
        'paths_to_plot': [paths_to_plot],
        'plot_variance': [plot_variance],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        # 'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_models_path_adaptive_loss_weights],
        'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_models_path_experiments],
        'use_cond_exp': [True],
        'input_current_t': [input_current_t],
        'periodic_current_t': [True],
        'validation_size': [validation_size],
        'train_data_perc': [train_data_perc],
        'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
    }
param_list_AD_OrnsteinUhlenbeckWithSeason_adaptive_loss_weights += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_1_adaptive_loss_weights)
# param_list_AD_OrnsteinUhlenbeckWithSeason_experiments += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_1_adaptive_loss_weights)

# with and without diff_t rescaling
AD_OrnsteinUhlenbeckWithSeason_models_path_time_rescaling = "{}saved_models_AD_OrnsteinUhlenbeckWithSeason_time_rescaling/".format(data_path)
param_list_AD_OrnsteinUhlenbeckWithSeason_time_rescaling = []
param_dict_AD_OrnsteinUhlenbeckWithSeason_1_time_rescaling = {
        'epochs': [50],
        'batch_size': [batch_size],
        'save_every': [1],
        'learning_rate': [learning_rate],
        'test_size': [test_size],
        'seed': [seed],
        'hidden_size': [hidden_size],
        'bias': [bias],
        'dropout_rate': [dropout_rate],
        'ode_nn': [ode_nn_1],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [use_rnn],
        'input_sig': [input_sig],
        'func_appl_X': [func_appl_X],
        'add_pred': [add_pred],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor],
        'weight': [1],
        'weight_evolve': [weight_evolve],
        'data_dict': ['AD_OrnsteinUhlenbeckWithSeason_3_dict'],
        'plot': [True],
        'which_loss': ['variance'],
        'which_eval_loss': ['eval_variance'],
        'evaluate': [True],
        'eval_metrics': [eval_metrics],
        'paths_to_plot': [paths_to_plot],
        'plot_variance': [plot_variance],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        # 'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_models_path_time_rescaling],
        'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_models_path_experiments],
        'use_cond_exp': [True],
        'input_current_t': [input_current_t],
        'periodic_current_t': [True],
        'validation_size': [validation_size],
        'train_data_perc': [train_data_perc],
        # 'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
    }
param_list_AD_OrnsteinUhlenbeckWithSeason_time_rescaling += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_1_time_rescaling)
# param_list_AD_OrnsteinUhlenbeckWithSeason_experiments += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_1_time_rescaling)

# with artificial training technique for the encoder / decoder (for small, i.e 200 path dataset)
AD_OrnsteinUhlenbeckWithSeason_models_path_pre_train = "{}saved_models_AD_OrnsteinUhlenbeckWithSeason_pre_train/".format(data_path)
param_list_AD_OrnsteinUhlenbeckWithSeason_pre_train = []
param_dict_AD_OrnsteinUhlenbeckWithSeason_1_pre_train = {
        'epochs': [20000],
        'batch_size': [batch_size],
        'save_every': [500],
        'learning_rate': [learning_rate],
        'test_size': [test_size],
        'seed': [seed],
        'hidden_size': [hidden_size],
        'bias': [True],
        'dropout_rate': [dropout_rate],
        'ode_nn': [ode_nn_1],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [use_rnn],
        'input_sig': [input_sig],
        'func_appl_X': [func_appl_X],
        'add_pred': [add_pred],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor],
        'weight': [weight],
        'weight_evolve': [weight_evolve],
        'data_dict': ['AD_OrnsteinUhlenbeckWithSeason_3_dict'],
        'plot': [True],
        'which_loss': ['variance'],
        'which_eval_loss': ['eval_variance'],
        'evaluate': [True],
        'eval_metrics': [['exp','std']],
        'paths_to_plot': [(0,1,2,3,4,)],
        'plot_variance': [True],
        'std_factor': [1.96],
        'plot_moments': [True],
        # 'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_models_path_pre_train],
        'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_models_path_experiments],
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
param_list_AD_OrnsteinUhlenbeckWithSeason_pre_train += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_1_pre_train)
param_list_AD_OrnsteinUhlenbeckWithSeason_experiments += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_1_pre_train)

overview_dict_AD_OrnsteinUhlenbeckWithSeason_experiments = dict(
    ids_from=1, ids_to=8, #len(param_list_AD_OrnsteinUhlenbeckWithSeason_experiments),
    path=AD_OrnsteinUhlenbeckWithSeason_models_path_experiments,
    params_extract_desc=('which_loss', 
                        'scale_dt', # not in all
                        'pre-train', # not in all
                        'weight_evolve', 
                        'train_data_perc', # / 'fixed_data_perc'
                        'fixed_data_perc',
                        'periodic_current_t'),
    val_test_params_extract=(
        ("min", "exp mean square error", "exp mean square error", "min expectation MSE"),
        ("min", "exp mean square error", "exp std square error", "min expectation MSE (std)"),
        ("min", "std mean square error", "std mean square error", "min standart deviation MSE"),
        ("min", "std mean square error", "std std square error", "min standart deviation MSE (std)"),
        ("last", "exp mean square error", "exp mean square error", "last expectation MSE"),
        ("last", "std mean square error", "std mean square error", "last standart deviation MSE"),
        ("average", "exp mean square error", "exp mean square error", "average expectation MSE"),
        ("average", "std mean square error", "std mean square error", "average standart deviation MSE"),
    ),
    sortby=None,
    save_file=None,
)

overview_dict_AD_OrnsteinUhlenbeckWithSeason_experiments_2 = dict(
    ids_from=2, ids_to=2, #len(param_list_AD_OrnsteinUhlenbeckWithSeason_experiments),
    path=AD_OrnsteinUhlenbeckWithSeason_models_path_experiments,
    params_extract_desc=('which_loss', 
                        'scale_dt', # not in all
                        'pre-train', # not in all
                        'weight_evolve', 
                        'train_data_perc', # / 'fixed_data_perc'
                        'fixed_data_perc',
                        'periodic_current_t'),
    val_test_params_extract=(
        ("min", "exp mean square error", "exp mean square error", "min expectation MSE"),
        ("min", "exp mean square error", "exp std square error", "min expectation MSE (std)"),
        ("min", "std mean square error", "std mean square error", "min standart deviation MSE"),
        ("min", "std mean square error", "std std square error", "min standart deviation MSE (std)"),
        ("last", "exp mean square error", "exp mean square error", "last expectation MSE"),
        ("last", "std mean square error", "std mean square error", "last standart deviation MSE"),
        ("average", "exp mean square error", "exp mean square error", "average expectation MSE"),
        ("average", "std mean square error", "std mean square error", "average standart deviation MSE"),
    ),
    sortby=None,
    save_file=None,
    select_every=80,
)

overview_dict_AD_OrnsteinUhlenbeckWithSeason_experiments_3 = dict(
    ids_from=9, ids_to=9, #len(param_list_AD_OrnsteinUhlenbeckWithSeason_experiments),
    path=AD_OrnsteinUhlenbeckWithSeason_models_path_experiments,
    params_extract_desc=('which_loss', 
                        'scale_dt', # not in all
                        'pre-train', # not in all
                        'weight_evolve', 
                        'train_data_perc', # / 'fixed_data_perc'
                        'fixed_data_perc',
                        'periodic_current_t'),
    val_test_params_extract=(
        ("min", "exp mean square error", "exp mean square error", "min expectation MSE"),
        ("min", "exp mean square error", "exp std square error", "min expectation MSE (std)"),
        ("min", "std mean square error", "std mean square error", "min standart deviation MSE"),
        ("min", "std mean square error", "std std square error", "min standart deviation MSE (std)"),
        ("last", "exp mean square error", "exp mean square error", "last expectation MSE"),
        ("last", "std mean square error", "std mean square error", "last standart deviation MSE"),
        ("average", "exp mean square error", "exp mean square error", "average expectation MSE"),
        ("average", "std mean square error", "std mean square error", "average standart deviation MSE"),
    ),
    sortby=None,
    save_file=None,
    select_every=400,
)

AD_OrnsteinUhlenbeckWithSeason_loss_metrics_artificial_points = {
    'model_ids':(3,9), #(2,3),
    'model_names': ("without artificial points","with artificial points"),
    #'cols': ("eval_loss_1","eval_loss_2","eval_loss_3","eval_loss_4","exp mean square error","std mean square error"),
    #'names': ("evaluation loss 1","evaluation loss 2","evaluation loss 3","evaluation loss 4","conditional expectation MSE",
    #          "conditional standard deviation MSE"),
    'cols': ("exp mean square error", "std mean square error"), # "std mean square error"),
    'names': ("conditional expectation MSE","conditional standard deviation MSE"), # "conditional standard deviation MSE"),
    'saved_models_path': AD_OrnsteinUhlenbeckWithSeason_models_path_experiments,
    'file_name': "artificial_points_metric_evol_{}.png",
    'from_epoch': 100,
    'to_epoch': 5000,
}

AD_OrnsteinUhlenbeckWithSeason_loss_metrics_varying_weights = {
    'model_ids':(1,7), #(2,3),
    'model_names': ("evolving weights","fixed weights"),
    #'cols': ("eval_loss_1","eval_loss_2","eval_loss_3","eval_loss_4","exp mean square error","std mean square error"),
    #'names': ("evaluation loss 1","evaluation loss 2","evaluation loss 3","evaluation loss 4","conditional expectation MSE",
    #          "conditional standard deviation MSE"),
    'cols': ("exp mean square error", "std mean square error"), # "std mean square error"),
    'names': ("conditional expectation MSE","conditional standard deviation MSE"), # "conditional standard deviation MSE"),
    'saved_models_path': AD_OrnsteinUhlenbeckWithSeason_models_path_experiments,
    'file_name': "varying_weights_metric_evol_{}.png",
}

AD_OrnsteinUhlenbeckWithSeason_loss_metrics_experiments = {
    'model_ids':(1,4), #(2,3),
    'model_names': ("base","with var. pos. term"),
    #'cols': ("eval_loss_1","eval_loss_2","eval_loss_3","eval_loss_4","exp mean square error","std mean square error"),
    #'names': ("evaluation loss 1","evaluation loss 2","evaluation loss 3","evaluation loss 4","conditional expectation MSE",
    #          "conditional standard deviation MSE"),
    'cols': ("eval_loss_1", "eval_loss_2", "eval_loss_3", "eval_loss_4", "exp mean square error", "std mean square error"), # "std mean square error"),
    'names': ("Evaluation loss 1","Evaluation loss 2","Evaluation loss 3","Evaluation loss 4","conditional expectation MSE","conditional standard deviation MSE"), # "conditional standard deviation MSE"),
    'saved_models_path': AD_OrnsteinUhlenbeckWithSeason_models_path_experiments,
}



# comparision the the loss with square inside and outside -> here the model only predicts the conditional expectation
AD_OrnsteinUhlenbeckWithSeason_models_path_modif_loss = "{}saved_models_AD_OrnsteinUhlenbeckWithSeason_modif_loss/".format(data_path)
param_list_AD_OrnsteinUhlenbeckWithSeason_modif_loss = []
param_dict_AD_OrnsteinUhlenbeckWithSeason_1_modif_loss = {
        'epochs': [50],
        'batch_size': [batch_size],
        'save_every': [1],
        'learning_rate': [learning_rate],
        'test_size': [test_size],
        'seed': [seed],
        'hidden_size': [hidden_size],
        'bias': [True],
        'dropout_rate': [dropout_rate],
        'ode_nn': [ode_nn_1],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [use_rnn],
        'input_sig': [input_sig],
        'func_appl_X': [func_appl_X],
        'add_pred': [[]],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor],
        'weight': [weight],
        'weight_evolve': [None],
        'data_dict': ['AD_OrnsteinUhlenbeckWithSeason_3_dict'],
        'plot': [True],
        'which_loss': ['easy_bis'], #, 'easy'],
        'evaluate': [True],
        'eval_metrics': [['exp', 'exp2']],
        'paths_to_plot': [paths_to_plot],
        'plot_variance': [plot_variance],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_models_path_modif_loss],
        'use_cond_exp': [True],
        'input_current_t': [input_current_t],
        'periodic_current_t': [True],
        'validation_size': [1000],
        'train_data_perc': [train_data_perc],
        'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
    }
param_list_AD_OrnsteinUhlenbeckWithSeason_modif_loss += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_1_modif_loss)

AD_OrnsteinUhlenbeckWithSeason_loss_metrics_modif_loss = {
    'model_ids':(9,10), #(2,3),
    'model_names': ("previous version", "new version"),
    #'cols': ("eval_loss_1","eval_loss_2","eval_loss_3","eval_loss_4","exp mean square error","std mean square error"),
    #'names': ("evaluation loss 1","evaluation loss 2","evaluation loss 3","evaluation loss 4","conditional expectation MSE",
    #          "conditional standard deviation MSE"),
    'cols': ("eval_loss", "exp mean square error", "exp2 mean square error"), # "std mean square error"),
    'names': ("Evaluation loss","conditional expectation MSE","conditional moment 2 MSE"), # "conditional standard deviation MSE"),
    'saved_models_path': AD_OrnsteinUhlenbeckWithSeason_models_path_modif_loss,
}

###### OTHER SYNTHETIC DATA

# two dimensions (with cross dependence)
AD_OrnsteinUhlenbeckWithSeason_models_path_dim2 = "{}saved_models_AD_OrnsteinUhlenbeckWithSeason_dim2/".format(data_path)
param_list_AD_OrnsteinUhlenbeckWithSeason_dim2 = []
param_dict_AD_OrnsteinUhlenbeckWithSeason_1_dim2 = {
        'epochs': [200],
        'batch_size': [batch_size],
        'save_every': [1],
        'learning_rate': [learning_rate],
        'test_size': [test_size],
        'seed': [seed],
        'hidden_size': [hidden_size],
        'bias': [True],
        'dropout_rate': [dropout_rate],
        'ode_nn': [ode_nn_2],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [use_rnn],
        'input_sig': [input_sig],
        'func_appl_X': [func_appl_X],
        'add_pred': [add_pred],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor],
        'weight': [weight],
        'weight_evolve': [weight_evolve],
        'data_dict': ['AD_OrnsteinUhlenbeckWithSeason_dim2_dict'],
        'plot': [True],
        # 'plot_only': [True],
        'which_loss': ['variance'],
        'which_eval_loss': ['eval_variance'],
        'evaluate': [True],
        'eval_metrics': [eval_metrics],
        'paths_to_plot': [paths_to_plot],
        'plot_variance': [plot_variance],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_models_path_dim2],
        'use_cond_exp': [True],
        'input_current_t': [input_current_t],
        'periodic_current_t': [True],
        'validation_size': [1000],
        'train_data_perc': [train_data_perc],
        'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
    }
param_list_AD_OrnsteinUhlenbeckWithSeason_dim2 += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_1_dim2)

# 5 seasons instead of 2 -> test for longer TS + test relevance of adding current t with features (cos, sin)
AD_OrnsteinUhlenbeckWithSeason_models_path_5seas = "{}saved_models_AD_OrnsteinUhlenbeckWithSeason_5seas/".format(data_path)
param_list_AD_OrnsteinUhlenbeckWithSeason_5seas = []
param_dict_AD_OrnsteinUhlenbeckWithSeason_1_5seas = {
        'epochs': [50],
        'batch_size': [batch_size],
        'save_every': [1],
        'learning_rate': [learning_rate],
        'test_size': [test_size],
        'seed': [seed],
        'hidden_size': [hidden_size],
        'bias': [bias],
        'dropout_rate': [dropout_rate],
        'ode_nn': [ode_nn_1],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [use_rnn],
        'input_sig': [input_sig],
        'func_appl_X': [func_appl_X],
        'add_pred': [add_pred],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor],
        'weight': [weight],
        'weight_evolve': [weight_evolve],
        'data_dict': ['AD_OrnsteinUhlenbeckWithSeason_5seas_dict'],
        'plot': [True],
        'which_loss': ['variance'],
        'which_eval_loss': ['eval_variance'],
        'evaluate': [True],
        'eval_metrics': [eval_metrics],
        'paths_to_plot': [(0,1,2,3,4,)],
        'plot_variance': [plot_variance],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_models_path_5seas],
        'use_cond_exp': [True],
        'input_current_t': [input_current_t],
        'periodic_current_t': [False],
        'validation_size': [validation_size],
        'train_data_perc': [train_data_perc],
        'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
    }
param_list_AD_OrnsteinUhlenbeckWithSeason_5seas += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_1_5seas)
param_dict_AD_OrnsteinUhlenbeckWithSeason_2_5seas = {
        'epochs': [50],
        'batch_size': [batch_size],
        'save_every': [1],
        'learning_rate': [learning_rate],
        'test_size': [test_size],
        'seed': [seed],
        'hidden_size': [hidden_size],
        'bias': [bias],
        'dropout_rate': [dropout_rate],
        'ode_nn': [ode_nn_1],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [use_rnn],
        'input_sig': [input_sig],
        'func_appl_X': [func_appl_X],
        'add_pred': [add_pred],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor],
        'weight': [weight],
        'weight_evolve': [weight_evolve],
        'data_dict': ['AD_OrnsteinUhlenbeckWithSeason_5seas_dict'],
        'plot': [True],
        'which_loss': ['variance'],
        'which_eval_loss': ['eval_variance'],
        'evaluate': [True],
        'eval_metrics': [eval_metrics],
        'paths_to_plot': [(0,1,2,3,4,)],
        'plot_variance': [plot_variance],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        'saved_models_path': [AD_OrnsteinUhlenbeckWithSeason_models_path_5seas],
        'use_cond_exp': [True],
        'input_current_t': [input_current_t],
        'periodic_current_t': [True],
        'validation_size': [validation_size],
        'train_data_perc': [train_data_perc],
        'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
    }
param_list_AD_OrnsteinUhlenbeckWithSeason_5seas += get_parameter_array(param_dict=param_dict_AD_OrnsteinUhlenbeckWithSeason_2_5seas)


overview_dict_AD_OrnsteinUhlenbeckWithSeason_5seas = dict(
    ids_from=4, ids_to=5, #len(param_list_AD_OrnsteinUhlenbeckWithSeason_experiments),
    path=AD_OrnsteinUhlenbeckWithSeason_models_path_5seas,
    params_extract_desc=('which_loss', 
                        'scale_dt', # not in all
                        'pre-train', # not in all
                        'weight_evolve', 
                        'train_data_perc', # / 'fixed_data_perc'
                        'fixed_data_perc',
                        'periodic_current_t'),
    val_test_params_extract=(
        ("min", "exp mean square error", "exp mean square error", "min expectation MSE"),
        ("min", "exp mean square error", "exp std square error", "min expectation MSE (std)"),
        ("min", "std mean square error", "std mean square error", "min standart deviation MSE"),
        ("min", "std mean square error", "std std square error", "min standart deviation MSE (std)"),
        ("last", "exp mean square error", "exp mean square error", "last expectation MSE"),
        ("last", "std mean square error", "std mean square error", "last standart deviation MSE"),
        ("average", "exp mean square error", "exp mean square error", "average expectation MSE"),
        ("average", "std mean square error", "std mean square error", "average standart deviation MSE"),
    ),
    sortby=None,
    save_file=None,
)


###### REAL DATA

### for cloud KPI
Cloud_KPI_daily_models_path = "{}saved_models_Cloud_KPI_daily/".format(data_path)
param_list_Cloud_KPI_daily = []
param_dict_Cloud_KPI_daily_1 = {
        'epochs': [10000],
        'batch_size': [160],
        'save_every': [25],
        'learning_rate': [learning_rate],
        'test_size': [test_size],
        'seed': [seed],
        'hidden_size': [hidden_size],
        'bias': [bias],
        'dropout_rate': [dropout_rate],
        'ode_nn': [ode_nn_1],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [use_rnn],
        'input_sig': [input_sig],
        'func_appl_X': [func_appl_X],
        'add_pred': [add_pred],
        'solver': [solver],
        'solver_delta_t_factor': [1],
        'weight': [0],
        'weight_evolve': [{'type':'linear', 'target': 1, 'reach': None}],
        'data_dict': ['Cloud_KPI_daily_transformed2_dict'],   # Cloud_KPI_daily_dict
        'plot': [True],
        'which_loss': ['variance'],
        'which_eval_loss': ['eval_variance'],
        'evaluate': [False],
        'eval_metrics': [eval_metrics],
        'paths_to_plot': [paths_to_plot],
        'plot_variance': [plot_variance],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        'saved_models_path': [Cloud_KPI_daily_models_path],
        'use_cond_exp': [False],
        'input_current_t': [input_current_t],
        'periodic_current_t': [True],
        'validation_size': [validation_size],
        'train_data_perc': [train_data_perc],
        'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
        'pre-train': [True]
    }
param_list_Cloud_KPI_daily += get_parameter_array(param_dict=param_dict_Cloud_KPI_daily_1)


