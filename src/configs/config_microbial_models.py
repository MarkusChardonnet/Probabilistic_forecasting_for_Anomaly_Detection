import numpy as np

from configs.config_utils import get_parameter_array, get_dataset_overview, \
    makedirs, data_path, training_data_path


epochs = 3000
save_every = 100
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
datasets_genus = ["microbial_genus", "microbial_genus_sig_div", "microbial_genus_sig_highab", "microbial_genus_sig_nonzero"]
datasets_otu = ["microbial_otu", "microbial_otu_sig_div", "microbial_otu_sig_highab", "microbial_otu_sig_nonzero"]
dataset_splits = ["all", "no_abx"]

hidden_size = 300
ode_nn = ((300, 'tanh'), (300, 'relu'))
ode_nn1 = ((300, 'tanh'), (300, 'tanh'))
enc_nn = ((200, 'tanh'), (300, 'tanh'))
readout_nn = ((300, 'tanh'), (200, 'tanh'))

# base setup
microbial_genus_models_path = "{}saved_models_microbial_genus/".format(data_path)
param_list_microbial_genus = []

# with rnn and signature
param_dict_microbial_genus_sig_rnn = {
        'dataset': datasets_genus,
        'dataset_split': dataset_splits,
        'epochs': [epochs],
        'batch_size': [batch_size],
        'save_every': [save_every],
        'learning_rate': [learning_rate],
        'seed': [seed],
        'hidden_size': [hidden_size],
        'bias': [bias],
        'dropout_rate': [dropout_rate],
        'ode_nn': [ode_nn],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [True],
        'input_sig': [True],
        'func_appl_X': [[]],              # [["power-2", "power-3", "power-4"]]
        'add_pred': [[]],
        'test': [test],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor],
        'weight': [0.5],
        'plot': [True],
        'which_loss': ['easy'],
        'which_val_loss': ['standard'],
        'evaluate': [False],
        'eval_metrics': [eval_metrics],
        'paths_to_plot': [paths_to_plot],
        'plot_variance': [False],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        'saved_models_path': [microbial_genus_models_path],
        'use_cond_exp': [True],
        'input_current_t': [input_current_t],
        'periodic_current_t': [True],
        'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
        'add_readout_activation': [add_readout_activation], # ('softmax',['id']) ('sum2one',['id'])
        'add_dynamic_cov': [True],
        'pre-train': [10000],
        'zero_weight_init': [False],
    }
param_list_microbial_genus += get_parameter_array(param_dict=param_dict_microbial_genus_sig_rnn)
# with rnn
param_dict_microbial_genus_rnn = {
        'dataset': datasets_genus,
        'dataset_id': dataset_splits,
        'epochs': [epochs],
        'batch_size': [batch_size],
        'save_every': [save_every],
        'learning_rate': [learning_rate],
        'seed': [seed],
        'hidden_size': [hidden_size],
        'bias': [bias],
        'dropout_rate': [dropout_rate],
        'ode_nn': [ode_nn],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [True],
        'input_sig': [False],
        'func_appl_X': [[]],              # [["power-2", "power-3", "power-4"]]
        'add_pred': [[]],
        'test': [test],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor],
        'weight': [0.5],
        'plot': [True],
        'which_loss': ['easy'],
        'which_val_loss': ['standard'],
        'evaluate': [False],
        'eval_metrics': [eval_metrics],
        'paths_to_plot': [paths_to_plot],
        'plot_variance': [False],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        'saved_models_path': [microbial_genus_models_path],
        'use_cond_exp': [True],
        'input_current_t': [input_current_t],
        'periodic_current_t': [True],
        'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
        'add_readout_activation': [add_readout_activation], # ('softmax',['id']) ('sum2one',['id'])
        'add_dynamic_cov': [True],
        'pre-train': [10000],
        'zero_weight_init': [False],
    }
param_list_microbial_genus += get_parameter_array(param_dict=param_dict_microbial_genus_rnn)
# with signature and residual connection
param_dict_microbial_genus_sig_res = {
        'dataset': datasets_genus,
        'dataset_id': dataset_splits,
        'epochs': [epochs],
        'batch_size': [batch_size],
        'save_every': [save_every],
        'learning_rate': [learning_rate],
        'seed': [seed],
        'hidden_size': [hidden_size],
        'bias': [bias],
        'dropout_rate': [dropout_rate],
        'ode_nn': [ode_nn],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [False],
        'input_sig': [True],
        'residual_enc_dec': [True],
        'func_appl_X': [[]],              # [["power-2", "power-3", "power-4"]]
        'add_pred': [[]],
        'test': [test],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor],
        'weight': [0.5],
        'plot': [True],
        'which_loss': ['easy'],
        'which_val_loss': ['standard'],
        'evaluate': [False],
        'eval_metrics': [eval_metrics],
        'paths_to_plot': [paths_to_plot],
        'plot_variance': [False],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        'saved_models_path': [microbial_genus_models_path],
        'use_cond_exp': [True],
        'input_current_t': [input_current_t],
        'periodic_current_t': [True],
        'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
        'add_readout_activation': [add_readout_activation], # ('softmax',['id']) ('sum2one',['id'])
        'add_dynamic_cov': [True],
        'pre-train': [0],
        'zero_weight_init': [True]
    }
param_list_microbial_genus += get_parameter_array(param_dict=param_dict_microbial_genus_sig_res)
# with residual connection
param_dict_microbial_genus_res = {
        'dataset': datasets_genus,
        'dataset_id': dataset_splits,
        'epochs': [epochs],
        'batch_size': [batch_size],
        'save_every': [save_every],
        'learning_rate': [learning_rate],
        'seed': [seed],
        'hidden_size': [hidden_size],
        'bias': [bias],
        'dropout_rate': [dropout_rate],
        'ode_nn': [ode_nn],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [False],
        'input_sig': [False],
        'residual_enc_dec': [True],
        'func_appl_X': [[]],              # [["power-2", "power-3", "power-4"]]
        'add_pred': [[]],
        'test': [test],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor],
        'weight': [0.5],
        'plot': [True],
        'which_loss': ['easy'],
        'which_val_loss': ['standard'],
        'evaluate': [False],
        'eval_metrics': [eval_metrics],
        'paths_to_plot': [paths_to_plot],
        'plot_variance': [False],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        'saved_models_path': [microbial_genus_models_path],
        'use_cond_exp': [True],
        'input_current_t': [input_current_t],
        'periodic_current_t': [True],
        'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
        'add_readout_activation': [add_readout_activation], # ('softmax',['id']) ('sum2one',['id'])
        'add_dynamic_cov': [True],
        'pre-train': [0],
        'zero_weight_init': [True],
    }
param_list_microbial_genus += get_parameter_array(param_dict=param_dict_microbial_genus_res)


microbial_otu_models_path = "{}saved_models_microbial_otu/".format(data_path)
param_list_microbial_otu = []
# with rnn and signature
param_dict_microbial_otu_sig_rnn = {
        'resume_training': [True],
        'dataset': datasets_otu,
        'dataset_split': dataset_splits,
        'epochs': [epochs],
        'batch_size': [batch_size],
        'save_every': [save_every],
        'learning_rate': [learning_rate],
        'seed': [seed],
        'hidden_size': [hidden_size],
        'bias': [bias],
        'dropout_rate': [dropout_rate],
        'ode_nn': [ode_nn],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [True],
        'input_sig': [True],
        'func_appl_X': [[]],              # [["power-2", "power-3", "power-4"]]
        'add_pred': [[]],
        'test': [test],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor],
        'weight': [0.5],
        'plot': [True],
        'which_loss': ['easy'],
        'which_val_loss': ['standard'],
        'evaluate': [False],
        'eval_metrics': [eval_metrics],
        'paths_to_plot': [paths_to_plot],
        'plot_variance': [False],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        'saved_models_path': [microbial_otu_models_path],
        'use_cond_exp': [True],
        'input_current_t': [input_current_t],
        'periodic_current_t': [True],
        'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
        'add_readout_activation': [add_readout_activation], # ('softmax',['id']) ('sum2one',['id'])
        'add_dynamic_cov': [True],
        'pre-train': [10000],
        'zero_weight_init': [False],
    }
param_list_microbial_otu += get_parameter_array(param_dict=param_dict_microbial_otu_sig_rnn)
# with rnn
param_dict_microbial_otu_rnn = {
        'resume_training': [True],
        'dataset': datasets_otu,
        'dataset_id': dataset_splits,
        'epochs': [epochs],
        'batch_size': [batch_size],
        'save_every': [save_every],
        'learning_rate': [learning_rate],
        'seed': [seed],
        'hidden_size': [hidden_size],
        'bias': [bias],
        'dropout_rate': [dropout_rate],
        'ode_nn': [ode_nn],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [True],
        'input_sig': [False],
        'func_appl_X': [[]],              # [["power-2", "power-3", "power-4"]]
        'add_pred': [[]],
        'test': [test],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor],
        'weight': [0.5],
        'plot': [True],
        'which_loss': ['easy'],
        'which_val_loss': ['standard'],
        'evaluate': [False],
        'eval_metrics': [eval_metrics],
        'paths_to_plot': [paths_to_plot],
        'plot_variance': [False],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        'saved_models_path': [microbial_otu_models_path],
        'use_cond_exp': [True],
        'input_current_t': [input_current_t],
        'periodic_current_t': [True],
        'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
        'add_readout_activation': [add_readout_activation], # ('softmax',['id']) ('sum2one',['id'])
        'add_dynamic_cov': [True],
        'pre-train': [10000],
        'zero_weight_init': [False],
    }
param_list_microbial_otu += get_parameter_array(param_dict=param_dict_microbial_otu_rnn)
# with signature and residual connection
param_dict_microbial_otu_sig_res = {
        'resume_training': [True],
        'dataset': datasets_otu,
        'dataset_id': dataset_splits,
        'epochs': [epochs],
        'batch_size': [batch_size],
        'save_every': [save_every],
        'learning_rate': [learning_rate],
        'seed': [seed],
        'hidden_size': [hidden_size],
        'bias': [bias],
        'dropout_rate': [dropout_rate],
        'ode_nn': [ode_nn],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [False],
        'input_sig': [True],
        'residual_enc_dec': [True],
        'func_appl_X': [[]],              # [["power-2", "power-3", "power-4"]]
        'add_pred': [[]],
        'test': [test],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor],
        'weight': [0.5],
        'plot': [True],
        'which_loss': ['easy'],
        'which_val_loss': ['standard'],
        'evaluate': [False],
        'eval_metrics': [eval_metrics],
        'paths_to_plot': [paths_to_plot],
        'plot_variance': [False],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        'saved_models_path': [microbial_otu_models_path],
        'use_cond_exp': [True],
        'input_current_t': [input_current_t],
        'periodic_current_t': [True],
        'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
        'add_readout_activation': [add_readout_activation], # ('softmax',['id']) ('sum2one',['id'])
        'add_dynamic_cov': [True],
        'pre-train': [0],
        'zero_weight_init': [True]
    }
param_list_microbial_otu += get_parameter_array(param_dict=param_dict_microbial_otu_sig_res)
# with residual connection
param_dict_microbial_otu_res = {
        'resume_training': [True],
        'dataset': datasets_otu,
        'dataset_id': dataset_splits,
        'epochs': [epochs],
        'batch_size': [batch_size],
        'save_every': [save_every],
        'learning_rate': [learning_rate],
        'seed': [seed],
        'hidden_size': [hidden_size],
        'bias': [bias],
        'dropout_rate': [dropout_rate],
        'ode_nn': [ode_nn],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [False],
        'input_sig': [False],
        'residual_enc_dec': [True],
        'func_appl_X': [[]],              # [["power-2", "power-3", "power-4"]]
        'add_pred': [[]],
        'test': [test],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor],
        'weight': [0.5],
        'plot': [True],
        'which_loss': ['easy'],
        'which_val_loss': ['standard'],
        'evaluate': [False],
        'eval_metrics': [eval_metrics],
        'paths_to_plot': [paths_to_plot],
        'plot_variance': [False],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        'saved_models_path': [microbial_otu_models_path],
        'use_cond_exp': [True],
        'input_current_t': [input_current_t],
        'periodic_current_t': [True],
        'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
        'add_readout_activation': [add_readout_activation], # ('softmax',['id']) ('sum2one',['id'])
        'add_dynamic_cov': [True],
        'pre-train': [0],
        'zero_weight_init': [True],
    }
param_list_microbial_otu += get_parameter_array(param_dict=param_dict_microbial_otu_res)

# OTU

# dataset: microbial_otu
# dataset split: all
# compare types of models
plot_metrics_otu_all = {
    # 'model_ids':(1,9,17,25),
    'model_ids':(17,25),
    'model_names':(# 'With signature and RNN',
                   # 'With RNN',
                   'With signature and residual connections',
                   'With residual connections'),
    'file_name':"loss_evol-otu-all-{}.png",
    'cols':('train_loss', 'eval_loss', 'val_loss'),
    'names':('train_loss', 'eval_loss', 'val_loss'),
    'saved_models_path':microbial_otu_models_path,
}
# dataset: microbial_otu
# dataset split: no_abx
# compare types of models
plot_metrics_otu_no_abx = {
    # 'model_ids':(2,10,18,26),
    'model_ids':(18,26),
    'model_names':(# 'With signature and RNN',
                   # 'With RNN',
                   'With signature and residual connections',
                   'With residual connections'),
    'file_name':"loss_evol-otu-no_abx-{}.png",
    'cols':('train_loss', 'eval_loss', 'val_loss'),
    'names':('train_loss', 'eval_loss', 'val_loss'),
    'saved_models_path':microbial_otu_models_path,
}

# dataset: microbial_otu_sig_div
# dataset split: all
# compare types of models
plot_metrics_otu_sig_div_all = {
    # 'model_ids':(3,11,19,27),
    'model_ids':(19,27),
    'model_names':(# 'With signature and RNN',
                   # 'With RNN',
                   'With signature and residual connections',
                   'With residual connections'),
    'file_name':"loss_evol-otu_sig_div-all-{}.png",
    'cols':('train_loss', 'eval_loss', 'val_loss'),
    'names':('train_loss', 'eval_loss', 'val_loss'),
    'saved_models_path':microbial_otu_models_path,
}
# dataset: microbial_otu_sig_div
# dataset split: no_abx
# compare types of models
plot_metrics_otu_sig_div_no_abx = {
    # 'model_ids':(4,12,20,28),
    'model_ids':(20,28),
    'model_names':(# 'With signature and RNN',
                   # 'With RNN',
                   'With signature and residual connections',
                   'With residual connections'),
    'file_name':"loss_evol-otu_sig_div-no_abx-{}.png",
    'cols':('train_loss', 'eval_loss', 'val_loss'),
    'names':('train_loss', 'eval_loss', 'val_loss'),
    'saved_models_path':microbial_otu_models_path,
}

# dataset: microbial_otu_sig_highab
# dataset split: all
# compare types of models
plot_metrics_otu_sig_highab_all = {
    # 'model_ids':(5,13,21,29),
    'model_ids':(21,29),
    'model_names':(# 'With signature and RNN',
                   # 'With RNN',
                   'With signature and residual connections',
                   'With residual connections'),
    'file_name':"loss_evol-otu_sig_highab-all-{}.png",
    'cols':('train_loss', 'eval_loss', 'val_loss'),
    'names':('train_loss', 'eval_loss', 'val_loss'),
    'saved_models_path':microbial_otu_models_path,
}
# dataset: microbial_otu_sig_highab
# dataset split: no_abx
# compare types of models
plot_metrics_otu_sig_highab_no_abx = {
    # 'model_ids':(6,14,22,30),
    'model_ids':(22,30),
    'model_names':(# 'With signature and RNN',
                   # 'With RNN',
                   'With signature and residual connections',
                   'With residual connections'),
    'file_name':"loss_evol-otu_sig_highab-no_abx-{}.png",
    'cols':('train_loss', 'eval_loss', 'val_loss'),
    'names':('train_loss', 'eval_loss', 'val_loss'),
    'saved_models_path':microbial_otu_models_path,
}

# dataset: microbial_otu_sig_nonzero
# dataset split: all
# compare types of models
plot_metrics_otu_sig_nonzero_all = {
    # 'model_ids':(7,15,23,31),
    'model_ids':(23,31),
    'model_names':(# 'With signature and RNN',
                   # 'With RNN',
                   'With signature and residual connections',
                   'With residual connections'),
    'file_name':"loss_evol-otu_sig_nonzero-all-{}.png",
    'cols':('train_loss', 'eval_loss', 'val_loss'),
    'names':('train_loss', 'eval_loss', 'val_loss'),
    'saved_models_path':microbial_otu_models_path,
}
# dataset: microbial_otu_sig_nonzero
# dataset split: no_abx
# compare types of models
plot_metrics_otu_sig_nonzero_no_abx = {
    # 'model_ids':(8,16,24,32),
    'model_ids':(24,32),
    'model_names':(# 'With signature and RNN',
                   # 'With RNN',
                   'With signature and residual connections',
                   'With residual connections'),
    'file_name':"loss_evol-otu_sig_nonzero-no_abx-{}.png",
    'cols':('train_loss', 'eval_loss', 'val_loss'),
    'names':('train_loss', 'eval_loss', 'val_loss'),
    'saved_models_path':microbial_otu_models_path,
}

# dataset: microbial_otu_sig_nonzero
# dataset split: all
# compare types of models
plot_metrics_otu_model_sig_res_all = {
    'model_ids':(17,19,21,23),
    'model_names':('Same features for signature',
                   'Diversity features for signature',
                   'High abundance features for signature',
                   'Nonzero features for signature'),
    'file_name':"loss_evol-otu-model_sig_res-all-{}.png",
    'cols':('train_loss', 'eval_loss', 'val_loss'),
    'names':('train_loss', 'eval_loss', 'val_loss'),
    'saved_models_path':microbial_otu_models_path,
}
# dataset: microbial_otu_sig_nonzero
# dataset split: no_abx
# compare types of models
plot_metrics_otu_model_sig_res_no_abx = {
    'model_ids':(18,20,22,24),
    'model_names':('Same features for signature',
                   'Diversity features for signature',
                   'High abundance features for signature',
                   'Nonzero features for signature'),
    'file_name':"loss_evol-otu-model_sig_res-no_abx-{}.png",
    'cols':('train_loss', 'eval_loss', 'val_loss'),
    'names':('train_loss', 'eval_loss', 'val_loss'),
    'saved_models_path':microbial_otu_models_path,
}

# GENUS

# dataset: microbial_genus
# dataset split: all
# compare types of models
plot_metrics_genus_all = {
    'model_ids':(1,9,17,25),
    # 'model_ids':(17,25),
    'model_names':('With signature and RNN',
                   'With RNN',
                   'With signature and residual connections',
                   'With residual connections'),
    'file_name':"loss_evol-genus-all-{}.png",
    'cols':('train_loss', 'eval_loss', 'val_loss'),
    'names':('train_loss', 'eval_loss', 'val_loss'),
    'saved_models_path':microbial_genus_models_path,
}
# dataset: microbial_genus
# dataset split: no_abx
# compare types of models
plot_metrics_genus_no_abx = {
    'model_ids':(2,10,18,26),
    # 'model_ids':(18,26),
    'model_names':('With signature and RNN',
                   'With RNN',
                   'With signature and residual connections',
                   'With residual connections'),
    'file_name':"loss_evol-genus-no_abx-{}.png",
    'cols':('train_loss', 'eval_loss', 'val_loss'),
    'names':('train_loss', 'eval_loss', 'val_loss'),
    'saved_models_path':microbial_genus_models_path,
}

# dataset: microbial_genus_sig_div
# dataset split: all
# compare types of models
plot_metrics_genus_sig_div_all = {
    'model_ids':(3,11,19,27),
    # 'model_ids':(19,27),
    'model_names':('With signature and RNN',
                   'With RNN',
                   'With signature and residual connections',
                   'With residual connections'),
    'file_name':"loss_evol-genus_sig_div-all-{}.png",
    'cols':('train_loss', 'eval_loss', 'val_loss'),
    'names':('train_loss', 'eval_loss', 'val_loss'),
    'saved_models_path':microbial_genus_models_path,
}
# dataset: microbial_genus_sig_div
# dataset split: no_abx
# compare types of models
plot_metrics_genus_sig_div_no_abx = {
    'model_ids':(4,12,20,28),
    # 'model_ids':(20,28),
    'model_names':('With signature and RNN',
                   'With RNN',
                   'With signature and residual connections',
                   'With residual connections'),
    'file_name':"loss_evol-genus_sig_div-no_abx-{}.png",
    'cols':('train_loss', 'eval_loss', 'val_loss'),
    'names':('train_loss', 'eval_loss', 'val_loss'),
    'saved_models_path':microbial_genus_models_path,
}

# dataset: microbial_genus_sig_highab
# dataset split: all
# compare types of models
plot_metrics_genus_sig_highab_all = {
    'model_ids':(5,13,21,29),
    # 'model_ids':(21,29),
    'model_names':('With signature and RNN',
                   'With RNN',
                   'With signature and residual connections',
                   'With residual connections'),
    'file_name':"loss_evol-genus_sig_highab-all-{}.png",
    'cols':('train_loss', 'eval_loss', 'val_loss'),
    'names':('train_loss', 'eval_loss', 'val_loss'),
    'saved_models_path':microbial_genus_models_path,
}
# dataset: microbial_genus_sig_highab
# dataset split: no_abx
# compare types of models
plot_metrics_genus_sig_highab_no_abx = {
    'model_ids':(6,14,22,30),
    # 'model_ids':(22,30),
    'model_names':('With signature and RNN',
                   'With RNN',
                   'With signature and residual connections',
                   'With residual connections'),
    'file_name':"loss_evol-genus_sig_highab-no_abx-{}.png",
    'cols':('train_loss', 'eval_loss', 'val_loss'),
    'names':('train_loss', 'eval_loss', 'val_loss'),
    'saved_models_path':microbial_genus_models_path,
}

# dataset: microbial_genus_sig_nonzero
# dataset split: all
# compare types of models
plot_metrics_genus_sig_nonzero_all = {
    'model_ids':(7,15,23,31),
    # 'model_ids':(23,31),
    'model_names':('With signature and RNN',
                   'With RNN',
                   'With signature and residual connections',
                   'With residual connections'),
    'file_name':"loss_evol-genus_sig_nonzero-all-{}.png",
    'cols':('train_loss', 'eval_loss', 'val_loss'),
    'names':('train_loss', 'eval_loss', 'val_loss'),
    'saved_models_path':microbial_genus_models_path,
}
# dataset: microbial_genus_sig_nonzero
# dataset split: no_abx
# compare types of models
plot_metrics_genus_sig_nonzero_no_abx = {
    'model_ids':(8,16,24,32),
    # 'model_ids':(24,32),
    'model_names':('With signature and RNN',
                   'With RNN',
                   'With signature and residual connections',
                   'With residual connections'),
    'file_name':"loss_evol-genus_sig_nonzero-no_abx-{}.png",
    'cols':('train_loss', 'eval_loss', 'val_loss'),
    'names':('train_loss', 'eval_loss', 'val_loss'),
    'saved_models_path':microbial_genus_models_path,
}

# dataset: microbial_genus_sig_nonzero
# dataset split: all
# compare types of models
plot_metrics_genus_model_sig_res_all = {
    'model_ids':(17,19,21,23),
    'model_names':('Same features for signature',
                   'Diversity features for signature',
                   'High abundance features for signature',
                   'Nonzero features for signature'),
    'file_name':"loss_evol-genus-model_sig_res-all-{}.png",
    'cols':('train_loss', 'eval_loss', 'val_loss'),
    'names':('train_loss', 'eval_loss', 'val_loss'),
    'saved_models_path':microbial_genus_models_path,
}
# dataset: microbial_genus_sig_nonzero
# dataset split: no_abx
# compare types of models
plot_metrics_genus_model_sig_res_no_abx = {
    'model_ids':(18,20,22,24),
    'model_names':('Same features for signature',
                   'Diversity features for signature',
                   'High abundance features for signature',
                   'Nonzero features for signature'),
    'file_name':"loss_evol-genus-model_sig_res-no_abx-{}.png",
    'cols':('train_loss', 'eval_loss', 'val_loss'),
    'names':('train_loss', 'eval_loss', 'val_loss'),
    'saved_models_path':microbial_genus_models_path,
}

param_list_microbial_otu_nn = []
nns = [((100, 'tanh'), (100, 'relu')),
       ((300, 'tanh'), (300, 'relu')),
       ((500, 'tanh'), (500, 'relu')),
       ((800, 'tanh'), (800, 'relu')),
       ((300, 'tanh'), (300, 'relu'), (300, 'relu')),]
for nn in nns:
    param_list_microbial_otu_nn_1 = {
            'resume_training': [True],
            'dataset': ["microbial_otu_sig_highab"],
            'dataset_id': dataset_splits,
            'epochs': [epochs],
            'batch_size': [batch_size],
            'save_every': [save_every],
            'learning_rate': [learning_rate],
            'seed': [seed],
            'hidden_size': [hidden_size],
            'bias': [bias],
            'dropout_rate': [dropout_rate],
            'ode_nn': [nn],
            'readout_nn': [nn],
            'enc_nn': [nn],
            'use_rnn': [False],
            'input_sig': [True, False],
            'residual_enc_dec': [True],
            'func_appl_X': [[]],              # [["power-2", "power-3", "power-4"]]
            'add_pred': [[]],
            'test': [test],
            'solver': [solver],
            'solver_delta_t_factor': [solver_delta_t_factor],
            'weight': [0.5],
            'plot': [True],
            'which_loss': ['easy'],
            'which_val_loss': ['standard'],
            'evaluate': [False],
            'eval_metrics': [eval_metrics],
            'paths_to_plot': [paths_to_plot],
            'plot_variance': [False],
            'std_factor': [std_factor],
            'plot_moments': [plot_moments],
            'saved_models_path': [microbial_otu_models_path],
            'use_cond_exp': [True],
            'input_current_t': [input_current_t],
            'periodic_current_t': [True],
            'scale_dt': [scale_dt],
            'enc_input_t': [enc_input_t],
            'add_readout_activation': [add_readout_activation], # ('softmax',['id']) ('sum2one',['id'])
            'add_dynamic_cov': [True],
            'pre-train': [0],
            'zero_weight_init': [True],
        }
    param_list_microbial_otu_nn += get_parameter_array(param_dict=param_list_microbial_otu_nn_1)

# dataset: microbial_otu_sig_highab
# dataset split: all
# model type : sig + res
# compare nn sizes
plot_metrics_otu_nnsizes_sig_res_all = {
    'model_ids':(53,57,61,65,69),
    'model_names':("((100, 'tanh'), (100, 'relu'))",
                    "((300, 'tanh'), (300, 'relu'))",
                    "((500, 'tanh'), (500, 'relu'))",
                    "((800, 'tanh'), (800, 'relu'))",
                    "((300, 'tanh'), (300, 'relu'), (300, 'relu'))"),
    'file_name':"loss_evol-otu-nnsizes-sig_res-all-{}.png",
    'cols':('train_loss', 'eval_loss', 'val_loss'),
    'names':('train_loss', 'eval_loss', 'val_loss'),
    'saved_models_path':microbial_otu_models_path,
}

# dataset: microbial_otu_sig_highab
# dataset split: no_abx
# model type : sig + res
# compare nn sizes
plot_metrics_otu_nnsizes_sig_res_no_abx = {
    'model_ids':(55,59,63,67,71),
    'model_names':("((100, 'tanh'), (100, 'relu'))",
                    "((300, 'tanh'), (300, 'relu'))",
                    "((500, 'tanh'), (500, 'relu'))",
                    "((800, 'tanh'), (800, 'relu'))",
                    "((300, 'tanh'), (300, 'relu'), (300, 'relu'))"),
    'file_name':"loss_evol-otu-nnsizes-sig_res-no_abx-{}.png",
    'cols':('train_loss', 'eval_loss', 'val_loss'),
    'names':('train_loss', 'eval_loss', 'val_loss'),
    'saved_models_path':microbial_otu_models_path,
}

# dataset: microbial_otu_sig_highab
# dataset split: all
# model type : res
# compare nn sizes
plot_metrics_otu_nnsizes_res_all = {
    'model_ids':(54,58,62,66,70),
    'model_names':("((100, 'tanh'), (100, 'relu'))",
                    "((300, 'tanh'), (300, 'relu'))",
                    "((500, 'tanh'), (500, 'relu'))",
                    "((800, 'tanh'), (800, 'relu'))",
                    "((300, 'tanh'), (300, 'relu'), (300, 'relu'))"),
    'file_name':"loss_evol-otu-nnsizes-res-all-{}.png",
    'cols':('train_loss', 'eval_loss', 'val_loss'),
    'names':('train_loss', 'eval_loss', 'val_loss'),
    'saved_models_path':microbial_otu_models_path,
}

# dataset: microbial_otu_sig_highab
# dataset split: no_abx
# model type : res
# compare nn sizes
plot_metrics_otu_nnsizes_res_no_abx = {
    'model_ids':(56,60,64,68,72),
    'model_names':("((100, 'tanh'), (100, 'relu'))",
                    "((300, 'tanh'), (300, 'relu'))",
                    "((500, 'tanh'), (500, 'relu'))",
                    "((800, 'tanh'), (800, 'relu'))",
                    "((300, 'tanh'), (300, 'relu'), (300, 'relu'))"),
    'file_name':"loss_evol-otu-nnsizes-res-no_abx-{}.png",
    'cols':('train_loss', 'eval_loss', 'val_loss'),
    'names':('train_loss', 'eval_loss', 'val_loss'),
    'saved_models_path':microbial_otu_models_path,
}


# ==============================================================================
# FLORIAN
# ==============================================================================

# ------------------------------------------------------------------------------
# testing on otu:
#   - different loss and eval loss function
#   - use RNN with residual connection to see whether there is really no
#       path-dependency
#   - only train on no_abx and only with highabundance signature features

microbial_otu_models_path2 = "{}saved_models_microbial_otu2/".format(data_path)
param_list_microbial_otu2 = []

param_dict_microbial_otu_sig_rnn = {
        'dataset': ["microbial_otu_sig_highab"],
        'dataset_split': ["no_abx"],
        'epochs': [epochs],
        'batch_size': [batch_size],
        'save_every': [save_every],
        'learning_rate': [learning_rate],
        'seed': [seed],
        'hidden_size': [hidden_size],
        'bias': [bias],
        'dropout_rate': [dropout_rate],
        'ode_nn': [ode_nn, ode_nn1],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [True, False],
        'input_sig': [True, False],
        'residual_enc_dec': [True, False],
        'func_appl_X': [[]],              # [["power-2", "power-3", "power-4"]]
        'add_pred': [[]],
        'test': [test],
        'solver': [solver],
        'solver_delta_t_factor': [solver_delta_t_factor, 1.],
        'weight': [0.5],
        'plot': [True],
        'which_loss': ['easy', 'noisy_obs'],
        'which_eval_loss': ['noisy_obs'],
        'evaluate': [False],
        'eval_metrics': [eval_metrics],
        'paths_to_plot': [paths_to_plot],
        'plot_variance': [False],
        'std_factor': [std_factor],
        'plot_moments': [plot_moments],
        'saved_models_path': [microbial_otu_models_path2],
        'use_cond_exp': [True],
        'input_current_t': [input_current_t],
        'periodic_current_t': [True],
        'scale_dt': [scale_dt],
        'enc_input_t': [enc_input_t],
        'add_readout_activation': [add_readout_activation], # ('softmax',['id']) ('sum2one',['id'])
        'add_dynamic_cov': [True],
        'pre-train': [10000],
        'zero_weight_init': [False],
    }
param_list_microbial_otu2 += get_parameter_array(param_dict=param_dict_microbial_otu_sig_rnn)

overview_dict_microbial_otu2 = dict(
    ids_from=1, ids_to=len(param_list_microbial_otu2),
    path=microbial_otu_models_path2,
    params_extract_desc=('dataset', 'dataset_split',
                         'ode_nn', 'enc_nn', 'readout_nn',
                         'dropout_rate', 'hidden_size', 'batch_size',
                         'which_loss', 'which_eval_loss',
                         'solver_delta_t_factor',
                         'residual_enc_dec', 'use_rnn',
                         'input_sig', 'level', ),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "eval_loss", "eval_loss", "eval_loss_min"),
    ),
    sortby=["eval_loss_min"],
)

plot_paths_microbial_otu2 = {
    'model_ids': [27, 25, 36], 'saved_models_path': microbial_otu_models_path2,
    'which': 'best', 'paths_to_plot': [0,1,2,3,4,5], 'wait_time': 5,
    'save_extras': {'bbox_inches': 'tight', 'pad_inches': 0.01},}


# ------------------------------------------------------------------------------
# also train variance estimator

microbial_otu_models_path3 = "{}saved_models_microbial_otu3/".format(data_path)
param_list_microbial_otu3 = []

for add_pred, which_loss in [
        [["var"], "variance"], [[], "moment_2"], [[], "easy_bis"],
        [[], "noisy_obs"], [["var"], "variance_bis"],
]:
        param_dict_microbial_otu_sig_rnn = {
                'dataset': ["microbial_otu_sig_highab"],
                'dataset_split': ["no_abx"],
                'epochs': [epochs],
                'batch_size': [batch_size],
                'save_every': [save_every],
                'learning_rate': [learning_rate],
                'seed': [seed],
                'hidden_size': [hidden_size],
                'bias': [bias],
                'dropout_rate': [dropout_rate],
                'ode_nn': [ode_nn, ode_nn1],
                'readout_nn': [readout_nn],
                'enc_nn': [enc_nn],
                'use_rnn': [True, False],
                'input_sig': [True, False],
                'residual_enc_dec': [True, False],
                'func_appl_X': [["power-2"]],              # [["power-2", "power-3", "power-4"]]
                'add_pred': [add_pred],
                'test': [test],
                'solver': [solver],
                'solver_delta_t_factor': [solver_delta_t_factor],
                'weight': [0.],
                'weight_evolve': [{'type': 'linear', 'target': 1, 'reach': None}],
                'plot': [True],
                'which_loss': [which_loss],
                'which_eval_loss': ['val_variance'],
                'evaluate': [False],
                'eval_metrics': [eval_metrics],
                'paths_to_plot': [(0,)],
                'plot_variance': [True],
                'std_factor': [std_factor],
                'plot_moments': [plot_moments],
                'saved_models_path': [microbial_otu_models_path3],
                'use_cond_exp': [True],
                'input_current_t': [input_current_t],
                'periodic_current_t': [True],
                'scale_dt': [scale_dt],
                'enc_input_t': [enc_input_t],
                'add_readout_activation': [add_readout_activation], # ('softmax',['id']) ('sum2one',['id'])
                'add_dynamic_cov': [True],
                'pre-train': [10000],
                'zero_weight_init': [False],
            }
        param_list_microbial_otu3 += get_parameter_array(param_dict=param_dict_microbial_otu_sig_rnn)

overview_dict_microbial_otu3 = dict(
    ids_from=1, ids_to=len(param_list_microbial_otu3),
    path=microbial_otu_models_path3,
    params_extract_desc=('dataset', 'dataset_split',
                         'ode_nn', 'enc_nn', 'readout_nn',
                         'dropout_rate', 'hidden_size', 'batch_size',
                         'which_loss', 'which_eval_loss', 'add_pred',
                         'solver_delta_t_factor',
                         'residual_enc_dec', 'use_rnn',
                         'input_sig', 'level', ),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "eval_loss", "eval_loss", "eval_loss_min"),
        ("min", "val_loss", "val_loss", "val_loss_min"),
    ),
    sortby=["val_loss_min"],
)

# ------------------------------------------------------------------------------
# do same for genus

microbial_genus_models_path3 = "{}saved_models_microbial_genus3/".format(data_path)
param_list_microbial_genus3 = []

for add_pred, which_loss in [
        [["var"], "variance"], [[], "moment_2"], [[], "easy_bis"],
        [[], "noisy_obs"], [["var"], "variance_bis"],
]:
        param_dict_microbial_genus_sig_rnn = {
                'dataset': ["microbial_genus_sig_highab"],
                'dataset_split': ["no_abx", "all"],
                'epochs': [epochs],
                'batch_size': [batch_size],
                'save_every': [save_every],
                'learning_rate': [learning_rate],
                'seed': [seed],
                'hidden_size': [hidden_size],
                'bias': [bias],
                'dropout_rate': [dropout_rate],
                'ode_nn': [ode_nn, ode_nn1],
                'readout_nn': [readout_nn],
                'enc_nn': [enc_nn],
                'use_rnn': [True, False],
                'input_sig': [True, False],
                'residual_enc_dec': [True, False],
                'func_appl_X': [["power-2"]],              # [["power-2", "power-3", "power-4"]]
                'add_pred': [add_pred],
                'test': [test],
                'solver': [solver],
                'solver_delta_t_factor': [solver_delta_t_factor],
                'weight': [0.],
                'weight_evolve': [{'type': 'linear', 'target': 1, 'reach': None}],
                'plot': [True],
                'which_loss': [which_loss],
                'which_eval_loss': ['val_variance'],
                'evaluate': [False],
                'eval_metrics': [eval_metrics],
                'paths_to_plot': [(0,)],
                'plot_variance': [True],
                'std_factor': [std_factor],
                'plot_moments': [plot_moments],
                'saved_models_path': [microbial_genus_models_path3],
                'use_cond_exp': [True],
                'input_current_t': [input_current_t],
                'periodic_current_t': [True],
                'scale_dt': [scale_dt],
                'enc_input_t': [enc_input_t],
                'add_readout_activation': [add_readout_activation], # ('softmax',['id']) ('sum2one',['id'])
                'add_dynamic_cov': [True],
                'pre-train': [10000],
                'zero_weight_init': [False],
            }
        param_list_microbial_genus3 += get_parameter_array(
                param_dict=param_dict_microbial_genus_sig_rnn)

overview_dict_microbial_genus3 = dict(
    ids_from=1, ids_to=len(param_list_microbial_genus3),
    path=microbial_genus_models_path3,
    params_extract_desc=('dataset', 'dataset_split',
                         'ode_nn', 'enc_nn', 'readout_nn',
                         'dropout_rate', 'hidden_size', 'batch_size',
                         'which_loss', 'which_eval_loss', 'add_pred',
                         'solver_delta_t_factor',
                         'residual_enc_dec', 'use_rnn',
                         'input_sig', 'level', ),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "eval_loss", "eval_loss", "eval_loss_min"),
        ("min", "val_loss", "val_loss", "val_loss_min"),
    ),
    sortby=["val_loss_min"],
)


# ------------------------------------------------------------------------------
# lower dim datasets, reducing the features with low variance

microbial_models_path_lowvar = "{}saved_models_microbial_lowvar/".format(data_path)
param_list_microbial_lowvar = []

for add_pred, which_loss in [
        [["var"], "variance_bis"],
]:
        param_dict_microbial_sig_rnn_lowvar = {
                'dataset': ["microbial_otu_sig_highab_lowvar5",
                            "microbial_otu_sig_highab_lowvar94q",
                            "microbial_genus_sig_highab_lowvar5",
                            "microbial_genus_sig_highab_lowvar94q"],
                'dataset_split': ["no_abx", "all"],
                'epochs': [epochs],
                'batch_size': [batch_size],
                'save_every': [save_every],
                'learning_rate': [learning_rate],
                'seed': [seed],
                'hidden_size': [hidden_size],
                'bias': [bias],
                'dropout_rate': [dropout_rate],
                'ode_nn': [ode_nn1],  # ode_nn, ode_nn1
                'readout_nn': [readout_nn],
                'enc_nn': [enc_nn],
                'use_rnn': [True, False],
                'input_sig': [True, False],
                'residual_enc_dec': [True, False],
                'func_appl_X': [["power-2"]],              # [["power-2", "power-3", "power-4"]]
                'add_pred': [add_pred],
                'test': [test],
                'solver': [solver],
                'solver_delta_t_factor': [solver_delta_t_factor],
                'weight': [0.],
                'weight_evolve': [{'type': 'linear', 'target': 1, 'reach': None}],
                'plot': [True],
                'which_loss': [which_loss],
                'which_eval_loss': ['val_variance'],
                'evaluate': [False],
                'eval_metrics': [eval_metrics],
                'paths_to_plot': [(0,)],
                'plot_variance': [True],
                'std_factor': [std_factor],
                'plot_moments': [plot_moments],
                'saved_models_path': [microbial_models_path_lowvar],
                'use_cond_exp': [True],
                'input_current_t': [input_current_t],
                'periodic_current_t': [True],
                'scale_dt': [scale_dt],
                'enc_input_t': [enc_input_t],
                'add_readout_activation': [('sum2one',['id']), (None, [])], # add_readout_activation # ('softmax',['id']) ('sum2one',['id'])
                'add_dynamic_cov': [True],
                'pre-train': [10000],
                'zero_weight_init': [False],
            }
        param_list_microbial_lowvar += get_parameter_array(
                param_dict=param_dict_microbial_sig_rnn_lowvar)

overview_dict_microbial_lowvar = dict(
    ids_from=1, ids_to=len(param_list_microbial_lowvar),
    path=microbial_models_path_lowvar,
    params_extract_desc=('dataset', 'dataset_split',
                         'ode_nn', 'enc_nn', 'readout_nn',
                         'dropout_rate', 'hidden_size', 'batch_size',
                         'which_loss', 'which_eval_loss', 'add_pred',
                         'solver_delta_t_factor', 'add_readout_activation',
                         'residual_enc_dec', 'use_rnn',
                         'input_sig', 'level', ),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "eval_loss", "eval_loss", "eval_loss_min"),
        ("min", "val_loss", "val_loss", "val_loss_min"),
    ),
    sortby=["dataset", "val_loss_min"],
)
# -> best performing:
#       - no_abx, residual_enc_dec=True, use_rnn=False, input_sig=False
#       - add_readout_activation no clear indication


# --------------
# other time steps; loss without variance=0 constraint; smaller networks
microbial_models_path_lowvar1 = "{}saved_models_microbial_lowvar1/".format(data_path)
param_list_microbial_lowvar1 = []

for add_pred, which_loss in [
        [["var"], "variance_bis"],
        [["var"], "variance_bis2"],
]:
        param_dict_microbial_sig_rnn_lowvar = {
                'dataset': ["microbial_otu_sig_highab_lowvar5",
                            "microbial_otu_sig_highab_lowvar94q",
                            "microbial_genus_sig_highab_lowvar5",
                            "microbial_genus_sig_highab_lowvar94q"],
                'dataset_split': ["no_abx",],
                'epochs': [epochs],
                'batch_size': [batch_size],
                'save_every': [save_every],
                'learning_rate': [learning_rate],
                'seed': [seed],
                'hidden_size': [100],
                'bias': [bias],
                'dropout_rate': [dropout_rate],
                'ode_nn': [((100, 'tanh'),)],  # ode_nn, ode_nn1
                'readout_nn': [((100, 'tanh'),)],
                'enc_nn': [((100, 'tanh'),)],
                'use_rnn': [False],
                'input_sig': [False],
                'residual_enc_dec': [True],
                'func_appl_X': [["power-2"]],              # [["power-2", "power-3", "power-4"]]
                'add_pred': [add_pred],
                'test': [test],
                'solver': [solver],
                'solver_delta_t_factor': [1/3., 1/7., 1/14.],
                'weight': [0.],
                'weight_evolve': [{'type': 'linear', 'target': 1, 'reach': None}],
                'plot': [True],
                'which_loss': [which_loss],
                'which_eval_loss': ['val_variance'],
                'evaluate': [False],
                'eval_metrics': [eval_metrics],
                'paths_to_plot': [(0,)],
                'plot_variance': [True],
                'std_factor': [std_factor],
                'plot_moments': [plot_moments],
                'saved_models_path': [microbial_models_path_lowvar1],
                'use_cond_exp': [True],
                'input_current_t': [input_current_t],
                'periodic_current_t': [True],
                'scale_dt': [scale_dt],
                'enc_input_t': [enc_input_t],
                'add_readout_activation': [('sum2one',['id']), (None, [])], # add_readout_activation # ('softmax',['id']) ('sum2one',['id'])
                'add_dynamic_cov': [True],
                'pre-train': [10000],
                'zero_weight_init': [False],
            }
        param_list_microbial_lowvar1 += get_parameter_array(
                param_dict=param_dict_microbial_sig_rnn_lowvar)

overview_dict_microbial_lowvar1 = dict(
    ids_from=1, ids_to=len(param_list_microbial_lowvar1),
    path=microbial_models_path_lowvar1,
    params_extract_desc=('dataset', 'dataset_split',
                         'ode_nn', 'enc_nn', 'readout_nn',
                         'dropout_rate', 'hidden_size', 'batch_size',
                         'which_loss', 'which_eval_loss', 'add_pred',
                         'solver_delta_t_factor', 'add_readout_activation',
                         'residual_enc_dec', 'use_rnn',
                         'input_sig', 'level', ),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "eval_loss", "eval_loss", "eval_loss_min"),
        ("min", "val_loss", "val_loss", "val_loss_min"),
        ("min", "sum--val_loss_1--val_loss_2--val_loss_4",
         "sum--val_loss_1--val_loss_2--val_loss_4", "corrected_val_loss_min"),
    ),
    sortby=["dataset", "corrected_val_loss_min"],
)


# --------------
# lr scheduler
microbial_models_path_lowvar2 = "{}saved_models_microbial_lowvar2/".format(data_path)
param_list_microbial_lowvar2 = []

for add_pred, which_loss in [
        [["var"], "variance_bis2"],
]:
        param_dict_microbial_sig_rnn_lowvar = {
                'dataset': ["microbial_otu_sig_highab_lowvar5",
                            "microbial_otu_sig_highab_lowvar94q",
                            "microbial_genus_sig_highab_lowvar5",
                            "microbial_genus_sig_highab_lowvar94q"],
                'dataset_split': ["no_abx",],
                'epochs': [epochs],
                'batch_size': [batch_size],
                'save_every': [save_every],
                'learning_rate': [0.001, 0.0001,],
                'lr_scheduler': [
                    {'step_size': 100, 'gamma': 0.1},
                    {'step_size': 100, 'gamma': 0.5},
                    {'step_size': 100, 'gamma': 0.8},
                    {'step_size': 100, 'gamma': 0.9},
                ],
                'seed': [seed],
                'hidden_size': [100],
                'bias': [bias],
                'dropout_rate': [dropout_rate],
                'ode_nn': [((100, 'tanh'),)],  # ode_nn, ode_nn1
                'readout_nn': [((100, 'tanh'),)],
                'enc_nn': [((100, 'tanh'),)],
                'use_rnn': [False],
                'input_sig': [False],
                'residual_enc_dec': [True],
                'func_appl_X': [["power-2"]],              # [["power-2", "power-3", "power-4"]]
                'add_pred': [add_pred],
                'test': [test],
                'solver': [solver],
                'solver_delta_t_factor': [1/7.,],
                'weight': [0.],
                'weight_evolve': [{'type': 'linear', 'target': 1, 'reach': None}],
                'plot': [True],
                'which_loss': [which_loss],
                'which_eval_loss': ['val_variance'],
                'evaluate': [False],
                'eval_metrics': [eval_metrics],
                'paths_to_plot': [(0,)],
                'plot_variance': [True],
                'std_factor': [std_factor],
                'plot_moments': [plot_moments],
                'saved_models_path': [microbial_models_path_lowvar2],
                'use_cond_exp': [True],
                'input_current_t': [input_current_t],
                'periodic_current_t': [True],
                'scale_dt': [scale_dt],
                'enc_input_t': [enc_input_t],
                'add_readout_activation': [(None, []),], # add_readout_activation # ('softmax',['id']) ('sum2one',['id'])
                'add_dynamic_cov': [True],
                'pre-train': [10000],
                'zero_weight_init': [False],
            }
        param_list_microbial_lowvar2 += get_parameter_array(
                param_dict=param_dict_microbial_sig_rnn_lowvar)

overview_dict_microbial_lowvar2 = dict(
    ids_from=1, ids_to=len(param_list_microbial_lowvar2),
    path=microbial_models_path_lowvar2,
    params_extract_desc=('dataset', 'dataset_split',
                         'ode_nn', 'enc_nn', 'readout_nn',
                         'dropout_rate', 'hidden_size', 'batch_size',
                         'which_loss', 'which_eval_loss', 'add_pred',
                         'solver_delta_t_factor', 'add_readout_activation',
                         'residual_enc_dec', 'use_rnn',
                         'input_sig', 'level', ),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "eval_loss", "eval_loss", "eval_loss_min"),
        ("min", "val_loss", "val_loss", "val_loss_min"),
        ("min", "sum--val_loss_1--val_loss_2--val_loss_4",
         "sum--val_loss_1--val_loss_2--val_loss_4", "corrected_val_loss_min"),
    ),
    sortby=["dataset", "corrected_val_loss_min"],
)




