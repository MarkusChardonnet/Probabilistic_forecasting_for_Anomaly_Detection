from configs.config_utils import get_parameter_array, get_dataset_overview, \
    makedirs, data_path, training_data_path

seed = 398


# ==============================================================================
# FLORIAN
# ==============================================================================

# ------------------------------------------------------------------------------
# AD for otu
AD_microbial_otu3 = "{}saved_models_microbial_otu3/".format(data_path)
AD_microbial_otu3_ids = [68]
param_list_AD_microbial_otu = []
param_dict_AD_microbial_otu = {
        'load_best': [True,],
        'nb_MC_samples': [10**4],
        'epsilon': [1e-8, 1e-6],
        'verbose': [True],
        'seed': [seed],
}
param_list_AD_microbial_otu += get_parameter_array(
        param_dict=param_dict_AD_microbial_otu)

param_dict_AD_microbial_otu = {
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
AD_microbial_genus3_ids = [135, 152, 92, 51, 17, 123]
param_list_AD_microbial_genus = []
param_dict_AD_microbial_genus = {
        'load_best': [True,],
        'nb_MC_samples': [10**4],
        'epsilon': [1e-6],
        'verbose': [True],
        'seed': [seed],
}
param_list_AD_microbial_genus += get_parameter_array(
        param_dict=param_dict_AD_microbial_genus)


# ------------------------------------------------------------------------------
# AD for lowvar
AD_microbial_lowvar = "{}saved_models_microbial_lowvar/".format(data_path)
AD_microbial_lowvar_ids = [38, 102, 120, 118, 70, 22, 86]
param_list_AD_microbial_lowvar = []
param_dict_AD_microbial_lowvar = {
        'load_best': [True,],
        'nb_MC_samples': [10**4],
        'epsilon': [1e-6],
        'verbose': [True],
        'seed': [seed],
}
param_list_AD_microbial_lowvar += get_parameter_array(
        param_dict=param_dict_AD_microbial_lowvar)


# AD for lowvar1
AD_microbial_lowvar1 = "{}saved_models_microbial_lowvar1/".format(data_path)
AD_microbial_lowvar1_ids = [44, 47, 38, 29]
param_list_AD_microbial_lowvar1 = []
param_dict_AD_microbial_lowvar1 = {
        'load_best': [True,],
        'nb_MC_samples': [10**4],
        'epsilon': [1e-6],
        'verbose': [True],
        'seed': [seed],
}
param_list_AD_microbial_lowvar1 += get_parameter_array(
        param_dict=param_dict_AD_microbial_lowvar1)



# AD for lowvar2
AD_microbial_lowvar2 = "{}saved_models_microbial_lowvar2/".format(data_path)
AD_microbial_lowvar2_ids = [19, 28, 4, 12]
param_list_AD_microbial_lowvar2 = []
param_dict_AD_microbial_lowvar2 = {
        'load_best': [True,],
        'nb_MC_samples': [10**4],
        'epsilon': [1e-6],
        'verbose': [True],
        'seed': [seed],
}
param_list_AD_microbial_lowvar2 += get_parameter_array(
        param_dict=param_dict_AD_microbial_lowvar2)

AD_microbial_lowvar2_ids_1 = [19,]
param_list_AD_microbial_lowvar2_1 = []
param_dict_AD_microbial_lowvar2_1 = {
        'load_best': [False,],
        'nb_MC_samples': [10**4],
        'epsilon': [1e-6],
        'verbose': [True],
        'seed': [seed],
}
param_list_AD_microbial_lowvar2_1 += get_parameter_array(
        param_dict=param_dict_AD_microbial_lowvar2_1)

AD_microbial_lowvar2_ids_2 = [19,]
param_list_AD_microbial_lowvar2_2 = []
param_dict_AD_microbial_lowvar2_2 = {
        'load_best': [True, False,],
        'scoring_distribution': ['beta'],
        'aggregation_method': ['mean', 'logistic'],
}
param_list_AD_microbial_lowvar2_2 += get_parameter_array(
        param_dict=param_dict_AD_microbial_lowvar2_2)


# ------------------------------------------------------------------------------
# AD for alpha diversity metric models
AD_microbial_alpha_div = "{}saved_models_microbial_alpha_div/".format(data_path)
AD_microbial_alpha_div_ids = [13, 102]
AD_microbial_alpha_div_ids_1 = [321, 327,]
param_list_AD_microbial_alpha_div = []
param_dict_AD_microbial_alpha_div = {
        'load_best': [True, False],
        'verbose': [True],
        'seed': [seed],
        'scoring_distribution': ['normal'],
}
param_list_AD_microbial_alpha_div += get_parameter_array(
        param_dict=param_dict_AD_microbial_alpha_div)

AD_microbial_alpha_div_ids_2 = [331]
param_list_AD_microbial_alpha_div_2 = []
param_dict_AD_microbial_alpha_div_2 = {
        'load_best': [True, False],
        'verbose': [True],
        'seed': [seed],
        'scoring_distribution': ['normal'],
        'aggregation_method': ['mean', 'max'],
}
param_list_AD_microbial_alpha_div_2 += get_parameter_array(
        param_dict=param_dict_AD_microbial_alpha_div_2)

AD_microbial_alpha_div_ids_3 = [102]
param_list_AD_microbial_alpha_div_3 = []
param_dict_AD_microbial_alpha_div_3 = {
        'load_best': [True, False],
        'verbose': [True],
        'seed': [seed],
        'scoring_distribution': ['normal'],
        'scoring_metric': ['left-tail'],
}
param_list_AD_microbial_alpha_div_3 += get_parameter_array(
        param_dict=param_dict_AD_microbial_alpha_div_3)



# TODO: best AD results for id 102
param_list_AD_microbial_alpha_div_4 = []
param_dict_AD_microbial_alpha_div_4 = {
        'load_best': [False],
        'verbose': [True],
        'seed': [seed],
        'scoring_distribution': ['lognormal'],
        'scoring_metric': ['left-tail'],
        'plot_cond_standardized_dist': [['normal', 'lognormal']],
        'only_jump_before_abx_exposure': [True],
}
param_list_AD_microbial_alpha_div_4 += get_parameter_array(
        param_dict=param_dict_AD_microbial_alpha_div_4)


# ------------------------------------------------------------------------------
# AD for novel alpha diversity metric models
# TODO: best AD results for id 2 & 37 at best with normal
AD_microbial_novel_alpha_div = "{}saved_models_microbial_novel_alpha_div/".format(data_path)
AD_microbial_novel_alpha_div_ids = [2,55,56]
param_list_AD_microbial_novel_alpha_div = []
param_dict_AD_microbial_novel_alpha_div = {
        'load_best': [True],
        'verbose': [True],
        'seed': [seed],
        'scoring_distribution': ['normal', 'lognormal'],
        'scoring_metric': ['left-tail'],
        'plot_cond_standardized_dist': [['normal', 'lognormal']],
        'only_jump_before_abx_exposure': [1,2,3],
        'use_dyn_cov_after_abx': [True],
}
param_list_AD_microbial_novel_alpha_div += get_parameter_array(
        param_dict=param_dict_AD_microbial_novel_alpha_div)

AD_microbial_rel_abund_ids = [37,59, 60]
param_list_AD_microbial_rel_abund = []
param_dict_AD_microbial_rel_abund = {
        'load_best': [True],
        'verbose': [True],
        'seed': [seed],
        'scoring_distribution': ['t-3'],
        'scoring_metric': ['right-tail'],
        'plot_cond_standardized_dist': [['t-3']],
        'only_jump_before_abx_exposure': [1,2,3],
        'use_dyn_cov_after_abx': [True],
}
param_list_AD_microbial_rel_abund += get_parameter_array(
        param_dict=param_dict_AD_microbial_rel_abund)


AD_microbial_joint_ids = [46, 50,]
param_list_AD_microbial_joint = []
param_dict_AD_microbial_joint = {
        'load_best': [True],
        'verbose': [True],
        'seed': [seed],
        'scoring_distribution': ['t-3'],
        'scoring_metric': ['left-tail'],
        'epsilon': [1e-8],
        'plot_cond_standardized_dist': [['normal', 't-3']],
        'only_jump_before_abx_exposure': [1,2,3],
        'aggregation_method': ['coord-0'],
}
param_list_AD_microbial_joint += get_parameter_array(
        param_dict=param_dict_AD_microbial_joint)
param_dict_AD_microbial_joint_1 = {
        'load_best': [True],
        'verbose': [True],
        'seed': [seed],
        'scoring_distribution': ['t-3'],
        'scoring_metric': ['right-tail'],
        'epsilon': [1e-8],
        'plot_cond_standardized_dist': [['normal', 't-3']],
        'only_jump_before_abx_exposure': [1,2,3],
        'aggregation_method': ['coord-1'],
}
param_list_AD_microbial_joint += get_parameter_array(
        param_dict=param_dict_AD_microbial_joint_1)

# ------------------------------------------------------------------------------
# reliability evaluation
AD_microbial_novel_alpha_div_ids_reliability_eval = [2, 56]
param_list_AD_microbial_novel_alpha_div_reliability_eval = []
param_dict_AD_microbial_novel_alpha_div = {
        'load_best': [True],
        'verbose': [True],
        'seed': [seed],
        'scoring_distribution': ['normal', 'lognormal'],
        'scoring_metric': ['left-tail'],
        'plot_cond_standardized_dist': [None],
        'only_jump_before_abx_exposure': [False],
        'use_dyn_cov_after_abx': [True],
        'reliability_eval_start_times': [list(range(0, 1162, 30))],
}
param_list_AD_microbial_novel_alpha_div_reliability_eval += get_parameter_array(
        param_dict=param_dict_AD_microbial_novel_alpha_div)

AD_microbial_novel_alpha_div_ids_reliability_eval_2 = [37, 60]
param_list_AD_microbial_novel_alpha_div_reliability_eval_2 = []
param_dict_AD_microbial_novel_alpha_div = {
        'load_best': [True],
        'verbose': [True],
        'seed': [seed],
        'scoring_distribution': ['t-3'],
        'scoring_metric': ['right-tail'],
        'plot_cond_standardized_dist': [None],
        'only_jump_before_abx_exposure': [False],
        'use_dyn_cov_after_abx': [True],
        'reliability_eval_start_times': [list(range(0, 1162, 30))],
}
param_list_AD_microbial_novel_alpha_div_reliability_eval_2 += get_parameter_array(
        param_dict=param_dict_AD_microbial_novel_alpha_div)


# ------------------------------------------------------------------------------
# AD for novel alpha diversity metric models -- new validation loss
AD_microbial_novel_alpha_div2 = "{}saved_models_microbial_novel_alpha_div2/".format(data_path)
AD_microbial_novel_alpha_div_ids2 = [8, 57, 55]
param_list_AD_microbial_novel_alpha_div2_scaling_factors = []

# params for getting the scaling factors
param_dict_AD_microbial_novel_alpha_div2sf = {
        'load_best': [True],
        'verbose': [True],
        'seed': [seed],
        'scoring_distribution': ['z_score'],
        'plot_cond_standardized_dist': [None],
        'only_jump_before_abx_exposure': [False],
        'use_dyn_cov_after_abx': [True],
        'reliability_eval_start_times': [list(range(0, 1162, 30))],
        'interval_length': [30],
        'shift_by': [1],
        'aggregation_method': ['coord-0'],
}
param_list_AD_microbial_novel_alpha_div2_scaling_factors += get_parameter_array(
        param_dict=param_dict_AD_microbial_novel_alpha_div2sf)

# params for the actual AD score computation using the scaling factors
param_list_AD_microbial_novel_alpha_div2 = []
param_dict_AD_microbial_novel_alpha_div2 = {
        'load_best': [True],
        'verbose': [True],
        'seed': [seed],
        'scoring_distribution': ['normal',],
        'scoring_metric': ['left-tail'],
        'plot_cond_standardized_dist': [['normal', 'lognormal']],
        'only_jump_before_abx_exposure': [1,2,3],
        'use_dyn_cov_after_abx': [True],
        'aggregation_method': ['coord-0'],
        'use_scaling_factors': [True],
        'preprocess_scaling_factors': [True],
}
param_list_AD_microbial_novel_alpha_div2 += get_parameter_array(
        param_dict=param_dict_AD_microbial_novel_alpha_div2)

# params for the reliability evaluation using the scaling factors
param_list_AD_microbial_novel_alpha_div2_reliability_eval = []
param_dict_AD_microbial_novel_alpha_div2 = {
        'load_best': [True],
        'verbose': [True],
        'seed': [seed],
        'scoring_distribution': ['normal',],
        'scoring_metric': ['left-tail'],
        'only_jump_before_abx_exposure': [False],
        'plot_cond_standardized_dist': [None],
        'use_dyn_cov_after_abx': [True],
        'reliability_eval_start_times': [list(range(0, 1162, 30))],
        'use_scaling_factors': [True],
        'preprocess_scaling_factors': [True],
}


