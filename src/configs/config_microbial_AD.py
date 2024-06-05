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





