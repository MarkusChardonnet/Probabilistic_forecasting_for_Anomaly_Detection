from configs.config_utils import get_parameter_array, get_dataset_overview, \
    makedirs, data_path, training_data_path

seed = 398

# ------------------------------------------------------------------------------
# AD for otu
AD_microbial_otu3 = "{}saved_models_microbial_otu3/".format(data_path)
AD_microbial_otu3_ids = [1000]
param_list_AD_microbial_otu = []
param_dict_AD_microbial_otu = {
        "dataset": ['microbial_otu_sig_highab'],
        'saved_models_path': [AD_microbial_otu3],
        'load_best': [True],
        'nb_MC_samples': [10**4],
        'verbose': [True],
        'seed': [seed],
        'validation': [False, True],
}
param_list_AD_microbial_otu += get_parameter_array(
        param_dict=param_dict_AD_microbial_otu)