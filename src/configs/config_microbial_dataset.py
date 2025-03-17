config_genus = {
    'dataset': 'ft_vat19_anomaly_v20240105_genus.tsv',
    'dataset_name': 'microbial_genus',
    'microbial_features': 'g_',
    'signature_features': None,
    'static_features': [],
    'dynamic_features': ["delivery_mode", "diet_milk", "diet_weaning"], # "sex", "geo_location_name"
    'val_size': 0.2,
    'seed': 398,
    'starting_date': 42,
    'init_val_method': ('group_feat_mean','delivery_mode'),
    'which_split': ['all', 'no_abx'],
}

config_genus_sig_highab = {
    'dataset': 'ft_vat19_anomaly_v20240105_genus.tsv',
    'dataset_name': 'microbial_genus_sig_highab',
    'microbial_features': 'g_',
    'signature_features': 'ft_sel_signature_high_abundance_v20240105g__.txt',
    'static_features': [],
    'dynamic_features': ["delivery_mode", "diet_milk", "diet_weaning"], # "sex", "geo_location_name"
    'val_size': 0.2,
    'seed': 398,
    'starting_date': 42,
    'init_val_method': ('group_feat_mean','delivery_mode'),
    'which_split': ['all', 'no_abx'],
}

config_genus_sig_nonzero = {
    'dataset': 'ft_vat19_anomaly_v20240105_genus.tsv',
    'dataset_name': 'microbial_genus_sig_nonzero',
    'microbial_features': 'g_',
    'signature_features': 'ft_sel_signature_high_nonzero_v20240105g__.txt',
    'static_features': [],
    'dynamic_features': ["delivery_mode", "diet_milk", "diet_weaning"], # "sex", "geo_location_name"
    'val_size': 0.2,
    'seed': 398,
    'starting_date': 42,
    'init_val_method': ('group_feat_mean','delivery_mode'),
    'which_split': ['all', 'no_abx'],
}

config_genus_sig_div = {
    'dataset': 'ft_vat19_anomaly_v20240105_genus.tsv',
    'dataset_name': 'microbial_genus_sig_div',
    'microbial_features': 'g_',
    'signature_features': 'ft_sel_signature_diversity_v20240105g__.txt',
    'static_features': [],
    'dynamic_features': ["delivery_mode", "diet_milk", "diet_weaning"], # "sex", "geo_location_name"
    'val_size': 0.2,
    'seed': 398,
    'starting_date': 42,
    'init_val_method': ('group_feat_mean','delivery_mode'),
    'which_split': ['all', 'no_abx'],
}

config_otu = {
    'dataset': 'ft_vat19_anomaly_v20240105_otu.tsv',
    'dataset_name': 'microbial_otu',
    'microbial_features': 'otu_',
    'signature_features': None,
    'static_features': [],
    'dynamic_features': ["delivery_mode", "diet_milk", "diet_weaning"], # "sex", "geo_location_name"
    'val_size': 0.2,
    'seed': 398,
    'starting_date': 42,
    'init_val_method': ('group_feat_mean','delivery_mode'),
    'which_split': ['all', 'no_abx'],
}

config_otu_sig_highab = {
    'dataset': 'ft_vat19_anomaly_v20240105_otu.tsv',
    'dataset_name': 'microbial_otu_sig_highab',
    'microbial_features': 'otu_',
    'signature_features': 'ft_sel_signature_high_abundance_v20240105otu__.txt',
    'static_features': [],
    'dynamic_features': ["delivery_mode", "diet_milk", "diet_weaning"], # "sex", "geo_location_name"
    'val_size': 0.2,
    'seed': 398,
    'starting_date': 42,
    'init_val_method': ('group_feat_mean','delivery_mode'),
    'which_split': ['all', 'no_abx'],
}

config_otu_sig_nonzero = {
    'dataset': 'ft_vat19_anomaly_v20240105_otu.tsv',
    'dataset_name': 'microbial_otu_sig_nonzero',
    'microbial_features': 'otu_',
    'signature_features': 'ft_sel_signature_high_nonzero_v20240105otu__.txt',
    'static_features': [],
    'dynamic_features': ["delivery_mode", "diet_milk", "diet_weaning"], # "sex", "geo_location_name"
    'val_size': 0.2,
    'seed': 398,
    'starting_date': 42,
    'init_val_method': ('group_feat_mean','delivery_mode'),
    'which_split': ['all', 'no_abx'],
}

config_otu_sig_div = {
    'dataset': 'ft_vat19_anomaly_v20240105_otu.tsv',
    'dataset_name': 'microbial_otu_sig_div',
    'microbial_features': 'otu_',
    'signature_features': 'ft_sel_signature_diversity_v20240105otu__.txt',
    'static_features': [],
    'dynamic_features': ["delivery_mode", "diet_milk", "diet_weaning"], # "sex", "geo_location_name"
    'val_size': 0.2,
    'seed': 398,
    'starting_date': 42,
    'init_val_method': ('group_feat_mean','delivery_mode'),
    'which_split': ['all', 'no_abx'],
}


# ------------------------------------------------------------------------------
# lower dim datasets, reducing the features with low variance

# otu
config_otu_sig_highab_lowvar5 = {
    'dataset': 'ft_vat19_anomaly_v20240105_otu_lowvar5.tsv',
    'dataset_name': 'microbial_otu_sig_highab_lowvar5',
    'microbial_features': 'otu_',
    'signature_features': 'ft_sel_signature_high_abundance_v20240105otu___lowvar5.txt',
    'static_features': [],
    'dynamic_features': ["delivery_mode", "diet_milk", "diet_weaning"], # "sex", "geo_location_name"
    'val_size': 0.2,
    'seed': 398,
    'starting_date': 42,
    'init_val_method': ('group_feat_mean','delivery_mode'),
    'which_split': ['all', 'no_abx'],
}

config_otu_sig_highab_lowvar94q = {
    'dataset': 'ft_vat19_anomaly_v20240105_otu_lowvar94q.tsv',
    'dataset_name': 'microbial_otu_sig_highab_lowvar94q',
    'microbial_features': 'otu_',
    'signature_features': 'ft_sel_signature_high_abundance_v20240105otu___lowvar94q.txt',
    'static_features': [],
    'dynamic_features': ["delivery_mode", "diet_milk", "diet_weaning"], # "sex", "geo_location_name"
    'val_size': 0.2,
    'seed': 398,
    'starting_date': 42,
    'init_val_method': ('group_feat_mean','delivery_mode'),
    'which_split': ['all', 'no_abx'],
}

# genus
config_genus_sig_highab_lowvar5 = {
    'dataset': 'ft_vat19_anomaly_v20240105_genus_lowvar5.tsv',
    'dataset_name': 'microbial_genus_sig_highab_lowvar5',
    'microbial_features': 'g_',
    'signature_features': 'ft_sel_signature_high_abundance_v20240105g___lowvar5.txt',
    'static_features': [],
    'dynamic_features': ["delivery_mode", "diet_milk", "diet_weaning"], # "sex", "geo_location_name"
    'val_size': 0.2,
    'seed': 398,
    'starting_date': 42,
    'init_val_method': ('group_feat_mean','delivery_mode'),
    'which_split': ['all', 'no_abx'],
}

config_genus_sig_highab_lowvar94q = {
    'dataset': 'ft_vat19_anomaly_v20240105_genus_lowvar94q.tsv',
    'dataset_name': 'microbial_genus_sig_highab_lowvar94q',
    'microbial_features': 'g_',
    'signature_features': 'ft_sel_signature_high_abundance_v20240105g___lowvar94q.txt',
    'static_features': [],
    'dynamic_features': ["delivery_mode", "diet_milk", "diet_weaning"], # "sex", "geo_location_name"
    'val_size': 0.2,
    'seed': 398,
    'starting_date': 42,
    'init_val_method': ('group_feat_mean','delivery_mode'),
    'which_split': ['all', 'no_abx'],
}


# ------------------------------------------------------------------------------
config_div_alpha_faith_pd_1 = {
    'dataset': 'ft_vat19_anomaly_v20240105_genus_lowvar94q.tsv',
    'dataset_name': 'microbial_div_alpha_faith_pd_1',
    'microbial_features': ['div_alpha_faith_pd'],
    'signature_features': None,  # sets to same as microbial_features
    'static_features': [],
    'dynamic_features': ["delivery_mode", "diet_milk", "diet_weaning"], # "sex", "geo_location_name"
    'val_size': 0.2,
    'seed': 398,
    'starting_date': 42,
    'init_val_method': ('group_feat_mean','delivery_mode'),
    'which_split': ['all', 'no_abx'],
}

config_div_alpha_faith_pd_2 = {
    'dataset': 'ft_vat19_anomaly_v20240105_genus_lowvar94q.tsv',
    'dataset_name': 'microbial_div_alpha_faith_pd_2',
    'microbial_features': ['div_alpha_faith_pd'],
    'signature_features': None,  # sets to same as microbial_features
    'static_features': [],
    'dynamic_features': ["delivery_mode", "diet_milk", "diet_weaning"], # "sex", "geo_location_name"
    'val_size': 0.2,
    'seed': 398,
    'starting_date': 0,
    'init_val_method': None,
    'which_split': ['all', 'no_abx'],
}

config_div_alpha_faith_pd_3 = {
    'dataset': 'ft_vat19_anomaly_v20240105_genus_lowvar94q.tsv',
    'dataset_name': 'microbial_div_alpha_faith_pd_3',
    'microbial_features': ['div_alpha_shannon'],
    'signature_features': None,  # sets to same as microbial_features
    'static_features': [],
    'dynamic_features': ["delivery_mode", "diet_milk", "diet_weaning"], # "sex", "geo_location_name"
    'val_size': 0.2,
    'seed': 398,
    'starting_date': 0,
    'init_val_method': None,
    'which_split': ['all', 'no_abx'],
}

config_div_alpha_faith_pd_4 = {
    'dataset': 'ft_vat19_anomaly_v20240105_genus_lowvar94q.tsv',
    'dataset_name': 'microbial_div_alpha_faith_pd_4',
    'microbial_features': ['div_alpha_observed_features'],
    'signature_features': None,  # sets to same as microbial_features
    'static_features': [],
    'dynamic_features': ["delivery_mode", "diet_milk", "diet_weaning"], # "sex", "geo_location_name"
    'val_size': 0.2,
    'seed': 398,
    'starting_date': 0,
    'init_val_method': None,
    'which_split': ['all', 'no_abx'],
}

config_div_alpha_faith_pd_5 = {
    'dataset': 'ft_vat19_anomaly_v20240105_genus_lowvar94q.tsv',
    'dataset_name': 'microbial_div_alpha_faith_pd_5',
    'microbial_features': ['div_alpha_shannon', 'div_alpha_observed_features', 'div_alpha_faith_pd'],
    'signature_features': None,  # sets to same as microbial_features
    'static_features': [],
    'dynamic_features': ["delivery_mode", "diet_milk", "diet_weaning"], # "sex", "geo_location_name"
    'val_size': 0.2,
    'seed': 398,
    'starting_date': 0,
    'init_val_method': None,
    'which_split': ['all', 'no_abx'],
}

# ------------------------------------------------------------------------------
config_novel_alpha_faith_pd = {
    'dataset': 'ft_vat19_anomaly_v20240806_entero_family.tsv',
    'dataset_name': 'microbial_novel_alpha_faith_pd',
    'microbial_features': ['div_alpha_faith_pd'],
    'signature_features': None,  # sets to same as microbial_features
    'static_features': [],
    'dynamic_features': ["delivery_mode", "diet_milk", "diet_weaning"],
    'val_size': 0.2,
    'seed': 398,
    'starting_date': 0,
    'init_val_method': None,
    'which_split': ['all', 'no_abx'],
}

config_novel_alpha_faith_pd_w_geo = {
    'dataset': 'ft_vat19_anomaly_v20240806_entero_family.tsv',
    'dataset_name': 'microbial_novel_alpha_faith_pd_w_geo',
    'microbial_features': ['div_alpha_faith_pd'],
    'signature_features': None,  # sets to same as microbial_features
    'static_features': [],
    'dynamic_features': ["delivery_mode", "diet_milk", "diet_weaning", "geo_location_name"],
    'val_size': 0.2,
    'seed': 398,
    'starting_date': 0,
    'init_val_method': None,
    'which_split': ['all', 'no_abx'],
}


# ------ enteropathogens ---------
config_entero_family = {
    'dataset': 'ft_vat19_anomaly_v20240806_entero_family.tsv',
    'dataset_name': 'microbial_rel_abd_enteropathogens_family',
    'microbial_features': ['rel_abd_enteropathogens_family'],
    'signature_features': None,  # sets to same as microbial_features
    'static_features': [],
    'dynamic_features': ["delivery_mode", "diet_milk", "diet_weaning",],
    'val_size': 0.2,
    'seed': 398,
    'starting_date': 0,
    'init_val_method': None,
    'which_split': ['all', 'no_abx'],
}

config_entero_family_w_geo = {
    'dataset': 'ft_vat19_anomaly_v20240806_entero_family.tsv',
    'dataset_name': 'microbial_rel_abd_enteropathogens_family_w_geo',
    'microbial_features': ['rel_abd_enteropathogens_family'],
    'signature_features': None,  # sets to same as microbial_features
    'static_features': [],
    'dynamic_features': ["delivery_mode", "diet_milk", "diet_weaning", "geo_location_name"],
    'val_size': 0.2,
    'seed': 398,
    'starting_date': 0,
    'init_val_method': None,
    'which_split': ['all', 'no_abx'],
}

config_entero_genus = {
    'dataset': 'ft_vat19_anomaly_v20240806_entero_genus.tsv',
    'dataset_name': 'microbial_rel_abd_enteropathogens_genus',
    'microbial_features': ['rel_abd_enteropathogens_genus'],
    'signature_features': None,  # sets to same as microbial_features
    'static_features': [],
    'dynamic_features': ["delivery_mode", "diet_milk", "diet_weaning"],
    'val_size': 0.2,
    'seed': 398,
    'starting_date': 0,
    'init_val_method': None,
    'which_split': ['all', 'no_abx'],
}

config_entero_genus_w_geo = {
    'dataset': 'ft_vat19_anomaly_v20240806_entero_genus.tsv',
    'dataset_name': 'microbial_rel_abd_enteropathogens_genus_w_geo',
    'microbial_features': ['rel_abd_enteropathogens_genus'],
    'signature_features': None,  # sets to same as microbial_features
    'static_features': [],
    'dynamic_features': ["delivery_mode", "diet_milk", "diet_weaning", "geo_location_name"],
    'val_size': 0.2,
    'seed': 398,
    'starting_date': 0,
    'init_val_method': None,
    'which_split': ['all', 'no_abx'],
}

# TODO: decide whether "geo_location_name" should be included in next model runs
# TODO: or not - if it doesn't hurt we should include it
config_novel_alpha_faith_pd_entero_family = {
    'dataset': 'ft_vat19_anomaly_v20240806_entero_family.tsv',
    'dataset_name': 'microbial_novel_alpha_faith_pd_entero_family',
    'microbial_features': ['div_alpha_faith_pd', 'rel_abd_enteropathogens_family'],
    'signature_features': None,  # sets to same as microbial_features
    'static_features': [],
    'dynamic_features': ["delivery_mode", "diet_milk", "diet_weaning", "geo_location_name"],
    'val_size': 0.2,
    'seed': 398,
    'starting_date': 0,
    'init_val_method': None,
    'which_split': ['all', 'no_abx'],
}

config_novel_alpha_faith_pd_entero_genus = {
    'dataset': 'ft_vat19_anomaly_v20240806_entero_genus.tsv',
    'dataset_name': 'microbial_novel_alpha_faith_pd_entero_genus',
    'microbial_features': ['div_alpha_faith_pd', 'rel_abd_enteropathogens_genus'],
    'signature_features': None,  # sets to same as microbial_features
    'static_features': [],
    'dynamic_features': ["delivery_mode", "diet_milk", "diet_weaning", "geo_location_name"],
    'val_size': 0.2,
    'seed': 398,
    'starting_date': 0,
    'init_val_method': None,
    'which_split': ['all', 'no_abx'],
}

config_novel_alpha_faith_pd_entero_genus_scaled = {
    'dataset': 'ft_vat19_anomaly_v20240806_entero_genus.tsv',
    'dataset_name': 'microbial_novel_alpha_faith_pd_entero_genus_scaled',
    'microbial_features': ['div_alpha_faith_pd', 'rel_abd_enteropathogens_genus'],
    'signature_features': None,  # sets to same as microbial_features
    'static_features': [],
    'dynamic_features': ["delivery_mode", "diet_milk", "diet_weaning", "geo_location_name"],
    'val_size': 0.2,
    'seed': 398,
    'starting_date': 0,
    'init_val_method': None,
    'which_split': ['all', 'no_abx'],
    'scaling': 'mean',
    'compute_scaling_on': 'train-noabx',
}

# ------------------------------------------------------------------------------
#TODO: maybe try a dataset with static instead of dynamic features

# ------------------------------------------------------------------------------
# Synthetic data

config_synthetic_novel_alpha_faith_noabx_pd = {
    'model_name': "Microbiome_OrnsteinUhlenbeck",
    'nb_paths': 300, 'nb_steps': 730,
    'dimension': 1, 'obs_perc': 0.01, 'S0': [0.],
    'maturity': 1., 'return_vol': False,
    'speed': 30.,
    'noise': {'type': "gaussian", 'cov': 0.},
    'volatility': {
        'vol_value': 10.,
    },
    'initial_value': False,
    'fct_params': {
        'type': 'invexp',
        'scale': 25,
        'decay': 3,
    },
    "dynamic_vars": [
        {
            "name": "delivery_mode",
            "type": "static",
            "nb_vals": 2,
            "val_names": ["vaginal", "ceasarean"],
            "probs": [0.9, 0.1],
            "factor": [1.1, 0.9],
            "duration": 180,
        },
        {
            "name": "diet_weaning",
            "type": "dynamic",
            "nb_vals": 3,
            "val_names": ["no", "yes", "unknown"],
            "goem_law": [0.0125, None, None],
            "max_dur": [150, 1000, 1000],
            "factor": [0.9, 1.1, 1.]
        },
        {
            "name": "diet_milk",
            "type": "dynamic",
            "nb_vals": 4,
            "val_names": ["bd", "mixed", "fd", "unknown"],
            "goem_law": [0.0125, 0.00625, 0.05, None],
            "max_dur": [150, 360, 40, 1000],
            "factor": [0.95, 1., 1.05, 1.]
        }
    ],
    'anomaly_params': {
        'type': None
    }
}

occurence_prob = 0.5
occurence_law = 'geometric'
occurence_law_param = 0.5
occurence_pos_law = 'uniform'
occurence_pos_range = (0.01,1.)
occurence_len_law = 'uniform'
occurence_len_range =  (0.05,0.25)

config_synthetic_novel_alpha_faith_abx_pd = {
    'model_name': "Microbiome_OrnsteinUhlenbeck",
    'nb_paths': 1000, 'nb_steps': 730,
    'dimension': 1, 'obs_perc': 0.01, 'S0': [0.],
    'maturity': 1., 'return_vol': False,
    'speed': 30., 
    'noise': {'type': "gaussian", 'cov': 0.},
    'volatility': {
        'vol_value': 10.,
    },
    'initial_value': False,
    'fct_params': {
        'type': 'invexp',
        'scale': 25,
        'decay': 3
    },
    "dynamic_vars": [
        {
            "name": "delivery_mode",
            "type": "static",
            "nb_vals": 2,
            "val_names": ["vaginal", "ceasarean"],
            "probs": [0.9, 0.1],
            "factor": [1.1, 0.9],
            "duration": 180,
        },
        {
            "name": "diet_weaning",
            "type": "dynamic",
            "nb_vals": 3,
            "val_names": ["no", "yes", "unknown"],
            "goem_law": [0.0125, None, None],
            "max_dur": [150, 1000, 1000],
            "factor": [0.9, 1.1, 1.]
        },
        {
            "name": "diet_milk",
            "type": "dynamic",
            "nb_vals": 4,
            "val_names": ["bd", "mixed", "fd", "unknown"],
            "goem_law": [0.0125, 0.00625, 0.05, None],
            "max_dur": [150, 360, 40, 1000],
            "factor": [0.95, 1., 1.05, 1.]
        }
    ],
    'anomaly_params': {
        'type': 'cutoff',
        'occurence_prob': 1,
        'occurence_law': occurence_law,
        'occurence_law_param': occurence_law_param,
        'occurence_pos_law': occurence_pos_law,
        'occurence_pos_range': occurence_pos_range,
        'occurence_len_law': occurence_len_law,
        'occurence_len_range': occurence_len_range,
        'cutoff_level_law': 'uniform',
        'cutoff_level_range': (0.2,0.5),
    }
}

config_synthetic_novel_alpha_faith_pd = {
    'model_name': "Microbiome_OrnsteinUhlenbeck",
    'nb_paths': 1000, 'nb_steps': 730,
    'dimension': 1, 'obs_perc': 0.01, 'S0': [0.],
    'maturity': 1., 'return_vol': False,
    'speed': 30., 
    'noise': {'type': "gaussian", 'cov': 0.},
    'volatility': {
        'vol_value': 10.,
    },
    'initial_value': False,
    'fct_params': {
        'type': 'invexp',
        'scale': 25,
        'decay': 3
    },
    "dynamic_vars": [
        {
            "name": "delivery_mode",
            "type": "static",
            "nb_vals": 2,
            "val_names": ["vaginal", "ceasarean"],
            "probs": [0.9, 0.1],
            "factor": [1.1, 0.9],
            "duration": 180,
        },
        {
            "name": "diet_weaning",
            "type": "dynamic",
            "nb_vals": 3,
            "val_names": ["no", "yes", "unknown"],
            "goem_law": [0.0125, None, None],
            "max_dur": [150, 1000, 1000],
            "factor": [0.9, 1.1, 1.]
        },
        {
            "name": "diet_milk",
            "type": "dynamic",
            "nb_vals": 4,
            "val_names": ["bd", "mixed", "fd", "unknown"],
            "goem_law": [0.0125, 0.00625, 0.05, None],
            "max_dur": [150, 360, 40, 1000],
            "factor": [0.95, 1., 1.05, 1.]
        }
    ],
    'anomaly_params': {
        'type': 'cutoff',
        'occurence_prob': occurence_prob,
        'occurence_law': occurence_law,
        'occurence_law_param': occurence_law_param,
        'occurence_pos_law': occurence_pos_law,
        'occurence_pos_range': occurence_pos_range,
        'occurence_len_law': occurence_len_law,
        'occurence_len_range': occurence_len_range,
        'cutoff_level_law': 'uniform',
        'cutoff_level_range': (0.2,0.5),
    }
}
