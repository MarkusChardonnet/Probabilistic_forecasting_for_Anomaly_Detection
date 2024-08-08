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

# ------------------------------------------------------------------------------
#TODO: maybe try a dataset with static instead of dynamic features
