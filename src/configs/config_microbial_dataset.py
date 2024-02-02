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