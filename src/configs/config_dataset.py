config_otu_abundance = {
    'dataset': 'ft_vat19_anomaly_v20240105_otu.tsv',
    'dataset_name': 'otu_high_abundance_sig',
    'microbial_features': 'otu_',
    'signature_features': 'ft_sel_signature_high_abundance_v20240105otu__.txt',
    'static_features': ["delivery_mode", "sex", "geo_location_name"],
    'dynamic_features': ["delivery_mode", "diet_milk", "diet_weaning"], # "sex", "geo_location_name"
    'val_size': 0.2,
    'seed': 398,
    'starting_date': 0,
    'init_val_method': ('group_feat_mean','delivery_mode'),
    'which_split': 'all',
}