import pandas
import numpy as np
import os
import json
import pickle
from scipy.stats import expon
from sklearn.model_selection import train_test_split
import argparse
from scipy.stats import gmean
from configs import config

data_path = config.data_path
train_data_path = config.training_data_path
original_data_path = config.original_data_path

def make_dataset(dataset,  # name of dataset file in data/original_data
                 dataset_name, # name of dataset that will be saved in data/training_data
                 microbial_features, # name of dataset subdivision (selected hosts for train/test/val)
                 signature_features, # dynamic features to be used for th computation of the signature
                 static_features, # static additional features
                 dynamic_features, # dynamic additional features
                 val_size = 0.2, # proportion of the validation set
                 seed = 398, # random seed for splitting into train/val
                 starting_date = 0,  # starting date of the time series in days
                 init_val_method = ('group_feat_mean','delivery_mode'), # method to set per host initial value
                    # choose among ('group_feat_mean','delivery_mode') or ('sample_weighted_sum','time_neg_exp',100.)
                 which_split = 'all',
                ):
    
    df = pandas.read_csv(os.path.join(original_data_path, dataset) ,sep='\t')


    ### PRE-PROCESSING ###

    # Features

    features = df.columns
    abx_feature_prefix = 'abx_'

    if isinstance(microbial_features, str):
        l = len(microbial_features)
        microbial_feature_prefix = microbial_features
        microbial_features = []
        for feature in features:
            print(feature)
            if len(feature) < l:
                continue
            if feature[:l] == microbial_feature_prefix:
                microbial_features.append(feature)
            elif feature[:l] == microbial_feature_prefix:
                microbial_features.append(feature)

    abx_features = []
    for feature in features:
        if len(feature) > len(abx_feature_prefix) and feature[:l] == abx_feature_prefix:
            abx_features.append(feature)


    
    if isinstance(signature_features, str):
        signature_features_filename = signature_features
        #signature_features = []
        with open(os.path.join(original_data_path, signature_features_filename), 'r') as f:
            signature_features = f.readlines()
        signature_features = [f.replace('\n', '') for f in signature_features]
        for f in signature_features:
            if f not in features:
                print("Warning : signature feature " + f + " is not part of the features in the dataset")

    microbial_features = np.array(microbial_features)
    nb_microbial_features = len(microbial_features)
    static_features = np.array(static_features)
    nb_static_features = len(static_features)
    dynamic_features = np.array(dynamic_features)
    nb_dynamic_features = len(dynamic_features)
    signature_features = np.array(signature_features)
    nb_signature_features = len(signature_features)
    abx_features = np.array(abx_features)

    # Raw data

    microbial_data = np.array(df[microbial_features])
    host_data_static = np.array(df[static_features])
    host_data_dynamic = np.array(df[dynamic_features])
    host_data_signature = np.array(df[signature_features])
    abx_data = np.array(df[abx_features])

    static_feature_values = np.array([list(set(host_data_static[:,i])) for i in range(nb_static_features)], dtype=object)
    static_feature_values_counter = np.array([len(static_feature_values[i]) for i in range(nb_static_features)])
    static_feature_digitized_size = len(sum(static_feature_values, []))

    dynamic_feature_values = np.array([list(set(host_data_dynamic[:,i])) for i in range(nb_dynamic_features)], dtype=object)
    dynamic_feature_values_counter = np.array([len(dynamic_feature_values[i]) for i in range(nb_dynamic_features)])
    dynamic_feature_digitized_size = len(sum(dynamic_feature_values, []))

    # Sample age

    sample_age_days = np.array(df[['age_days']])
    sample_age_days = sample_age_days.astype(int)
    sample_age_days_max = sample_age_days.max()

    # Hosts

    sample_host = np.array(df[['host_id']]).reshape(-1)
    hosts = np.array(list(set(sample_host)))
    nb_host = len(hosts)

    # Antibiotics presence

    abx_max_count_ever = np.array(df[['abx_max_count_ever']]).reshape(-1)
    abx_any_last_t_dmonths = np.array(df[['abx_any_last_t_dmonths']]).reshape(-1)

    ### DATASET DIMENSIONS ###

    print("Number of infant hosts : ", nb_host)
    print("--> number of paths")
    print("Number of microbial features : ", nb_microbial_features)
    print("--> dimension")
    print("Maximum age (in days) of an infant on which a sample was collected : ", sample_age_days_max)
    print("--> number of steps")
    print("Total number of samples (among all hosts) : ", df.shape[0])
    print("--> total number of data points")

    ### DATASET CREATION ###

    paths = np.zeros((nb_host,nb_microbial_features,sample_age_days_max+1), dtype=np.float32)
    observed_dates = np.zeros((nb_host,sample_age_days_max+1), dtype=np.int32)

    static = np.zeros((nb_host,static_feature_digitized_size), dtype=np.float32)
    dynamic = np.zeros((nb_host,dynamic_feature_digitized_size,sample_age_days_max+1), dtype=np.float32)
    signature = np.zeros((nb_host,nb_signature_features,sample_age_days_max+1), dtype=np.float32)

    abx_any = np.zeros(nb_host, dtype=np.bool_)
    abx_observed = np.zeros(nb_host, dtype=np.bool_)

    # Time series creation

    for i in range(df.shape[0]):
        idx = list(hosts).index(sample_host[i])
        time = sample_age_days[i][0]
        observed_dates[idx,time]=1
        paths[idx,:,time] = microbial_data[i]
        signature[idx,:,time] = host_data_signature[i]
        host_feature = np.zeros(dynamic_feature_digitized_size)
        c = 0
        for f in range(nb_static_features):
            feature = host_data_dynamic[i,f]
            feature_idx = dynamic_feature_values[f].index(feature)
            host_feature[c + feature_idx] = 1.
            c += dynamic_feature_values_counter[f]
        dynamic[idx,:,time] = host_feature
        if abx_max_count_ever[i] > 0.:
            abx_any[idx] = 1
        if not np.isnan(abx_any_last_t_dmonths[i]):
            abx_observed[idx] = 1
    nb_obs = np.sum(observed_dates, axis=1)

    count = 0
    for i in range(df.shape[0]):
        idx = list(hosts).index(sample_host[i])
        host_feature = np.zeros(static_feature_digitized_size)
        c = 0
        for f in range(nb_static_features):
            feature = host_data_static[i,f]
            feature_idx = static_feature_values[f].index(feature)
            host_feature[c + feature_idx] = 1.
            c += static_feature_values_counter[f]
        if np.sum(static[idx]) != 0 and (static[idx] != host_feature).any():
            count += 1
            print("Problem : different host static data encountered among samples ! ")
            print(idx)
            print(static[idx])
            print(host_feature)
        static[idx] = host_feature

    print("Number of host who took antibiotics before at least one sample was taken : ", abx_observed.sum())
    print("Number of host who where never observed with antibiotics : ", nb_host - abx_observed.sum())


    ### SETTING OF PATH INITIAL VALUES ###

    if isinstance(init_val_method, tuple) and init_val_method[0] == 'group_feat_mean':
        grouping_feature = [init_val_method[1]]
        host_data_group = np.array(df[grouping_feature]).reshape(-1)
        grouping_feature_values = list(set(host_data_group))
        groups = [[] for i in range(len(grouping_feature_values))]
        for i in range(df.shape[0]):
            idx = list(hosts).index(sample_host[i])
            for n, v in enumerate(grouping_feature_values):
                if host_data_group[i] == v:
                    groups[n].append(idx)
        groups = [np.array(list(set(g))) for g in groups]
        for n,g in enumerate(groups):
            group_points = paths[g,:,:starting_date].transpose(0,2,1).reshape(-1,nb_microbial_features)
            group_point_nan_idx = observed_dates[g,:starting_date].reshape(-1).astype(np.bool)
            mean_group = np.mean(group_points[group_point_nan_idx],axis=0)
            for idx in g:
                if observed_dates[idx,starting_date] == 0:
                    paths[idx,:,starting_date] = mean_group
    
    elif isinstance(init_val_method, tuple) and init_val_method[0] == 'sample_weighted_sum':
        paths_0 = np.zeros((nb_host, nb_microbial_features))
        weight_method = init_val_method[1]
        weight_param = init_val_method[2]
        times = np.arange(sample_age_days_max+1)
        if weight_method == "time_neg_exp":
            weights = expon.pdf(times, loc=0., scale=weight_param)
        for i in range(nb_host):
            paths_0[i] = np.sum(paths[i] * np.tile((weights * observed_dates[i]).reshape(1,-1),(nb_microbial_features,1)),axis=1) / np.sum(weights * observed_dates[i])
        paths[:,:,starting_date] = paths_0

    paths = paths[:,:,starting_date:]
    observed_dates = observed_dates[:,starting_date:]
    dynamic = dynamic[:,:,starting_date:]
    signature = signature[:,:,starting_date:]
    observed_dates[:,0] = 1
    nb_obs = np.sum(observed_dates, axis=1)
    nb_steps = int(sample_age_days_max - starting_date)

    ### CHECKS ###

    print("Check whether the time of samples is more precise than days :")
    # Check if 'age_days' values are just int
    if np.max(np.abs(np.array(df[['age_days']]) - np.array(df[['age_days']]).astype(int))) > 0:
        print("--> Yes")
    else:
        print("--> No")

    print("Check if the sum of features is 1 for each data point : ")
    if np.max(np.abs(np.sum(microbial_data, axis=1)-1)) > 1e-10:
        print("--> No")
    else:
        print("--> Yes")

    print("Check if the sum of features is 1 for each estimated S0 : ")
    if np.max(np.abs(np.sum(paths[:,:,0], axis=1)-1)) > 1e-5:
        print("--> No")
    else:
        print("--> Yes")

    print("Minimal Time-difference (in days) : ")
    min_dt_per_host = np.empty(nb_host)
    min_dt_per_host.fill(np.inf)
    for i in range(nb_host):
        times = np.where(observed_dates[i] == 1.)[0]
        if len(times) > 1:
            min_dt_per_host[i] = np.min(times[1:] - times[:-1])
    # print(min_dt_per_host)
    print(np.min(min_dt_per_host))

    ### PATHS ###

    dataset_path = os.path.join(train_data_path, dataset_name)
    if not os.path.isdir(dataset_path):
        os.mkdir(dataset_path)

    ### SAVING FILES ###

    with open(os.path.join(dataset_path, 'data.npy'), 'wb') as f:
        np.save(f, paths)
        np.save(f, observed_dates)
        np.save(f, nb_obs)
        np.save(f, signature)
        np.save(f, dynamic)
        np.save(f, static)

    
    metadata_dict = {"S0": None,
            "dimension": nb_microbial_features,
            "dynamic_cov_dim": dynamic_feature_digitized_size,
            "dt": 1 / float(nb_steps),
            "maturity": 1.,
            "model_name": "microbial_genus",
            "nb_paths": nb_host,
            "nb_steps": nb_steps,
            "period": 1.}

    with open(os.path.join(dataset_path, 'metadata.txt'), 'w') as f:
        json.dump(metadata_dict, f, sort_keys=True)

    ### CREATE DATASET SUBDIVISION TRAIN/TEST/VAL ###

    if which_split == 'all':
        train_idx, val_idx = train_test_split(np.arange(nb_host), test_size=val_size, random_state=seed)
    elif which_split == 'no_abx':
        train_idx, val_idx = train_test_split(np.where(~abx_observed)[0], test_size=val_size, random_state=seed)

    idx_dataset_path = os.path.join(dataset_path, which_split)
    if not os.path.isdir(idx_dataset_path):
        os.mkdir(idx_dataset_path)

    with open(os.path.join(idx_dataset_path, 'train_idx.npy'), 'wb') as f:
        np.save(f, train_idx)
    with open(os.path.join(idx_dataset_path, 'val_idx.npy'), 'wb') as f:
        np.save(f, val_idx)
        
    # eval_ad_idx = np.where(abx_observed)[0]
    # with open(idx_dataset_path + 'eval_ad_idx.npy', 'wb') as f:
    #     np.save(f, eval_ad_idx)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config', type=str,
                    help='config to generate the dataset')

    args = parser.parse_args()
    dataset_config = eval("config."+args.dataset_config)
    make_dataset(**dataset_config)
