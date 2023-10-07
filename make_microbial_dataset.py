import pandas
import numpy as np
import os
import json
import pickle
from scipy.stats import expon
from sklearn.model_selection import train_test_split
import argparse

def make_dataset(dataset = 'ft_vat19_anomaly_v20230824_genus',
                 val_size = 0.2,
                seed = 398):
    df = pandas.read_csv('data/original_data/' + dataset + '.tsv',sep='\t')


    ### PRE-PROCESSING ###

    # Features

    cols = df.columns
    microbial_features = []
    abx_features = []
    static_features = ["delivery_mode", "sex", "geo_location_name"]
    dynamic_features = ["delivery_mode", "sex", "geo_location_name", "diet_milk", "diet_weaning"]

    for col in cols:
        print(col)
        if col[:2] == 'g_':
            microbial_features.append(col)
        if len(col) > 4 and col[:4] == 'abx_':
            abx_features.append(col)
            
    microbial_features = np.array(microbial_features)
    nb_microbial_features = len(microbial_features)
    static_features = np.array(static_features)
    nb_static_features = len(static_features)
    dynamic_features = np.array(dynamic_features)
    nb_dynamic_features = len(dynamic_features)
    abx_features = np.array(abx_features)

    # Raw data

    microbial_data = np.array(df[microbial_features])
    host_data_1 = np.array(df[static_features])
    host_data_2 = np.array(df[dynamic_features])
    abx_data = np.array(df[abx_features])

    static_feature_values = np.array([list(set(host_data_1[:,i])) for i in range(nb_static_features)], dtype=object)
    static_feature_values_counter = np.array([len(static_feature_values[i]) for i in range(nb_static_features)])
    static_feature_digitized_size = len(sum(static_feature_values, []))

    dynamic_feature_values = np.array([list(set(host_data_2[:,i])) for i in range(nb_dynamic_features)], dtype=object)
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

    abx_any = np.zeros(nb_host, dtype=np.bool_)
    abx_observed = np.zeros(nb_host, dtype=np.bool_)

    # Time series creation

    for i in range(df.shape[0]):
        idx = list(hosts).index(sample_host[i])
        time = sample_age_days[i][0]
        observed_dates[idx,time]=1
        paths[idx,:,time] = microbial_data[i]
        host_feature = np.zeros(dynamic_feature_digitized_size)
        c = 0
        for f in range(nb_static_features):
            feature = host_data_2[i,f]
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
            feature = host_data_1[i,f]
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


    ### ESTIMATION OF PATH INITIAL VALUES ###

    weight_method = "time_neg_exponential"
    weight_param = 100.
    times = np.arange(sample_age_days_max+1)
    if weight_method == "time_neg_exponential":
        weights = expon.pdf(times, loc=0., scale=weight_param)

    paths_0 = np.zeros((nb_host, nb_microbial_features))
    for i in range(nb_host):
        paths_0[i] = np.sum(paths[i] * np.tile((weights * observed_dates[i]).reshape(1,-1),(nb_microbial_features,1)),axis=1) / np.sum(weights * observed_dates[i])
            
    paths[:,:,0] = paths_0
    observed_dates[:,0] = 1
    nb_obs = np.sum(observed_dates, axis=1)

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
    if np.max(np.abs(np.sum(paths_0, axis=1)-1)) > 1e-5:
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

    dataset_name = "microbial_genus/"
    dataset_path = os.path.join('data/training_data/', dataset_name)
    if not os.path.isdir(dataset_path):
        os.mkdir(dataset_path)

    ### SAVING FILES ###

    with open(dataset_path + 'data.npy', 'wb') as f:
        np.save(f, paths)
        np.save(f, observed_dates)
        np.save(f, nb_obs)
        np.save(f, dynamic)
        np.save(f, static)
    
    metadata_dict = {"S0": None,
            "dimension": nb_microbial_features,
            "dynamic_cov_dim": dynamic_feature_digitized_size,
            "dt": 1. / float(sample_age_days_max),
            "maturity": 1.,
            "model_name": "microbial_genus",
            "nb_paths": nb_host,
            "nb_steps": int(sample_age_days_max),
            "period": 1.}

    with open(dataset_path + "metadata.txt", 'w') as f:
        json.dump(metadata_dict, f, sort_keys=True)

    ### CREATE DATASET SUBDIVISION TRAIN/TEST/VAL ###

    train_idx, val_idx = train_test_split(np.where(~abx_observed)[0], test_size=val_size, random_state=seed)
    eval_ad_idx = np.where(abx_observed)[0]

    idx_dataset_path = os.path.join(dataset_path, "no_abx/")
    if not os.path.isdir(idx_dataset_path):
        os.mkdir(idx_dataset_path)

    with open(idx_dataset_path + 'train_idx.npy', 'wb') as f:
        np.save(f, train_idx)
    with open(idx_dataset_path + 'val_idx.npy', 'wb') as f:
        np.save(f, val_idx)
        
    with open(idx_dataset_path + 'eval_ad_idx.npy', 'wb') as f:
        np.save(f, eval_ad_idx)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ft_vat19_anomaly_v20230824_genus',
                    help='name of dataset file in data/original_data')

    parser.add_argument('--which_train', type=str,
                    help='criteria for the training data of the forecasting module')
    parser.add_argument('--which_eval', type=str,
                    help='criteria for the evaluation data of anomaly detection')

    parser.add_argument('--val_size', type=float, default=0.2,
                    help='proportion of the training dataset (for the forecasting module) used for validation')
    parser.add_argument('--seed', type=int, default=398,
                    help='radomness seed')

    args = parser.parse_args()
    dataset = args.dataset
    seed = args.seed
    val_size = args.val_size

    make_dataset(dataset=dataset,
                 seed=seed,
                 val_size=val_size)
