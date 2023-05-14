import numpy as np
import torch
import os
import json
from datetime import datetime
from matplotlib import pyplot as plt
from scipy.stats import norm

'''a = 0
    b = 1
    bins = 100
    points = np.linspace(-5,5,bins)
    
    x = norm.rvs(a, b, size=1000, random_state=123)
    
    loc1, scale1 = norm.fit(x)
    pdf = norm.pdf(points,loc=loc1,scale=scale1)
    plt.hist(x, bins=bins, density=True)
    plt.plot(points, pdf, color='r')
    plt.savefig('lala')
    plt.close()'''

'''a = 0
    b = 0.2

    x = np.arange(200)
    
    y = norm.rvs(a, b, size=200, random_state=123)
    y[100:150] += 2

    y_ = np.ma.masked_where(np.logical_and(x>=100, x<150), y)
    z_ = np.ma.masked_where(np.logical_or(x<99, x>150), y)

    plt.plot(y_, 'b', label='Normal stage')
    plt.plot(z_, 'r', label='Anomaly')
    plt.legend()
    plt.title("Gaussian white noise sample, with shift anomaly")
    plt.savefig('levy_anomly_shift.png')
    plt.close()'''

path = 'PD-NJODE/data/training_data/AD_OrnsteinUhlenbeckWithSeason-30/'

with open('{}data.npy'.format(path), 'rb') as f:
    data_paths = np.load(f)
    observed_dates = np.load(f) # [nb_paths, time_steps]
    nb_obs = np.load(f)  # array f ints with size [nb_paths]
    ad_labels = np.load(f) # anomaly labels [nb_paths, dim, time_steps]
    deter_paths = np.load(f)  # [nb_paths, dimension, time_steps]
    seasonal_function = np.load(f) # anomaly labels [nb_paths, dim, time_steps]
with open('{}metadata.txt'.format(path), 'r') as f:
    hyperparam_dict = json.load(f)

T = hyperparam_dict['maturity']
delta_t = hyperparam_dict['dt']  # copy metadata
t_period = hyperparam_dict['period']

anomaly_type = hyperparam_dict['anomaly_params']['type']

# print(data_paths)

ad_labels_1 = ad_labels.copy()
ad_labels_1[:,:,1:][np.logical_and(ad_labels[:,:,1:]==0,ad_labels[:,:,:-1]==1)] = 1
ad_labels_1[:,:,:-1][np.logical_and(ad_labels[:,:,1:]==1,ad_labels[:,:,:-1]==0)] = 1

# print(np.sum(ad_labels - ad_labels_1))

indices = np.arange(data_paths.shape[2])
times = np.linspace(0., T, int(np.round(T / delta_t)) + 1)
mask_data_path = np.ma.masked_where((ad_labels == 1), data_paths)
mask_data_path_w_anomaly = np.ma.masked_where((ad_labels_1 == 0), data_paths)
# mask_data_path_base = np.ma.masked_where((ad_labels == 1), deter_paths)
# mask_data_path_w_anomaly_base = np.ma.masked_where((ad_labels == 0), deter_paths)

dim = data_paths.shape[1]
path = "plots/AD_OrnsteinUhlenbeckWithSeason/"
for i in range(data_paths.shape[0]):
    if i < 5:
        fig, axs = plt.subplots(dim, figsize=(8, 4*dim))
        if dim == 1:
            axs = [axs]
        for j in range(data_paths.shape[1]):
            axs[j].plot(times, mask_data_path[i,j], color='b', label='True path, no anomaly')
            if anomaly_type == 'prout':
                path_t_obs = []
                path_X_obs = []
                for k, od in enumerate(ad_labels[i,j]):
                    if od == 1:
                        path_t_obs.append(times[k])
                        path_X_obs.append(data_paths[i, j, k])
                path_t_obs = np.array(path_t_obs)
                path_X_obs = np.array(path_X_obs)
                print(path_t_obs)
                print(path_X_obs)
                axs[j].scatter(path_t_obs, path_X_obs, color='r', label='True path, anomaly')
                #axs[j].scatter(times[i,j][ad_labels[i,j]==1], data_paths[i,j][ad_labels[i,j]==1], 'r', label='True path, anomaly')
            else:
                axs[j].plot(times, mask_data_path_w_anomaly[i,j], color='r', label='True path, anomaly')
            
            axs[j].plot(times, deter_paths[i,j], color='g', label='Deterministic path \n without anomaly')
            axs[j].plot(times, seasonal_function[i,j], 'y', label='Drift function')
            #if anomaly_type is None:
                #axs[j].plot(times, seasonal_function[i,j], 'y', label='Drift function')
            '''
            # plt.plot(indices, data_paths[i,j], 'b')
            # plt.plot(indices, mask_data_path_base[i,j], 'g')
            # plt.plot(indices, mask_data_path_w_anomaly_base[i,j], 'm')
            else:
                plt.plot(indices, data_paths[i,j], 'g')
                plt.plot(indices, data_paths_wo_noise[i,j], 'm')
            '''
        plt.legend(bbox_to_anchor=(1.02, dim/2.+0.1), loc="center left")
        plt.subplots_adjust(right=0.7)
        plt.xlabel('$t$')
        plt.savefig(path + "sample_{}_{}.png".format(i,anomaly_type))
        plt.close()

# if __name__== "__main__" :
 #   pass
