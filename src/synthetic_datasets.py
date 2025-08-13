"""
author: Florian Krach & Calypso Herrera

code to generate synthetic data from stock-model SDEs
"""

# ==============================================================================
from math import sqrt, exp, isnan, gamma
import numpy as np
import tqdm
from scipy.integrate import quad
from scipy.linalg import expm
import matplotlib.pyplot as plt
import copy, os, sys
import pickle
from fbm import fbm, fgn  # import fractional brownian motion package
import torch

import matplotlib.animation as animation
from tslearn.preprocessing import TimeSeriesResampler

from AD_synthetic_utils import RMDF, Season_NN


# ==============================================================================
# CLASSES
class StockModel:
    """
    mother class for all stock models defining the variables and methods shared
    amongst all of them, some need to be defined individually
    """

    def __init__(self, drift, volatility, S0, nb_paths, nb_steps,
                 maturity, sine_coeff, **kwargs):
        self.drift = drift
        self.volatility = volatility
        self.S0 = S0
        self.nb_paths = nb_paths
        self.nb_steps = nb_steps
        self.maturity = maturity
        self.dimensions = np.size(S0)
        if sine_coeff is None:
            self.periodic_coeff = lambda t: 1
        else:
            self.periodic_coeff = lambda t: (1 + np.sin(sine_coeff * t))
        self.loss = None
        self.path_t = None
        self.path_y = None

    def generate_paths(self, **options):
        """
        generate random paths according to the model hyperparams
        :return: stock paths as np.array, dim: [nb_paths, data_dim, nb_steps]
        """
        raise ValueError("not implemented yet")

    def next_cond_exp(self, *args, **kwargs):
        """
        compute the next point of the conditional expectation starting from
        given point for given time_delta
        :return: cond. exp. at next time_point (= current_time + time_delta)
        """
        raise ValueError("not implemented yet")

    def compute_cond_exp(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                         n_obs_ot, return_path=True, get_loss=False,
                         weight=0.5, store_and_use_stored=True,
                         start_time=None, moments = [], dim=None,
                         **kwargs):
        """
        compute conditional expectation similar to computing the prediction in
        the model.NJODE.forward
        ATTENTION: Works correctly only for non-masked data!
        :param times: see model.NJODE.forward
        :param time_ptr: see model.NJODE.forward
        :param X: see model.NJODE.forward, as np.array
        :param obs_idx: see model.NJODE.forward, as np.array
        :param delta_t: see model.NJODE.forward, as np.array
        :param T: see model.NJODE.forward
        :param start_X: see model.NJODE.forward, as np.array
        :param n_obs_ot: see model.NJODE.forward, as np.array
        :param return_path: see model.NJODE.forward
        :param get_loss: see model.NJODE.forward
        :param weight: see model.NJODE.forward
        :param store_and_use_stored: bool, whether the loss, and cond exp path
            should be stored and reused when calling the function again
        :param start_time: None or float, if float, this is first time point
        :param kwargs: unused, to allow for additional unused inputs
        :return: float (loss), if wanted paths of t and y (np.arrays)
        """
        if return_path and store_and_use_stored:
            if get_loss:
                if self.path_t is not None and self.loss is not None:
                    return self.loss, self.path_t, self.path_y
            else:
                if self.path_t is not None:
                    return self.loss, self.path_t, self.path_y
        elif store_and_use_stored:
            if get_loss:
                if self.loss is not None:
                    return self.loss

        y = start_X
        batch_size = start_X.shape[0]
        current_time = 0.0
        if start_time:
            current_time = start_time

        loss = 0

        if return_path:
            if start_time:
                path_t = []
                path_y = []
            else:
                path_t = [0.]
                path_y = [y]

        for i, obs_time in enumerate(times):
            # the following is needed for the combined stock model datasets
            if obs_time > T + 1e-10*delta_t:
                break
            if obs_time <= current_time:
                continue
            # Calculate conditional expectation stepwise
            while current_time < (obs_time - 1e-10*delta_t):
                if current_time < obs_time - delta_t:
                    delta_t_ = delta_t
                else:
                    delta_t_ = obs_time - current_time
                y = self.next_cond_exp(y, delta_t_, current_time)
                current_time = current_time + delta_t_

                # Storing the conditional expectation
                if return_path:
                    path_t.append(current_time)
                    path_y.append(y)

            # Reached an observation - set new interval
            start = time_ptr[i]
            end = time_ptr[i + 1]
            X_obs = X[start:end]
            i_obs = obs_idx[start:end]

            # Update h. Also updating loss, tau and last_X
            Y_bj = y
            temp = copy.copy(y)
            temp[i_obs] = X_obs
            y = temp
            Y = y

            if get_loss:
                loss = loss + compute_loss(X_obs=X_obs, Y_obs=Y[i_obs],
                                           Y_obs_bj=Y_bj[i_obs],
                                           n_obs_ot=n_obs_ot[i_obs],
                                           batch_size=batch_size, weight=weight)

            if return_path:
                path_t.append(obs_time)
                path_y.append(y)

        # after every observation has been processed, propagating until T
        while current_time < T - 1e-10 * delta_t:
            if current_time < T - delta_t:
                delta_t_ = delta_t
            else:
                delta_t_ = T - current_time
            y = self.next_cond_exp(y, delta_t_, current_time)
            current_time = current_time + delta_t_

            # Storing the predictions.
            if return_path:
                path_t.append(current_time)
                path_y.append(y)

        if get_loss and store_and_use_stored:
            self.loss = loss
        if return_path and store_and_use_stored:
            self.path_t = np.array(path_t)
            self.path_y = np.array(path_y)

        if return_path:
            # path dimension: [time_steps, batch_size, output_size]
            return loss, np.array(path_t), np.array(path_y)
        else:
            return loss

    def get_optimal_loss(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                         n_obs_ot, weight=0.5, M=None, mult=None, functions=None):
        if mult is not None and mult > 1:
            bs, dim = start_X.shape
            _dim = round(dim / mult)
            X = X[:, :_dim]
            start_X = start_X[:, :_dim]
            if M is not None:
                M = M[:, :_dim]

        loss, _, _ = self.compute_cond_exp(
            times, time_ptr, X, obs_idx, delta_t, T, start_X, n_obs_ot,
            return_path=True, get_loss=True, weight=weight, M=M)
        return loss
    
class Microbiome_OrnsteinUhlenbeck(StockModel):

    def __init__(self, volatility, nb_paths, nb_steps, dimension, S0, noise,
                 fct_params, anomaly_params, dynamic_vars, speed, maturity, sine_coeff=None, **kwargs):
        if S0 is not None:
            S0 = np.array(S0)
        super(Microbiome_OrnsteinUhlenbeck, self).__init__(
            volatility=volatility, nb_paths=nb_paths, drift=None,
            nb_steps=nb_steps, S0=S0, maturity=maturity,
            sine_coeff=sine_coeff
        )
        self.dynamic_vars = dynamic_vars
        self.anomaly_params = anomaly_params
        self.anomaly_type = self.anomaly_params['type']
        self.get_fct_generator(fct_params)

        # to change
        if speed is not None:
            if isinstance(speed, float):
                self.speed = speed * np.eye(self.dimensions)
            if isinstance(speed, list) and isinstance(speed[0], float):
                self.speed = np.array(speed) * np.eye(self.dimensions)
            if isinstance(speed, list) and isinstance(speed[0], list):
                self.speed = np.array(speed)
            assert(self.speed.shape == (self.dimensions, self.dimensions))
        else:
            NotImplementedError

        if noise is not None:
            self.noise_type = noise['type']
            if isinstance(noise['cov'], float):
                self.noise_cov = noise['cov'] * np.eye(self.dimensions)
            if isinstance(noise['cov'], list) and isinstance(noise['cov'][0], float):
                self.noise_cov = np.array(noise['cov']) * np.eye(self.dimensions)
            if isinstance(noise['cov'], list) and isinstance(noise['cov'][0], list):
                self.noise_cov = np.array(noise['cov'])
            assert(self.noise_cov.shape == (self.dimensions, self.dimensions))
        else:
            self.noise_cov = np.zeros((self.dimensions, self.dimensions))

        if volatility is not None:
            if isinstance(volatility['vol_value'], float):
                self.volatility = volatility['vol_value'] * np.eye(self.dimensions)
            if isinstance(volatility['vol_value'], list) and isinstance(volatility['vol_value'][0], float):
                self.volatility = np.array(volatility['vol_value']) * np.eye(self.dimensions)
            if isinstance(volatility['vol_value'], list) and isinstance(volatility['vol_value'][0], list):
                self.volatility = np.array(volatility['vol_value'])
            assert(self.noise_cov.shape == (self.dimensions, self.dimensions))
        else:
            self.volatility = np.zeros((self.dimensions, self.dimensions))

    def get_fct_generator(self, fct_params):

        self.fct_type = fct_params['type']

        if self.fct_type == 'invexp':
            self.fct_scale = fct_params['scale']
            self.fct_decay = fct_params['decay']
            self.fct = [lambda x : self.fct_scale * (1 - np.exp(-self.fct_decay * x)) for j in range(self.dimensions)]

        return 0


    def get_components(self):

        if self.fct_type == 'invexp':
            # same procedure as for RMDF, generate already array
            # times = np.expand_dims(np.arange(self.nb_steps+1), axis=1)
            times = np.expand_dims(np.linspace(0., 1., self.nb_steps + 1), axis=1)
            self.fct_pattern = np.transpose(np.concatenate([self.fct[j](times)
                                                                for j in range(self.dimensions)],axis=1),axes=(1,0))

        self.ad_labels = np.zeros(self.nb_steps + 1)

        if self.S0 is None:
            self.S0 = self.fct_pattern[:,0]

        # self.fct = lambda t: self.fct_patterns[:,int(t*self.nb_steps/self.maturity)]
        # self.anomalies = lambda t: self.ad_labels[int(t*self.nb_steps/self.maturity)]
        # self.drift = lambda x, t: - self.periodic_coeff(t) * self.speed @ (x - self.fct(t))
        self.diffusion = lambda x, t: self.volatility @ np.sqrt(np.maximum(x,1e-5))
        if self.noise_type == 'gaussian':
            def noise(x, t):
                if np.all(self.noise_cov == 0):
                    return np.zeros(self.dimensions)
                eps = np.random.normal(x, x, self.dimensions)
                return self.noise_cov @ eps
            self.noise = noise

    
    def compute_cond_exp(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                         n_obs_ot, return_path=True, get_loss=False,
                         weight=0.5, store_and_use_stored=False,
                         start_time=None, functions = None,
                         **kwargs):
        
        if return_path and store_and_use_stored:
            if get_loss:
                if self.path_t is not None and self.loss is not None:
                    return self.loss, self.path_t, self.path_y
            else:
                if self.path_t is not None:
                    return self.loss, self.path_t, self.path_y
        elif store_and_use_stored:
            if get_loss:
                if self.loss is not None:
                    return self.loss

        y = start_X
        batch_size = start_X.shape[0]

        current_time = 0.0
        if start_time:
            current_time = start_time

        loss = 0

        if return_path:
            if start_time:
                path_t = []
                path_y = []
            else:
                path_t = [0.]
                path_y = [y]

        last_times = current_time * np.ones(batch_size)

        for i, obs_time in enumerate(times):
            # the following is needed for the combined stock model datasets
            if obs_time > T + 1e-10*delta_t:
                break
            if obs_time <= current_time:
                continue

            # Calculate conditional expectation stepwise
            while current_time < (obs_time - 1e-10*delta_t):
                if current_time < obs_time - delta_t:
                    delta_t_ = delta_t
                else:
                    delta_t_ = obs_time - current_time

                current_time = current_time + delta_t_
                diff_t = current_time * np.ones(batch_size) - last_times
                y = self.next_cond_moments(y, diff_t, delta_t, current_time, functions)

                # Storing the conditional expectation
                if return_path:
                    path_t.append(current_time)
                    path_y.append(y)

            # Reached an observation - set new interval
            start = time_ptr[i]
            end = time_ptr[i + 1]
            X_obs = X[start:end]
            i_obs = obs_idx[start:end]
            last_times[i_obs] = current_time * np.ones(len(i_obs))

            # Update h. Also updating loss, tau and last_X
            Y_bj = y
            temp = copy.copy(y)
            temp[i_obs] = X_obs
            y = temp
            Y = y

            if get_loss:
                loss = loss + compute_loss(X_obs=X_obs, Y_obs=Y[i_obs],
                                           Y_obs_bj=Y_bj[i_obs],
                                           n_obs_ot=n_obs_ot[i_obs],
                                           batch_size=batch_size, weight=weight)

            if return_path:
                path_t.append(obs_time)
                path_y.append(y)

        last_time = current_time
        # after every observation has been processed, propagating until T
        while current_time < T - 1e-10 * delta_t:
            if current_time < T - delta_t:
                delta_t_ = delta_t
            else:
                delta_t_ = T - current_time
            
            '''
            y1 = y[:,:self.dimensions]
            y1 = self.next_cond_exp(y1, delta_t_, current_time)
            current_time = current_time + delta_t_
            diff_t = current_time * np.ones(batch_size) - last_times
            
            cond_moments = self.cond_moments(cond_exp=y1, diff_t=diff_t, current_t=current_time, 
                                        functions=functions)
            y = np.concatenate([y1,cond_moments], axis=1)
            '''
            current_time = current_time + delta_t_
            diff_t = current_time * np.ones(batch_size) - last_times
            y = self.next_cond_moments(y, diff_t, delta_t, current_time, functions)
            
            # Storing the predictions.
            if return_path:
                path_t.append(current_time)
                path_y.append(y)

        if get_loss and store_and_use_stored:
            self.loss = loss
        if return_path and store_and_use_stored:
            self.path_t = np.array(path_t)
            self.path_y = np.array(path_y)

        # print(np.array(path_y)[:,0,1] - np.power(np.array(path_y)[:,0,0],2))

        if return_path:
            # path dimension: [time_steps, batch_size, output_size]
            return loss, np.array(path_t), np.array(path_y)
        else:
            return loss
    
        
    def get_optimal_loss(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                         n_obs_ot, weight=0.5, M=None, mult=None):

        if mult is not None and mult > 1:
            bs, dim = start_X.shape
            _dim = round(dim / mult)
            X = X[:, :_dim]
            start_X = start_X[:, :_dim]
            if M is not None:
                M = M[:, :_dim]

        loss, _, _ = self.compute_cond_exp(
            times, time_ptr, X, obs_idx, delta_t, T, start_X, n_obs_ot,
            return_path=True, get_loss=True, weight=weight, M=M)
        return loss

    def next_cond_moments(self, y, diff_t, delta_t, current_t, functions=None):  # and in higher dimension ????

        self.get_components()
        if functions is None:
            functions = ['id']

        dim = self.dimensions
        batch_size = y.shape[0]

        res = np.zeros((batch_size, dim * len(functions)))

        factor = np.expand_dims(self.periodic_coeff(current_t) *  self.speed, axis=0)
        diff_t = np.expand_dims(diff_t, axis=(1,2))
        vol = np.matmul(self.volatility, np.transpose(self.volatility)).reshape(1, dim, dim)

        
        cond_exp_integral = delta_t * np.matmul(np.tile(factor, (batch_size, 1, 1)), 
                                                np.tile(np.expand_dims(self.fct(current_t),axis=(0,2)), 
                                                        (batch_size, 1, 1))).reshape(batch_size,dim)

        mat_diff_t = - np.tile(factor, (batch_size, 1, 1)) * np.tile(diff_t, (1, dim, dim))
        exp_term_diff_t = np.concatenate([expm(m).reshape(1,dim,dim) for m in mat_diff_t], axis=0)
        cond_var_add = np.matmul(np.matmul(exp_term_diff_t, np.tile(vol, (batch_size, 1, 1))), exp_term_diff_t.transpose((0,2,1)))
        cond_var_add = delta_t * np.diagonal(cond_var_add, axis1=1, axis2=2)

        mat_delta_t = - np.tile(factor, (batch_size, 1, 1)) * np.tile(np.array([delta_t]).reshape(1,1,1), (batch_size, dim, dim))
        exp_term_delta_t = np.concatenate([expm(m).reshape(1,dim,dim) for m in mat_delta_t], axis=0)

        which = np.argmax(np.array(functions) == 'id')
        prev_con_exp = y[:,dim*which:dim*(which+1)]
        cond_exp = np.matmul(exp_term_delta_t, prev_con_exp.reshape(batch_size,dim,1)).reshape(batch_size,dim) + cond_exp_integral
        res[:,dim*which:dim*(which+1)] = cond_exp

        for i,f in enumerate(functions):
            if f == "power-2":
                #cond_var = y[:,(i+1)*dim:(i+2)*dim] - np.power(prev_con_exp,2)
                cond_var = y[:,i*dim:(i+1)*dim] - np.power(prev_con_exp,2)
                cond_var += cond_var_add
                cond_exp_2 = cond_var + np.power(cond_exp, 2)
                #res[:,(i+1)*dim:(i+2)*dim] = cond_exp_2
                res[:,i*dim:(i+1)*dim] = cond_exp_2
        return res

    def get_anomaly_fcts(self):

        self.spike = False

        if self.anomaly_type is None:
            return None, None, []

        # if self.anomaly_type in ['scale', 'diffusion', 'noise', 'trend', 'cutoff']:
        if self.anomaly_type in ['cutoff']:
            self.occ_prob = self.anomaly_params['occurence_prob']
            self.occ_pos_range = self.anomaly_params['occurence_pos_range']
            self.occ_pos_law = self.anomaly_params['occurence_pos_law']
            self.occ_len_range = self.anomaly_params['occurence_len_range']
            self.occ_len_law = self.anomaly_params['occurence_len_law']
            self.occ_law = self.anomaly_params['occurence_law']
            self.occ_law_param = self.anomaly_params['occurence_law_param']

            r = np.random.binomial(1, self.occ_prob, 1)
            if r == 0:
                return None, None, []

            pos_list = []
            olr0, olr1 = self.occ_len_range
            opr0, opr1 = self.occ_pos_range
            if self.occ_law == 'single':
                if self.occ_len_law == 'uniform':
                    length = float(np.random.uniform(olr0,olr1,1))
                if self.occ_pos_law == 'uniform':
                    pos = float(np.random.uniform(opr0,opr1-length,1))
                for j in range(self.dimensions):
                    l = []
                    l.append((pos, pos+length))
                    pos_list.append(l)

            elif self.occ_law == 'geometric':
                n = np.random.geometric(self.occ_law_param, 1)
                pos_list = [[] for j in range(self.dimensions)]
                for o in range(int(n)):
                    if self.occ_len_law == 'uniform':
                        length = float(np.random.uniform(olr0,olr1,1))
                    if self.occ_pos_law == 'uniform':
                        pos = float(np.random.uniform(opr0,opr1-length,1))
                    for j in range(self.dimensions):
                        pos_list[j].append((pos, pos+length))

            exposure_steps = [int(p[0] * self.nb_steps) for p in pos_list[0]]
        
        # if self.anomaly_type == 'diffusion':
        #     ad_labels = copy.copy(self.ad_labels)

        #     diff_change = self.anomaly_params['diffusion_change']
        #     diff_deviation = self.anomaly_params['diffusion_deviation']

        #     diffusion_pattern = np.tile(np.expand_dims(self.volatility, 0), (self.nb_steps+1,1,1))
        #     for j in range(self.dimensions):
        #         for p in pos_list[j]:
        #             p0 = int(p[0] * self.nb_steps)
        #             p1 = int(p[1] * self.nb_steps)
        #             if diff_change == 'multiplicative':
        #                 diffusion_pattern[p0:p1,j,j] *= diff_deviation
        #             elif diff_change == 'additive':
        #                 diffusion_pattern[p0:p1,j,j] += diff_deviation
        #             ad_labels[j,p0:p1] = 1

        #     anomalies = lambda t: ad_labels[:,int(t*self.nb_steps/self.maturity)]
        #     diffusion = lambda x, t: diffusion_pattern[int(t*self.nb_steps/self.maturity)]
                
        #     return None, diffusion, None, anomalies, exposure_steps
        
        # if self.anomaly_type == 'noise':
        #     ad_labels = copy.copy(self.ad_labels)

        #     noise_change = self.anomaly_params['noise_change']
        #     noise_deviation = self.anomaly_params['noise_deviation']
                
        #     noise_cov_pattern = np.tile(np.expand_dims(self.noise_cov, 0), (self.nb_steps+1,1,1))
        #     for j in range(self.dimensions):
        #         for p in pos_list[j]:
        #             p0 = int(p[0] * self.nb_steps)
        #             p1 = int(p[1] * self.nb_steps)
        #             if noise_change == 'multiplicative':
        #                 noise_cov_pattern[p0:p1,j,j] *= noise_deviation
        #             elif noise_change == 'additive':
        #                 noise_cov_pattern[p0:p1,j,j] += noise_deviation
        #             ad_labels[j,p0:p1] = 1

        #     anomalies = lambda t: ad_labels[:,int(t*self.nb_steps/self.maturity)]
        #     noise = lambda x, t: noise_cov_pattern[int(t*self.nb_steps/self.maturity)] @ np.random.normal(0, 1, self.dimensions)
                
        #     return None, None, noise, anomalies, exposure_steps
        
        # elif self.anomaly_type == 'scale':

        #     fct_patterns = copy.copy(self.fct_patterns)
        #     ad_labels = copy.copy(self.ad_labels)

        #     scale_level_law = self.anomaly_params['scale_level_law']

        #     for j in range(self.dimensions):
        #         for p in pos_list[j]:
        #             p0 = int(p[0] * self.nb_steps)
        #             p1 = int(p[1] * self.nb_steps)
        #             if scale_level_law == 'uniform':
        #                 c0, c1 = self.anomaly_params['scale_level_range']
        #                 scale_level = float(np.random.uniform(c0,c1,1))
        #             fct_patterns[j,p0:p1] *= scale_level
        #             ad_labels[j,p0:p1] = 1

        #     seasons = lambda t: fct_patterns[:,int(t*self.nb_steps/self.maturity)]
        #     anomalies = lambda t: ad_labels[:,int(t*self.nb_steps/self.maturity)]
        #     drift = lambda x, t: - self.periodic_coeff(t) * self.speed @ (x - seasons(t))

        #     return drift, None, None, anomalies, exposure_steps
        
        # elif self.anomaly_type == 'trend':

        #     fct_patterns = copy.copy(self.fct_patterns)
        #     ad_labels = copy.copy(self.ad_labels)

        #     trend_level_law = self.anomaly_params['trend_level_law']

        #     for j in range(self.dimensions):
        #         for p in pos_list[j]:
        #             p0 = int(p[0] * self.nb_steps)
        #             p1 = int(p[1] * self.nb_steps)
        #             if trend_level_law == 'uniform':
        #                 c0, c1 = self.anomaly_params['trend_level_range']
        #                 trend_level = float(np.random.uniform(c0,c1,1))
        #                 if self.anomaly_params['trend_level_sign'] == 'both':
        #                     s = np.random.binomial(1, 0.5, 1)
        #                     if s == 0:
        #                         trend_level = -trend_level
        #                 elif self.anomaly_params['trend_level_sign'] == 'minus':
        #                     trend_level = -trend_level

        #             length = p1-p0
        #             fct_patterns[j,p0:p1] += np.arange(length) * trend_level / self.nb_steps
        #             ad_labels[j,p0:p1] = 1

        #     seasons = lambda t: fct_patterns[:,int(t*self.nb_steps/self.maturity)]
        #     anomalies = lambda t: ad_labels[:,int(t*self.nb_steps/self.maturity)]
        #     drift = lambda x, t: - self.periodic_coeff(t) * self.speed @ (x - seasons(t))

        #     return drift, None, None, anomalies, exposure_steps
        
        if self.anomaly_type == 'cutoff':

            fct_pattern = copy.copy(self.fct_pattern)
            ad_labels = copy.copy(self.ad_labels)

            cutoff_level_law = self.anomaly_params['cutoff_level_law']

            for j in range(self.dimensions):
                for p in pos_list[j]:
                    p0 = int(p[0] * self.nb_steps)
                    p1 = int(p[1] * self.nb_steps)
                    if cutoff_level_law == 'uniform':
                        c0, c1 = self.anomaly_params['cutoff_level_range']
                        cutoff_level = float(np.random.uniform(c0,c1,1))
                    elif cutoff_level_law == 'current_level':
                        cutoff_level = self.fct_pattern(p[0])
                    fct_pattern[j,p0:p1] = cutoff_level * fct_pattern[j,p0]
                    ad_labels[p0:p1] = 1

            # seasons = lambda t: fct_patterns[:,int(t*self.nb_steps/self.maturity)]
            # anomalies = lambda t: ad_labels[int(t*self.nb_steps/self.maturity)]
            # drift = lambda x, t: - self.periodic_coeff(t) * self.speed @ (x - seasons(t))

            # return drift, None, None, anomalies, exposure_steps

            return fct_pattern, ad_labels, exposure_steps
        

    def get_dynamic_vars(self, fct_pattern):

        dynamic_vars = np.zeros((self.nb_steps + 1, 0))

        for var in self.dynamic_vars:
            if var["type"] == "static":
                x = np.random.random()
                s = var["probs"][0]
                v = 0
                while s < x:
                    v += 1
                    s += var["probs"][v]
                var_values = np.zeros((self.nb_steps + 1, var["nb_vals"]))
                var_values[:,v] = 1
                fct_pattern[:,:var["duration"]] *= var["factor"][v]
                dynamic_vars = np.concatenate([dynamic_vars, var_values],axis=1)
            if var["type"] == "dynamic":
                durations = [0]
                var_values = np.zeros((self.nb_steps + 1, var["nb_vals"]))
                for v in range(var["nb_vals"]):
                    if var["goem_law"][v] is not None:
                        x = np.random.geometric(var["goem_law"][v])
                    else:
                        x = self.nb_steps + 1
                    durations.append(durations[-1] + min(x, var["max_dur"][v]))
                    if durations[-2] < self.nb_steps:
                        var_values[durations[-2]:min(durations[-1],self.nb_steps + 1),v] = 1
                        fct_pattern[:,durations[-2]:durations[-1]] *= var["factor"][v]

                dynamic_vars = np.concatenate([dynamic_vars, var_values],axis=1)

        return fct_pattern, dynamic_vars


    def generate_paths(self, start_X=None, no_S0=True):
        # Diffusion of the variance: dv = -k(v-season(t))*dt + vol*dW
        if no_S0:
            self.S0 = None

        self.get_components()

        spot_paths = np.empty((self.nb_paths, self.dimensions, self.nb_steps + 1))
        deter_paths = np.empty_like(spot_paths)
        final_paths = np.empty_like(spot_paths)
        ad_label_paths = np.zeros((self.nb_paths, self.nb_steps + 1))
        path_fct_patterns = np.empty_like(spot_paths)
        path_exposure_steps = np.empty(self.nb_paths, dtype=object)
        dynamic = np.zeros((self.nb_paths, sum([var["nb_vals"] for var in self.dynamic_vars]), self.nb_steps + 1))

        dt = self.maturity / self.nb_steps

        if start_X is not None:
            spot_paths[:, :, 0] = start_X
        for i in tqdm.tqdm(range(self.nb_paths), total=self.nb_paths):

            fct_pattern, ad_labels, exposure_steps = self.get_anomaly_fcts()
            diffusion = self.diffusion
            noise = self.noise
            if ad_labels is None:
                ad_labels = self.ad_labels
            anomalies = lambda t: ad_labels[int(t*self.nb_steps/self.maturity)]
            if fct_pattern is None:
                fct_pattern = copy.copy(self.fct_pattern)
            path_exposure_steps[i] = exposure_steps

            fct_pattern, dynamic_vars = self.get_dynamic_vars(fct_pattern)
            dynamic[i] = np.transpose(dynamic_vars)

            fct = lambda t: fct_pattern[:,int(t*self.nb_steps/self.maturity)]
            drift = lambda x, t: - self.periodic_coeff(t) * self.speed @ (x - fct(t))

            if start_X is None:
                spot_paths[i, :, 0] = self.S0
                deter_paths[i, :, 0] = self.S0
                final_paths[i, :, 0] = (spot_paths[i, :, 0] + noise(spot_paths[i, :, 0], (0) * dt)) # @ eps)
                path_fct_patterns[i, :, 0] = (fct(0.))
            for k in range(1, self.nb_steps + 1):
                random_numbers_bm = np.random.normal(0, 1, self.dimensions)
                dW = random_numbers_bm * np.sqrt(dt)
                spot_paths[i, :, k] = (
                        spot_paths[i, :, k - 1]
                        + drift(spot_paths[i, :, k - 1], (k - 1) * dt) * dt
                        + diffusion(spot_paths[i, :, k - 1], (k) * dt) @ dW)
                final_paths[i, :, k] = (spot_paths[i, :, k] + noise(spot_paths[i, :, k], (k) * dt)) # @ eps)
                deter_paths[i, :, k] = (
                        deter_paths[i, :, k - 1]
                        + drift(deter_paths[i, :, k - 1], (k - 1) * dt) * dt)
                path_fct_patterns[i,:,k] = (fct((k - 1) * dt))
                ad_label_paths[i, k] = (anomalies((k - 1) * dt))
        
        # stock_path, final_paths, deter_paths, seasonal_function, ad_labels : [nb_paths, dimension, time_steps]
        # return season_pattern, ad_labels
        return final_paths, ad_label_paths, deter_paths, path_fct_patterns, dt, path_exposure_steps, dynamic

class AD_OrnsteinUhlenbeckWithSeason(StockModel):

    def __init__(self, volatility, nb_paths, nb_steps, dimension, S0, noise,
                 season_params, anomaly_params, speed, maturity, sine_coeff=None, **kwargs):
        if S0 is None:
            S0 = np.zeros(dimension)
        super(AD_OrnsteinUhlenbeckWithSeason, self).__init__(
            volatility=volatility, nb_paths=nb_paths, drift=None,
            nb_steps=nb_steps, S0=S0, maturity=maturity,
            sine_coeff=sine_coeff
        )
        self.anomaly_params = anomaly_params
        self.anomaly_type = self.anomaly_params['type']
        self.get_season_generator(season_params)

        # to change
        if speed is not None:
            if isinstance(speed, float):
                self.speed = speed * np.eye(self.dimensions)
            if isinstance(speed, list) and isinstance(speed[0], float):
                self.speed = np.array(speed) * np.eye(self.dimensions)
            if isinstance(speed, list) and isinstance(speed[0], list):
                self.speed = np.array(speed)
            assert(self.speed.shape == (self.dimensions, self.dimensions))
        else:
            NotImplementedError

        if noise is not None:
            self.noise_type = noise['type']
            if isinstance(noise['cov'], float):
                self.noise_cov = noise['cov'] * np.eye(self.dimensions)
            if isinstance(noise['cov'], list) and isinstance(noise['cov'][0], float):
                self.noise_cov = np.array(noise['cov']) * np.eye(self.dimensions)
            if isinstance(noise['cov'], list) and isinstance(noise['cov'][0], list):
                self.noise_cov = np.array(noise['cov'])
            assert(self.noise_cov.shape == (self.dimensions, self.dimensions))
        else:
            self.noise_cov = np.zeros((self.dimensions, self.dimensions))

        if volatility is not None:
            if isinstance(volatility['vol_value'], float):
                self.volatility = volatility['vol_value'] * np.eye(self.dimensions)
            if isinstance(volatility['vol_value'], list) and isinstance(volatility['vol_value'][0], float):
                self.volatility = np.array(volatility['vol_value']) * np.eye(self.dimensions)
            if isinstance(volatility['vol_value'], list) and isinstance(volatility['vol_value'][0], list):
                self.volatility = np.array(volatility['vol_value'])
            assert(self.noise_cov.shape == (self.dimensions, self.dimensions))
        else:
            self.volatility = np.zeros((self.dimensions, self.dimensions))

    def get_season_generator(self, season_params):

        self.season_type = season_params['seas_type']
        self.new_season = season_params['new_seas']
        self.season_path = season_params['seas_path']
        self.changing_season = season_params['changing_seas']
        self.season_nb = season_params['seas_nb']
        self.season_length = season_params['seas_length']
        self.anomaly_type = self.anomaly_params['type']
        self.seas_amp = season_params['seas_amp']
        if 'seas_range' in season_params:
            self.seas_range = season_params['seas_range']

        if self.season_type == 'RMDF':
            self.season_generator = [RMDF(depth=season_params['depth']) for j in range(self.dimensions)]
            self.forking_depth = season_params['forking_depth']
            if self.new_season:
                for j in range(self.dimensions):
                    self.season_generator[j].gen_anchor()
                    self.season_generator[j].save_anchor(self.season_path + '_dim{}.pkl'.format(j))
            else:
                for j in range(self.dimensions):
                    self.season_generator[j].load_anchor(self.season_path + '_dim{}.pkl'.format(j))
        if self.season_type == 'NN':
            self.season_nn_layers = season_params['nn_layers']
            self.season_nn_bias = season_params['nn_bias']
            self.season_nn_input = season_params['nn_input']
            self.season_generator = [Season_NN(self.season_nn_layers, self.season_nn_input, 1, 
                                               self.season_nn_bias) for j in range(self.dimensions)]
            seas_path = os.path.dirname(self.season_path)
            if not os.path.exists(seas_path):
                os.makedirs(seas_path)
                self.new_season = True
            if self.new_season:
                for j in range(self.dimensions):
                    self.season_generator[j].gen_weigths()
                    self.season_generator[j].save(self.season_path + '_dim{}.pt'.format(j))
            else:
                for j in range(self.dimensions):
                    self.season_generator[j].load(self.season_path + '_dim{}.pt'.format(j))
            for j in range(self.dimensions):
                self.season_generator[j].eval()

        return 0


    def get_components(self):

        if self.season_type == 'RMDF':
            if self.forking_depth == 0:
                base_season_patterns = np.concatenate([np.expand_dims(self.season_generator[j].gen(self.forking_depth, 
                                                                                                   self.season_length),axis=0) for j in range(self.dimensions)],axis=0)
                season_patterns = [base_season_patterns for s in range(self.season_nb)]
            else:
                season_patterns = []
                for s in range(self.season_nb):
                    season_patterns.append(np.concatenate([np.expand_dims(self.season_generator[j].gen(self.forking_depth, 
                                                                                                   self.season_length),axis=0) for j in range(self.dimensions)],axis=0))
            
        if self.season_type == 'NN':
            # same procedure as for RMDF, generate already array
            times = torch.tensor(2. * np.pi * np.linspace(0., 1., self.season_length)).unsqueeze(1)
            base_season_patterns = np.transpose(np.concatenate([self.season_generator[j](times.clone().detach()).detach().numpy() 
                                                                for j in range(self.dimensions)],axis=1),axes=(1,0))
            season_patterns = [copy.copy(base_season_patterns) for s in range(self.season_nb)]

        ad_labels = [np.zeros_like(base_season_patterns) for s in range(self.season_nb)]

        if self.seas_amp == 'automatic':
            min_seas = np.min(season_patterns[0])
            max_seas = np.max(season_patterns[0])
            min_exp = self.seas_range[0]
            max_exp = self.seas_range[1]

        # if self.anomaly_type == 'deformation':
        #         season_patterns, ad_labels = self.inject_deformation_anomaly(season_patterns, ad_labels)

        season_patterns = np.concatenate(season_patterns, axis=1)
        season_patterns = np.reshape(TimeSeriesResampler(sz=self.nb_steps+1).fit_transform(season_patterns),(self.dimensions,-1))
        if self.seas_amp == 'automatic':
            factor = (max_exp - min_exp) / (max_seas - min_seas)
            season_patterns = (season_patterns - min_seas) * factor + min_exp
        else:
            season_patterns = self.seas_amp * season_patterns

        ad_labels = np.concatenate(ad_labels, axis=1)
        ad_labels = np.reshape(TimeSeriesResampler(sz=self.nb_steps+1).fit_transform(ad_labels),(self.dimensions,-1))
        ad_labels[ad_labels > 0] = 1
        self.season_patterns = season_patterns
        self.ad_labels = ad_labels
        if self.S0 is None:
            self.S0 = season_patterns[:,0]

        self.seasons = lambda t: self.season_patterns[:,int(t*self.nb_steps/self.maturity)]
        self.anomalies = lambda t: self.ad_labels[:,int(t*self.nb_steps/self.maturity)]
        self.drift = lambda x, t: - self.periodic_coeff(t) * self.speed @ (x - self.seasons(t))
        self.diffusion = lambda x, t: self.volatility
        if self.noise_type == 'gaussian':
            def noise(x, t):
                eps = np.random.normal(0, 1, self.dimensions)
                return self.noise_cov @ eps
            self.noise = noise

    
    def compute_cond_exp(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                         n_obs_ot, return_path=True, get_loss=False,
                         weight=0.5, store_and_use_stored=False,
                         start_time=None, functions = None,
                         **kwargs):
        
        if return_path and store_and_use_stored:
            if get_loss:
                if self.path_t is not None and self.loss is not None:
                    return self.loss, self.path_t, self.path_y
            else:
                if self.path_t is not None:
                    return self.loss, self.path_t, self.path_y
        elif store_and_use_stored:
            if get_loss:
                if self.loss is not None:
                    return self.loss

        y = start_X
        batch_size = start_X.shape[0]
        '''dim_obs = y.shape[1]
        dim_tot = self.dimensions * len(functions)
        if dim_obs != dim_tot:
            y = np.concatenate([y, np.zeros((batch_size, dim_tot-dim_obs))],axis=1)'''

        current_time = 0.0
        if start_time:
            current_time = start_time

        loss = 0

        if return_path:
            if start_time:
                path_t = []
                path_y = []
            else:
                path_t = [0.]
                path_y = [y]

        last_times = current_time * np.ones(batch_size)

        for i, obs_time in enumerate(times):
            # the following is needed for the combined stock model datasets
            if obs_time > T + 1e-10*delta_t:
                break
            if obs_time <= current_time:
                continue

            # Calculate conditional expectation stepwise
            while current_time < (obs_time - 1e-10*delta_t):
                if current_time < obs_time - delta_t:
                    delta_t_ = delta_t
                else:
                    delta_t_ = obs_time - current_time

                '''
                y1 = y[:,:self.dimensions]
                y1 = self.next_cond_exp(y1, delta_t_, current_time)
                current_time = current_time + delta_t_
                diff_t = current_time * np.ones(batch_size) - last_times
                cond_moments = self.cond_moments(cond_exp=y1, diff_t=diff_t, current_t=current_time, 
                                            functions=functions)
                y = np.concatenate([y1,cond_moments], axis=1)
                '''
                current_time = current_time + delta_t_
                diff_t = current_time * np.ones(batch_size) - last_times
                y = self.next_cond_moments(y, diff_t, delta_t, current_time, functions)

                # Storing the conditional expectation
                if return_path:
                    path_t.append(current_time)
                    path_y.append(y)

            # Reached an observation - set new interval
            start = time_ptr[i]
            end = time_ptr[i + 1]
            X_obs = X[start:end]
            i_obs = obs_idx[start:end]
            last_times[i_obs] = current_time * np.ones(len(i_obs))

            # Update h. Also updating loss, tau and last_X
            Y_bj = y
            temp = copy.copy(y)
            temp[i_obs] = X_obs
            y = temp
            Y = y

            if get_loss:
                loss = loss + compute_loss(X_obs=X_obs, Y_obs=Y[i_obs],
                                           Y_obs_bj=Y_bj[i_obs],
                                           n_obs_ot=n_obs_ot[i_obs],
                                           batch_size=batch_size, weight=weight)

            if return_path:
                path_t.append(obs_time)
                path_y.append(y)

        last_time = current_time
        # after every observation has been processed, propagating until T
        while current_time < T - 1e-10 * delta_t:
            if current_time < T - delta_t:
                delta_t_ = delta_t
            else:
                delta_t_ = T - current_time
            
            '''
            y1 = y[:,:self.dimensions]
            y1 = self.next_cond_exp(y1, delta_t_, current_time)
            current_time = current_time + delta_t_
            diff_t = current_time * np.ones(batch_size) - last_times
            
            cond_moments = self.cond_moments(cond_exp=y1, diff_t=diff_t, current_t=current_time, 
                                        functions=functions)
            y = np.concatenate([y1,cond_moments], axis=1)
            '''
            current_time = current_time + delta_t_
            diff_t = current_time * np.ones(batch_size) - last_times
            y = self.next_cond_moments(y, diff_t, delta_t, current_time, functions)
            
            # Storing the predictions.
            if return_path:
                path_t.append(current_time)
                path_y.append(y)

        if get_loss and store_and_use_stored:
            self.loss = loss
        if return_path and store_and_use_stored:
            self.path_t = np.array(path_t)
            self.path_y = np.array(path_y)

        # print(np.array(path_y)[:,0,1] - np.power(np.array(path_y)[:,0,0],2))

        if return_path:
            # path dimension: [time_steps, batch_size, output_size]
            return loss, np.array(path_t), np.array(path_y)
        else:
            return loss
    
        
    def get_optimal_loss(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                         n_obs_ot, weight=0.5, M=None, mult=None):

        if mult is not None and mult > 1:
            bs, dim = start_X.shape
            _dim = round(dim / mult)
            X = X[:, :_dim]
            start_X = start_X[:, :_dim]
            if M is not None:
                M = M[:, :_dim]

        loss, _, _ = self.compute_cond_exp(
            times, time_ptr, X, obs_idx, delta_t, T, start_X, n_obs_ot,
            return_path=True, get_loss=True, weight=weight, M=M)
        return loss

    def next_cond_moments(self, y, diff_t, delta_t, current_t, functions=None):  # and in higher dimension ????

        self.get_components()
        if functions is None:
            functions = ['id']

        dim = self.dimensions
        batch_size = y.shape[0]

        res = np.zeros((batch_size, dim * len(functions)))

        factor = np.expand_dims(self.periodic_coeff(current_t) *  self.speed, axis=0)
        diff_t = np.expand_dims(diff_t, axis=(1,2))
        vol = np.matmul(self.volatility, np.transpose(self.volatility)).reshape(1, dim, dim)

        
        cond_exp_integral = delta_t * np.matmul(np.tile(factor, (batch_size, 1, 1)), 
                                                np.tile(np.expand_dims(self.seasons(current_t),axis=(0,2)), 
                                                        (batch_size, 1, 1))).reshape(batch_size,dim)

        mat_diff_t = - np.tile(factor, (batch_size, 1, 1)) * np.tile(diff_t, (1, dim, dim))
        exp_term_diff_t = np.concatenate([expm(m).reshape(1,dim,dim) for m in mat_diff_t], axis=0)
        cond_var_add = np.matmul(np.matmul(exp_term_diff_t, np.tile(vol, (batch_size, 1, 1))), exp_term_diff_t.transpose((0,2,1)))
        cond_var_add = delta_t * np.diagonal(cond_var_add, axis1=1, axis2=2)

        mat_delta_t = - np.tile(factor, (batch_size, 1, 1)) * np.tile(np.array([delta_t]).reshape(1,1,1), (batch_size, dim, dim))
        exp_term_delta_t = np.concatenate([expm(m).reshape(1,dim,dim) for m in mat_delta_t], axis=0)

        which = np.argmax(np.array(functions) == 'id')
        prev_con_exp = y[:,dim*which:dim*(which+1)]
        cond_exp = np.matmul(exp_term_delta_t, prev_con_exp.reshape(batch_size,dim,1)).reshape(batch_size,dim) + cond_exp_integral
        res[:,dim*which:dim*(which+1)] = cond_exp

        for i,f in enumerate(functions):
            if f == "power-2":
                #cond_var = y[:,(i+1)*dim:(i+2)*dim] - np.power(prev_con_exp,2)
                cond_var = y[:,i*dim:(i+1)*dim] - np.power(prev_con_exp,2)
                cond_var += cond_var_add
                cond_exp_2 = cond_var + np.power(cond_exp, 2)
                #res[:,(i+1)*dim:(i+2)*dim] = cond_exp_2
                res[:,i*dim:(i+1)*dim] = cond_exp_2
        return res
    
    '''    
    def cond_moments(self, cond_exp, diff_t, current_t, functions):  # and in higher dimension ????
        # if functions[0] == 'id':
        #     functions = functions[1:]
        nb_moments = len(functions)
        dim = self.dimensions
        factor = self.speed * self.periodic_coeff(current_t)
        # cond_var = (self.volatility ** 2) * (1-np.exp(-2. * factor * diff_t)) / (2. * factor)
        vol = np.diagonal(self.volatility @ np.transpose(self.volatility))
        cond_var_factor = (1. - np.exp(-2. * factor * diff_t)) / (2. * factor)
        cond_var_factor = cond_var_factor.reshape(-1,1)
        dim0 = cond_var_factor.shape[0]
        cond_var = np.tile(cond_var_factor,(1,dim)) * np.tile(vol.reshape(1,dim),(dim0,1))
        cond_var = cond_var.reshape(-1,dim)
        res = np.zeros((cond_exp.shape[0], cond_exp.shape[1] * nb_moments))
        for i,f in enumerate(functions):
            if f == "power-2":
                cond_exp_2 = cond_var + np.power(cond_exp, 2)
                res[:,i*dim:(i+1)*dim] = cond_exp_2
            # if f == "var":
            #    res[:,i*dim:(i+1)*dim] = cond_var
        return res

    def next_cond_exp(self, y, delta_t, current_t): # dimension of self.speed, in case batch !!!
        self.get_components()
        factor = self.speed * self.periodic_coeff(current_t)
        integral = delta_t * self.speed * np.exp(factor * current_t) * self.seasons(current_t)    # redundant operation
        res = y * np.exp(- factor * delta_t) + np.exp(- factor * current_t) * integral
        return res
    '''

    def get_anomaly_fcts(self):
        assert(self.season_type == 'NN')

        self.spike = False

        if self.anomaly_type is None:
            return None, None, None, None, None

        if self.anomaly_type in ['deformation', 'scale', 'diffusion', 'noise', 'trend', 'cutoff']:

            self.occ_prob = self.anomaly_params['occurence_prob']
            self.occ_pos_range = self.anomaly_params['occurence_pos_range']
            self.occ_pos_law = self.anomaly_params['occurence_pos_law']
            self.occ_len_range = self.anomaly_params['occurence_len_range']
            self.occ_len_law = self.anomaly_params['occurence_len_law']
            self.dim_occ_pos = self.anomaly_params['dim_occurence_pos']
            self.dim_occ_law = self.anomaly_params['dim_occurence_law']
            self.dim_occ_prob = self.anomaly_params['dim_occurence_prob']

            pos_list = []
            
            r = np.random.binomial(1, self.occ_prob, 1)
            if r == 1:
                olr0, olr1 = self.occ_len_range
                opr0, opr1 = self.occ_pos_range
                if self.dim_occ_pos == 'same':
                    if self.occ_len_law == 'uniform':
                        length = float(np.random.uniform(olr0,olr1,1))
                    if self.occ_pos_law == 'uniform':
                        pos = float(np.random.uniform(opr0,opr1-length,1))
                    for j in range(self.dimensions):
                        l = []
                        l.append(pos, pos+length)
                        pos_list.append(l)
                elif self.dim_occ_pos == 'indep':
                    for j in range(self.dimensions):
                        if self.occ_len_law == 'uniform':
                            length = float(np.random.uniform(olr0,olr1,1))
                        if self.occ_pos_law == 'uniform':
                            pos = float(np.random.uniform(opr0,opr1-length,1))
                        l = []
                        l.append((pos, pos+length))
                        pos_list.append(l)

                for j in range(self.dimensions):
                    for p in range(len(pos_list[j])):
                        r = np.random.binomial(1, self.dim_occ_prob, 1)
                        if r == 0:
                            del pos_list[j][p]

        if self.anomaly_type == 'deformation':
            season_patterns = copy.copy(self.season_patterns)
            ad_labels = copy.copy(self.ad_labels)

            for j in range(self.dimensions):
                for p in pos_list[j]:

                    anomaly_season_generator = Season_NN(self.season_nn_layers, self.season_nn_input, 1, 
                                                            self.season_nn_bias)
                    anomaly_season_generator.gen_weigths()
                    times = torch.tensor(2. * np.pi * np.linspace(0., 1., self.season_length)).unsqueeze(1)
                    base_season_pattern = anomaly_season_generator(times.clone().detach()).squeeze().detach().numpy()
                    anomaly_season_pattern = [copy.copy(base_season_pattern) for s in range(self.season_nb)]
                    if self.seas_amp == 'automatic':
                        min_seas = np.min(anomaly_season_pattern[0])
                        max_seas = np.max(anomaly_season_pattern[0])
                        min_exp = self.seas_range[0]
                        max_exp = self.seas_range[1]
                    anomaly_season_pattern = np.concatenate(anomaly_season_pattern, axis=0)
                    anomaly_season_pattern = np.reshape(TimeSeriesResampler(sz=self.nb_steps+1).fit_transform(anomaly_season_pattern),(-1))
                    # anomaly_season_pattern = np.squeeze(anomaly_season_pattern)
                    if self.seas_amp == 'automatic':
                        factor = (max_exp - min_exp) / (max_seas - min_seas)
                        anomaly_season_pattern = (anomaly_season_pattern - min_seas) * factor + min_exp
                    else:
                        anomaly_season_pattern = self.seas_amp * anomaly_season_pattern
                    # if self.S0 is None:
                    #     self.S0 = anomaly_season_pattern[:,0]

                    p0 = int(p[0] * self.nb_steps)
                    p1 = int(p[1] * self.nb_steps)
                    season_patterns[j,p0:p1] = anomaly_season_pattern[p0:p1]
                    ad_labels[j,p0:p1] = 1

            seasons = lambda t: season_patterns[:,int(t*self.nb_steps/self.maturity)]
            anomalies = lambda t: ad_labels[:,int(t*self.nb_steps/self.maturity)]
            drift = lambda x, t: - self.periodic_coeff(t) * self.speed @ (x - seasons(t))

            return drift, None, None, anomalies, None
        
        elif self.anomaly_type == 'diffusion':
            ad_labels = copy.copy(self.ad_labels)

            diff_change = self.anomaly_params['diffusion_change']
            diff_deviation = self.anomaly_params['diffusion_deviation']

            diffusion_pattern = np.tile(np.expand_dims(self.volatility, 0), (self.nb_steps+1,1,1))
            for j in range(self.dimensions):
                for p in pos_list[j]:
                    p0 = int(p[0] * self.nb_steps)
                    p1 = int(p[1] * self.nb_steps)
                    if diff_change == 'multiplicative':
                        diffusion_pattern[p0:p1,j,j] *= diff_deviation
                    elif diff_change == 'additive':
                        diffusion_pattern[p0:p1,j,j] += diff_deviation
                    ad_labels[j,p0:p1] = 1

            anomalies = lambda t: ad_labels[:,int(t*self.nb_steps/self.maturity)]
            diffusion = lambda x, t: diffusion_pattern[int(t*self.nb_steps/self.maturity)]
                
            return None, diffusion, None, anomalies, None
        
        if self.anomaly_type == 'noise':
            ad_labels = copy.copy(self.ad_labels)

            noise_change = self.anomaly_params['noise_change']
            noise_deviation = self.anomaly_params['noise_deviation']
                
            noise_cov_pattern = np.tile(np.expand_dims(self.noise_cov, 0), (self.nb_steps+1,1,1))
            for j in range(self.dimensions):
                for p in pos_list[j]:
                    p0 = int(p[0] * self.nb_steps)
                    p1 = int(p[1] * self.nb_steps)
                    if noise_change == 'multiplicative':
                        noise_cov_pattern[p0:p1,j,j] *= noise_deviation
                    elif noise_change == 'additive':
                        noise_cov_pattern[p0:p1,j,j] += noise_deviation
                    ad_labels[j,p0:p1] = 1

            anomalies = lambda t: ad_labels[:,int(t*self.nb_steps/self.maturity)]
            noise = lambda x, t: noise_cov_pattern[int(t*self.nb_steps/self.maturity)] @ np.random.normal(0, 1, self.dimensions)
                
            return None, None, noise, anomalies, None
        
        elif self.anomaly_type == 'scale':

            season_patterns = copy.copy(self.season_patterns)
            ad_labels = copy.copy(self.ad_labels)

            scale_level_law = self.anomaly_params['scale_level_law']

            for j in range(self.dimensions):
                for p in pos_list[j]:
                    p0 = int(p[0] * self.nb_steps)
                    p1 = int(p[1] * self.nb_steps)
                    if scale_level_law == 'uniform':
                        c0, c1 = self.anomaly_params['scale_level_range']
                        scale_level = float(np.random.uniform(c0,c1,1))
                    season_patterns[j,p0:p1] *= scale_level
                    ad_labels[j,p0:p1] = 1

            seasons = lambda t: season_patterns[:,int(t*self.nb_steps/self.maturity)]
            anomalies = lambda t: ad_labels[:,int(t*self.nb_steps/self.maturity)]
            drift = lambda x, t: - self.periodic_coeff(t) * self.speed @ (x - seasons(t))

            return drift, None, None, anomalies, None
        
        elif self.anomaly_type == 'trend':

            season_patterns = copy.copy(self.season_patterns)
            ad_labels = copy.copy(self.ad_labels)

            trend_level_law = self.anomaly_params['trend_level_law']

            for j in range(self.dimensions):
                for p in pos_list[j]:
                    p0 = int(p[0] * self.nb_steps)
                    p1 = int(p[1] * self.nb_steps)
                    if trend_level_law == 'uniform':
                        c0, c1 = self.anomaly_params['trend_level_range']
                        trend_level = float(np.random.uniform(c0,c1,1))
                        if self.anomaly_params['trend_level_sign'] == 'both':
                            s = np.random.binomial(1, 0.5, 1)
                            if s == 0:
                                trend_level = -trend_level
                        elif self.anomaly_params['trend_level_sign'] == 'minus':
                            trend_level = -trend_level

                    length = p1-p0
                    season_patterns[j,p0:p1] += np.arange(length) * trend_level / self.nb_steps
                    ad_labels[j,p0:p1] = 1

            seasons = lambda t: season_patterns[:,int(t*self.nb_steps/self.maturity)]
            anomalies = lambda t: ad_labels[:,int(t*self.nb_steps/self.maturity)]
            drift = lambda x, t: - self.periodic_coeff(t) * self.speed @ (x - seasons(t))

            return drift, None, None, anomalies, None
        
        elif self.anomaly_type == 'cutoff':

            season_patterns = copy.copy(self.season_patterns)
            ad_labels = copy.copy(self.ad_labels)

            cutoff_level_law = self.anomaly_params['cutoff_level_law']

            for j in range(self.dimensions):
                for p in pos_list[j]:
                    p0 = int(p[0] * self.nb_steps)
                    p1 = int(p[1] * self.nb_steps)
                    if cutoff_level_law == 'uniform':
                        c0, c1 = self.anomaly_params['cutoff_level_range']
                        cutoff_level = float(np.random.uniform(c0,c1,1))
                    elif cutoff_level_law == 'current_level':
                        cutoff_level = self.seasons(p[0])
                    season_patterns[j,p0:p1] = cutoff_level
                    ad_labels[j,p0:p1] = 1

            seasons = lambda t: season_patterns[:,int(t*self.nb_steps/self.maturity)]
            anomalies = lambda t: ad_labels[:,int(t*self.nb_steps/self.maturity)]
            drift = lambda x, t: - self.periodic_coeff(t) * self.speed @ (x - seasons(t))

            return drift, None, None, anomalies, None
        
        elif self.anomaly_type == 'spike':
            self.spike = True

            ad_labels = copy.copy(self.ad_labels)
            
            r1, r2 = self.anomaly_params['occurence_pos_range']
            prob = self.anomaly_params['occurence_prob']
            amp_law = self.anomaly_params['spike_amp_law']
            a1, a2 = self.anomaly_params['spike_amp_range']
            
            range_mask = np.zeros((self.dimensions, self.nb_steps+1), dtype=np.bool)
            range_mask[int(r1*self.nb_steps/self.maturity):int(r2*self.nb_steps/self.maturity)] = True
            if amp_law == 'uniform':
                values = np.random.uniform(a1, a2, (self.dimensions, self.nb_steps+1))
                neg = 2. * np.random.binomial(1,0.5,(self.dimensions, self.nb_steps+1)) - 1.
                values *= neg
            else:
                NotImplementedError
            pos = np.random.binomial(1,1-prob,(self.dimensions, self.nb_steps+1)).astype(np.bool)
            mask = np.logical_or(pos, range_mask)
            values[mask] = 0.
            ad_labels[~mask] = 1

            spikes = {'values': values,
                      'labels': ad_labels}

            return None, None, None, None, spikes


    def generate_paths(
        self, start_X=None, no_S0=True, plot_paths=None, plot_save_path=None):
        # Diffusion of the variance: dv = -k(v-season(t))*dt + vol*dW
        if no_S0:
            self.S0 = None

        self.get_components()

        spot_paths = np.empty((self.nb_paths, self.dimensions, self.nb_steps + 1))
        deter_paths = np.empty_like(spot_paths)
        final_paths = np.empty_like(spot_paths)
        ad_labels = np.empty_like(spot_paths)
        seasonal_function = np.empty_like(spot_paths)

        dt = self.maturity / self.nb_steps
        period = self.maturity / self.season_nb

        if start_X is not None:
            spot_paths[:, :, 0] = start_X
        for i in tqdm.tqdm(range(self.nb_paths)):
            drift, diffusion, noise, anomalies, spikes = self.get_anomaly_fcts()
            if drift is None:
                drift = self.drift
            if diffusion is None:
                diffusion = self.diffusion
            if noise is None:
                noise = self.noise
            if anomalies is None:
                anomalies = self.anomalies

            if start_X is None:
                spot_paths[i, :, 0] = self.S0
                deter_paths[i, :, 0] = self.S0
                final_paths[i, :, 0] = (spot_paths[i, :, 0] + noise(spot_paths[i, :, 0], (0) * dt)) # @ eps)
                seasonal_function[i, :, 0] = (self.seasons(0.))
            for k in range(1, self.nb_steps + 1):
                random_numbers_bm = np.random.normal(0, 1, self.dimensions)
                dW = random_numbers_bm * np.sqrt(dt)
                spot_paths[i, :, k] = (
                        spot_paths[i, :, k - 1]
                        + drift(spot_paths[i, :, k - 1], (k - 1) * dt) * dt
                        + diffusion(spot_paths[i, :, k - 1], (k) * dt) @ dW)
                final_paths[i, :, k] = (spot_paths[i, :, k] + noise(spot_paths[i, :, k], (k) * dt)) # @ eps)
                deter_paths[i, :, k] = (
                        deter_paths[i, :, k - 1]
                        + self.drift(deter_paths[i, :, k - 1], (k - 1) * dt) * dt)
                seasonal_function[i,:,k] = (self.seasons((k - 1) * dt))
                ad_labels[i, :, k] = (anomalies((k - 1) * dt))
            if self.spike:
                final_paths[i] += spikes['values']
                ad_labels[i] = spikes['labels']
            if plot_paths and i in plot_paths:
                # only plots dim=0 path
                plt.figure()
                ad_labels_plot = ad_labels.copy()
                ad_labels_plot[i, :, 1:][
                    np.logical_and(ad_labels[i, :, 1:] == 0,
                                   ad_labels[i, :, :-1] == 1)] = 1
                ad_labels_plot[i, :, :-1][
                    np.logical_and(ad_labels[i, :, 1:] == 1,
                                   ad_labels[i, :, :-1] == 0)] = 1
                plot_t = np.linspace(0, self.maturity, self.nb_steps+1)
                mask_data_path_w_anomaly = np.ma.masked_where(
                    (ad_labels_plot[i] == 0), final_paths[i])
                plt.plot(plot_t, final_paths[i, 0, :], label="True path, no anomaly")
                plt.plot(plot_t, mask_data_path_w_anomaly[0,:], label="True path, anomaly")
                plt.plot(plot_t, deter_paths[i, 0, :], label="Deterministic path (without anomaly)", alpha=0.3)
                plt.plot(plot_t, seasonal_function[i, 0, :], label="Drift function", alpha=0.3, color="olive")
                plt.xlabel("$t$")
                if self.get_anomaly_fcts()  == (None, None, None, None, None):
                    plt.legend(
                        bbox_to_anchor=(1.05, 0.5),
                        loc='center left', borderaxespad=0.)
                fname = "{}path-{}.pdf".format(plot_save_path, i)
                plt.savefig(fname, bbox_inches='tight', pad_inches=0.01)

        # stock_path, final_paths, deter_paths, seasonal_function, ad_labels : [nb_paths, dimension, time_steps]
        # return season_pattern, ad_labels
        return final_paths, ad_labels, deter_paths, seasonal_function, dt, period



# ==============================================================================
# this is needed for computing the loss with the true conditional expectation
def compute_loss(X_obs, Y_obs, Y_obs_bj, n_obs_ot, batch_size, eps=1e-10,
                 weight=0.5, M_obs=None):
    """
    compute the loss of the true conditional expectation, as in
    model.compute_loss
    """
    if M_obs is None:
        inner = (2 * weight * np.sqrt(np.sum((X_obs - Y_obs) ** 2, axis=1) + eps) +
                 2 * (1 - weight) * np.sqrt(np.sum((Y_obs_bj - Y_obs) ** 2, axis=1)
                                            + eps)) ** 2
    else:
        inner = (2 * weight * np.sqrt(
            np.sum(M_obs * (X_obs - Y_obs)**2, axis=1) + eps) +
                 2 * (1 - weight) * np.sqrt(
                    np.sum(M_obs * (Y_obs_bj - Y_obs)**2, axis=1) + eps))**2
    outer = np.sum(inner / n_obs_ot)
    return outer / batch_size


# ==============================================================================
# dict for the supported stock models to get them from their name
DATASETS = {
    "AD_OrnsteinUhlenbeckWithSeason": AD_OrnsteinUhlenbeckWithSeason,
    "Microbiome_OrnsteinUhlenbeck": Microbiome_OrnsteinUhlenbeck,
}
# ==============================================================================


hyperparam_test_stock_models = {
    'drift': 0.2, 'volatility': 0.3, 'mean': 0.5, "poisson_lambda": 3.,
    'speed': 0.5, 'correlation': 0.5, 'nb_paths': 10, 'nb_steps': 100,
    'S0': 1, 'maturity': 1., 'dimension': 1}


def draw_stock_model(stock_model_name):
    hyperparam_test_stock_models['model_name'] = stock_model_name
    stockmodel = DATASETS[stock_model_name](**hyperparam_test_stock_models)
    stock_paths, dt = stockmodel.generate_paths()
    filename = '{}.pdf'.format(stock_model_name)

    # draw a path
    one_path = stock_paths[0, 0, :]
    dates = np.array([i for i in range(len(one_path))])
    cond_exp = np.zeros(len(one_path))
    cond_exp[0] = hyperparam_test_stock_models['S0']
    cond_exp_const = hyperparam_test_stock_models['S0']
    for i in range(1, len(one_path)):
        if i % 3 == 0:
            cond_exp[i] = one_path[i]
        else:
            cond_exp[i] = cond_exp[i - 1] * exp(
                hyperparam_test_stock_models['drift'] * dt)

    plt.plot(dates, one_path, label='stock path')
    plt.plot(dates, cond_exp, label='conditional expectation')
    plt.legend()
    plt.savefig(filename)
    plt.close()


if __name__ == '__main__':
    pass


