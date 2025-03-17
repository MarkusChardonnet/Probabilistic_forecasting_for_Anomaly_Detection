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

'''
sys.path.append('../..')
from state_space_model import GaussianLinearInvariantSSM
sys.path.append('../../TSAGen')
import Assembler as assem
import generator.trend_generator as tg
import generator.noise_generator as ng
import generator.season_generator as sg
from shape.RMDF import RMDF
'''

# ==============================================================================
# CLASSES
'''
class AD_GLISSM(GaussianLinearInvariantSSM):

    def __init__(self, nb_paths, nb_steps, maturity, dimension, d_state, d_noise, model_file, **kwargs):
        super(AD_GLISSM, self).__init__(
            d_obs=dimension, d_state=d_state, d_noise=d_noise
        )
        self.nb_paths = nb_paths
        self.nb_steps = nb_steps
        self.maturity = maturity
        model_path = "configs/models/" + model_file
        super().load_model(model_path)

    def generate_paths(self, **options):

        dt = self.maturity / self.nb_steps
        paths = self.generate_data(nsamples=self.nb_paths, ts_length=self.nb_steps+1).detach().numpy()
        paths = paths.transpose(0,2,1)
        return paths, None, dt
    
    
class AD_TSAGen():

    def __init__(self, nb_paths, nb_steps, maturity, noise_type, dimension, parameters, anomaly_params, new_shape = True, shape_path = None, **kwargs):
        self.nb_paths = nb_paths
        self.nb_steps = nb_steps
        self.maturity = maturity
        self.parameters = parameters
        self.anomaly_params = anomaly_params
        self.noise_type = noise_type
        self.shape_path = shape_path
        self.new_shape = new_shape

    def generate_paths(self, **options):

        dt = self.maturity / self.nb_steps

        ### TS Components Generators
        # Noise Component
        anomaly_type = self.anomaly_params['type']
        if anomaly_type in ['variance']:
            noise_anomaly_type = anomaly_type
        else:
            noise_anomaly_type = None
        noise_generator = ng.NoiseGeneratorFactory().get_generator(noise_type=self.noise_type, 
                                anomaly_type=noise_anomaly_type,
                                mu=self.parameters['noise_mu'],
                                sigma=self.parameters['noise_std'],
                                skew=self.parameters['noise_skew'],
                                kurt=self.parameters['noise_kurt'],
                                anomaly_params=self.anomaly_params)
        # Trend Component
        trend_generator = tg.TrendGenerator()
        # Season Component
        if anomaly_type in ['deformation', 'vanish']:
            season_anomaly_type = anomaly_type
        else:
            season_anomaly_type = None

        if self.shape_path is None:
                self.shape_path = "configs/models/TSAG_season_generator.pickle"

        season_generator = sg.SeasonGeneratorFactory().get_generator(cycle_num=self.parameters['seas_nb'],
                                                        amplitude=self.parameters['seas_amp'],
                                                        cycle_len=self.parameters['seas_length'],
                                                        drift_a=self.parameters['drift_amp'],
                                                        drift_f=self.parameters['drift_freq'],
                                                        forking_depth=self.parameters['forking_depth'],
                                                        depth=self.parameters['depth'],
                                                        anomaly_type=season_anomaly_type,
                                                        anomaly_params=self.anomaly_params)
        if self.new_shape:
            season_generator.save_anchor(self.shape_path)
        else:
            season_generator.load_anchor(self.shape_path)
        
        ### Generate Components for each TS
        season_list = []
        noise_list = []
        trend_list = []
        for i in range(self.nb_paths):
            season = season_generator.gen_season()
            noise = noise_generator.gen(size=len(season[0]))
            trend = trend_generator.gen(ba=self.parameters['trend_level'],
                                        zeta=self.parameters['trend_slope'],
                                        size=len(season[0]))
            season_list.append(season)
            noise_list.append(noise)
            trend_list.append(trend)
            if i % 10 == 0:
                print("Generated {} paths".format(i+1))

        ### Assemble Components
        assembler = assem.AbstractAssembler(season_list,noise_list,trend_list)
        assembler.assemble()
        results = assembler.get_results()

        ### Process to fit length
        paths = np.empty(shape=(self.nb_paths,1,self.nb_steps+1))
        ad_labels = np.empty(shape=(self.nb_paths,1,self.nb_steps+1))
        for i,res in enumerate(results):
            ts = np.squeeze(np.array(res)[0])
            ls = np.squeeze(np.array(res)[1])
            new_ts = np.squeeze(TimeSeriesResampler(sz=self.nb_steps+1).fit_transform(ts))
            new_ls = np.squeeze(TimeSeriesResampler(sz=self.nb_steps+1).fit_transform(ls))
            new_ls[new_ls > 0] = 1
            paths[i,0,:] = new_ts
            ad_labels[i,0,:] = new_ls

        out_path = '../../plots/AD_TSAGen/{}'.format(self.anomaly_params['type'])
        assembler.save(path=out_path)

        return paths, ad_labels, dt

    # for Orstein Uhlenbeck
    def inject_deformation_anomaly(self, season_patterns, ad_labels):

        pos_list = []
        self.occ_law = self.anomaly_params['occurence_law']
        if self.occ_law == 'single':
            prob = self.anomaly_params['occurence_prob']
            real = np.random.binomial(1, prob, 1)
            if real == 1:
                self.occ_range = self.anomaly_params['occurence_range']
                self.occ_pos_law = self.anomaly_params['occurence_pos_law']
                if self.occ_pos_law == 'uniform':
                    pos = int(np.random.uniform(self.occ_range[0],self.occ_range[1],1))
                    pos_list.append(pos)
        if 'forking_depth_law' in self.anomaly_params:
            self.fork_depth_law = self.anomaly_params['forking_depth_law']
        pos_dim_list = []
        for p in pos_list:
            dim_list = []
            dim_occurence_law = self.anomaly_params['dim_occurence_law']
            if dim_occurence_law == 'indep':
                prob = self.anomaly_params['dim_occurence_prob']
                for j in range(self.dimensions):
                    real = np.random.binomial(1, prob, 1)
                    if real == 1:
                        dim_list.append(j)
            pos_dim_list.append(dim_list)
        
        if self.season_type == 'RMDF':
            if self.fork_depth_law == 'delta':
                self.ad_forking_depth = self.anomaly_params['forking_depth']
            for i,pos in enumerate(pos_list):
                for j in pos_dim_list[i]:
                    anomaly_cycle = self.season_generator[j].gen(self.ad_forking_depth, int(self.season_length))
                    season_patterns[pos][j,:] = anomaly_cycle
                    ad_labels[pos][j,:] = 1
        if self.season_type == 'NN':
            times = torch.tensor(2. * np.pi * np.linspace(0., 1., self.season_length)).unsqueeze(1)
            for i,pos in enumerate(pos_list):
                for j in pos_dim_list[i]:

                    anomaly_season_generator = Season_NN(self.season_nn_layers, self.season_nn_input, 1, 
                                                    self.season_nn_bias)
                    anomaly_season_generator.gen_weigths()
                    anomaly_season_generator.eval()
                    anomaly_cycle = anomaly_season_generator(times.clone().detach()).squeeze().detach().numpy()
                    season_patterns[pos][j,:] = anomaly_cycle
                    ad_labels[pos][j,:] = 1
        
        return season_patterns, ad_labels
'''

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
        for i in range(self.nb_paths):
            if i % 100 == 0 and i != 0:
                print("Generated {} paths".format(i))

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


    def generate_paths(self, start_X=None, no_S0=True):
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
        for i in range(self.nb_paths):
            if i % 100 == 0 and i != 0:
                print("Generated {} paths".format(i))

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
        
        # stock_path, final_paths, deter_paths, seasonal_function, ad_labels : [nb_paths, dimension, time_steps]
        # return season_pattern, ad_labels
        return final_paths, ad_labels, deter_paths, seasonal_function, dt, period


class Heston(StockModel):
    """
    the Heston model, see: https://en.wikipedia.org/wiki/Heston_model
    a basic stochastic volatility stock price model
    """

    def __init__(self, drift, volatility, mean, speed, correlation, nb_paths,
                 nb_steps, S0, maturity, sine_coeff=None, **kwargs):
        super(Heston, self).__init__(
            drift=drift, volatility=volatility, nb_paths=nb_paths,
            nb_steps=nb_steps,
            S0=S0, maturity=maturity,
            sine_coeff=sine_coeff
        )
        self.mean = mean
        self.speed = speed
        self.correlation = correlation

    def next_cond_exp(self, y, delta_t, current_t):
        return y * np.exp(self.drift * self.periodic_coeff(current_t) * delta_t)

    def generate_paths(self, start_X=None):
        # Diffusion of the spot: dS = mu*S*dt + sqrt(v)*S*dW
        spot_drift = lambda x, t: self.drift * self.periodic_coeff(t) * x
        spot_diffusion = lambda x, v, t: np.sqrt(v) * x

        # Diffusion of the variance: dv = -k(v-vinf)*dt + sqrt(v)*v*dW
        var_drift = lambda v, t: - self.speed * (v - self.mean)
        var_diffusion = lambda v, t: self.volatility * np.sqrt(v)

        spot_paths = np.empty(
            (self.nb_paths, self.dimensions, self.nb_steps + 1))
        var_paths = np.empty(
            (self.nb_paths, self.dimensions, self.nb_steps + 1))

        dt = self.maturity / self.nb_steps
        if start_X is not None:
            spot_paths[:, :, 0] = start_X
        for i in range(self.nb_paths):
            if start_X is None:
                spot_paths[i, :, 0] = self.S0
            var_paths[i, :, 0] = self.mean
            for k in range(1, self.nb_steps + 1):
                normal_numbers_1 = np.random.normal(0, 1, self.dimensions)
                normal_numbers_2 = np.random.normal(0, 1, self.dimensions)
                dW = normal_numbers_1 * np.sqrt(dt)
                dZ = (self.correlation * normal_numbers_1 + np.sqrt(
                    1 - self.correlation ** 2) * normal_numbers_2) * np.sqrt(dt)

                var_paths[i, :, k] = (
                        var_paths[i, :, k - 1]
                        + var_drift(var_paths[i, :, k - 1], (k) * dt) * dt
                        + var_diffusion(var_paths[i, :, k - 1], (k) * dt) * dZ)

                spot_paths[i, :, k] = (
                        spot_paths[i, :, k - 1]
                        + spot_drift(spot_paths[i, :, k - 1], (k - 1) * dt) * dt
                        + spot_diffusion(spot_paths[i, :, k - 1],
                                                var_paths[i, :, k],
                                                (k) * dt) * dW)
        # stock_path dimension: [nb_paths, dimension, time_steps]
        return spot_paths, dt


class HestonWOFeller(StockModel):
    """
    the Heston model, see: https://en.wikipedia.org/wiki/Heston_model
    a basic stochastic volatility stock price model, that can be used
    even if Feller condition is not satisfied
    Feller condition: 2*speed*mean > volatility**2
    """

    def __init__(self, drift, volatility, mean, speed, correlation, nb_paths,
                 nb_steps, S0, maturity, scheme='euler', return_vol=False,
                 v0=None, sine_coeff=None, **kwargs):
        super(HestonWOFeller, self).__init__(
            drift=drift, volatility=volatility, nb_paths=nb_paths,
            nb_steps=nb_steps,
            S0=S0, maturity=maturity,
            sine_coeff=sine_coeff
        )
        self.mean = mean
        self.speed = speed
        self.correlation = correlation

        self.scheme = scheme
        self.retur_vol = return_vol
        if v0 is None:
            self.v0 = self.mean
        else:
            self.v0 = v0

    def next_cond_exp(self, y, delta_t, current_t):
        if self.retur_vol:
            s, v = np.split(y, indices_or_sections=2, axis=1)
            s = s * np.exp(self.drift * self.periodic_coeff(current_t) * delta_t)
            exp_delta = np.exp(-self.speed * delta_t)
            v = v * exp_delta + self.mean * (1 - exp_delta)
            y = np.concatenate([s, v], axis=1)
            return y
        else:
            return y * np.exp(self.drift * self.periodic_coeff(current_t) * delta_t)

    def generate_paths(self, start_X=None):
        if self.scheme == 'euler':
            # Diffusion of the spot: dS = mu*S*dt + sqrt(v)*S*dW
            log_spot_drift = lambda v, t: \
                (self.drift * self.periodic_coeff(t) - 0.5 * np.maximum(v, 0))
            log_spot_diffusion = lambda v: np.sqrt(np.maximum(v, 0))

            # Diffusion of the variance: dv = -k(v-vinf)*dt + sqrt(v)*v*dW
            var_drift = lambda v: - self.speed * (np.maximum(v, 0) - self.mean)
            var_diffusion = lambda v: self.volatility * np.sqrt(np.maximum(v, 0))

            spot_paths = np.empty(
                (self.nb_paths, self.dimensions, self.nb_steps + 1))
            var_paths = np.empty(
                (self.nb_paths, self.dimensions, self.nb_steps + 1))

            dt = self.maturity / self.nb_steps
            if start_X is not None:
                spot_paths[:, :, 0] = start_X
            for i in range(self.nb_paths):
                if start_X is None:
                    spot_paths[i, :, 0] = self.S0
                var_paths[i, :, 0] = self.v0
                for k in range(1, self.nb_steps + 1):
                    normal_numbers_1 = np.random.normal(0, 1, self.dimensions)
                    normal_numbers_2 = np.random.normal(0, 1, self.dimensions)
                    dW = normal_numbers_1 * np.sqrt(dt)
                    dZ = (self.correlation * normal_numbers_1 + np.sqrt(
                        1 - self.correlation ** 2) * normal_numbers_2) * np.sqrt(dt)

                    spot_paths[i, :, k] = np.exp(
                        np.log(spot_paths[i, :, k - 1])
                        + log_spot_drift(
                            var_paths[i, :, k - 1], (k - 1) * dt) * dt
                        + log_spot_diffusion(var_paths[i, :, k - 1]) * dW
                    )
                    var_paths[i, :, k] = (
                            var_paths[i, :, k - 1]
                            + var_drift(var_paths[i, :, k - 1]) * dt
                            + var_diffusion(var_paths[i, :, k - 1]) * dZ
                    )
            if self.retur_vol:
                spot_paths = np.concatenate([spot_paths, var_paths], axis=1)
            # stock_path dimension: [nb_paths, dimension, time_steps]
            return spot_paths, dt

        else:
            raise ValueError('unknown sampling scheme')


class BlackScholes(StockModel):
    """
    standard Black-Scholes model, see:
    https://en.wikipedia.org/wiki/Black–Scholes_model
    https://en.wikipedia.org/wiki/Geometric_Brownian_motion
    """

    def __init__(self, drift, volatility, nb_paths, nb_steps, S0,  # initialize relevant parameters
                 maturity, sine_coeff=None, **kwargs):
        super(BlackScholes, self).__init__(
            drift=drift, volatility=volatility, nb_paths=nb_paths,
            nb_steps=nb_steps, S0=S0, maturity=maturity,
            sine_coeff=sine_coeff
        )

    def next_cond_exp(self, y, delta_t, current_t):
        return y * np.exp(self.drift * self.periodic_coeff(current_t) * delta_t)

    def generate_paths(self, start_X=None):
        drift = lambda x, t: self.drift * self.periodic_coeff(t) * x
        diffusion = lambda x, t: self.volatility * x
        spot_paths = np.empty(
            (self.nb_paths, self.dimensions, self.nb_steps + 1))
        dt = self.maturity / self.nb_steps
        if start_X is not None:
            spot_paths[:, :, 0] = start_X
        for i in range(self.nb_paths):
            if start_X is None:
                spot_paths[i, :, 0] = self.S0
            for k in range(1, self.nb_steps + 1):
                random_numbers = np.random.normal(0, 1, self.dimensions)
                dW = random_numbers * np.sqrt(dt)
                spot_paths[i, :, k] = (
                        spot_paths[i, :, k - 1]
                        + drift(spot_paths[i, :, k - 1], (k - 1) * dt) * dt
                        + diffusion(spot_paths[i, :, k - 1], (k) * dt) * dW)
        # stock_path dimension: [nb_paths, dimension, time_steps]
        return spot_paths, dt


class PoissonPointProcess(StockModel):
    """
    standard Poisson Point Process model, see:
    https://en.wikipedia.org/wiki/Poisson_point_process
    """

    def __init__(self, poisson_lambda, nb_paths, nb_steps, S0,
                 maturity, sine_coeff=None, **kwargs):
        super().__init__(
            drift=None, volatility=None, nb_paths=nb_paths,
            nb_steps=nb_steps, S0=S0, maturity=maturity,
            sine_coeff=sine_coeff
        )
        self.poisson_lambda = poisson_lambda

    def next_cond_exp(self, y, delta_t, current_t):
        return y + self.poisson_lambda*delta_t

    def generate_paths(self, start_X=None):
        spot_paths = np.empty(
            (self.nb_paths, self.dimensions, self.nb_steps + 1))
        dt = self.maturity / self.nb_steps
        if start_X is not None:
            spot_paths[:, :, 0] = start_X
        else:
            spot_paths[:, :, 0] = self.S0
        # generate arrival times
        exp_rvs = np.random.exponential(
            scale=1/self.poisson_lambda,
            size=(self.nb_paths, self.dimensions,
                  int(self.poisson_lambda*self.maturity)+1))
        while np.any(np.sum(exp_rvs, axis=2) <= self.maturity):
            exp_rvs_app = np.random.exponential(
                scale=1/self.poisson_lambda,
                size=(self.nb_paths, self.dimensions,
                      int(self.poisson_lambda*self.maturity)+1))
            exp_rvs = np.concatenate([exp_rvs, exp_rvs_app], axis=2)
        exp_rvs = np.cumsum(exp_rvs, axis=2)

        for k in range(1, self.nb_steps + 1):
            spot_paths[:, :, k] = np.argmin(exp_rvs <= k*dt, axis=2) + \
                                  spot_paths[:, :, 0]

        # stock_path dimension: [nb_paths, dimension, time_steps]
        return spot_paths, dt


class BM(StockModel):
    """
    Brownian Motion
    """

    def __init__(self, nb_paths, nb_steps, maturity, dimension, **kwargs):
        super().__init__(
            drift=None, volatility=None, nb_paths=nb_paths,
            nb_steps=nb_steps, S0=0., maturity=maturity,
            sine_coeff=None,)
        assert dimension == 1

    def next_cond_exp(self, y, delta_t, current_t):
        next_y = y
        return next_y

    def generate_paths(self, start_X=None):
        spot_paths = np.zeros(
            (self.nb_paths, 1, self.nb_steps + 1))
        dt = self.maturity / self.nb_steps

        random_numbers = np.random.normal(
            0, 1, (self.nb_paths, 1, self.nb_steps)) * np.sqrt(dt)
        W = np.cumsum(random_numbers, axis=2)

        spot_paths[:, 0, 1:] = W[:, 0, :]

        # stock_path dimension: [nb_paths, dimension, time_steps]
        return spot_paths, dt


class BMandVar(StockModel):
    """
    Brownian Motion and its square
    """

    def __init__(self, nb_paths, nb_steps, maturity, dimension, **kwargs):
        super().__init__(
            drift=None, volatility=None, nb_paths=nb_paths,
            nb_steps=nb_steps, S0=np.array([0,0]), maturity=maturity,
            sine_coeff=None,)
        assert dimension == 2

    def next_cond_exp(self, y, delta_t, current_t):
        next_y = y
        next_y[:, 1] += delta_t
        return next_y

    def generate_paths(self, start_X=None):
        spot_paths = np.zeros(
            (self.nb_paths, 2, self.nb_steps + 1))
        dt = self.maturity / self.nb_steps

        random_numbers = np.random.normal(
            0, 1, (self.nb_paths, 1, self.nb_steps)) * np.sqrt(dt)
        W = np.cumsum(random_numbers, axis=2)

        spot_paths[:, 0, 1:] = W[:, 0, :]
        spot_paths[:, 1, 1:] = W[:, 0, :]**2

        # stock_path dimension: [nb_paths, dimension, time_steps]
        return spot_paths, dt


class BM2DCorr(StockModel):
    """
    2-dim Brownian Motion with correlation and correct cond. expectation
    for incomplete observations
    https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    """

    def __init__(self, nb_paths, nb_steps, maturity, alpha_sq, dimension,
                 **kwargs):
        super(BM2DCorr, self).__init__(
            drift=None, volatility=None, nb_paths=nb_paths,
            nb_steps=nb_steps, S0=np.array([0,0]), maturity=maturity,
            sine_coeff=None
        )
        self.alpha_sq = alpha_sq
        self.beta_sq = 1 - alpha_sq
        assert 0 < alpha_sq < 1, "alpha_sq needs to be in (0,1)"
        assert dimension == 2, "dimension has to be set to 2 for this dataset"
        self.path_t = None
        self.loss = None

    def next_cond_exp(self, y, delta_t, current_t):
        return y

    def get_mu(self, jj, which_coord_obs):
        sig = np.diag(self.observed_t_all_inc[jj]*3)
        M0 = np.tri(N=len(self.observed_0[jj]), k=0)[
            np.array(self.observed_0[jj])==1]
        M1 = np.tri(N=len(self.observed_1[jj]), k=0)[
            np.array(self.observed_1[jj])==1]
        r1, c1 = M0.shape
        r2, c2 = M1.shape
        M = np.zeros((r1+r2, c1*3))
        M[:r1, :c1] = np.sqrt(self.alpha_sq)*M0
        M[:r1, c1:c1*2] = np.sqrt(self.beta_sq)*M0
        M[r1:, :c1] = np.sqrt(self.alpha_sq)*M1
        M[r1:, c1*2:c1*3] = np.sqrt(self.beta_sq)*M1
        sig_bar_22_inv = np.linalg.inv(
            np.dot(np.dot(M, sig), np.transpose(M)))
        m = np.zeros((1,c1*3))
        m[0, :c1] = np.sqrt(self.alpha_sq)
        if which_coord_obs == 0:
            m[0, c1*2:c1*3] = np.sqrt(self.beta_sq)
        else:
            m[0, c1:c1*2] = np.sqrt(self.beta_sq)
        sig_bar_12 = np.dot(np.dot(m, sig), np.transpose(M))
        obs_arr = np.array(
            self.observed_X0[jj]+self.observed_X1[jj])
        mu = np.dot(np.dot(sig_bar_12, sig_bar_22_inv), obs_arr)
        return mu

    def compute_cond_exp(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                         n_obs_ot, return_path=True, get_loss=False,
                         weight=0.5, store_and_use_stored=True,
                         start_time=None, M=None,
                         **kwargs):
        """
        Compute conditional expectation
        :param times: np.array, of observation times
        :param time_ptr: list, start indices of X and obs_idx for a given
                observation time, first element is 0, this pointer tells how
                many (and which) of the observations of X along the batch-dim
                belong to the current time, and obs_idx then tells to which of
                the batch elements they belong. In particular, not each batch-
                element has to jump at the same time, and only those elements
                which jump at the current time should be updated with a jump
        :param X: np.array, data tensor
        :param obs_idx: list, index of the batch elements where jumps occur at
                current time
        :param delta_t: float, time step for Euler
        :param T: float, the final time
        :param start_X: np.array, the starting point of X
        :param n_obs_ot: np.array, the number of observations over the
                entire time interval for each element of the batch
        :param return_path: bool, whether to return the path of h
        :param get_loss: bool, whether to compute the loss, otherwise 0 returned
        :param until_T: bool, whether to continue until T (for eval) or only
                until last observation (for training)
        :param M: None or np.array, if not None: the mask for the data, same
                size as X, with 0 or 1 entries
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

        bs = start_X.shape[0]
        self.observed_0 = [[] for x in range(bs)]
        self.observed_1 = [[] for x in range(bs)]
        self.observed_X0 = [[] for x in range(bs)]
        self.observed_X1 = [[] for x in range(bs)]
        self.observed_t_all = [[] for x in range(bs)]
        self.observed_t_all_inc = [[] for x in range(bs)]

        y = start_X
        batch_size = bs
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
            # Calculate conditional expectation stepwise
            while current_time < (obs_time - 1e-10 * delta_t):
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
            M_obs = M[start:end]

            # Update h. Also updating loss, tau and last_X
            Y_bj = y
            temp = copy.copy(y)
            for j, jj in enumerate(i_obs):
                if M_obs[j, 0] == 1:
                    self.observed_X0[jj].append(X_obs[j, 0])
                    self.observed_0[jj].append(1)
                else:
                    self.observed_0[jj].append(0)
                if M_obs[j, 1] == 1:
                    self.observed_X1[jj].append(X_obs[j, 1])
                    self.observed_1[jj].append(1)
                else:
                    self.observed_1[jj].append(0)
                assert M_obs[j, :].sum() > 0
                self.observed_t_all[jj].append(obs_time)
                last = [0.] + self.observed_t_all[jj]
                self.observed_t_all_inc[jj].append(obs_time - last[-2])

                if M_obs[j, 0] == 1 and M_obs[j, 1] == 1:
                    temp[jj, 0] = X_obs[j, 0]
                    temp[jj, 1] = X_obs[j, 1]
                else:

                    if M_obs[j, 0] == 1:
                        temp[jj, 0] = X_obs[j, 0]
                        temp[jj, 1] = self.get_mu(jj=jj, which_coord_obs=0)
                    else:
                        temp[jj, 1] = X_obs[j, 1]
                        temp[jj, 0] = self.get_mu(jj=jj, which_coord_obs=1)
            y = temp
            Y = y

            if get_loss:
                loss = loss + compute_loss(
                    X_obs=X_obs, Y_obs=Y[i_obs], Y_obs_bj=Y_bj[i_obs],
                    n_obs_ot=n_obs_ot[i_obs], batch_size=batch_size,
                    weight=weight, M_obs=M_obs)
            if return_path:
                path_t.append(obs_time)
                path_y.append(y)

        # after every observation has been processed, propagating until T
        while current_time < T - 1e-10 * delta_t:
            if current_time < T - delta_t:
                delta_t_ = delta_t
            else:
                delta_t_ = T - current_time
            y = self.next_cond_exp(y, delta_t_)
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

    def generate_paths(self):
        spot_paths = np.zeros(
            (self.nb_paths, 2, self.nb_steps + 1))
        dt = self.maturity / self.nb_steps

        random_numbers = np.random.normal(
            0, 1, (self.nb_paths, 3, self.nb_steps)) * np.sqrt(dt)
        W = np.cumsum(random_numbers, axis=2)


        spot_paths[:, 0, 1:] = W[:, 0, :] * np.sqrt(self.alpha_sq) + \
                               W[:, 1, :] * np.sqrt(self.beta_sq)
        spot_paths[:, 1, 1:] = W[:, 0, :] * np.sqrt(self.alpha_sq) + \
                               W[:, 2, :] * np.sqrt(self.beta_sq)

        # stock_path dimension: [nb_paths, dimension, time_steps]
        return spot_paths, dt


class BMFiltering(BM2DCorr):
    """
    A Brownian Motion filtering example. The Signal process is a BM $X$ and the
    observation process is given by
        $ Y = \\alpha X + W, $
    where $W$ is also a BM independent of $X$ and $\\alpha \\in \\R$.
    $ Z = (Y, X) $, i.e. the first coordinate is $Y$ and is always observed.
    expectation for incomplete observations
    https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    """

    def __init__(self, nb_paths, nb_steps, maturity, alpha, dimension,
                 **kwargs):
        super(BM2DCorr, self).__init__(
            drift=None, volatility=None, nb_paths=nb_paths,
            nb_steps=nb_steps, S0=np.array([0,0]), maturity=maturity,
            sine_coeff=None
        )
        self.alpha = alpha
        assert dimension == 2, "dimension has to be set to 2 for this dataset"
        self.path_t = None
        self.loss = None

    def get_mu(self, jj, which_coord_obs):
        sig = np.diag(self.observed_t_all_inc[jj]*2)
        M0 = np.tri(N=len(self.observed_0[jj]), k=0)[
            np.array(self.observed_0[jj])==1]
        M1 = np.tri(N=len(self.observed_1[jj]), k=0)[
            np.array(self.observed_1[jj])==1]
        r1, c1 = M0.shape
        r2, c2 = M1.shape
        M = np.zeros((r1+r2, c1*2))
        M[:r1, :c1] = np.sqrt(self.alpha)*M0
        M[:r1, c1:c1*2] = M0
        M[r1:, :c1] = M1
        sig_bar_22_inv = np.linalg.inv(
            np.dot(np.dot(M, sig), np.transpose(M)))
        m = np.zeros((1,c1*2))
        m[0, :c1] = 1
        sig_bar_12 = np.dot(np.dot(m, sig), np.transpose(M))
        obs_arr = np.array(
            self.observed_X0[jj]+self.observed_X1[jj])
        mu = np.dot(np.dot(sig_bar_12, sig_bar_22_inv), obs_arr)
        return mu

    def compute_cond_exp(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                         n_obs_ot, return_path=True, get_loss=False,
                         weight=0.5, store_and_use_stored=True,
                         start_time=None, M=None,
                         **kwargs):
        """
        Compute conditional expectation
        :param times: np.array, of observation times
        :param time_ptr: list, start indices of X and obs_idx for a given
                observation time, first element is 0, this pointer tells how
                many (and which) of the observations of X along the batch-dim
                belong to the current time, and obs_idx then tells to which of
                the batch elements they belong. In particular, not each batch-
                element has to jump at the same time, and only those elements
                which jump at the current time should be updated with a jump
        :param X: np.array, data tensor
        :param obs_idx: list, index of the batch elements where jumps occur at
                current time
        :param delta_t: float, time step for Euler
        :param T: float, the final time
        :param start_X: np.array, the starting point of X
        :param n_obs_ot: np.array, the number of observations over the
                entire time interval for each element of the batch
        :param return_path: bool, whether to return the path of h
        :param get_loss: bool, whether to compute the loss, otherwise 0 returned
        :param until_T: bool, whether to continue until T (for eval) or only
                until last observation (for training)
        :param M: None or np.array, if not None: the mask for the data, same
                size as X, with 0 or 1 entries
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

        bs = start_X.shape[0]
        self.observed_0 = [[] for x in range(bs)]
        self.observed_1 = [[] for x in range(bs)]
        self.observed_X0 = [[] for x in range(bs)]
        self.observed_X1 = [[] for x in range(bs)]
        self.observed_t_all = [[] for x in range(bs)]
        self.observed_t_all_inc = [[] for x in range(bs)]

        y = start_X
        batch_size = bs
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
            # Calculate conditional expectation stepwise
            while current_time < (obs_time - 1e-10 * delta_t):
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
            M_obs = M[start:end]

            # Update h. Also updating loss, tau and last_X
            Y_bj = y
            temp = copy.copy(y)
            for j, jj in enumerate(i_obs):
                if M_obs[j, 0] == 1:
                    self.observed_X0[jj].append(X_obs[j, 0])
                    self.observed_0[jj].append(1)
                else:
                    self.observed_0[jj].append(0)
                if M_obs[j, 1] == 1:
                    self.observed_X1[jj].append(X_obs[j, 1])
                    self.observed_1[jj].append(1)
                else:
                    self.observed_1[jj].append(0)
                assert M_obs[j, :].sum() > 0
                self.observed_t_all[jj].append(obs_time)
                last = [0.] + self.observed_t_all[jj]
                self.observed_t_all_inc[jj].append(obs_time - last[-2])

                if M_obs[j, 0] == 1 and M_obs[j, 1] == 1:
                    temp[jj, 0] = X_obs[j, 0]
                    temp[jj, 1] = X_obs[j, 1]
                else:
                    assert M_obs[j, 0] == 1
                    temp[jj, 0] = X_obs[j, 0]
                    temp[jj, 1] = self.get_mu(jj=jj, which_coord_obs=0)
            y = temp
            Y = y

            if get_loss:
                loss = loss + compute_loss(
                    X_obs=X_obs, Y_obs=Y[i_obs], Y_obs_bj=Y_bj[i_obs],
                    n_obs_ot=n_obs_ot[i_obs], batch_size=batch_size,
                    weight=weight, M_obs=M_obs)
            if return_path:
                path_t.append(obs_time)
                path_y.append(y)

        # after every observation has been processed, propagating until T
        while current_time < T - 1e-10 * delta_t:
            if current_time < T - delta_t:
                delta_t_ = delta_t
            else:
                delta_t_ = T - current_time
            y = self.next_cond_exp(y, delta_t_)
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

    def generate_paths(self):
        spot_paths = np.zeros(
            (self.nb_paths, 2, self.nb_steps + 1))
        dt = self.maturity / self.nb_steps

        random_numbers = np.random.normal(
            0, 1, (self.nb_paths, 2, self.nb_steps)) * np.sqrt(dt)
        W = np.cumsum(random_numbers, axis=2)

        spot_paths[:, 0, 1:] = W[:, 0, :] * self.alpha + W[:, 1, :]
        spot_paths[:, 1, 1:] = W[:, 0, :]

        # stock_path dimension: [nb_paths, dimension, time_steps]
        return spot_paths, dt


class BMwithTimeLag(BM2DCorr):
    """
    A Brownian Motion and its time-lagged version. In particular, the first
    coordinate is a BM $X$ and the second coordinate is $Y_t = X_{t - \alpha}$
    for some $\alpha > 0$.

    For computing conditional expectations, the first coordinate is assumed to
    be always observed (otherwise not working correctly). However, this is
    therefore only needed on the test set (where cond. exp. are computed) but
    not on the training set (where they are not computed).
    """

    def __init__(self, nb_paths, nb_steps, maturity, alpha_in_dt_steps,
                 dimension, **kwargs):
        """
        Args:
            nb_paths:
            nb_steps:
            maturity:
            alpha_in_dt_steps: int, this defines via
                alpha = dt * alpha_in_dt_steps
            dimension:
            **kwargs:
        """
        super(BM2DCorr, self).__init__(
            drift=None, volatility=None, nb_paths=nb_paths,
            nb_steps=nb_steps, S0=np.array([0,0]), maturity=maturity,
            sine_coeff=None
        )
        self.alpha_in_dt_steps = alpha_in_dt_steps
        self.dt = self.maturity / self.nb_steps
        self.alpha = self.dt * self.alpha_in_dt_steps
        assert dimension == 2, "dimension has to be set to 2 for this dataset"
        self.path_t = None
        self.loss = None

    def next_cond_exp(self, y, delta_t, current_t):
        t = current_t + delta_t
        s = t - self.alpha
        next_y = copy.deepcopy(y)
        for j in range(y.shape[0]):
            next_y[j, 1] = self.get_next_y1(j, y[j, 1], t, s)
        return next_y

    def get_next_y1(self, j, y1, t, s):
        if len(self.observed_t_0[j]) == 0:
            return 0.
        last_obs_time = np.max(self.observed_t_0[j])
        if t <= self.alpha:
            next_y1 = 0.
        elif t >= last_obs_time + self.alpha:
            next_y1 = y1
        else:
            # in this case: s < last_obs_time
            i = np.argmax((np.array(self.observed_t_0[j]) - s) > 0)
            t_obs_after = self.observed_t_0[j][i]
            X0_obs_after = self.observed_X0[j][i]
            if i==0:
                t_obs_before = 0.
                X0_obs_before = 0.
            else:
                t_obs_before = self.observed_t_0[j][i-1]
                X0_obs_before = self.observed_X0[j][i-1]
            if len(self.observed_t_1[j])>0 and \
                    self.observed_t_1[j][-1] - self.alpha > t_obs_before:
                t_obs_before = self.observed_t_1[j][-1] - self.alpha
                X0_obs_before = self.observed_X1[j][-1]
            w = (s - t_obs_before)/(t_obs_after - t_obs_before)
            next_y1 = (1-w)*X0_obs_before + w*X0_obs_after
        return next_y1

    def compute_cond_exp(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                         n_obs_ot, return_path=True, get_loss=False,
                         weight=0.5, store_and_use_stored=True,
                         start_time=None, M=None,
                         **kwargs):
        """
        Compute conditional expectation
        :param times: np.array, of observation times
        :param time_ptr: list, start indices of X and obs_idx for a given
                observation time, first element is 0, this pointer tells how
                many (and which) of the observations of X along the batch-dim
                belong to the current time, and obs_idx then tells to which of
                the batch elements they belong. In particular, not each batch-
                element has to jump at the same time, and only those elements
                which jump at the current time should be updated with a jump
        :param X: np.array, data tensor
        :param obs_idx: list, index of the batch elements where jumps occur at
                current time
        :param delta_t: float, time step for Euler
        :param T: float, the final time
        :param start_X: np.array, the starting point of X
        :param n_obs_ot: np.array, the number of observations over the
                entire time interval for each element of the batch
        :param return_path: bool, whether to return the path of h
        :param get_loss: bool, whether to compute the loss, otherwise 0 returned
        :param until_T: bool, whether to continue until T (for eval) or only
                until last observation (for training)
        :param M: None or np.array, if not None: the mask for the data, same
                size as X, with 0 or 1 entries
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

        bs = start_X.shape[0]
        self.observed_0 = [[] for x in range(bs)]
        self.observed_1 = [[] for x in range(bs)]
        self.observed_X0 = [[] for x in range(bs)]
        self.observed_X1 = [[] for x in range(bs)]
        self.observed_t_0 = [[] for x in range(bs)]
        self.observed_t_1 = [[] for x in range(bs)]

        y = start_X
        batch_size = bs
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
            # Calculate conditional expectation stepwise
            while current_time < (obs_time - 1e-10 * delta_t):
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
            M_obs = M[start:end]

            # Update h. Also updating loss, tau and last_X
            Y_bj = y
            temp = copy.copy(y)
            for j, jj in enumerate(i_obs):
                if M_obs[j, 0] == 1:
                    self.observed_X0[jj].append(X_obs[j, 0])
                    self.observed_0[jj].append(1)
                    self.observed_t_0[jj].append(obs_time)
                else:
                    self.observed_0[jj].append(0)
                if M_obs[j, 1] == 1:
                    self.observed_X1[jj].append(X_obs[j, 1])
                    self.observed_1[jj].append(1)
                    self.observed_t_1[jj].append(obs_time)
                else:
                    self.observed_1[jj].append(0)
                assert M_obs[j, :].sum() > 0

                if M_obs[j, 0] == 1 and M_obs[j, 1] == 1:
                    temp[jj, 0] = X_obs[j, 0]
                    temp[jj, 1] = X_obs[j, 1]
                elif M_obs[j, 0] == 1:
                    temp[jj, 0] = X_obs[j, 0]
                    temp[jj, 1] = self.get_next_y1(
                        jj, y[jj, 1], obs_time, obs_time-self.alpha)
                else:
                    temp[jj, 1] = X_obs[j, 1]

            y = temp
            Y = y

            if get_loss:
                loss = loss + compute_loss(
                    X_obs=X_obs, Y_obs=Y[i_obs], Y_obs_bj=Y_bj[i_obs],
                    n_obs_ot=n_obs_ot[i_obs], batch_size=batch_size,
                    weight=weight, M_obs=M_obs)
            if return_path:
                path_t.append(obs_time)
                path_y.append(y)

        # after every observation has been processed, propagating until T
        while current_time < T - 1e-10 * delta_t:
            if current_time < T - delta_t:
                delta_t_ = delta_t
            else:
                delta_t_ = T - current_time
            y = self.next_cond_exp(y, delta_t_)
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

    def generate_paths(self):
        spot_paths = np.zeros(
            (self.nb_paths, 2, self.nb_steps + 1))
        dt = self.maturity / self.nb_steps

        random_numbers = np.random.normal(
            0, 1, (self.nb_paths, 1, self.nb_steps)) * np.sqrt(dt)
        W = np.cumsum(random_numbers, axis=2)

        spot_paths[:, 0, 1:] = W[:, 0, :]
        spot_paths[:, 1, 1+self.alpha_in_dt_steps:] = \
            W[:, 0, :-self.alpha_in_dt_steps]

        # stock_path dimension: [nb_paths, dimension, time_steps]
        return spot_paths, dt


class FracBM(StockModel):
    """
    Implementing FBM via FBM package
    """

    def __init__(self, nb_paths, nb_steps, S0, maturity, hurst,
                 method="daviesharte", **kwargs):
        """Instantiate the FBM"""
        self.nb_paths = nb_paths
        self.nb_steps = nb_steps
        self.S0 = S0
        self.maturity = maturity
        self.hurst = hurst
        self.method = method
        self.dimensions = np.size(S0)
        self.loss = None
        self.path_t = None
        self.path_y = None

    def r_H(self, t, s):
        return 0.5 * (t**(2*self.hurst) + s**(2*self.hurst) -
                      np.abs(t-s)**(2*self.hurst))

    def get_cov_mat(self, times):
        m = np.array(times).reshape((-1,1)).repeat(len(times), axis=1)
        return self.r_H(m, np.transpose(m))

    def next_cond_exp(self, y, delta_t, current_t):
        t = current_t+delta_t
        next_y = np.zeros_like(y)
        for ii in range(y.shape[0]):
            if self.obs_cov_mat_inv[ii] is not None:
                r = self.r_H(np.array(self.observed_t[ii]), t)
                next_y[ii] = np.dot(r, np.matmul(
                    self.obs_cov_mat_inv[ii], np.array(self.observed_X[ii])))
        return next_y

    def compute_cond_exp(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                         n_obs_ot, return_path=True, get_loss=False,
                         weight=0.5, store_and_use_stored=True,
                         start_time=None,
                         **kwargs):
        """
        Compute conditional expectation
        :param times: np.array, of observation times
        :param time_ptr: list, start indices of X and obs_idx for a given
                observation time, first element is 0, this pointer tells how
                many (and which) of the observations of X along the batch-dim
                belong to the current time, and obs_idx then tells to which of
                the batch elements they belong. In particular, not each batch-
                element has to jump at the same time, and only those elements
                which jump at the current time should be updated with a jump
        :param X: np.array, data tensor
        :param obs_idx: list, index of the batch elements where jumps occur at
                current time
        :param delta_t: float, time step for Euler
        :param T: float, the final time
        :param start_X: np.array, the starting point of X
        :param n_obs_ot: np.array, the number of observations over the
                entire time interval for each element of the batch
        :param return_path: bool, whether to return the path of h
        :param get_loss: bool, whether to compute the loss, otherwise 0 returned
        :param until_T: bool, whether to continue until T (for eval) or only
                until last observation (for training)
        :param M: None or torch.tensor, if not None: the mask for the data, same
                size as X, with 0 or 1 entries
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

        assert self.dimensions == 1, "cond. exp. computation of FBM only for 1d"
        assert self.S0 == 0, "cond. exp. computation of FBM only for S0=0"

        bs = start_X.shape[0]
        self.observed_t = [[] for x in range(bs)]
        self.observed_X = [[] for x in range(bs)]
        self.obs_cov_mat = [None for x in range(bs)]
        self.obs_cov_mat_inv = [None for x in range(bs)]

        y = start_X
        batch_size = bs
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
            # Calculate conditional expectation stepwise
            while current_time < (obs_time - 1e-10 * delta_t):
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

            # add to observed
            for j, ii in enumerate(i_obs):
                self.observed_t[ii].append(obs_time)
                self.observed_X[ii].append(X_obs[j, 0])
                self.obs_cov_mat[ii] = self.get_cov_mat(self.observed_t[ii])
                self.obs_cov_mat_inv[ii] = np.linalg.inv(self.obs_cov_mat[ii])

            # Update h. Also updating loss, tau and last_X
            Y_bj = y
            temp = copy.copy(y)
            temp[i_obs] = X_obs
            y = temp
            Y = y

            if get_loss:
                loss = loss + compute_loss(
                    X_obs=X_obs, Y_obs=Y[i_obs], Y_obs_bj=Y_bj[i_obs],
                    n_obs_ot=n_obs_ot[i_obs], batch_size=batch_size,
                    weight=weight)
            if return_path:
                path_t.append(obs_time)
                path_y.append(y)

        # after every observation has been processed, propagating until T
        while current_time < T - 1e-10 * delta_t:
            if current_time < T - delta_t:
                delta_t_ = delta_t
            else:
                delta_t_ = T - current_time
            y = self.next_cond_exp(y, delta_t_)
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

    def generate_paths(self, start_X=None):
        spot_paths = np.empty(
            (self.nb_paths, self.dimensions, self.nb_steps + 1))
        dt = self.maturity / self.nb_steps
        if start_X is not None:
            spot_paths[:, :, 0] = start_X
        else:
            spot_paths[:, :, 0] = self.S0
        for i in range(self.nb_paths):
            for j in range(self.dimensions):
                fgn_sample = fgn(n=self.nb_steps, hurst=self.hurst,
                             length=self.maturity, method=self.method)
                spot_paths[i, j, 1:] = np.cumsum(fgn_sample)+spot_paths[i, j, 0]
        # stock_path dimension: [nb_paths, dimension, time_steps]
        return spot_paths, dt


class SP500(StockModel):
    """
    Data from SP500: https://www.kaggle.com/camnugent/sandp500
    """
    def __init__(self, **kwargs):
        self.nb_paths = 470

    def generate_paths(self, start_X=None):
        spot_paths = np.genfromtxt("SP500data.csv", delimiter=",")
        spot_paths = np.expand_dims(spot_paths, axis=1)
        # stock_path dimension: [nb_paths, dimension, time_steps]
        return spot_paths, 1

class OrnsteinUhlenbeck(StockModel):
    """
    Ornstein-Uhlenbeeck stock model, see:
    https://en.wikipedia.org/wiki/Ornstein–Uhlenbeck_process
    """

    def __init__(self, volatility, nb_paths, nb_steps, S0,
                 mean, speed, maturity, sine_coeff=None, **kwargs):
        super(OrnsteinUhlenbeck, self).__init__(
            volatility=volatility, nb_paths=nb_paths, drift=None,
            nb_steps=nb_steps, S0=S0, maturity=maturity,
            sine_coeff=sine_coeff
        )
        self.mean = mean
        self.speed = speed

    def next_cond_exp(self, y, delta_t, current_t):
        exp_delta = np.exp(-self.speed * self.periodic_coeff(current_t) * delta_t)
        return y * exp_delta + self.mean * (1 - exp_delta)

    def generate_paths(self, start_X=None):
        # Diffusion of the variance: dv = -k(v-vinf)*dt + vol*dW
        drift = lambda x, t: - self.speed * self.periodic_coeff(t) * (x - self.mean)
        diffusion = lambda x, t: self.volatility

        spot_paths = np.empty(
            (self.nb_paths, self.dimensions, self.nb_steps + 1))
        dt = self.maturity / self.nb_steps
        if start_X is not None:
            spot_paths[:, :, 0] = start_X
        for i in range(self.nb_paths):
            if start_X is None:
                spot_paths[i, :, 0] = self.S0
            for k in range(1, self.nb_steps + 1):
                random_numbers = np.random.normal(0, 1, self.dimensions)
                dW = random_numbers * np.sqrt(dt)
                spot_paths[i, :, k] = (
                        spot_paths[i, :, k - 1]
                        + drift(spot_paths[i, :, k - 1], (k - 1) * dt) * dt
                        + diffusion(spot_paths[i, :, k - 1], (k) * dt) * dW)
        # stock_path dimension: [nb_paths, dimension, time_steps]
        return spot_paths, dt


class Combined(StockModel):
    def __init__(self, stock_model_names, hyperparam_dicts, **kwargs):
        self.stock_model_names = stock_model_names
        self.hyperparam_dicts = hyperparam_dicts

    def compute_cond_exp(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                         n_obs_ot, return_path=True, get_loss=False,
                         weight=0.5, **kwargs):
        # get first stockmodel
        stockmodel = DATASETS[self.stock_model_names[0]](
            **self.hyperparam_dicts[0])
        T = self.hyperparam_dicts[0]['maturity']
        loss, path_t, path_y = stockmodel.compute_cond_exp(
            times, time_ptr, X, obs_idx, delta_t,
            T, start_X,
            n_obs_ot, return_path=True, get_loss=get_loss,
            weight=weight, store_and_use_stored=False)
        for i in range(1, len(self.stock_model_names)):
            start_X = path_y[-1, :, :]
            start_time = path_t[-1]
            T += self.hyperparam_dicts[i]['maturity']
            stockmodel = DATASETS[self.stock_model_names[i]](
                **self.hyperparam_dicts[i])
            _loss, _path_t, _path_y = stockmodel.compute_cond_exp(
                times, time_ptr, X, obs_idx, delta_t,
                T, start_X,
                n_obs_ot, return_path=True, get_loss=get_loss,
                weight=weight, start_time=start_time,
                store_and_use_stored=False)
            loss += _loss
            path_t = np.concatenate([path_t, _path_t])
            path_y = np.concatenate([path_y, _path_y], axis=0)

        if return_path:
            # path dimension: [time_steps, batch_size, output_size]
            return loss, np.array(path_t), np.array(path_y)
        else:
            return loss


class DoublePendulum:
    """
    see formulas at https://math24.net/double-pendulum.html
    Runge-Kutta-4 is used to solve the ODE numerically
    """
    def __init__(self, start_alpha, length, mass_ratio, step_size=0.01):
        self.start_alpha = start_alpha
        self.reset()
        self.l = length
        self.m1 = 1
        self.mu = mass_ratio
        self.step_size = step_size
        self.g = 9.81

    def reset(self):
        self.Z = np.array(
            [self.start_alpha,
             self.start_alpha,
             np.zeros_like(self.start_alpha),
             np.zeros_like(self.start_alpha)])

    def set(self, Z):
        self.Z = Z

    def diff_equ(self, Z):
        return np.array([self.f1(Z), self.f2(Z), self.f3(Z), self.f4(Z)])

    @staticmethod
    def a1(Z):
        return Z[0]

    @staticmethod
    def a2(Z):
        return Z[1]

    @staticmethod
    def p1(Z):
        return Z[2]

    @staticmethod
    def p2(Z):
        return Z[3]

    def d(self, Z):
        return self.m1*(self.l**2)*(1+self.mu*np.sin(self.a1(Z)-self.a2(Z))**2)

    def f1(self, Z):
        n = self.p1(Z)-self.p2(Z)*np.cos(self.a1(Z)-self.a2(Z))
        d = self.d(Z)
        return n/d

    def f2(self, Z):
        n = self.p2(Z)*(1+self.mu)-self.p1(Z)*self.mu*np.cos(self.a1(Z)-self.a2(Z))
        d = self.d(Z)
        return n/d

    def A1(self, Z):
        n = self.p1(Z)*self.p2(Z)*np.sin(self.a1(Z)-self.a2(Z))
        d = self.d(Z)
        return n/d

    def A2(self, Z):
        d = 2*self.d(Z)*(1+self.mu*np.sin(self.a1(Z)-self.a2(Z))**2)
        n1 = self.p1(Z)**2*self.mu-\
             2*self.p1(Z)*self.p2(Z)*self.mu*np.cos(self.a1(Z)-self.a2(Z))+\
             self.p2(Z)**2*(1+self.mu)
        n2 = np.sin(2*(self.a1(Z)-self.a2(Z)))
        return n1*n2/d

    def f3(self, Z):
        n1 = -self.m1*(1+self.mu)*self.g*self.l*np.sin(self.a1(Z))
        return n1-self.A1(Z)+self.A2(Z)

    def f4(self, Z):
        n1 = -self.m1*self.mu*self.g*self.l*np.sin(self.a2(Z))
        return n1+self.A1(Z)-self.A2(Z)

    def update(self):
        """
        uses RK4 method to update the state vektor Z
        """
        Y1 = self.step_size*self.diff_equ(self.Z)
        Y2 = self.step_size*self.diff_equ(self.Z+Y1/2)
        Y3 = self.step_size*self.diff_equ(self.Z+Y2/2)
        Y4 = self.step_size*self.diff_equ(self.Z+Y3)
        Znew = self.Z + (Y1+2*Y2+2*Y3+Y4)/6
        self.Z = Znew
        return Znew


def plot_pendulum(nb_steps, start_alpha=0., length=1., mass_ratio=1.,
                  step_size=0.01, interval=10):
    l = length
    fig, ax = plt.subplots()
    p0, = ax.plot([0], [0], marker="o")
    x1 = l*np.cos(start_alpha-np.pi/2)
    y1 = l*np.sin(start_alpha-np.pi/2)
    x2 = x1 + l*np.cos(start_alpha-np.pi/2)
    y2 = y1 + l*np.sin(start_alpha-np.pi/2)
    p1, = ax.plot([x1], [y1], marker="o")
    p2, = ax.plot([x2], [y2], marker="o")
    l1, = ax.plot([0, x1], [0, y1], marker="", lw="1")
    l2, = ax.plot([x1, x2], [y1, y2], marker="", lw="1")

    def init():
        ax.set_ylim(-3, 3)
        ax.set_xlim(-3, 3)

    def data_gen():
        s = 0
        DP = DoublePendulum(start_alpha=start_alpha, length=length,
                            mass_ratio=mass_ratio, step_size=step_size)
        while s < nb_steps:
            s += 1
            yield DP.update()

    def run(data):
        Z = data
        x1 = l*np.cos(Z[0]-np.pi/2)
        y1 = l*np.sin(Z[0]-np.pi/2)
        x2 = x1 + l*np.cos(Z[1]-np.pi/2)
        y2 = y1 + l*np.sin(Z[1]-np.pi/2)
        p1.set_data([x1], [y1])
        p2.set_data([x2], [y2])
        l1.set_data([0, x1], [0, y1])
        l2.set_data([x1, x2], [y1, y2])

    ani = animation.FuncAnimation(
        fig, run, data_gen, blit=False, interval=interval,
        repeat=False, init_func=init)

    plt.show()


class DoublePendulumDataset(StockModel):
    def __init__(
            self, start_alpha_mean, start_alpha_std, length, mass_ratio,
            nb_paths, maturity=1., sampling_step_size=0.01,
            sampling_nb_steps=1000, use_every_n=10, **kwargs):
        """
        Args:
            start_alpha_mean: mean for normal distribution of which starting
                angle is sampled
            start_alpha_std: std for normal distribution of which starting
                angle is sampled
            length: length of pendulum components
            mass_ratio: ratio m2/m1 for mass of first (m1) and second (m2)
                pendulum
            nb_paths: int, number of paths to sample
            nb_steps: int, number of steps
            step_size:
            use_every_n:
        """
        self.path_t = None
        self.path_y = None

        self.start_alpha_mean = start_alpha_mean
        self.start_alpha_std = start_alpha_std
        self.nb_paths = nb_paths
        self.maturity = maturity
        self.sampling_step_size = sampling_step_size
        self.sampling_nb_steps = sampling_nb_steps
        self.use_every_n = use_every_n
        self.nb_steps = self.sampling_nb_steps/use_every_n
        self.dt = maturity/self.nb_steps
        self.dimensions = 4

        self.length = length
        self.mass_ratio = mass_ratio

    def generate_paths(self, start_X=None):
        start_alphas = np.random.normal(
            self.start_alpha_mean, self.start_alpha_std, self.nb_paths)

        paths = np.empty(
            (self.nb_paths, self.dimensions, self.sampling_nb_steps+1))
        DP = DoublePendulum(
            start_alpha=start_alphas, length=self.length,
            mass_ratio=self.mass_ratio, step_size=self.sampling_step_size)
        paths[:, :, 0] = np.transpose(DP.Z)
        for j in range(self.sampling_nb_steps):
            paths[:, :, j+1] = np.transpose(DP.update())

        # stock_path dimension: [nb_paths, dimension, time_steps]
        return paths[:, :, ::self.use_every_n], self.dt

    def next_cond_exp(self, y, delta_t_, current_time):
        DP = DoublePendulum(
            start_alpha=0., length=self.length, mass_ratio=self.mass_ratio,
            step_size=delta_t_*self.sampling_step_size*
                      self.sampling_nb_steps/self.maturity)
        DP.set(Z=np.transpose(y))
        next_y = np.transpose(DP.update())
        return next_y








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
    # "AD_GLISSM": AD_GLISSM,
    # "AD_TSAGen": AD_TSAGen,
    "AD_OrnsteinUhlenbeckWithSeason": AD_OrnsteinUhlenbeckWithSeason,
    "Microbiome_OrnsteinUhlenbeck": Microbiome_OrnsteinUhlenbeck,
    "BlackScholes": BlackScholes,
    "Heston": Heston,
    "OrnsteinUhlenbeck": OrnsteinUhlenbeck,
    "HestonWOFeller": HestonWOFeller,
    "combined": Combined,
    "sine_BlackScholes": BlackScholes,
    "sine_Heston": Heston,
    "sine_OrnsteinUhlenbeck": OrnsteinUhlenbeck,
    "PoissonPointProcess": PoissonPointProcess,
    "FBM": FracBM,
    "BM2DCorr": BM2DCorr,
    "BMandVar": BMandVar,
    "BM": BM,
    "SP500": SP500,
    "DoublePendulum": DoublePendulumDataset,
    "BMFiltering": BMFiltering,
    "BMwithTimeLag": BMwithTimeLag,
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
    # draw_stock_model("BlackScholes")

    # ------------------------------
    # sm = FracBM(1, 100, 0, 1, 0.05)
    # paths = sm.generate_paths()
    # t = np.linspace(0, 1, 101)
    # rand_ind = np.random.random(len(t))
    # rand_ind = rand_ind > 0.9
    # rand_ind = rand_ind.tolist()
    # x = paths[0,0,rand_ind]
    # obs = t[rand_ind]
    # plt.plot(t, paths[0,0])
    # plt.plot(obs, x)
    # plt.show()
    # sm.observed_t = [obs[:7]]
    # sm.observed_X = [x[:7]]
    # sm.obs_cov_mat = [sm.get_cov_mat(sm.observed_t[0])]
    # sm.obs_cov_mat_inv = [np.linalg.inv(sm.obs_cov_mat[0])]
    # sm.next_cond_exp(x[6:7], 0, obs[6])
    # sm.next_cond_exp(x[6:7], 0.1, obs[6])
    # sm.next_cond_exp(x[6:7], 1e-16, obs[6])

    # ------------------------------------
    # plot_pendulum(1000, start_alpha=np.pi-0.0013, length=1., mass_ratio=1.,
    #               step_size=0.01, interval=1)
    # plot_pendulum(100, start_alpha=np.pi-0.2, length=1., mass_ratio=1.,
    #               step_size=0.1, interval=100)
    plot_pendulum(100, start_alpha=np.pi-0.05, length=1., mass_ratio=1.,
                  step_size=0.025, interval=100)

    pass


