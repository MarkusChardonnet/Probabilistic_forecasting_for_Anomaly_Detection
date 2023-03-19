import torch
import scipy
import numpy as np


def gaussian_scoring(obs, cond_exp, cond_exp_2, scoring_metric = 'p-value', min_std_val = 1e-6):
    # obs : [nb_steps, nb_samples, dimension]
    # cond_exp : [nb_steps, nb_samples, dimension, steps_ahead]
    # cond_exp_2 : [nb_steps, nb_samples, dimension, steps_ahead]
    d4 = cond_exp.shape[3]
    cond_var = cond_exp_2 - np.power(cond_exp, 2)   # make sure it is positive
    if np.any(cond_var < 0):
        print('WARNING: some predicted cond. variances below 0 -> clip')
        cond_var = np.maximum(min_std_val, cond_var)
    cond_std = np.sqrt(cond_var)
    z_scores = (np.tile(np.expand_dims(obs,axis=3),(1,1,1,d4)) - cond_exp) / cond_std
    if scoring_metric == 'p-value':
        p_vals = 2*scipy.stats.norm.sf(z_scores) # computes survival function, 2 factor for two sided
        scores = 1 - p_vals
    # scores : [nb_steps, nb_samples, dimension, steps_ahead]
    return scores


class AD_module(torch.nn.Module): # AD_module_1D, AD_module_ND
    def __init__(self, steps_ahead,
                 scoring_metric = 'p-value',
                 distribution_class = 'gaussian',
                 smoothing = 0,
                 smooth_padding = 0,
                 activation_fct = 'sigmoid'
                 ):
        super(AD_module, self).__init__()
        self.scoring_metric = scoring_metric
        self.distribution_class = distribution_class
        self.steps_ahead = steps_ahead
        self.steps_weighting = torch.nn.Linear(in_features=self.steps_ahead, out_features=1, bias=False)
        self.smoothing = smoothing
        self.smooth_padding = smooth_padding
        if self.smoothing > 0:
            self.smoothing_weights = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2*self.smoothing+1, 
                                                     bias=False, padding=smooth_padding)
        if activation_fct == 'sigmoid':
            self.activation = torch.nn.Sigmoid()

    def forward(self, obs, cond_moments,
                    washout_border = 'automatic'):
        # obs : [nb_steps, nb_samples, dimension]
        # times : [nb_samples]
        # cond_moments : [nb_steps, nb_samples, dimension, nb_moments, steps_ahead]
        nb_steps = obs.size()[0]
        nb_samples = obs.size()[1]
        dimension = obs.size()[2]
        valid_first = 0
        if washout_border == 'automatic':
            isvalid = torch.all(~torch.isnan(cond_moments).reshape(nb_steps,-1), dim=1)
            obs_valid = obs[isvalid]
            cond_moments_valid = cond_moments[isvalid]
            valid_enum = torch.arange(nb_steps)
            valid_enum[~isvalid] = nb_steps
            valid_first = torch.argmin(valid_enum)

        if self.distribution_class == 'gaussian':
            cond_exp = cond_moments_valid[:,:,:,0,:]
            cond_exp_2 = cond_moments_valid[:,:,:,1,:]
            scores_valid = gaussian_scoring(obs_valid.numpy(), cond_exp.numpy(), cond_exp_2.numpy(), self.scoring_metric)
            scores_valid = torch.tensor(scores_valid, dtype=torch.float32)

        scores_valid = torch.squeeze(self.steps_weighting(scores_valid),dim=3)
        scores = torch.zeros_like(obs)
        scores[isvalid] = scores_valid

        mask = isvalid
        if self.smoothing > 0:
            scores = scores.permute(1,2,0).reshape(-1,1,nb_steps)
            aggregated_scores = self.smoothing_weights(scores).reshape(nb_samples,dimension,-1)
            aggregated_scores = aggregated_scores.permute(2,0,1)
            scores = torch.zeros_like(obs)
            scores[self.smoothing:-self.smoothing] = aggregated_scores
            conv_mask = torch.ones(nb_steps, dtype=torch.bool)
            conv_mask[:self.smoothing - self.smooth_padding + valid_first] = 0
            conv_mask[self.smooth_padding - self.smoothing:] = 0
            mask = torch.bitwise_and(mask, conv_mask)

        scores = self.activation(scores)
        # scores : [nb_steps, nb_samples, dimension]
        # mask : [nb_steps, nb_samples, dimension]
        return scores, mask

class AD_module_2(torch.nn.Module): # AD_module_1D, AD_module_ND
    def __init__(self, steps_ahead,
                 scoring_metric = 'p-value',
                 distribution_class = 'gaussian',
                 smoothing = 0,
                 smooth_padding = 0,
                 activation_fct = 'sigmoid'
                 ):
        super(AD_module, self).__init__()
        self.scoring_metric = scoring_metric
        self.distribution_class = distribution_class
        self.steps_ahead = steps_ahead
        self.steps_weighting = torch.nn.Linear(in_features=self.steps_ahead, out_features=1, bias=False)
        self.smoothing = smoothing
        self.smooth_padding = smooth_padding
        self.weights = torch.nn.Conv1d(in_channels=steps_ahead, out_channels=1, kernel_size=2*self.smoothing+1, 
                                                     bias=False, padding=smooth_padding)
        if activation_fct == 'sigmoid':
            self.activation = torch.nn.Sigmoid()

    def forward(self, obs, cond_moments,
                    washout_border = 'automatic'):
        # obs : [nb_steps, nb_samples, dimension]
        # cond_moments : [nb_steps, nb_samples, dimension, nb_moments, steps_ahead]
        nb_steps = obs.size()[0]
        nb_samples = obs.size()[1]
        dimension = obs.size()[2]
        valid_first = 0
        if washout_border == 'automatic':
            isvalid = torch.all(~torch.isnan(cond_moments).reshape(nb_steps,-1), dim=1)
            obs_valid = obs[isvalid]
            cond_moments_valid = cond_moments[isvalid]
            valid_enum = torch.arange(nb_steps)
            valid_enum[~isvalid] = nb_steps
            valid_first = torch.argmin(valid_enum)

        if self.distribution_class == 'gaussian':
            cond_exp = cond_moments_valid[:,:,:,0,:]
            cond_exp_2 = cond_moments_valid[:,:,:,1,:]
            scores_valid = gaussian_scoring(obs_valid.numpy(), cond_exp.numpy(), cond_exp_2.numpy(), self.scoring_metric)
            scores_valid = torch.tensor(scores_valid, dtype=torch.float32)

        scores = torch.zeros((nb_steps, nb_samples, dimension, self.steps_ahead))
        scores[isvalid] = scores_valid

        mask = isvalid
        scores = scores.permute(1,2,3,0).reshape(-1,self.steps_ahead,nb_steps)
        aggregated_scores = self.weights(scores).reshape(nb_samples,dimension,-1).permute(2,0,1)
        scores = torch.zeros_like(obs)
        scores[self.smoothing:-self.smoothing] = aggregated_scores
        conv_mask = torch.ones(nb_steps, dtype=torch.bool)
        conv_mask[:self.smoothing - self.smooth_padding + valid_first] = 0
        conv_mask[self.smooth_padding - self.smoothing:] = 0
        mask = torch.bitwise_and(mask, conv_mask)

        scores = self.activation(scores)
        # scores : [nb_steps, nb_samples, dimension]
        # mask : [nb_steps, nb_samples, dimension]
        return scores, mask

    def linear_solver(self, obs, labels, cond_moments, washout_border='automatic'):
        # obs : [nb_steps, nb_samples, dimension]
        # labels : [nb_steps, nb_samples, dimension]
        # cond_moments : [nb_steps, nb_samples, dimension, nb_moments, steps_ahead]
        nb_steps = obs.size()[0]
        nb_samples = obs.size()[1]
        dimension = obs.size()[2]
        valid_first = 0
        if washout_border == 'automatic':
            isvalid = torch.all(~torch.isnan(cond_moments).reshape(nb_steps,-1), dim=1)
            obs_valid = obs[isvalid]
            cond_moments_valid = cond_moments[isvalid]
            valid_enum = torch.arange(nb_steps)
            valid_enum[~isvalid] = nb_steps
            valid_first = torch.argmin(valid_enum)

        if self.distribution_class == 'gaussian':
            cond_exp = cond_moments_valid[:,:,:,0,:]
            cond_exp_2 = cond_moments_valid[:,:,:,1,:]
            scores_valid = gaussian_scoring(obs_valid.numpy(), cond_exp.numpy(), cond_exp_2.numpy(), self.scoring_metric)
            scores_valid = torch.tensor(scores_valid, dtype=torch.float32)

        scores = torch.zeros((nb_steps, nb_samples, dimension, self.steps_ahead))
        scores[isvalid] = scores_valid

        mask = isvalid
        conv_mask = torch.ones(nb_steps, dtype=torch.bool)
        conv_mask[:self.smoothing - self.smooth_padding + valid_first] = 0
        conv_mask[self.smooth_padding - self.smoothing:] = 0
        mask = torch.bitwise_and(mask, conv_mask)
