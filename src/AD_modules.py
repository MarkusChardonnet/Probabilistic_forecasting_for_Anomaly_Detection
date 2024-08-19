import torch
import scipy
import numpy as np
import os
import torch.nn as nn
import tqdm
import scipy.stats as stat
import sklearn.linear_model as lm


def get_corrected_var(var, min_var_val=1e-4, replace_var=None):
    if np.any(var < 0):
        if replace_var is not None:
            condition = ~np.isnan(var)
            condition = np.logical_and(condition, var <= 0)
            var[condition] = replace_var
        else:
            var = np.maximum(min_var_val, var)
    return var

def gaussian_scoring(
        obs, cond_exp, cond_var,
        observed_dates=None,
        scoring_metric = 'p-value', 
        min_var_val = 1e-4,
        replace_var = None):
    # obs : [nb_steps, nb_samples, dimension]
    # cond_exp : [nb_steps, nb_samples, dimension, steps_ahead]
    # cond_var : [nb_steps, nb_samples, dimension, steps_ahead]

    nb_steps_ahead = cond_exp.shape[3]
    cond_var = get_corrected_var(cond_var, min_var_val, replace_var)
    cond_std = np.sqrt(cond_var)
    z_scores = (np.tile(np.expand_dims(obs,axis=3),(1,1,1,nb_steps_ahead)) - cond_exp) / cond_std
    if scoring_metric in ['p-value', 'two-sided']:
        p_vals = 2*scipy.stats.norm.sf(np.abs(z_scores)) # computes survival function, 2 factor for two sided
        scores = -np.log(p_vals + 1e-10)
    elif scoring_metric == 'left-tail':
        p_vals = scipy.stats.norm.cdf(z_scores)
        scores = -np.log(p_vals + 1e-10)
    elif scoring_metric == 'right-tail':
        p_vals = scipy.stats.norm.sf(z_scores)
        scores = -np.log(p_vals + 1e-10)
    else:
        raise ValueError('scoring_metric not supported')
    # scores : [nb_steps, nb_samples, dimension, steps_ahead]
    if observed_dates is not None:
        scores[~observed_dates] = np.nan
    return scores


def lognorm_scoring(
        obs, cond_exp, cond_var,
        observed_dates=None,
        scoring_metric = 'p-value',
        min_var_val = 1e-4,
        replace_var = None):
    # obs : [nb_steps, nb_samples, dimension]
    # cond_exp : [nb_steps, nb_samples, dimension, steps_ahead]
    # cond_var : [nb_steps, nb_samples, dimension, steps_ahead]

    nb_steps_ahead = cond_exp.shape[3]
    cond_var = get_corrected_var(cond_var, min_var_val, replace_var)

    # method of moments estimator
    mu = np.log(cond_exp) - 0.5 * np.log(1 + cond_var / cond_exp ** 2)
    sigma = np.sqrt(np.log(1 + cond_var / cond_exp ** 2))
    obs = np.tile(np.expand_dims(obs, axis=3),(1,1,1,nb_steps_ahead))
    sf = scipy.stats.lognorm.sf(obs, s=sigma, scale=np.exp(mu))
    cdf = scipy.stats.lognorm.cdf(obs, s=sigma, scale=np.exp(mu))
    if scoring_metric in ['p-value', 'two-sided']:
        p_vals = 2 * np.minimum(cdf, sf)
        scores = -np.log(p_vals + 1e-10)
    elif scoring_metric == 'left-tail':
        p_vals = cdf
        scores = -np.log(p_vals + 1e-10)
    elif scoring_metric == 'right-tail':
        p_vals = sf
        scores = -np.log(p_vals + 1e-10)
    else:
        raise ValueError('scoring_metric not supported')
    # scores : [nb_steps, nb_samples, dimension, steps_ahead]
    if observed_dates is not None:
        scores[~observed_dates] = np.nan
    return scores


def beta_scoring(
        obs, cond_exp, cond_var,
        observed_dates=None,
        scoring_metric = 'p-value', 
        min_var_val = 1e-4,
        epsilon = 1e-6,
        replace_var = None):
    # obs : [nb_steps, nb_samples, dimension]
    # cond_exp : [nb_steps, nb_samples, dimension, steps_ahead]
    # cond_var : [nb_steps, nb_samples, dimension, steps_ahead]

    nb_samples = cond_exp.shape[1]
    dimension = cond_exp.shape[2]
    nb_steps_ahead = cond_exp.shape[3]
    cond_var = get_corrected_var(cond_var, min_var_val, replace_var)

    # clip expectation into admissible range [0,1]
    cond_exp = np.clip(cond_exp, epsilon, 1-epsilon)
    # compute alpha beta from moments
    idx = cond_var <= cond_exp * (1-cond_exp) - epsilon
    terms = np.zeros_like(cond_exp)
    terms[idx] = cond_exp[idx] * (1-cond_exp[idx]) / cond_var[idx] - 1
    terms[~idx] = (cond_exp[~idx] * (1-cond_exp[~idx])) / (cond_exp[~idx] * (1-cond_exp[~idx]) - epsilon) - 1
    alphas = cond_exp * terms
    betas = (1-cond_exp) * terms
    # reshape vars
    reshaped_alphas = alphas.reshape(-1)
    reshaped_betas = betas.reshape(-1)
    reshaped_obs = np.tile(np.expand_dims(obs,axis=3),(1,1,1,nb_steps_ahead)).reshape(-1)
    # compute survival / accumulation function -> p-value
    if scoring_metric in ['p-value', 'two-sided']:
        # compute the CDF and SF values
        cdf_values = scipy.stats.beta.cdf(reshaped_obs, reshaped_alphas, reshaped_betas)
        sf_values = scipy.stats.beta.sf(reshaped_obs, reshaped_alphas, reshaped_betas)
        # compute the two-sided p-values
        p_vals = 2 * np.minimum(cdf_values, sf_values)
        # infer score
        scores = - np.log(p_vals + 1e-10)
    scores = scores.reshape(-1,nb_samples,dimension,nb_steps_ahead)
    if observed_dates is not None:
        scores[~observed_dates] = np.nan
    return scores


def dirichlet_scoring(
        obs, cond_exp, cond_var, observed_dates,
        epsilon=1e-6, nb_samples=10**5,
        min_var_val=1e-5, replace_var=None, verbose=False,
        seed=1, coord=None):
    # obs : [nb_steps, nb_samples, dimension]
    # cond_exp : [nb_steps, nb_samples, dimension, 1]
    # cond_var : [nb_steps, nb_samples, dimension, 1]
    # observed_dates : [nb_steps, nb_samples]

    assert cond_exp.shape == cond_var.shape
    assert len(cond_exp.shape) == 3
    time_steps = cond_exp.shape[0]
    samples = cond_exp.shape[1]
    cond_var = get_corrected_var(cond_var, min_var_val, replace_var)

    scores = np.zeros((time_steps, samples,))
    scores[~observed_dates] = np.nan
    for t in tqdm.tqdm(range(time_steps), disable=not verbose):
        for s in range(samples):
            if observed_dates[t,s]:
                if seed is not None:
                    np.random.seed(seed)
                E = np.maximum(cond_exp[t,s], epsilon)  # only positive values
                E = E/np.sum(E)  # normalize s.t. sum = 1
                # TODO: here we could try to find the ind with the best variance prediction
                factors = (E*(1-E))/cond_var[t,s] - 1
                if coord is not None and factors[coord] > 0:
                    factor = factors[coord]
                else:
                    factor = np.median(factors[(factors > 0) & (cond_var[t,s] > 0)])
                # use_ind = 0
                # factor = -1
                # while factor <= 0:
                #     use_ind += 1
                #     factor = (E[use_ind]*(1-E[use_ind]))/cond_var[t,s,use_ind] - 1
                #     if use_ind == dimension-1:
                #         factor = 1
                #         break
                # print(factor, E[coord], cond_var[t,s,coord])
                alpha = E * factor
                diri = stat.dirichlet(alpha)
                rvs = diri.rvs(size=nb_samples)
                pdfs = np.array(
                    [diri.logpdf(check_dirichlet_input(r, epsilon)) for r in rvs])
                obs_pdf = diri.logpdf(check_dirichlet_input(obs[t,s], epsilon))
                pval = np.mean(obs_pdf > pdfs)
                scores[t,s] = -np.log(pval + epsilon)

    # scores : [nb_steps, nb_samples]
    return scores


def check_dirichlet_input(sample, epsilon=1e-14):
    """
    check if the input sample is a valid dirichlet distribution
    :param sample: np.array, the sample
    """
    if np.any(sample <= 0):
        # print(np.min(sample))
        sample = np.maximum(sample, epsilon)
    if np.sum(sample) != 1:
        # to avoid rounding errors, which can break the code
        sample = sample.astype(np.double)
        sample = sample/np.sum(sample)
    s = np.sum(sample)
    if s > 1:
        d = s - 1.
        sample[np.argmax(sample)] -= d
    if s < 1:
        d = 1. - s
        sample[np.argmax(sample)] += d
    return sample


def save_checkpoint(model, optimizer, path, epoch):
    """
    save a trained torch model and the used optimizer at the given path, s.t.
    training can be resumed at the exact same point
    :param model: a torch model, e.g. instance of NJODE
    :param optimizer: a torch optimizer
    :param path: str, the path where to save the model
    :param epoch: int, the current epoch
    """
    if not os.path.exists(path):
        os.makedirs(path)
    filename = os.path.join(path, 'checkpt.tar')
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
               filename)

def get_ckpt_model(ckpt_path, model, optimizer, device):
    """
    load a saved torch model and its optimizer, inplace
    :param ckpt_path: str, path where the model is saved
    :param model: torch model instance, of which the weights etc. should be
            reloaded
    :param optimizer: torch optimizer, which should be loaded
    :param device: the device to which the model should be loaded
    """
    if not os.path.exists(ckpt_path):
        raise Exception("Checkpoint " + ckpt_path + " does not exist.")
    # Load checkpoint.
    checkpt = torch.load(ckpt_path)
    state_dict = checkpt['model_state_dict']
    optimizer.load_state_dict(checkpt['optimizer_state_dict'])
    model.load_state_dict(state_dict)
    model.to(device)



class AD_module(torch.nn.Module): # AD_module_1D, AD_module_ND
    def __init__(self, 
                 output_vars,
                 steps_ahead,
                 scoring_metric = 'p-value',
                 distribution_class = 'gaussian',
                 smoothing = 0,
                 smooth_padding = 0,
                 activation_weights = 'sigmoid',
                 activation_fct = 'sigmoid',
                 replace_values = None,
                 class_thres = 0.5,
                 factorize_weights = True,
                 ):
        super(AD_module, self).__init__()
        self.output_vars = output_vars
        self.scoring_metric = scoring_metric
        self.distribution_class = distribution_class
        self.replace_values = replace_values
        self.steps_ahead = steps_ahead
        self.nb_steps_ahead = len(steps_ahead)
        self.threshold = class_thres
        self.smoothing = smoothing
        self.smooth_padding = smooth_padding

        # random init of weights ?
        self.factorize_weights = factorize_weights
        ran = (-10.,-3.)
        if factorize_weights: 
            # 2*self.smoothing+1
            self.steps_weights = torch.nn.Parameter(torch.FloatTensor(self.nb_steps_ahead).uniform_(ran[0], ran[1]))
            self.smooth_weights = torch.nn.Parameter(torch.FloatTensor(2*self.smoothing+1).uniform_(0., 1.))

            # self.weights = self.smooth_weights.repeat(self.nb_steps_ahead,1) * self.steps_weights.repeat(2*self.smoothing+1,1).transpose(1,0)
        else:
            self.weights = torch.nn.Parameter(torch.FloatTensor(self.nb_steps_ahead,2*self.smoothing+1).uniform_(ran[0], ran[1]))

        if activation_fct == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif activation_fct == 'id':
            self.act = torch.nn.Identity()

        if activation_weights == 'sigmoid':
            self.act_weights = torch.nn.Sigmoid()
        elif activation_weights == 'softmax':
            self.act_weights = torch.nn.Softmax(dim=0)
        elif activation_weights == 'relu':
            self.act_weights = torch.nn.ReLU()
        elif activation_weights == 'identity':
            self.act_weights = torch.nn.Identity()

    def get_washout_mask(self, cond_moments, washout_border = 'automatic'):
        nb_steps = cond_moments.size()[0]
        if washout_border == 'automatic':
            mask = torch.all(~torch.isnan(cond_moments).reshape(nb_steps,-1), dim=1)
        return mask
    
    def get_operation_mask(self, mask, nb_steps):
        valid_enum = torch.arange(nb_steps)
        valid_enum[~mask] = nb_steps
        valid_first = torch.argmin(valid_enum)
        conv_mask = torch.ones(nb_steps, dtype=torch.bool)
        conv_mask[:self.smoothing - self.smooth_padding + valid_first] = 0
        conv_mask[self.smooth_padding - self.smoothing:] = 0
        mask = torch.bitwise_and(mask, conv_mask)

        return mask

    def get_individual_scores(self, cond_moments, obs):
        if self.distribution_class == 'gaussian':
            assert(('id' in self.output_vars) and (('var' in self.output_vars) or ('power-2' in self.output_vars)))
            which = np.argmax(np.array(self.output_vars) == 'id')
            cond_exp = cond_moments[:,:,:,which,:].cpu().numpy()
            if 'var' in self.output_vars:
                which = np.argmax(np.array(self.output_vars) == 'var')
                cond_exp_2 = cond_moments[:,:,:,which,:].cpu().numpy()
                cond_var = cond_exp_2 - cond_exp ** 2
            elif 'power-2' in self.output_vars:
                which = np.argmax(np.array(self.output_vars) == 'power-2')
                cond_var = cond_moments[:,:,:,which,:].cpu().numpy()
            scores_valid = gaussian_scoring(obs=obs.numpy(), cond_exp=cond_exp, cond_var=cond_var, 
                                                      scoring_metric=self.scoring_metric, replace_var=self.replace_values['var'])
            scores_valid = torch.tensor(scores_valid, dtype=torch.float32)

        return scores_valid

    def get_weights(self):
        if self.factorize_weights:
            smooth_weights = self.smooth_weights.repeat(self.nb_steps_ahead,1)
            steps_weights = self.steps_weights.repeat(2*self.smoothing+1,1).transpose(1,0)
            weights = torch.mul(smooth_weights, steps_weights)
        else:
            weights = self.weights
        # weights = self.act_weights(weights.reshape(-1)).reshape(self.nb_steps_ahead,2*self.smoothing+1)
        weights = self.act_weights(weights)
        return weights
    
    def loss(self, ad_scores, ad_labels):
        criterion = nn.CrossEntropyLoss()

        loss = criterion(ad_scores, ad_labels)

        loss_regularizer = True
        if loss_regularizer:
            loss += torch.sum(self.get_weights() ** 2)

        return loss

    def forward(self, obs, cond_moments,
                    washout_border = 'automatic'):
        # obs : [nb_steps, nb_samples, dimension]
        # cond_moments : [nb_steps, nb_samples, dimension, nb_moments, steps_ahead]
        nb_steps = obs.size()[0]
        nb_samples = obs.size()[1]
        dimension = obs.size()[2]

        mask = self.get_washout_mask(cond_moments, washout_border)
        obs_valid = obs[mask]
        cond_moments_valid = cond_moments[mask]

        scores = self.get_individual_scores(cond_moments_valid, obs_valid)

        # scores = torch.zeros((nb_steps, nb_samples, dimension, self.nb_steps_ahead))
        # scores[mask] = scores_valid
        scores = scores.permute(1,2,0,3)

        # print(torch.mean(scores.reshape(-1,self.nb_steps_ahead),dim=0))

        weights = self.get_weights().reshape(-1,1)

        unfold_scores = scores.unfold(dimension=2, size=2*self.smoothing+1, step=1)
        unfold_scores = unfold_scores.reshape(-1, (2 * self.smoothing + 1) * self.nb_steps_ahead)
        aggregated_scores = torch.matmul(unfold_scores, weights).reshape((nb_samples, dimension, -1)).permute(2,0,1)

        # eventually put this into another method called get_operation_mask
        mask = self.get_operation_mask(mask, nb_steps)

        final_scores = torch.zeros((nb_steps, nb_samples, dimension))
        final_scores[mask] = self.act(aggregated_scores - 2)
        # scores : [nb_steps, nb_samples, dimension]
        # mask : [nb_steps, nb_samples, dimension]
        return final_scores, mask

    def get_predicted_label(self, obs, cond_moments,
                    washout_border = 'automatic'):
        
        scores, mask = self(obs, cond_moments, washout_border)

        # scores = scores.detach().cpu().numpy()
        labels = torch.zeros_like(scores)
        masked_scores = scores[mask]
        masked_labels = labels[mask]
        masked_labels[masked_scores > self.threshold] = 1
        labels[mask] = masked_labels

        return labels, scores

    def linear_solver(self, obs, labels, cond_moments, washout_border='automatic'):
        # obs : [nb_steps, nb_samples, dimension]
        # labels : [nb_steps, nb_samples, dimension]
        # cond_moments : [nb_steps, nb_samples, dimension, nb_moments, steps_ahead]
        assert(isinstance(self.activation,torch.nn.Identity))

        nb_steps = obs.size()[0]
        nb_samples = obs.size()[1]
        dimension = obs.size()[2]
        
        mask = self.get_washout_mask(cond_moments, washout_border)
        obs_valid = obs[mask]
        cond_moments_valid = cond_moments[mask]

        scores_valid = self.get_individual_scores(cond_moments_valid, obs_valid)

        scores = torch.zeros((nb_steps, nb_samples, dimension, self.nb_steps_ahead))
        scores[mask] = scores_valid

        op_mask = self.get_operation_mask(mask, nb_steps)
        
        labels = labels[op_mask].permute(1,2,0)
        scores = scores[mask].permute(1,2,0,3)

        unfold_scores = scores.unfold(dimension=2, size=2*self.smoothing+1, step=1).reshape(-1, (2 * self.smoothing + 1) * self.nb_steps_ahead)
        unfold_labels = labels.reshape(-1).unsqueeze(1)

        ls_weights = torch.linalg.lstsq(unfold_scores,unfold_labels).solution
        ls_weights = ls_weights.reshape(2 * self.smoothing + 1, self.nb_steps_ahead).permute(1,0).unsqueeze(0)

        self.weights.weight = torch.nn.Parameter(ls_weights)


AGGREGATION_METHODS = {
    'mean': np.mean,
    'min': np.min,
    'max': np.max,
}


def get_cond_exp_var(cond_moments, output_vars):
    assert(('id' in output_vars) and (
            ('var' in output_vars) or ('power-2' in output_vars)))
    which = np.argmax(np.array(output_vars) == 'id')
    cond_exp = cond_moments[:,:,:,which]
    if 'var' in output_vars:
        which = np.argmax(np.array(output_vars) == 'var')
        cond_var = cond_moments[:, :, :, which]
    elif 'power-2' in output_vars:
        which = np.argmax(np.array(output_vars) == 'power-2')
        cond_exp_2 = cond_moments[:,:,:,which]
        cond_var = cond_exp_2 - cond_exp ** 2
    return cond_exp, cond_var


class Simple_AD_module(torch.nn.Module):  # AD_module_1D, AD_module_ND
    def __init__(self, 
                 output_vars,
                 scoring_metric='p-value',
                 distribution_class='gaussian',
                 score_factor=1.,
                 activation_fct='sigmoid',
                 replace_values=None,
                 class_thres=0.5,
                 nb_MC_samples=10**5,
                 epsilon=1e-6,
                 dirichlet_use_coord=None,
                 aggregation_method='mean',
                 seed=None,
                 verbose=False,):
        super(Simple_AD_module, self).__init__()
        self.output_vars = output_vars
        self.scoring_metric = scoring_metric
        self.distribution_class = distribution_class
        self.replace_values = replace_values
        self.threshold = class_thres
        self.nb_samples = nb_MC_samples
        self.verbose = verbose
        self.seed = seed
        self.epsilon = epsilon
        self.dirichlet_use_coord = dirichlet_use_coord
        self.weight = score_factor
        self.aggregation_method = aggregation_method

        if activation_fct == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif activation_fct == 'id':
            self.act = torch.nn.Identity()

    def get_individual_scores(
            self, cond_moments, obs, observed_dates=None):

        cond_exp, cond_var = get_cond_exp_var(cond_moments, self.output_vars)

        if self.distribution_class == 'gaussian':
            cond_exp = np.expand_dims(cond_exp, 3)
            cond_var = np.expand_dims(cond_var, 3)
            # scores : [nb_steps, nb_samples, dimension, steps_ahead=1]
            scores_valid = gaussian_scoring(
                obs=obs, cond_exp=cond_exp, cond_var=cond_var,
                observed_dates=observed_dates,
                scoring_metric=self.scoring_metric, replace_var=None)
            # scores : [nb_samples, nb_steps, dimension, steps_ahead=1]
            scores_valid = scores_valid.transpose(1,0,2,3)
        elif self.distribution_class in ['normal', 'lognormal']:
            scoring_methods = {
                'normal': gaussian_scoring,
                'lognormal': lognorm_scoring,
            }
            cond_exp = np.expand_dims(cond_exp, 3)
            cond_var = np.expand_dims(cond_var, 3)
            if self.replace_values is not None:
                self.replace_values = np.expand_dims(self.replace_values, 1)
            # scores : [nb_steps, nb_samples, dimension, steps_ahead]
            scores_valid = scoring_methods[self.distribution_class](
                obs=obs, cond_exp=cond_exp, cond_var=cond_var,
                observed_dates=observed_dates,
                scoring_metric=self.scoring_metric,
                replace_var=self.replace_values,)
            if scores_valid.shape[2] > 1:
                scores_valid = AGGREGATION_METHODS[self.aggregation_method](
                    scores_valid, axis=2, keepdims=True)
            assert scores_valid.shape[2] == 1 and scores_valid.shape[3] == 1
            # scores : [nb_samples, nb_steps]
            scores_valid = scores_valid.squeeze(3).squeeze(2).transpose(1,0)
        elif self.distribution_class == 'dirichlet':
            # scores : [nb_steps, nb_samples]
            scores_valid = dirichlet_scoring(
                obs=obs, cond_exp=cond_exp, cond_var=cond_var,
                observed_dates=observed_dates,
                nb_samples=self.nb_samples, replace_var=self.replace_values,
                min_var_val=0., coord=self.dirichlet_use_coord,
                verbose=self.verbose, seed=self.seed)
            scores_valid = scores_valid.transpose(1,0)
        elif self.distribution_class == 'beta':
            cond_exp = np.expand_dims(cond_exp, 3)
            cond_var = np.expand_dims(cond_var, 3)
            # scores : [nb_steps, nb_samples, dimension, steps_ahead=1]
            scores_valid = beta_scoring(
                obs=obs, cond_exp=cond_exp, cond_var=cond_var,
                observed_dates=observed_dates,
                scoring_metric=self.scoring_metric, replace_var=None)
            # scores : [nb_samples, nb_steps, dimension, steps_ahead=1]
            scores_valid = scores_valid.transpose(1, 0, 2, 3)
        else:
            raise ValueError('distribution_class not supported')
        return scores_valid

    def forward(self, obs, cond_moments, observed_dates=None):
        # obs : [nb_steps, nb_samples, dimension]
        # cond_moments : [nb_steps, nb_samples, dimension, nb_moments]

        scores = self.get_individual_scores(cond_moments, obs, observed_dates)

        # scores : [nb_steps, nb_samples, dimension] or [nb_samples, nb_steps]
        return scores

    def get_predicted_label(self, obs, cond_moments, observed_dates=None):
        
        scores = self(obs, cond_moments, observed_dates)

        # scores = scores.detach().cpu().numpy()
        labels = torch.zeros_like(scores)
        labels[scores > self.threshold] = 1

        return labels, scores
    

class DimAcc_AD_module(torch.nn.Module):
    def __init__(self, 
                 output_vars,
                 dimension,
                 scoring_metric='p-value',
                 distribution_class='gaussian',
                 activation_fct='sigmoid',
                 activation_weights='sigmoid',
                 replace_values=None,
                 class_thres=0.5,
                 aggregation_method='mean',
                 train_labels=None,
                 **kwargs,):
        """
        Args:
            aggregation_method: one of
                - 'mean': mean of the scores
                - 'logistic': logistic regression
            train_labels: label for the training data to fit the logistic
                regression
        """
        super(DimAcc_AD_module, self).__init__()
        self.output_vars = output_vars
        self.scoring_metric = scoring_metric
        self.distribution_class = distribution_class
        self.replace_values = replace_values
        self.threshold = class_thres
        # self.nb_samples = nb_MC_samples
        # self.verbose = verbose
        # self.seed = seed
        # self.dirichlet_use_coord = dirichlet_use_coord
        self.dimension = dimension
        self.aggregation_method = aggregation_method
        self.train_labels = train_labels
        self.logreg = None

        if activation_fct == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif activation_fct == 'id':
            self.act = torch.nn.Identity()

        if activation_weights == 'sigmoid':
            self.act_weights = torch.nn.Sigmoid()
        elif activation_weights == 'identity':
            self.act_weights = torch.nn.Identity()

        ran = (-10.,-3.)
        self.score_weights = torch.nn.Parameter(torch.FloatTensor(dimension).uniform_(ran[0], ran[1]))
        self.act_intercept = torch.nn.Parameter(torch.FloatTensor(1).uniform_(ran[0], ran[1]))

    def get_individual_scores(
            self, cond_moments, obs, observed_dates=None):

        assert(('id' in self.output_vars) and (
                ('var' in self.output_vars) or ('power-2' in self.output_vars)))
        which = np.argmax(np.array(self.output_vars) == 'id')
        cond_exp = cond_moments[:,:,:,which]
        if 'var' in self.output_vars:
            which = np.argmax(np.array(self.output_vars) == 'var')
            cond_var = cond_moments[:, :, :, which]
        elif 'power-2' in self.output_vars:
            which = np.argmax(np.array(self.output_vars) == 'power-2')
            cond_exp_2 = cond_moments[:,:,:,which]
            cond_var = cond_exp_2 - cond_exp ** 2

        if self.distribution_class == 'gaussian':
            cond_exp = np.expand_dims(cond_exp, 3)
            cond_var = np.expand_dims(cond_var, 3)
            # scores : [nb_steps, nb_samples, dimension]
            scores_valid = gaussian_scoring(
                obs=obs, cond_exp=cond_exp, cond_var=cond_var,
                observed_dates=observed_dates,
                scoring_metric=self.scoring_metric, replace_var=None)
            # scores : [nb_samples, nb_steps, dimension]
            scores_valid = scores_valid.squeeze(axis=3).transpose(1,0,2)
        elif self.distribution_class == 'beta':
            cond_exp = np.expand_dims(cond_exp, 3)
            cond_var = np.expand_dims(cond_var, 3)
            # scores : [nb_steps, nb_samples, dimension]
            scores_valid = beta_scoring(
                obs=obs, cond_exp=cond_exp, cond_var=cond_var,
                observed_dates=observed_dates,
                scoring_metric=self.scoring_metric, replace_var=None)
            # scores : [nb_samples, nb_steps, dimension]
            scores_valid = scores_valid.squeeze(axis=3).transpose(1,0,2)

        scores_valid = torch.tensor(scores_valid, dtype=torch.float32)
        return scores_valid
    
    def get_weights(self):
        return self.act_weights(self.score_weights)
    
    def loss(self, ad_scores, ad_labels):
        criterion = nn.CrossEntropyLoss()

        loss = criterion(ad_scores, ad_labels)

        loss_regularizer = True
        if loss_regularizer:
            loss += torch.sum(self.get_weights() ** 2)

        return loss

    def forward(self, obs, cond_moments, observed_dates=None):
        # obs : [nb_steps, nb_samples, dimension]
        # cond_moments : [nb_steps, nb_samples, dimension, nb_moments]
        nb_steps = cond_moments.shape[0]

        # scores : [nb_samples, nb_steps, dimension]
        scores = self.get_individual_scores(cond_moments, obs, observed_dates)

        if self.aggregation_method == 'mean':
            scores = torch.mean(scores, dim=2)
            scores = scores.detach().cpu().numpy()
        elif self.aggregation_method == 'logistic':
            scores = scores.detach().cpu().numpy()
            X = scores.reshape(-1, self.dimension)
            which = ~np.isnan(X).any(axis=1)
            y = self.train_labels.reshape(-1, 1).repeat(
                nb_steps, axis=1).reshape(-1)

            if self.logreg is None:
                self.logreg = lm.LogisticRegression()
                self.logreg.fit(X=X[which], y=y[which])

            _scores = self.logreg.predict_proba(X[which])[:,1]
            scores = np.zeros_like(X[:,0]) * np.nan
            scores[which] = _scores
            scores = scores.reshape(-1, nb_steps)
        else:
            raise ValueError('aggregation_method not supported')

        # nb_steps = cond_moments.shape[0]
        # nb_samples = cond_moments.shape[1]
        # weights = self.get_weights()
        # scores = torch.matmul(scores.reshape(-1, self.dimension), weights.view(-1,1))
        # scores = scores.reshape(nb_samples, nb_steps)
        # scores = self.act(scores - self.act_intercept)

        # scores : [nb_samples, nb_steps]
        return scores

    def get_predicted_label(self, obs, cond_moments, observed_dates=None):
        
        scores = self(obs, cond_moments, observed_dates)

        labels = torch.zeros_like(scores)
        labels[scores > self.threshold] = 1

        return labels, scores
