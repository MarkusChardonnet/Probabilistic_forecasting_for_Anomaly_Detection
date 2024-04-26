import torch
import scipy
import numpy as np
import os
import torch.nn as nn
import tqdm
import scipy.stats as stat


def gaussian_scoring_2_moments(obs, 
                               cond_exp,
                               cond_var, 
                               scoring_metric = 'p-value', 
                               min_var_val = 1e-4,
                               replace_var = None):
    # obs : [nb_steps, nb_samples, dimension]
    # cond_exp : [nb_steps, nb_samples, dimension, steps_ahead]
    # cond_var : [nb_steps, nb_samples, dimension, steps_ahead]
    dimension = cond_exp.shape[2]
    nb_steps_ahead = cond_exp.shape[3]
    if np.any(cond_var < 0):
        # print('WARNING: some predicted cond. variances below 0 -> clip')
        if replace_var is not None:
            for d in range(dimension):
                for s in range(nb_steps_ahead):
                    condition = ~np.isnan(cond_var[:,:,d,s])
                    condition = np.logical_and(condition, cond_var[:,:,d,s] <= 0)
                    cond_var[:,:,d,s][condition] = replace_var[d,s]
        else:
            cond_var = np.maximum(min_var_val, cond_var)
    cond_std = np.sqrt(cond_var)
    z_scores = np.abs((np.tile(np.expand_dims(obs,axis=3),(1,1,1,nb_steps_ahead)) - cond_exp)) / cond_std
    if scoring_metric == 'p-value':
        p_vals = 2*scipy.stats.norm.sf(z_scores) # computes survival function, 2 factor for two sided
        scores = - np.log(p_vals + 1e-10)
    # scores : [nb_steps, nb_samples, dimension, steps_ahead]
    return scores


def dirichlet_scoring(
        obs, cond_exp, cond_var, nb_samples=10**5,
        min_var_val=1e-5, replace_var=None, verbose=False):
    # obs : [nb_steps, nb_samples, dimension]
    # cond_exp : [nb_steps, nb_samples, dimension]
    # cond_var : [nb_steps, nb_samples, dimension]

    assert cond_exp.shape == cond_var.shape
    assert len(cond_exp.shape) == 3
    dimension = cond_exp.shape[2]
    time_steps = cond_exp.shape[0]
    samples = cond_exp.shape[1]
    if np.any(cond_var < 0):
        # print('WARNING: some predicted cond. variances below 0 -> clip')
        if replace_var is not None:
            for d in range(dimension):
                condition = ~np.isnan(cond_var[:,:,d])
                condition = np.logical_and(condition, cond_var[:,:,d] <= 0)
                cond_var[:,:,d][condition] = replace_var[d]
        else:
            cond_var = np.maximum(min_var_val, cond_var)

    scores = np.zeros((time_steps, samples,))
    for t in tqdm.tqdm(range(time_steps), disable=not verbose):
        for s in range(samples):
            E = np.maximum(cond_exp[t,s], 1e-10)  # only positive values
            E = E/np.sum(E)  # normalize s.t. sum = 1
            # TODO: here we could try to find the ind with the best variance prediction
            factors = (E*(1-E))/cond_var[t,s] - 1
            factor = np.median(factors[factors > 0])
            # use_ind = 0
            # factor = -1
            # while factor <= 0:
            #     use_ind += 1
            #     factor = (E[use_ind]*(1-E[use_ind]))/cond_var[t,s,use_ind] - 1
            #     if use_ind == dimension-1:
            #         factor = 1
            #         break
            alpha = E * factor
            diri = stat.dirichlet(alpha)
            rvs = diri.rvs(size=nb_samples)
            pdfs = np.array([diri.logpdf(r) for r in rvs])
            obs_pdf = diri.logpdf(obs[t,s])
            pval = np.mean(obs_pdf <= pdfs)
            scores[t,s] = -np.log(pval + 1e-10)

    # scores : [nb_steps, nb_samples]
    return scores


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
            scores_valid = gaussian_scoring_2_moments(obs=obs.numpy(), cond_exp=cond_exp, cond_var=cond_var, 
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



'''
class AD_module(torch.nn.Module): # AD_module_1D, AD_module_ND
    def __init__(self, steps_ahead,
                 scoring_metric = 'p-value',
                 distribution_class = 'gaussian',
                 smoothing = 0,
                 smooth_padding = 0,
                 activation_fct = 'sigmoid',
                 class_thres = 0.5,
                 ):
        super(AD_module, self).__init__()
        self.scoring_metric = scoring_metric
        self.distribution_class = distribution_class
        self.steps_ahead = steps_ahead
        self.steps_weighting = torch.nn.Linear(in_features=self.steps_ahead, out_features=1, bias=False)
        self.smoothing = smoothing
        self.smooth_padding = smooth_padding
        self.threshold = class_thres
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
            scores_valid = gaussian_scoring_2_moments(obs_valid.numpy(), cond_exp.numpy(), cond_exp_2.numpy(), self.scoring_metric)
            scores_valid = torch.tensor(scores_valid, dtype=torch.float32)

        scores_valid = torch.squeeze(self.steps_weighting(scores_valid),dim=3)
        scores = torch.zeros_like(obs)
        scores[isvalid] = scores_valid

        mask = isvalid

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
    
'''


'''
class AD_module(torch.nn.Module): # AD_module_1D, AD_module_ND
    def __init__(self, 
                 output_vars,
                 steps_ahead,
                 scoring_metric = 'p-value',
                 distribution_class = 'gaussian',
                 smoothing = 0,
                 smooth_padding = 0,
                 activation_fct = 'sigmoid',
                 replace_values = None,
                 class_thres = 0.5,
                 ):
        super(AD_module, self).__init__()
        self.output_vars = output_vars
        self.scoring_metric = scoring_metric
        self.distribution_class = distribution_class
        self.replace_values = replace_values
        self.steps_ahead = steps_ahead
        self.nb_steps = len(steps_ahead)
        self.threshold = class_thres
        self.steps_weighting = torch.nn.Linear(in_features=self.nb_steps, out_features=1, bias=False)
        self.smoothing = smoothing
        self.smooth_padding = smooth_padding

        self.weights = torch.nn.Conv1d(in_channels=self.nb_steps, out_channels=1, kernel_size=2*self.smoothing+1, 
                                                     bias=False, padding=smooth_padding)
        if activation_fct == 'sigmoid':
            self.activation = torch.nn.Sigmoid()
        if activation_fct == 'id':
            self.activation = torch.nn.Identity()

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
            scores_valid = gaussian_scoring_2_moments(obs=obs.numpy(), cond_exp=cond_exp, cond_var=cond_var, 
                                                      scoring_metric=self.scoring_metric, replace_var=self.replace_values['var'])
            scores_valid = torch.tensor(scores_valid, dtype=torch.float32)

        return scores_valid
    
    def get_weights(self):
        return self.weights.weight

    def forward(self, obs, cond_moments,
                    washout_border = 'automatic'):
        # obs : [nb_steps, nb_samples, dimension]
        # cond_moments : [nb_steps, nb_samples, dimension, nb_moments, steps_ahead]
        nb_steps = obs.size()[0]
        nb_samples = obs.size()[1]
        dimension = obs.size()[2]

        # eventually put this into another method called get_washout_mask
        mask = self.get_washout_mask(cond_moments, washout_border)
        obs_valid = obs[mask]
        cond_moments_valid = cond_moments[mask]

        scores_valid = self.get_individual_scores(cond_moments_valid, obs_valid)

        scores = torch.zeros((nb_steps, nb_samples, dimension, self.nb_steps))
        scores[mask] = scores_valid

        scores = scores.permute(1,2,3,0).reshape(-1,self.nb_steps,nb_steps)
        aggregated_scores = self.weights(scores).reshape(nb_samples,dimension,-1).permute(2,0,1)
        scores = torch.zeros_like(obs)
        scores[self.smoothing:-self.smoothing] = aggregated_scores

        # eventually put this into another method called get_operation_mask
        mask = self.get_operation_mask(mask, nb_steps)

        scores = self.activation(scores)
        # scores : [nb_steps, nb_samples, dimension]
        # mask : [nb_steps, nb_samples, dimension]
        return scores, mask

    def get_predicted_label(self, obs, cond_moments,
                    washout_border = 'automatic'):
        
        scores, mask = self(obs, cond_moments, washout_border)

        # scores = scores.detach().cpu().numpy()
        labels = torch.zeros_like(scores)
        masked_scores = scores[mask]
        masked_labels = labels[mask]
        masked_labels[masked_scores > self.threshold] = 1
        labels[mask] = masked_labels

        print(self.threshold)

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

        scores = torch.zeros((nb_steps, nb_samples, dimension, self.nb_steps))
        scores[mask] = scores_valid

        op_mask = self.get_operation_mask(mask, nb_steps)
        
        labels = labels[op_mask].permute(1,2,0)
        scores = scores[mask].permute(1,2,0,3)

        unfold_scores = scores.unfold(dimension=2, size=2*self.smoothing+1, step=1).reshape(-1, (2 * self.smoothing + 1) * self.nb_steps)
        unfold_labels = labels.reshape(-1).unsqueeze(1)

        ls_weights = torch.linalg.lstsq(unfold_scores,unfold_labels).solution
        ls_weights = ls_weights.reshape(2 * self.smoothing + 1, self.nb_steps).permute(1,0).unsqueeze(0)

        self.weights.weight = torch.nn.Parameter(ls_weights)

'''

class Simple_AD_module(torch.nn.Module): # AD_module_1D, AD_module_ND
    def __init__(self, 
                 output_vars,
                 scoring_metric = 'p-value',
                 distribution_class = 'gaussian',
                 score_factor=1.,
                 activation_fct = 'sigmoid',
                 replace_values = None,
                 class_thres = 0.5,
                 ):
        super(Simple_AD_module, self).__init__()
        self.output_vars = output_vars
        self.scoring_metric = scoring_metric
        self.distribution_class = distribution_class
        self.replace_values = replace_values
        self.threshold = class_thres

        self.weight = score_factor

        if activation_fct == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif activation_fct == 'id':
            self.act = torch.nn.Identity()

    def get_individual_scores(self, cond_moments, obs):
        if self.distribution_class == 'gaussian':
            assert(('id' in self.output_vars) and (('var' in self.output_vars) or ('power-2' in self.output_vars)))
            which = np.argmax(np.array(self.output_vars) == 'id')
            cond_exp = np.expand_dims(cond_moments[:,:,:,which].cpu().numpy(),3)
            if 'var' in self.output_vars:
                which = np.argmax(np.array(self.output_vars) == 'var')
                cond_var = np.expand_dims(cond_moments[:,:,:,which].cpu().numpy(),3)
            elif 'power-2' in self.output_vars:
                which = np.argmax(np.array(self.output_vars) == 'power-2')
                cond_exp_2 = np.expand_dims(cond_moments[:,:,:,which].cpu().numpy(),3)
                cond_var = cond_exp_2 - cond_exp ** 2
            scores_valid = gaussian_scoring_2_moments(obs=obs.numpy(), cond_exp=cond_exp, cond_var=cond_var, 
                                                      scoring_metric=self.scoring_metric, replace_var=None)
            scores_valid = torch.tensor(scores_valid, dtype=torch.float32)

        return scores_valid

    def forward(self, obs, cond_moments):
        # obs : [nb_steps, nb_samples, dimension]
        # cond_moments : [nb_steps, nb_samples, dimension, nb_moments]

        scores = self.get_individual_scores(cond_moments, obs)

        # scores = scores.permute(1,2,0,3)

        # scores : [nb_steps, nb_samples, dimension]
        return scores

    def get_predicted_label(self, obs, cond_moments):
        
        scores = self(obs, cond_moments)

        # scores = scores.detach().cpu().numpy()
        labels = torch.zeros_like(scores)
        labels[scores > self.threshold] = 1

        return labels, scores