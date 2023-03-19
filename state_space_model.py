import torch
import torch.nn as nn
import torch.nn.functional as F

# def make_SSM_parameters():

'''
state equation : z(t+1) = a + A*z(t) + W*u(t)
observation equation : x(t) = b + B*z(t) + v(t)
v is a Gaussian white noise with covariance matrix V
u is a Gaussian white noise with covariance matrix U
Dimensions :
x(t) : d
z(t) : n
v(t) : d
u(t) : m
'''

class GaussianLinearInvariantSSM(nn.Module):
    def __init__(self, 
                d_obs, 
                d_state,
                d_noise,
                initial_params = 'default'):
        super(GaussianLinearInvariantSSM, self).__init__()

        # dimensions of observation, state, and noise spaces
        self.d_obs = d_obs
        self.d_state = d_state
        self.d_noise = d_noise

        self.initialize_params(initial_params)

    def initialize_params(self, initial_params = 'default'):
        # initial_params : dictionary containing all parameters (if provided), with appripriate keys
        
        if initial_params == 'default':
            self.params = nn.ParameterDict({
                'a': nn.Parameter(torch.zeros(self.d_state)),
                'b': nn.Parameter(torch.zeros(self.d_obs)),
                'A': nn.Parameter(torch.eye(self.d_state, self.d_state)),
                'B': nn.Parameter(torch.ones(self.d_obs, self.d_state)),
                'U': nn.Parameter(torch.eye(self.d_noise, self.d_noise)),
                'V': nn.Parameter(torch.eye(self.d_obs, self.d_obs)),
                'W': nn.Parameter(torch.eye(self.d_state, self.d_noise)),
                'a0': nn.Parameter(torch.zeros(self.d_state)),
                'A0': nn.Parameter(torch.eye(self.d_state, self.d_state))
            })
        elif initial_params is not None:
            self.params = nn.ParameterDict({ 
                'a': nn.Parameter(initial_params['a']),
                'b': nn.Parameter(initial_params['b']),
                'A': nn.Parameter(initial_params['A']),
                'B': nn.Parameter(initial_params['B']),
                'U': nn.Parameter(initial_params['U']),
                'V': nn.Parameter(initial_params['V']),
                'W': nn.Parameter(initial_params['W']),
                'a0': nn.Parameter(initial_params['a0']),
                'A0': nn.Parameter(initial_params['A0'])
            })

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def generate_data(self, nsamples, ts_length, noise_type = 'gaussian'):

        # make inital zero state (no observation at this stage)
        current_state = torch.zeros(nsamples, self.d_state)

        observations = []

        for t in range(ts_length):
            state_noise = torch.normal(mean=torch.zeros(nsamples, self.d_noise), std=torch.ones(nsamples, self.d_noise))
            state_noise = torch.matmul(torch.matmul(self.params['W'],self.params['U']), torch.unsqueeze(state_noise, dim=2)).reshape(nsamples, self.d_state)
            new_state = torch.unsqueeze(self.params['a'], dim=0).repeat(nsamples,1)
            new_state += torch.matmul(self.params['A'], torch.unsqueeze(current_state, dim=2)).reshape(nsamples, self.d_state)
            new_state += state_noise
            obs = torch.unsqueeze(self.params['b'], dim=0).repeat(nsamples,1)
            obs += torch.matmul(self.params['B'], torch.unsqueeze(new_state, dim=2)).reshape(nsamples, self.d_obs)
            obs_noise = torch.normal(mean=torch.zeros(nsamples, self.d_obs), std=torch.ones(nsamples, self.d_obs))
            obs_noise = torch.matmul(self.params['V'], torch.unsqueeze(obs_noise, dim=2)).reshape(nsamples, self.d_obs)
            obs += obs_noise
            observations.append(obs)
            current_state = new_state

        observations = torch.cat([obs.unsqueeze(1) for obs in observations],dim=1)

        return observations


    def freeze_params(self, param_name_list):
        # param_name_list : list of parameter keys to freeze
        for p in param_name_list:
            if p not in self.params.keys():
                print("{} is not a parameter of the model ! ".format(p))
            else:
                self.params[p].requires_grad = False

    def compute_innovation(self, obs, prior_state_exp, prior_state_var):
        # obs : [b,d]
        # prior_state_exp [b,n]
        # prior_state_var [b,n,n]

        # Performs the following operations :
        # e(t) = x(t) - B*z(t|t-1) - b
        # R(t) = B*S(t|t-1)*B' + V
        # where e is the innovation, and R its variance

        b = obs.size(dim=0)

        innov = obs - torch.matmul(self.params['B'], prior_state_exp.unsqueeze(2)).view(b,-1) - self.params['b'].unsqueeze(0).repeat(b,1)
        innov_var = torch.matmul(torch.matmul(self.params['B'], prior_state_var), self.params['B'].transpose(1,0)) + self.params['V'].unsqueeze(0).repeat(b,1,1)

        # innov : [b,d]
        # innov_var : [b,d,d]
        return innov, innov_var

    def compute_posterior_state(self, innov, innov_var, prior_state_exp, prior_state_var):
        # innov : [b,d]
        # innov_var : [b,d,d]
        # prior_state_exp [b,n]
        # prior_state_var [b,n,n]

        # Performs the following operations :
        # z(t|t) = z(t|t-1) - S(t|t-1)*B'*R(t)^(-1)*e(t)
        # S(t|t) = S(t|t-1) - S(t|t-1)*B'*R(t)^(-1)*B*S(t|t-1)
        # where z(t|t) and S(t|t) are the posterior state expectation and variance respectively

        b = innov.size(dim=0)

        SBR = torch.bmm(prior_state_var, torch.matmul(self.params['B'].transpose(0,1),torch.linalg.inv(innov_var)))
        post_state_exp = prior_state_exp + torch.bmm(SBR, innov.unsqueeze(2)).view(b,-1)
        post_state_var = prior_state_var - torch.bmm(SBR, torch.matmul(self.params['B'], prior_state_var))

        # post_state_exp : [b,n]
        # post_state_var : [b,n,n]
        return post_state_exp, post_state_var

    def compute_prior_state(self, post_state_exp, post_state_var):
        # innov : [b,d]
        # innov_var : [b,d,d]
        # prior_state_exp [b,n]
        # prior_state_var [b,n,n]

        # Performs the following operations :
        # z(t+1|t) = A*z(t|t) + a
        # S(t+1|t) = A*S(t|t)*A' + W*U*W'
        # where z(t+1|t) and S(t+1|t) are the prior state expectation and variance respectively

        b = post_state_exp.size(dim=0)

        prior_state_exp = self.params['a'].unsqueeze(0).repeat(b,1) + torch.matmul(self.params['A'], post_state_exp.unsqueeze(2)).view(b,-1)
        post_var_term = torch.matmul(self.params['A'], torch.matmul(post_state_var, self.params['A'].transpose(1,0)))
        current_var_term = torch.matmul(self.params['W'], torch.matmul(self.params['U'], self.params['W'].transpose(1,0))).unsqueeze(0).repeat(b,1,1)
        prior_state_var = post_var_term + current_var_term

        # prior_state_exp : [b,n]
        # prior_state_var : [b,n,n]
        return prior_state_exp, prior_state_var

    def negloglikelihood(self, input):
        # input [b,T,d]
        # b : batch
        # T : length (time)
        # d : dimension

        # Computes the negative log likelihood of the observations given the curent model parameters
        # This is an objective function for learning the model parameters

        b = input.size(0)
        T = input.size(1)
        innovs, innov_vars, _, _ = self.filter_sequence(input, T)

        pi = torch.acos(torch.zeros(1)).item()
        negloglikelihood = torch.zeros(b)
        for t in range(T):
            negloglikelihood += 0.5*torch.logdet(2.*pi*innov_vars[t]).view(b)
            negloglikelihood += 0.5*torch.bmm(innovs[t].unsqueeze(1),torch.bmm(torch.linalg.inv(innov_vars[t]),innovs[t].unsqueeze(2))).view(b)

        return negloglikelihood

    def filter_one_step(self, obs, prior_state_exp, prior_state_var):
        # Computes filterning quantities at time t
        # Returns innovation and posterior of state at time t given observations until time t
        # and prior state at time t+1 given observations until time t

        innov, innov_var = self.compute_innovation(obs, prior_state_exp, prior_state_var)
        post_state_exp, post_state_var = self.compute_posterior_state(innov, innov_var, prior_state_exp, prior_state_var)
        prior_state_exp, prior_state_var = self.compute_prior_state(post_state_exp, post_state_var)

        return innov, innov_var, post_state_exp, post_state_var, prior_state_exp, prior_state_var

    def filter_sequence(self, input, time_horizon):
        # input [b,T,d]
        # b : batch
        # T : length (time)
        # d : dimension

        # Computes filterning quantities for the input until the time horizon
        # Returns innovations and the last prior of state (see filter_one_step)

        b = input.size(0)
        T = input.size(1)

        prior_state_exps = []
        prior_state_vars = []
        innovs = []
        innov_vars = []
        post_state_exps = []
        post_state_vars = []
        
        prior_state_exp = self.params['a0'].unsqueeze(0).repeat(b,1)
        prior_state_var = self.params['A0'].unsqueeze(0).repeat(b,1,1)

        for t in range(time_horizon):
            obs = input[:,t,:]
            innov, innov_var, post_state_exp, post_state_var, prior_state_exp, prior_state_var = self.filter_one_step(obs, prior_state_exp, prior_state_var)
            
            innovs.append(innov)
            innov_vars.append(innov_var)
            post_state_exps.append(post_state_exp)
            post_state_vars.append(post_state_var)
            prior_state_exps.append(prior_state_exp)
            prior_state_vars.append(prior_state_var)

        last_prior_state_exp = prior_state_exps[-1]
        last_prior_state_var = prior_state_vars[-1]

        return innovs, innov_vars, last_prior_state_exp, last_prior_state_var

    def forecast_one_step(self, prior_state_exp, prior_state_var):

        b = prior_state_exp.size(0)

        prior_state_exp = self.params['a'].unsqueeze(0).repeat(b,1) + torch.matmul(self.params['A'], prior_state_exp.unsqueeze(2)).view(b,-1)
        post_var_term = torch.matmul(self.params['A'], torch.matmul(prior_state_var, self.params['A'].transpose(1,0)))
        current_var_term = torch.matmul(self.params['W'], torch.matmul(self.params['U'], self.params['W'].transpose(1,0))).unsqueeze(0).repeat(b,1,1)
        prior_state_var = post_var_term + current_var_term
        estim_exp = self.params['b'].unsqueeze(0).repeat(b,1) + torch.matmul(self.params['B'], prior_state_exp.unsqueeze(2)).view(b,-1)
        estim_var = torch.matmul(self.params['B'], torch.matmul(prior_state_var, self.params['B'].transpose(1,0))) + self.params['V'].unsqueeze(0).repeat(b,1,1)

        return prior_state_exp, prior_state_var, estim_exp, estim_var

    def forecast_current(self, prior_state_exp, prior_state_var, time_horizon):

        b = prior_state_exp.size(0)

        prior_state_exps = [prior_state_exp]
        prior_state_vars = [prior_state_var]

        estim_exp = self.params['b'].unsqueeze(0).repeat(b,1) + torch.matmul(self.params['B'], prior_state_exp.unsqueeze(2)).view(b,-1)
        estim_var = torch.matmul(self.params['B'], torch.matmul(prior_state_var, self.params['B'].transpose(1,0))) + self.params['V'].unsqueeze(0).repeat(b,1,1)

        estim_exps = [estim_exp]
        estim_vars = [estim_var]

        for t in range(time_horizon-1):
            prior_state_exp, prior_state_var, estim_exp, estim_var = self.forecast_one_step(prior_state_exp, prior_state_var)

            prior_state_exps.append(prior_state_exp)
            prior_state_vars.append(prior_state_var)
            estim_exps.append(estim_exp)
            estim_vars.append(estim_var)

        return estim_exps, estim_vars

    def forecast(self, input, steps_ahead = 1, washout = 20, time_horizon = None):

        if time_horizon is None:
            time_horizon = input.size(1)

        _, _, prior_state_exp, prior_state_var = self.filter_sequence(input, washout)

        estim_exps = []
        estim_vars = []

        for t in range(washout, time_horizon - steps_ahead + 1):
            estim_exp, estim_var = self.forecast_current(prior_state_exp, prior_state_var, steps_ahead)
            estim_exps.append(estim_exp[-1])
            estim_vars.append(estim_var[-1])

            obs = input[:,t,:]
            _, _, _, _, prior_state_exp, prior_state_var = self.filter_one_step(obs, prior_state_exp, prior_state_var)

        return estim_exps, estim_vars

    






        
        

