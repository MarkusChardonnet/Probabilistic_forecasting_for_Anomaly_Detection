#!python3.6

# Created by Chengyu on 2020/5/14。
# Noise Generator.

import numpy as np
# import matlab
# import matlab.engine

class Gaussian():
    def __init__(self, mu=0, sigma=1, skew=0, kurt=3):
        self.noise = []
        self.label = []
        self.mu = mu
        self.sigma = sigma
        pass
    
    def _inject(self):
        pass

    def gen(self, size=1):
        self.noise = np.random.normal(self.mu, self.sigma, size)
        self.label = np.zeros(size, dtype=np.int)
        self._inject()
        return (self.noise, self.label)

class GaussianWithVarianceAnomalies(Gaussian):
    def __init__(self, mu=0, sigma=1, skew=0, kurt=3, anomaly_params = None):
        super(GaussianWithVarianceAnomalies, self).__init__(
            mu=mu, sigma=sigma
        )
        self.anomaly_params = anomaly_params

    def _inject(self):
        pos_list = []
        alen_list = []
        fvar_list = []
        self.occ_law = self.anomaly_params['occurence_law']
        if self.occ_law == 'single':
            prob = self.anomaly_params['occurence_prob']
            real = np.random.binomial(1, prob, 1)
            if real == 1:
                self.occ_range = self.anomaly_params['occurence_range']
                self.occ_pos_law = self.anomaly_params['occurence_pos_law']
                if self.occ_pos_law == 'uniform':
                    pos = np.random.uniform(self.occ_range[0],self.occ_range[1],1)
                    pos_list.append(pos)
        self.len_law = self.anomaly_params['length_law']
        for pos in pos_list:
            if self.len_law == 'uniform':
                self.len_range = self.anomaly_params['length_range']
                alen = int(np.random.uniform(self.len_range[0],self.len_range[1],1))
                alen_list.append(alen)
        self.var_law = self.anomaly_params['variance_factor_law']
        for pos in pos_list:
            if self.var_law == 'uniform':
                self.var_range = self.anomaly_params['variance_factor_range']
                fvar = np.random.uniform(self.var_range[0],self.var_range[1],1)
                fvar_list.append(fvar)
        for i,pos in enumerate(pos_list):
            alen = alen_list[i]
            fvar = fvar_list[i]
            position = int(pos*len(self.noise))
            a_segment = np.random.normal(self.mu, self.sigma*fvar, alen)
            self.noise[position:position+alen] = a_segment
            self.label[position:position+alen] = np.ones(len(a_segment),dtype=np.int)

"""
print("starting matlab.")
engine = matlab.engine.start_matlab()

class Pearson:
    def __init__(self):
        # print("starting matlab.")
        self.engine = engine # Start MATLAB process
        # engine = matlab.engine.start_matlab("-desktop") # Start MATLAB process with graphic UI

        self.noise = []
        self.label = []

    def gen(self,mu,sigma,skew,kurt,size):
        
        self.mu = mu
        self.sigma = sigma
        self.skew = skew
        self.kurt = kurt

        self.noise = self.engine.pearsrnd(matlab.double([mu]),
                                    matlab.double([sigma]),
                                    matlab.double([skew]),
                                    matlab.double([kurt]),
                                    matlab.double([1]),
                                    matlab.double([size]))[0]
                                    
        self.label = np.zeros(size, dtype=np.int)
        self._inject()
        return (np.array(self.noise), np.array(self.label))
    
    def _inject(self):
        pass

class PearsonWithChangePoints(Pearson):
    def _inject(self):
        pos_list = [0.5,0.8]
        a_len = 20
        for pos in pos_list:
            position = int(pos*len(self.noise))
            a_segment = np.random.normal(self.mu, self.sigma*10, a_len)
            self.noise[position:position+a_len] = a_segment
            self.label[position:position+a_len] = np.ones(len(a_segment),dtype=np.int)
"""

class NoiseGeneratorFactory():

    def __init__(self):
        pass

    def get_generator(self, mu=0, sigma=1, skew = 0, kurt = 3, noise_type="gaussian", anomaly_type=None, anomaly_params = None):
        if anomaly_type is None:
            if noise_type == "gaussian":
                return Gaussian(mu=mu, sigma=sigma)
            #elif noise_type == "pearson":
            #    return Pearson()
        elif anomaly_type == 'variance':
            if noise_type == "gaussian":
                return GaussianWithVarianceAnomalies(mu=mu, sigma=sigma, anomaly_params=anomaly_params)
            #elif noise_type == "pearson":
            #    return PearsonWithChangePoints()