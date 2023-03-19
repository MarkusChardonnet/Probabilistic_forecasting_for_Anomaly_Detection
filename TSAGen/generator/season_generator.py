from generator.abstract_generator import AbstractSeasonGenerator
import matplotlib.pyplot as plt
import numpy as np

class NormalSeasonGenerator(AbstractSeasonGenerator):
    def _inject(self):
        # do nothing
        pass

class SeasonGeneratorWithShapeDeformation(AbstractSeasonGenerator):
    def __init__(self, cycle_num, amplitude, cycle_len, drift_a, drift_f, forking_depth, depth, anomaly_params):
        super(SeasonGeneratorWithShapeDeformation, self).__init__(
            cycle_num=cycle_num, amplitude=amplitude, cycle_len=cycle_len, drift_a=drift_a, drift_f=drift_f, 
            forking_depth=forking_depth, depth=depth
        )
        self.anomaly_params = anomaly_params

    def _inject(self):
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
        self.fork_depth_law = self.anomaly_params['forking_depth_law']
        if self.fork_depth_law == 'delta':
            self.ad_forking_depth = self.anomaly_params['forking_depth']
        for pos in pos_list:
            amplitude = self.amplitude
            drift_a = self.drift_a_for_every_cycle[pos]
            length_d = self.drift_f_for_every_cycle[pos]
            anomaly_cycle = drift_a*amplitude*self.cycle_generator.gen(self.ad_forking_depth, int(length_d*self.cycle_len))
            self.cycle_list[pos] = anomaly_cycle
            self.label_list[pos] = np.ones(len(anomaly_cycle),dtype=np.int)

class SeasonGeneratorWithCycleVanish(AbstractSeasonGenerator):
    def _inject(self):
        pos_list = [5]
        forking_depth = 9
        for pos in pos_list:
            amplitude = self.drift_a_for_every_cycle[pos]
            length_d = self.drift_f_for_every_cycle[pos]
            # anomaly_cycle = amplitude*self.cycle_generator.gen(forking_depth, int(length_d*self.cycle_len))
            anomaly_cycle = np.zeros(int(length_d*self.cycle_len))
            self.cycle_list[pos] = anomaly_cycle
            self.label_list[pos] = np.ones(len(anomaly_cycle),dtype=np.int)
            
class example_season_generator(AbstractSeasonGenerator):
    def _inject(self):
        pos_list = [5,6]
        forking_depth = 10
        for pos in pos_list:
            amplitude = self.drift_a_for_every_cycle[pos]
            length_d = self.drift_f_for_every_cycle[pos]
            anomaly_cycle = amplitude*self.cycle_generator.gen(forking_depth, int(length_d*self.cycle_len))
            self.cycle_list[pos] = anomaly_cycle
            self.label_list[pos] = np.ones(len(anomaly_cycle),dtype=np.int)


class SeasonGeneratorFactory():

    def __init__(self):
        '''self.cycle_num = cycle_num
        self.amplitude = amplitude
        self.cycle_len = cycle_len
        self.drift_a = drift_a
        self.drift_f = drift_f
        self.forking_depth = forking_depth
        self.depth = depth'''
        
    def get_generator(self,cycle_num, amplitude, cycle_len, drift_a=0, drift_f=0, forking_depth=6, depth=10, anomaly_type=None, anomaly_params = None):
        if anomaly_type is None:
            return AbstractSeasonGenerator(cycle_num=cycle_num, amplitude=amplitude, cycle_len=cycle_len, 
                                           drift_a=drift_a, drift_f=drift_f, forking_depth=forking_depth, depth=depth)
        elif anomaly_type == 'deformation':
            return SeasonGeneratorWithShapeDeformation(cycle_num=cycle_num, amplitude=amplitude, cycle_len=cycle_len, 
                                                       drift_a=drift_a, drift_f=drift_f, forking_depth=forking_depth, depth=depth, anomaly_params=anomaly_params)
        #elif anomaly_type == 'vanish':
        #    return SeasonGeneratorWithCycleVanish(self.cycle_num,self.amplitude,self.cycle_len,self.drift_a,self.drift_f,self.forking_depth,self.depth)