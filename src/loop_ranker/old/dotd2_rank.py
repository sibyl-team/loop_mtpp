import numpy as np
from .template_rank import AbstractRanker

class DotdRanker(AbstractRanker):

    def __init__(self):
        self.description = "class for random tests of openABM loop"
        self.rng = np.random.RandomState(1)

    def init(self, N, T):
        self.T = T
        self.N = N

        return True

    def rank(self, t_day, daily_contacts, daily_obs, data):
        '''
        computing rank of infected individuals
        return: list -- [(index, value), ...]
        '''
#        rank = [[i,self.rng.random()] for i in range(self.N)]
	rank = [[i,1.0] for i in range(self.N)]
        return rank
