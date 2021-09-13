import sys
import pandas as pd
import numpy as np

 #sys.path.insert(3,"../../../ViraTrace-Model/contagion_sim/")

from .template_rank import AbstractRanker
from continuous_fast import MultisimRankModel

class ViraTraceRanker(AbstractRanker):

    def __init__(self,n_nodes,n_sims, lamda, mu, p_seed, contacts_directed=True,tqdm=None):
        self.description = "Rank using ViraTrace simulations"
        self.author = """
        algorithm: github.com/ViraTrace
        implementation: Fabio Mazza
        """
        self.p_seed = p_seed
        self.p_infection = lamda
        self.mu = mu
        self.n_nodes = n_nodes
        self.n_sims = n_sims
        self.contacts_directed = contacts_directed

        self.model = None
        self.tqdm = tqdm

    def init(self, N, T):
        self.N = N
        self.n_days = T

        self.state_avg = np.zeros((3,T))

        self.model = MultisimRankModel(self.n_nodes,self.p_infection, self.mu,
                                       self.n_days, self.n_sims, self.contacts_directed, 
                                       tqdm=self.tqdm)
        
        
    def rank(self, day, daily_contacts, daily_obs, data):
        '''
        Order the individuals by the probability to be infected
        
        input
        ------
        t: int - 
            day of rank
        
        daily_contacts: list (i, j, t, value)
            list of daily contacts
        daily_obs: list (i, state, t)
            list of daily observations

        return
        ------
        list -- [(index, value), ...]
        '''
        contacts_df = pd.DataFrame([c[:2] for c in daily_contacts],columns=["a","b"])

        self.model.set_max_days(day)
        self.model.add_contacts_day(day,contacts_df)

        tested_I = set()
        tested_not_I = set()
        
        for (i,state,t_obs) in daily_obs:
            if t_obs != day-1:
                raise ValueError("Times not coinciding")
            if state == 1:
                tested_I.add(i)
            else:
                tested_not_I.add(i)
        
        #print(tested_I, tested_not_I)
                
        tested_positive = np.full(self.n_nodes,False)
        if len(tested_I) > 0:
            tested_positive[list(tested_I)] = True
        tested_negative = np.full(self.n_nodes,False)
        if len(tested_not_I) > 0:
            tested_negative[list(tested_not_I)] = True
        
        
        self.model.set_daily_observations(day-1, tested_negative, tested_positive)
        ## We never have observations for the current day,
        # Only for the previous day
        #self.model._propagate_susceptibles()
        
        #print("Got {} init tests".format(self.model.daily_positives[0].sum()))
        self.model.start_sim(self.p_seed)
        #self.model.start_sim(0.0)

        self.state_avg[0,day] = self.model.S.mean()
        self.state_avg[1,day] = self.model.I.mean()
        self.state_avg[2,day] = self.n_nodes - self.state_avg[0,day] - self.state_avg[1,day]
        
        #if day >= self.T-1:
        data["estim_S"] = self.state_avg[0]
        data["estim_I"] = self.state_avg[1]
        data["estim_R"] = self.state_avg[2]

        rank = pd.Series(self.model.I.sum(1))

        rank = (rank / rank.max()).sort_values()
        
        return list(rank.items())
