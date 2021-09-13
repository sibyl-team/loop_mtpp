import sib
import numpy as np
from .template_rank import AbstractRanker

def compact_printer(t,err,f):
    print(f"{t} {err:1.3e}      ",end="")
    if err < 1e-1:
        print("\r",end="")
    else:
        print("\n",end="")

class SibRanker(AbstractRanker):
    def __init__(self,
                 params = sib.Params(),
                 maxit0 = 10,
                 maxit1 = 10,
                 damp0 = 0,
                 damp1 = 0.5,
                 tol = 1e-3,
                 err_weight = 0,
                 print_callback = lambda t,e,f: print(t,e),
                 array_obs_symp = None
                ):
        self.description = "class for BP inference of openABM loop"
        self.authors = "Indaco Biazzo, Alessandro Ingrosso, Alfredo Braunstein"
        self.params = params
        self.maxit0 = maxit0
        self.maxit1 = maxit1
        self.damp0 = damp0
        self.damp1 = damp1
        self.tol = tol
        self.err_weight = err_weight
        self.array_obs_symp = array_obs_symp
        self.print_callback = print_callback

    def init(self, N, T):
        self.T = T
        self.N = N
        f = sib.FactorGraph(params=self.params,  observations=[(i,-1,0) for i in range(self.N)])
        self.f = f
        self.contacts = []
        self.obs = []
        self.bpIs = np.full(T, np.nan)
        self.bpRs = np.full(T, np.nan)
        self.bpseeds = np.full(T, np.nan)
        self.lls = np.full(T, np.nan)


    def rank(self, t_day, daily_contacts, daily_obs, data):

                #print(daily_obs)
        for obs in daily_obs:
            self.obs.append(obs)

        for obs in daily_obs:
            self.f.append_observation(obs[0],obs[1], obs[2])
            if self.array_obs_symp is not None:
                if obs[1] == 1:
                    node = self.f.nodes[obs[0]]
                    t_obs = obs[2]
                        #print("t_obs", t_obs, list(node.times).index(t_obs))
                    for t_indx in range(list(node.times).index(t_obs)):
                        node.ht[t_indx] *= self.array_obs_symp[int(t_obs - node.times[t_indx])]
                            #print(t_indx, node.ht[t_indx], t_obs, self.func_obs_symp(t_obs - node.times[t_indx]))
                    #print(node.ht, t_obs)
        ### add fake obs
        for i in range(self.N):
            self.f.append_observation(i,-1,t_day)

        for c in daily_contacts:
            self.f.append_contact(*c)

        sib.iterate(self.f, maxit=self.maxit0, damping=self.damp0, tol=self.tol, callback=self.print_callback)
        sib.iterate(self.f, maxit=self.maxit1, damping=self.damp1, tol=self.tol, callback=self.print_callback)

        marg = np.array([sib.marginal_t(n, t_day) for n in self.f.nodes])
        bpS, bpI, bpR = sum(m[0] for m in marg), sum(m[1] for m in marg), sum(m[2] for m in marg)
        nseed = sum(n.bt[0] for n in self.f.nodes)
        ll = self.f.loglikelihood()

        data["logger"].info(f"BP: (S,I,R): ({bpS:.1f}, {bpI:.1f}, {bpR:.1f}), seeds: {nseed:.1f}, ll: {ll:.1f}")

        self.bpIs[t_day] = bpI
        self.bpRs[t_day] = bpR
        self.bpseeds[t_day] = nseed
        self.lls[t_day] = ll

        data["<I>"] = self.bpIs
        data["<IR>"] = self.bpIs + self.bpRs
        data["<seeds>"] = self.bpseeds
        data["lls"] = self.lls

        inf_prob = [[i, m[1]+self.err_weight*e] for (i,(m,e)) in enumerate(zip(marg,[n.err for n in self.f.nodes]))]
        rank = list(sorted(inf_prob, key=lambda tup: tup[1], reverse=True))
        return rank

