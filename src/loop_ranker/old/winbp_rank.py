import sib
import numpy as np
from .template_rank import AbstractRanker



class WinBPRanker(AbstractRanker):
    def __init__(self,
                params = sib.Params(),
                window_length = 14,
                maxit0 = 10,
                maxit1 = 10,
                damp0 = 0,
                damp1 = 0.5,
                tol = 1e-3,
                with_prior = False,
                print_callback = lambda t,err,f: print(t,err)
                ):
        self.description = "class for BP inference of openABM loop"
        self.authors = "Indaco Biazzo, Alessandro Ingrosso, Alfredo Braunstein"
        self.params = params
        self.window_length = window_length
        self.maxit0 = maxit0
        self.maxit1 = maxit1
        self.damp0 = damp0
        self.damp1 = damp1
        self.tol = 1e-3
        self.window_length = window_length
        self.print_callback = print_callback
        self.with_prior = with_prior


    def init(self, N, T):
        self.T = T
        self.N = N
        self.f = sib.FactorGraph(params=self.params)
        self.contacts = []
        self.bi = np.zeros((T + 2, N))
        self.br = np.zeros((T + 2, N))
        self.bpSs = np.full(T, np.nan)
        self.bpIs = np.full(T, np.nan)
        self.bpRs = np.full(T, np.nan)
        self.bpseeds = np.full(T, np.nan)
        self.lls = np.full(T, np.nan)
        self.all_obs = [[] for t in range(T + 1)]


    def rank(self, t_day, daily_contacts, daily_obs, data):

        for obs in daily_obs:
            self.f.append_observation(obs[0], obs[1], obs[2])
            self.all_obs[obs[2]] += [obs]

        ### add fake obs
        for i in range(self.N):
            self.f.append_observation(i,-1,t_day)

        for c in daily_contacts:
            self.f.append_contact(*c)

        if t_day >= self.window_length:

            t_start = t_day - self.window_length
            print("...drop first time and reset observations")
            self.f.drop_time(t_start)
            self.f.reset_observations(sum(self.all_obs[t_start + 1:], []))

            if self.with_prior:
                mi = sum(self.bi[t_start,:])
                mr = sum(self.br[t_start + 1,:])
                print(f"...applying prior ({self.N-mr-mi:.2f},{mi:.2f},{mr:.2f})")

                self.f.params.pseed = 1/3
                self.f.params.psus = 1/3
                for i in range(self.N):
                    self.f.nodes[i].ht[0] *= self.bi[t_start,     i]
                    self.f.nodes[i].hg[0] *= self.br[t_start + 1, i]
            for i in range(self.N):
                self.f.nodes[i].ht[0] = max(self.f.nodes[i].ht[0], 1e-5)
                self.f.nodes[i].hg[0] = max(self.f.nodes[i].hg[0], 1e-5)


        print(f"sib.iterate(maxit={self.maxit0}, tol={self.tol}, damping={self.damp0}):")
        sib.iterate(self.f, maxit=self.maxit0, damping=self.damp0, tol=self.tol, callback=self.print_callback)
        print(f"\nsib.iterate(maxit={self.maxit1}, tol={self.tol}, damping={self.damp1}):")
        sib.iterate(self.f, maxit=self.maxit1, damping=self.damp1, tol=self.tol, callback=self.print_callback)
        print()


        marg = np.array([sib.marginal_t(n, t_day) for n in self.f.nodes])

        for i in range(self.N):
            self.bi[t_day, i] = marg[i][1]
            self.br[t_day, i] = marg[i][2]

        bpS, bpI, bpR = sum(m[0] for m in marg), sum(m[1] for m in marg), sum(m[2] for m in marg)
        nseed = sum(n.bt[0] for n in self.f.nodes)
        ll = self.f.loglikelihood()

        data["logger"].info(f"winBP: (S,I,R): ({bpS:.1f}, {bpI:.1f}, {bpR:.1f}), seeds: {nseed:.1f}, ll: {ll:.1f}")

        self.bpSs[t_day] = bpS
        self.bpIs[t_day] = bpI
        self.bpRs[t_day] = bpR
        self.bpseeds[t_day] = nseed
        self.lls[t_day] = ll

        data["<I>"] = self.bpIs
        data["<IR>"] = self.bpRs + self.bpIs
        data["<seeds>"] = self.bpseeds
        data["lls"] = self.lls

        inf_prob = [[i, marg[i][1]] for i in range(self.N)]
        #inf_prob = [[i, 1-marg[i][0]] for i in range(self.N)]
        rank = list(sorted(inf_prob, key=lambda tup: tup[1], reverse=True))
        return rank

