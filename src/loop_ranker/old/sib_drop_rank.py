import sib
import numpy as np
from .template_rank import AbstractRanker


def flatten_contacts(contact_window):
    return [c for contacts in contact_window for c in contacts]


def set_h(bnew, bold):
    assert(bnew>=0)
    if bnew == 0:
        assert(bold == 0)
        return 1
    if bnew > 0:
        return bold / bnew


class SibDropRanker(AbstractRanker):
    def __init__(self,
                prob_i = None,
                prob_r = None,
                prob_seed = 0.1,
                prob_sus = 0.5,
                window_length = 14,
                maxit_sc = 5,
                maxit_bp = 30,
                damping_sc = 0.05,
                damping_bp = 0.,
                tol_sc = 1e-3,
                pautoinf = 1e-10,
                maxit0 = 10,
                maxit1 = 10,
                damp0 = 0,
                damp1 = 0.5,
                tol = 1e-3,
                print_callback = False
                ):
        self.description = "class for BP inference of openABM loop"
        self.authors = "Indaco Biazzo, Alessandro Ingrosso, Alfredo Braunstein"
        self.params = sib.Params(prob_i=prob_i, prob_r=prob_r, pautoinf=pautoinf)
        self.params.pseed = prob_seed / (2 - prob_seed)
        self.params.psus = prob_sus * (1 - self.params.pseed)
        self.window_length = window_length
        self.maxit_sc = maxit_sc
        self.maxit_bp = maxit_bp
        self.damping_sc = damping_sc
        self.damping_bp = damping_bp
        self.tol_sc = tol_sc
        self.pautoinf = pautoinf
        self.maxit0 = maxit0
        self.maxit1 = maxit1
        self.damp0 = damp0
        self.damp1 = damp1
        self.tol = 1e-3
        self.print_callback = print_callback


    def init(self, N, T):
        self.T = T
        self.N = N
        self.f = sib.FactorGraph(params=self.params)
        self.contacts = []
        self.hts = np.zeros((N, self.T + 2))
        self.hgs = np.zeros((N, self.T + 2))
        self.bpSs = np.full(T, np.nan)
        self.bpIs = np.full(T, np.nan)
        self.bpRs = np.full(T, np.nan)
        self.bpseeds = np.full(T, np.nan)
        self.lls = np.full(T, np.nan)


    def rank(self, t_day, daily_contacts, daily_obs, data):

        for obs in daily_obs:
            assert(obs[2]==t_day-1)
            self.f.append_observation(obs[0], obs[1], obs[2])

        ### add fake obs
        for i in range(self.N):
            self.f.append_observation(i,-1,t_day)

        for c in daily_contacts:
            # undirected contacts
            if c[0] < c[1]:
                self.f.append_contact(c[0], c[1], c[2], c[3], c[3])

        sib.iterate(self.f, maxit=self.maxit0, damping=self.damp0, tol=self.tol)
        sib.iterate(self.f, maxit=self.maxit1, damping=self.damp1, tol=self.tol)

        if t_day >= self.window_length:

            delta_t = t_day - self.window_length

            print("...drop contacts and set self consistent fields")

            # store beliefs
            #bts_target = []
            #bgs_target = []
            #for i in range(self.N):
            #    bts_target.append(list(self.f.nodes[i].bt))
            #    bgs_target.append(list(self.f.nodes[i].bg))

            # drop contact
            #self.f.drop_contacts(delta_t)

            # set field self-consistently
            #self.set_sc_fields(t_day, delta_t, bts_target, bgs_target)
            (it, err_bp, err_sc) = self.f.drop_sc(delta_t, 
                                       maxit_bp = self.maxit_bp,
                                       tol_bp = self.tol,
                                       damping_bp = self.damping_bp,
                                       maxit_sc = self.maxit_sc, 
                                       tol_sc = self.tol_sc,
                                       damping_sc = self.damping_sc)

            print(f"drop_sc: {it} {err_bp} {err_sc}")
        marg = np.array([sib.marginal_t(n,t_day) for n in self.f.nodes])
        bpS, bpI, bpR = sum(m[0] for m in marg), sum(m[1] for m in marg), sum(m[2] for m in marg)
        nseed = sum(n.bt[0] for n in self.f.nodes)
        ll = self.f.loglikelihood()

        data["logger"].info(f"BP_res: (S,I,R): ({bpS:.1f}, {bpI:.1f}, {bpR:.1f}), seeds: {nseed:.1f}, ll: {ll:.1f}")

        self.bpSs[t_day] = bpS
        self.bpIs[t_day] = bpI
        self.bpRs[t_day] = bpR
        self.bpseeds[t_day] = nseed
        self.lls[t_day] = ll

        data["<I>"] = self.bpIs
        data["<IR>"] = self.bpRs + self.bpIs
        data["<seeds>"] = self.bpseeds
        data["lls"] = self.lls

        inf_prob = [[i, m[1]] for (i,m) in enumerate(marg)]
        rank = list(sorted(inf_prob, key=lambda tup: tup[1], reverse=True))
        return rank

    def set_sc_fields(self, t, delta_t, bts_target, bgs_target):

        for ep in range(self.maxit_sc):
            print(f"...iter sc: {ep}")

            # iterate BP
            sib.iterate(self.f, maxit=self.maxit_bp, damping=self.damping_bp)

            # compute error
            err = 0.0
            for i in range(self.N):
                for ti in range(delta_t, t + 1 + 2):
                    a = len(bts_target[i])
                    b = len(self.f.nodes[i].bt)
                    if a != b:
                        print(i, a, b)
                    a = len(bgs_target[i])
                    b = len(self.f.nodes[i].bg)
                    if a != b:
                        print(i, a, b)
                    if ti >= len(bgs_target[i]):
                        print(delta_t)
                        print(len(bgs_target[i]))
                        print(ti)
                        print(i)
                        bts_target[i][ti]
                        self.f.nodes[i].bt[ti]
                    err += np.abs(bts_target[i][ti] - self.f.nodes[i].bt[ti])
                    err += np.abs(bgs_target[i][ti] - self.f.nodes[i].bg[ti])
            err /= self.N * (self.window_length + 1 + 2)
            print(f"\n\tERR: {err}\n")

            if err < self.tol_sc:
                print("...self consistent")
                break

            # adjust fields
            for i in range(self.N):
                ht = self.f.nodes[i].ht
                bt = self.f.nodes[i].bt
                hg = self.f.nodes[i].hg
                bg = self.f.nodes[i].bg
                for ti in range(t + 1 + 2):
                    if bt[ti]<0:
                        print(i, ti, bt[ti])
                        bt[ti] = 0
                    if bg[ti]<0:
                        print(i, ti, bg[ti])
                        bg[ti] = 0
                    ht[ti] *= set_h(bt[ti], bts_target[i][ti])**self.damping_sc
                    hg[ti] *= set_h(bg[ti], bgs_target[i][ti])**self.damping_sc
