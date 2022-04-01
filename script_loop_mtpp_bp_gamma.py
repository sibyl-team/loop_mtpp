import sys
import os

sys.path.insert(0,'src/')
sys.path.insert(0, '../sib/')
sys.path.insert(0, '../simulator/sim/lib/') # we need distributions.py
sys.path.insert(0, '../epidemic_mitigation/src/')
simulator_path = "../simulator/sim/" ##change simulation path here 
sys.path.insert(0,simulator_path)

import random
from pathlib import Path
import numpy as np, pandas as pd
#, matplotlib.pyplot as plt
import json, log, logging, loop_mtpp
from importlib import reload
import imp
from math import exp
import pickle
from lib.mobilitysim import MobilitySimulator
import loop_mtpp
import rankers
import argparse
from distributions import CovidDistributions

parser = argparse.ArgumentParser(description="Run Loop MTPP")

parser.add_argument('--adp_frac', type=float, default=1.0, dest="af", help="adoption fraction of the app")
parser.add_argument('--fn_rate', type=float, default=0.0, dest="fnr", help="false negative rate")
parser.add_argument('-n', type=int, default=500, dest="obs", help="number of obs")
parser.add_argument('-s', type=int, default=0, dest="seed", help="seed")
parser.add_argument('-m', type=int, default=50, dest="seed_mob", help="seed mob")
parser.add_argument('-b', type=float, default=0.55, dest="beta", help="infection prob rate")
parser.add_argument('-i', type=int, default=7, dest="ti", help="waiting time before intervention")
parser.add_argument('--tau', type=int, default=7, dest="tau", help="tau > 0 use \sum_t in tau b[t]. tau = 0 use prob[x = I]")
args = parser.parse_args()

#logging
data_path = '../simulator/sim/lib/mobility/'

distr = CovidDistributions("GER")
output_dir = "output_Tubingen_pop1_site1/"
fold_out = Path(output_dir)
if not fold_out.exists():
    fold_out.mkdir(parents=True)

reload(log)
logger = log.setup_logger()

fnr = args.fnr
beta = args.beta # from paper
country = 'GER'
#with open(data_path + 'Isle_of_Wight_settings_pop10_site5.pk', 'rb') as fp:
with open(data_path + 'Tubingen_settings_pop1_site1.pk', 'rb') as fp:
    mob_kwargs = pickle.load(fp)

mob_kwargs["delta"] = 0.2554120904376099
T = 100
seed_mob = args.seed_mob
random.seed(seed_mob)
np.random.seed(seed_mob)
t_unit = 24
t_res = 0.25 # drop contacts with a duration < t_res (in hours)
max_time = T * t_unit  
mob = MobilitySimulator(**mob_kwargs)
mob.verbose = True
out = mob.simulate(max_time=max_time, seed=seed_mob)
#contacts_df = pd.DataFrame(contacts_cg(mob, t_res, t_unit, first_filter = True),columns = ['i','j','t','deltat'])
N = mob.num_people 
#print(N)
#contacts_df = pd.DataFrame(contacts_cg(mob, t_res, t_unit, first_filter = False),columns = ['i','j','t','deltat'])

#contacts_df = contacts_df.sort_values(by=["t","i","j"])

#n_indiv=np.ceil(mob_kwargs['num_people_unscaled']/mob_kwargs['downsample_pop'])
n_seeds = {'expo': 3, 'iasy':4,'ipre':5} # select initial infected seeds
num_test_random = 0 #number of random tests per day
fraction_sym_obs = 0.5 #fraction of Symptomatic tested positive
initial_steps = args.ti #starting time of intervention
delta_days = 1 # intervention every delta_days days (for the moment keep to 1)
#assert initial_steps % delta_days == 0
test_HH = False
quarantine_HH = True
adoption_fraction = args.af

import sib,  scipy

#from loop_ranker import sib_rank, sib_drop_rank, greedy_rank, dotd_rank, mean_field_rank, tracing_rank, winbp_rank, winbp_prob0_rank
from rankers import dotd_rank, greedy_rank, mean_field_rank, sib_rank
from tqdm.notebook import tqdm
from scipy.stats import gamma
os.environ['NUMEXPR_MAX_THREADS'] = '8'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
sib.set_num_threads(8)

#import matplotlib.pyplot as plt

mu = 1/12
prob_seed = 1/N
prob_sus = 0.5
pseed = prob_seed / (2 - prob_seed)
psus = prob_sus * (1 - pseed)
#if adoption_fraction < 1.0:
#    pautoinf = 1e-4
#else:
pautoinf = 1e-6
#pautoinf = 1e-6
fp_rate = 0.0
fn_rate = fnr

rankers = {}

k_rec_gamma = 62.484380808876004
scale_rec_gamma = 0.2992112296058585
t0 = distr.incubation_mean_of_lognormal - distr.median_infectious_without_symptom
alpha = 2.0
tau =  args.tau
rankers["BP_gamma"] = sib_rank.SibRanker(
                 params = sib.Params(
                                 #prob_i = sib.Uniform(1.0), 
                                 #prob_r = sib.Exponential(1/30), 
                                 #prob_i = sib.PriorDiscrete(list(scipy.special.expit(alpha*(range(T+1) -t0*np.ones(T+1))))), 
                                 #prob_r = sib.PriorDiscrete(list(scipy.stats.gamma.sf(range(T+1), k_rec_gamma, scale=scale_rec_gamma))),
                                 prob_i = sib.PiecewiseLinear(sib.RealParams(list(scipy.special.expit(alpha*(range(T+1) -t0*np.ones(T+1)))))), 
                                 prob_r = sib.PiecewiseLinear(sib.RealParams(list(scipy.stats.gamma.sf(range(T+1), k_rec_gamma, scale=scale_rec_gamma)))),
                                 pseed = pseed,
                                 psus = psus,
                                 fp_rate = fp_rate,
                                 fn_rate = fn_rate,
                                 pautoinf = pautoinf),
                 maxit0 = 20,
                 maxit1 = 20,
                 tol = 1e-3,
                 memory_decay = 1e-5,
                 window_length = 21,
                 tau=tau,
                 fnr=fn_rate,
                 fpr=fp_rate,
)

ress = {}
num_test_algo = args.obs
print("Obs: ", num_test_algo, "n_seeds: ", n_seeds)
for s in list(rankers.keys()):
    data = {"algo":s}
    if s== "no_intervention":
        res_s = loop_mtpp.free_mtpp(mob,
        country  = country,
        beta = beta,
        T = T,
        seed=args.seed,
        logger = logging.getLogger(f"iteration.{s}"),
        data = data,
        initial_counts = n_seeds,
        name_file_res = s + f"_N_{N}_T_{T}_obs_{num_test_algo}_sym_obs_{fraction_sym_obs}_seed_{seed_mob}",

        output_dir = output_dir,
        )
    else:
        res_s = loop_mtpp.loop_mtpp(mob,
        rankers[s],
        country  = country,
        T = T,
        seed=args.seed,
        logger = logging.getLogger(f"iteration.{s}"),
        data = data,
        initial_steps = initial_steps, 
        num_test_random = num_test_random,
        num_test_algo = num_test_algo,
        fraction_sym_obs = fraction_sym_obs,
        initial_counts = n_seeds,
        beta = beta,
        fn_rate = fnr,
        test_HH = test_HH,
        quarantine_HH = quarantine_HH,
        name_file_res = s + f"_N_{N}_T_{T}_obs_{num_test_algo}_ti_{initial_steps}_sym_obs_{fraction_sym_obs}_af_{adoption_fraction}_fnr_{fnr}_seed_{seed_mob}_tau_{tau}",
        output_dir = output_dir,
        save_every_iter = 1,
        adoption_fraction = adoption_fraction
        )
    ress[s] = res_s    
    del res_s



