from pathlib import Path
import sys 

import json
import random
import pandas as pd
import numpy as np
import networkx as nx
import copy
import scipy as sp
import math
import seaborn
import pickle
import warnings
import os
from os.path import join

from lib.distributions import CovidDistributions
#from lib.slim_dynamics import get_seeds, run, is_state_at
from lib.slim_dynamics import DiseaseModel
#from lib.summary import is_state_at

from mtpp_utils import contacts_cg, get_households_contacts
from analysis_utils import ranker_I, check_fn_I, ranker_IR, check_fn_IR, roc_curve, events_list
#from abm_utils import status_to_state, listofhouses, dummy_logger, quarantine_households
from mtpp_utils import dummy_logger

status_legend = {'susc': 0, 'expo': 10, 'ipre': 11, 'isym': 12, 'iasy': 13,  'hosp':19, 'resi': 20, 'dead': 21} #,'quar':30}
status_legend_inv = {0:'susc', 10: 'expo', 11: 'ipre', 12: 'isym', 13: 'iasy', 19:'hosp', 20:'resi', 21: 'dead'} #,30: 'quar'}

TEST_LAG = 0

def is_state_at(sim,state, t):
    #if state == 'posi' or state == 'nega':
    #    return (sim.state_started_at[state] - TEST_LAG <= t) & (sim.state_ended_at[state] - TEST_LAG > t)
    #else:
    return (sim.state_started_at[state] <= t) & (sim.state_ended_at[state] > t)

def get_status(sim_state,t,N):
    #return np.sum(np.stack([status_legend[k]*is_state_at(sim_state,k,t) for k in status_legend.keys()]),axis = 0)
    s_dict = {k : list(np.where(is_state_at(sim_state,k,t) == True)[0]) for k in status_legend.keys()}
    s_vec = np.zeros(N,dtype = int)
    for k in s_dict.keys():
        s_vec[s_dict[k]] = status_legend[k]
    return s_dict,s_vec

def get_seeds(num,initial_counts):
    # init state variables with seeds
    total_seeds = sum(v for v in initial_counts.values())
    initial_people = np.random.choice(num, size=total_seeds, replace=False)

    ptr = 0
    initial_seeds = dict()
    for k, v in initial_counts.items():
        initial_seeds[k] = initial_people[ptr:ptr + v].tolist()
        ptr += v 
    return initial_seeds



def status_to_state_(current_status, hosp_as_rec = False):
    return (current_status+hosp_as_rec) // 10

status_to_state = np.vectorize(status_to_state_)

def loop_mtpp(mob,
             contacts_df,
             inference_algo,
             T = 30, #days of simulation
             beta = 0.55,
             country = 'GBR',
             initial_counts = {'expo': 5, 'isym': 5 , 'iasy': 5}, #in the form {'expo': 7, 'isym_posi': 2, 'iasy': 2},
             logger = dummy_logger(),
             seed=1,
             initial_steps = 0,
             num_test_random = 50,
             num_test_algo = 50,
             fraction_sym_obs = 0.4,
             t_unit = 24,
             delta_days = 1, # quarantine every tot days
             t_res = 0.25, #in hours,
             test_HH = False,
             quarantine_HH = True,
             contact_duration_households = 3, # hours
             name_file_res = "res",
             output_dir = "./output_mtpp/",
             save_every_iter = 5,
             stop_zero_I = True,
             adoption_fraction = 1.0,
             fp_rate = 0.0,
             fn_rate = 0.0,
             callback = lambda x : None,
             data = {}
            ):
    '''
    Simulate interventions strategy on the MTPP epidemic simulation.

    input
    -----
    mob_kwargs: Dict
            Dictonary with settings of mobility simulations (town, queries, downsampling, sites/houses location and simulation params)
    inference_algo: Class (rank_template)
            Class for order the nodes according to the prob to be infected
            logger = logger for printing intermediate steps
    results:
        print on file true configurations and transmission
    '''
    
    ### create output_dir if missing
    fold_out = Path(output_dir)
    if not fold_out.exists():
        fold_out.mkdir(parents=True)

    ### initialize a separate random stream
    rng = np.random.RandomState()
    rng.seed(seed)
    np.random.seed(seed)
    #quarantine_HH = True # at the moment, this is the only possible option. solved ?         
    
    N = mob.num_people
    
    distributions = CovidDistributions(country=country)
    intensity_params = {'betas' : {'education': beta,'social': beta,'bus_stop': beta,'office': beta,'supermarket': beta}, 'beta_household' : beta }
    initial_seeds=get_seeds(N, initial_counts)
    print("Starting with guys: ",initial_seeds)
    house = mob.people_household
    housedict = mob.households
    has_app = (rng.random(N) <= adoption_fraction)
    
    
    contacts_df = contacts_df.append(get_households_contacts(housedict,T,hh_deltat = contact_duration_households))
    contacts_df['lambda'] = 1-np.exp(-beta*contacts_df['deltat'].to_numpy()/t_unit)
    
    ### init data and data_states
    data_states = {}
    data_states["true_conf"] = np.zeros((T,N))
    data_states["statuses"] = np.zeros((T,N))
    data_states["tested_algo"] = []
    data_states["tested_random"] = []
    data_states["tested_sym"] = []
    for col_name in ["num_quarantined", "q_sym", "q_algo", "q_random", "q_all", "infected_free", "S", "I", "R", "IR", "aurI", "prec1%", "prec5%"]:
        data[col_name] = np.full(T,np.nan)
    data["logger"] = logger

    ### init inference algo
    inference_algo.init(N, T)
    
    ### running variables
    indices = np.arange(N, dtype=int)
    excluded = np.zeros(N, dtype=bool)
    daily_obs = []
    all_obs = []
    all_quarantined = []
    freebirds = 0
    num_quarantined = 0

    status_dict = initial_seeds
    delta_th = delta_days * t_unit # quarantine every tot days


    sim = DiseaseModel(mob, distributions)
    sim.init_all(intensity_params, initial_seeds)
    
    t = 0
    sim.t0 = 0
    for t in range(T):
        th = t*t_unit
        ### advance one time step
        
        sim.run_one_step_dyn(th + delta_th, excluded)
        status_dict, status = get_status(sim,th+delta_th,N)

        state = status_to_state(status) ## 0, 1, 2, 3 (quar)
        data_states["true_conf"][t] = state
        data_states["statuses"][t] = status
        nS, nI, nR = (state == 0).sum(), (state == 1).sum(), (state == 2).sum()
        if nI == 0 and stop_zero_I:
            logger.info("stopping simulation as there are no more infected individuals")
            break
        if t == initial_steps:
            logger.info("\nobservation-based inference algorithm starts now\n")
        logger.info(f'time:{t}')
        ### extract contacts
        daily_contacts = contacts_df[['i','j','t','lambda']][contacts_df['t'] == t].to_records(index = False)
        #daily_contacts = [x for x in daily_contacts if status[x[0]]!=status_legend['quar'] or status[x[1]]!=status_legend['quar']] # drop contacts of quarantined nodes
        
        # TO DO: figure out what to do with hospitalised nodes and
        logger.info(f"number of unique contacts: {len(daily_contacts)}")
        
        ### compute potential test results for all
        if fp_rate or fn_rate:
            noise = rng.random(N)
            f_state = (state==1)*(noise > fn_rate) + (state==0)*(noise < fp_rate) + 2*(state==2)
        else:
            f_state = state
        
        to_quarantine = []
        all_test = []
        excluded_now = excluded.copy()
         
       
        # add hospitalized people to the excluded nodes
        for q in status_dict['hosp']:
            excluded_now[q] = True
            excluded[q] = True

        def test_and_quarantine(rank, num):
            nonlocal to_quarantine, excluded_now, all_test, all_quarantined
            test_rank = []
            for i in rank:
                if len(test_rank) == num:
                    break;
                if excluded_now[i]:
                    continue
                test_rank += [i]
                if f_state[i] == 1:
                    q = housedict[house[i]] if quarantine_HH else [i]
                    excluded_now[q] = True
                    all_quarantined += q
                    to_quarantine += q
                    excluded[q] = True
                    if test_HH:
                        tests += q
                    all_test += q
                else:
                    excluded_now[i] = True
                    all_test += [i]
            return test_rank
                
        
        ### compute rank from algorithm
        num_test_algo_today = num_test_algo
        if t < initial_steps:
            daily_obs = []
            num_test_algo_today = 0            
        
        weighted_contacts = [c for c in daily_contacts if (has_app[c[0]] and has_app[c[1]]) and (c[0] not in all_quarantined and c[1] not in all_quarantined)]
        rank_algo = inference_algo.rank(t, weighted_contacts, daily_obs, data)
        rank = np.array(sorted(rank_algo, key= lambda tup: tup[1], reverse=True))
        rank = [int(tup[0]) for tup in rank]
        
        ### test num_test_algo_today individuals
        test_algo = test_and_quarantine(rank, num_test_algo_today)

        ### compute roc now, only excluding past tests
        eventsI = events_list(t, [(i,1,t) for (i,tf) in enumerate(excluded) if tf], data_states["true_conf"], check_fn = check_fn_I)
        xI, yI, aurI, sortlI = roc_curve(dict(rank_algo), eventsI, lambda x: x)
        
        ### test a fraction of sym
        sym = indices[status == status_legend['isym']]
        sym = test_and_quarantine(rng.permutation(sym), int(len(sym) * fraction_sym_obs))

        ### do num_test_random extra random tests
        test_random = test_and_quarantine(rng.permutation(N), num_test_random)

        ### quarantine infected individuals
        num_quarantined += len(to_quarantine)
        #covid19.intervention_quarantine_list(model.model.c_model, to_quarantine, T+1)
            
        ### update observations
        for i in all_test:
            if f_state[i]==3:
                print("ERROR-> adding observation of a quarantined node")
                
        daily_obs = [(int(i), int(f_state[i]), int(t)) for i in all_test]
        all_obs += daily_obs

        ### exclude forever nodes that are observed recovered
        rec = [i[0] for i in daily_obs if f_state[i[0]] == 2]
        excluded[rec] = True

        ### update data 
        data_states["tested_algo"].append(test_algo)
        data_states["tested_random"].append(test_random)
        data_states["tested_sym"].append(sym)
        data_states["statuses"][t] = status
        data["S"][t] = nS
        data["I"][t] = nI
        data["R"][t] = nR
        data["IR"][t] = nR+nI
        data["aurI"][t] = aurI
        prec = lambda f: yI[int(f/100*len(yI))]/int(f/100*len(yI)) if len(yI) else np.nan
        ninfq = sum(state[to_quarantine]>0)
        nfree = int(nI - sum(excluded[state == 1]))
        data["aurI"][t] = aurI
        data["prec1%"][t] = prec(1)
        data["prec5%"][t] = prec(5)
        data["num_quarantined"][t] = num_quarantined
        data["q_sym"][t] = len(sym)
        inf_test_algo = sum(state[test_algo]==1)
        sus_test_algo = sum(state[test_algo]==0)
        rec_test_algo = sum(state[test_algo]==2)
        inf_test_random = sum(state[test_random]==1)
        data["q_algo"][t] = inf_test_algo
        data["q_random"][t] = sum(state[test_random]==1)
        data["infected_free"][t] = nfree
        asbirds = 'a bird' if nfree == 1 else 'birds'

        ### show output
        logger.info(f"True  : (S,I,R): ({nS:.1f}, {nI:.1f}, {nR:.1f})")
        logger.info(f"AUR_I : {aurI:.3f}, prec100: {yI[100]}, prec(1% of {len(yI)}): {prec(1):.2f}, prec5%: {prec(5):.2f}")
        logger.info(f"sym: {len(sym)}, results test algo (S,I,R): ({sus_test_algo},{inf_test_algo},{rec_test_algo}), infected test random: {inf_test_random}/{num_test_random}")
        logger.info(f"...quarantining {len(to_quarantine)} guys -> got {ninfq} infected, {nfree} free as {asbirds} ({nfree-freebirds:+d})")
        freebirds = nfree

        ### callback
        callback(data)

        if t % save_every_iter == 0:
            df_save = pd.DataFrame.from_records(data, exclude=["logger"])
            df_save.to_csv(output_dir + name_file_res + "_res.gz")

    # save files
    del sim
    df_save = pd.DataFrame.from_records(data, exclude=["logger"])
    df_save.to_csv(output_dir + name_file_res + "_res.gz")
    with open(output_dir + name_file_res + "_states.pkl", mode="wb") as f_states:
        pickle.dump(data_states, f_states)
    return df_save




def free_mtpp(mob,
              T = 30, #days of simulation
             beta = 0.55,
            country = 'GBR',
            initial_counts = {'expo': 5, 'isym': 5 , 'iasy': 5}, #in the form {'expo': 7, 'isym_posi': 2, 'iasy': 2},
             logger = dummy_logger(),
             seed=1,
              t_unit = 24,
              delta_days = 1, # quarantine every tot days
              t_res = 0.25, #in hours,
              contact_duration_households = 3, # hours
             name_file_res = "res",
             output_dir = "./output_mtpp/",
             save_every_iter = 1,
             stop_zero_I = True,
             callback = lambda x : None,
             data = {}
            ):
    '''
    Simulate interventions strategy on the MTPP epidemic simulation.

    input
    -----
    mob_kwargs: Dict
            Dictonary with settings of mobility simulations (town, queries, downsampling, sites/houses location and simulation params)
    inference_algo: Class (rank_template)
            Class for order the nodes according to the prob to be infected
            logger = logger for printing intermediate steps
    results:
        print on file true configurations and transmission
    '''
    
    ### create output_dir if missing
    fold_out = Path(output_dir)
    if not fold_out.exists():
        fold_out.mkdir(parents=True)

    np.random.seed(seed)
    
    N = mob.num_people
    
    distributions = CovidDistributions(country=country)
    intensity_params = {'betas' : {'education': beta,'social': beta,'bus_stop': beta,'office': beta,'supermarket': beta}, 'beta_household' : beta }
    initial_seeds=get_seeds(N, initial_counts)
    
    for col_name in ["I", "IR","infected_free","num_quarantined"]:
        data[col_name] = np.full(T,np.nan)
    data["logger"] = logger

    status_dict = initial_seeds
    delta_th = delta_days * t_unit # quarantine every tot days
   
    sim = DiseaseModel(mob, distributions)
    sim.init_all(intensity_params, initial_seeds)
    
    num_quarantined = 0
    excluded = np.zeros(N, dtype=bool)
    t = 0
    sim.t0 = 0
    for t in range(T):
        th = t*t_unit
        ### advance one time step

        sim.run_one_step_dyn(th + delta_th, excluded)
        status_dict, status = get_status(sim,th+delta_th,N)
        state = status_to_state(status) ## 0, 1, 2, 3 (quar)

        nS, nI, nR = (state == 0).sum(), (state == 1).sum(), (state == 2).sum()
        if nI == 0 and stop_zero_I:
            logger.info("stopping simulation as there are no more infected individuals")
            break
        logger.info(f'time:{t}')
        ### extract contacts
        
        data["I"][t] = nI
        data["IR"][t] = nR+nI
        data["infected_free"][t] = nI
        data["num_quarantined"][t] = num_quarantined
        
        ### callback
        callback(data)

        if t % save_every_iter == 0:
            df_save = pd.DataFrame.from_records(data, exclude=["logger"])
    del sim
    df_save.to_csv(output_dir + name_file_res + "_res.gz")
    #with open(output_dir + name_file_res + "_states.pkl", mode="wb") as f_states:
    #    pickle.dump(data_states, f_states)
    return df_save
    print("End of Simulation")
    
    # save files
    
    return df_save

