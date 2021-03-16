import numpy as np, pandas as pd


def contacts_cg(mob, t_res, t_unit, first_filter = True):
    # check asymmetric contacts (same t_from and different t_to OR same t_to and different t_from). Take as t_from the minimum
    # value among repeated contacts and as t_to the maximum value
    contacts_raw = []
    for i in range(mob.num_people ):
        contacts_raw +=list(mob.find_contacts_of_indiv(i,tmin = 0,tmax = np.inf).find((0,np.inf)))
        
    print("Check asymmetric contacts")
    t_to_unique = dict()
    t_from_unique = dict()
    for h in contacts_raw:
        if h.indiv_i > h.indiv_j:
            idxi = h.indiv_j
            idxj = h.indiv_i
        else:
            idxi = h.indiv_i
            idxj = h.indiv_j
        link_from = (idxi, idxj, h.t_from, h.site) # check contacts with the same starting time
        link_to = (idxi, idxj, h.t_to_direct, h.site) # check contacts with the same ending time
        if link_from in t_to_unique:
            if t_to_unique[link_from] != h.t_to_direct:
                print(idxi, idxj, h.t_from, "but", h.t_to_direct, "!=", t_to_unique[link_from])
            t_to_unique[link_from] = max(t_to_unique[link_from], h.t_to_direct)
        else:
            t_to_unique[link_from] = h.t_to_direct
        if link_to in t_from_unique:
            if t_from_unique[link_to] != h.t_from:
                print(idxi, idxj, h.t_to_direct, "but", h.t_from, "!=", t_from_unique[link_to])
            t_from_unique[link_to] = min(t_from_unique[link_to], h.t_from)
        else:
            t_from_unique[link_to] = h.t_from

    # collect contacts by site and match t_from and t_to
    site_cont = dict()
    for link in t_to_unique:
        a = (link[0], link[1], link[3]) # (i,j,site)
        if a in site_cont:
            site_cont[a].append( (link[2], t_to_unique[link]) ) if (link[2], t_to_unique[link]) not in site_cont[a] else site_cont[a] 
        else:
            site_cont[a] = [(link[2], t_to_unique[link])] 
            
    for link in t_from_unique:
        a = (link[0], link[1], link[3]) # (i,j,site)
        if a in site_cont:
            site_cont[a].append( (t_from_unique[link] ,link[2]) ) if (t_from_unique[link] ,link[2]) not in site_cont[a] else site_cont[a]
        else:
            site_cont[a] = [(t_from_unique[link], link[2])]
    
    # drop all contacts with duration less than t_res independently on the day they occur (if flag True)
    cont=[]
    for link in site_cont:
        all_times = site_cont[link]
        for t in all_times:
            if first_filter:
                if t[1]-t[0] > t_res:
                    cont.append( (link[0], link[1], t[0], t[1], t[1]-t[0], link[2], mob.site_type[link[2]]) )  
            else:
                cont.append( (link[0], link[1], t[0], t[1], t[1]-t[0], link[2], mob.site_type[link[2]]) )  

                
    # build DataFrame
    contact_raw = pd.DataFrame(data=cont, columns=['indiv_i', 'indiv_j', 't_from', 't_to', 'deltat','site','site_type'])
        
    # filter them by duration
    n_contacts = len(contact_raw)
    indiv_i = contact_raw.indiv_i.to_numpy()
    indiv_j = contact_raw.indiv_j.to_numpy()
    t_from = contact_raw.t_from.to_numpy()
    t_to = contact_raw.t_to.to_numpy()
    dt = contact_raw.deltat.to_numpy()
    sites = contact_raw.site.to_numpy()

     # duration of all contacts, unique (i,j)
    dt_dict = {}
    print("All raw (i < j) contacts", n_contacts)

    for i in range(n_contacts):
        if i % 500000 == 0 :
            print(round(i / n_contacts * 100, 2), "%")
        day_start = int(t_from[i] // t_unit)
        day_end = int(t_to[i] // t_unit)
        idxi = indiv_i[i]
        idxj = indiv_j[i]
        assert(idxi < idxj)
        if day_start == day_end:
            if (idxi, idxj, day_start) in dt_dict:
                dt_dict[(idxi, idxj, day_start)] += dt[i]
            else:
                dt_dict[(idxi, idxj, day_start)] = dt[i]
        else:
            if (idxi, idxj, day_start) in dt_dict:
                dt_dict[(idxi, idxj, day_start)] += (day_start+1)*t_unit - t_from[i]
            else:
                dt_dict[(idxi, idxj, day_start)] = (day_start+1)*t_unit - t_from[i]
            if (idxi, idxj, day_end) in dt_dict:
                dt_dict[(idxi, idxj, day_end)] += t_to[i] - day_end*t_unit
            else:
                dt_dict[(idxi, idxj, day_end)] = t_to[i] - day_end*t_unit
            if day_end - day_start > 1:
                for t in np.arange(day_start+1,day_end,1):
                    if (idxi, idxj, t) in dt_dict:
                        dt_dict[(idxi, idxj, t)] += t_unit
                    else:
                        dt_dict[(idxi, idxj, t)] = t_unit

    # filter and double them
    cont_sqzd_ls = []
    for a in dt_dict:
        if dt_dict[a] > t_res:
            cont_sqzd_ls.append([a[1], a[0], a[2], dt_dict[a]])
            cont_sqzd_ls.append([a[0], a[1], a[2], dt_dict[a]])

    print("Coarse-grained contacts", int(len(cont_sqzd_ls)/2))

    return cont_sqzd_ls


def extract_hospitalized_contacts(contacts_df,hosp_df):
    to_drop = []
    for r in hosp_df.iloc:
        i,ti,tf = r[0],r[1],r[2]
        temp_i = list(contacts_df[(contacts_df['i']==i) & (contacts_df['t']>=ti) & (contacts_df['t']<tf)].index)
        temp_j = list(contacts_df[(contacts_df['j']==i) & (contacts_df['t']>=ti) & (contacts_df['t']<tf)].index)
        to_drop += temp_i
        to_drop += temp_j 
    return contacts_df.loc[to_drop]

def check_sym_contacts(contacts_df,N,t,tol = 1e-15):
    A = np.zeros((N,N))
    contacts_t_df = contacts_df[contacts_df['t']==t]
    for r in contacts_t_df.iloc:
        i,j,_,l = int(r[0]),int(r[1]),int(r[2]),int(r[3])
        if A[i,j]!=0:
            print(f"contact ({i}, {j}) already present at time {t}")
        A[i,j] = l
        
    return np.all(np.abs(A-A.T)<tol)


def get_households_contacts(housedict,max_day,hh_deltat = 3):
    #duration is in hours
    contacts_households = []
   
    for house in housedict.keys():
        people_h = housedict[house]
        for i in people_h:
            for j in people_h:
                if i!=j:
                    contacts_households += [(i,j,t) for t in range(max_day)]

    contacts_households_df = pd.DataFrame(contacts_households,columns = ['i','j','t'])
    contacts_households_df['deltat'] = hh_deltat
    
    return contacts_households_df
    
    
def status_to_state_(status, hosp_as_rec = False):
    return (status+hosp_as_rec) // 10

status_to_state = np.vectorize(status_to_state_)
