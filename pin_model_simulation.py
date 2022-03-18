# -*- coding: utf-8 -*-

# numpy for matrix algebra
import numpy as np
from numpy import log, exp

# log(sum(exp))
from scipy.special import logsumexp
from scipy.linalg import inv
import scipy.optimize as op
from tqdm import tqdm

# import common functions
from common import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
from joblib import Parallel, delayed
import multiprocessing

class PINModel(object):

    def __init__(self,a,d,es,eb,u,n=1,t=252):
        """Initializes parameters of an Easley and O'Hara Sequential Trade Model
        
        a : $\alpha$, the unconditional probability of an information event
        d : $\delta$, the unconditional probability of good news
        es : $\epsilon_s$, the average number of sells on a day with no news
        eb : $\epsilon_b$, the average number of buys on a day with no news
        mu : $\mu$, the average number of (additional) trades on a day with news

        n : the number of stocks to simulate, default 1
        t : the number of periods to simulate, default 252 (one trading year)
        """

        # Assign model parameters
        self.a, self.d, self.es, self.eb, self.u, self.N, self.T = a, d, es, eb, u, n, t
        self.states = self._draw_states()
        self.buys = np.random.poisson((eb+(self.states == 1)*u)) # T x N
        self.sells = np.random.poisson((es+(self.states == -1)*u)) # T x N
        self.alpha = compute_alpha(a, d, eb, es, u, self.buys, self.sells)

    def _draw_states(self):
        """Draws the states for N stocks and T periods.

        In the Easley and O'Hara sequential trade model at the beginning of each period nature determines whether there is an information event with probability $\alpha$ (a). If there is information, nature determines whether the signal is good news with probability $\delta$ (d) or bad news $1-\delta$ (1-d).

        A quick way to implement this is to draw all of the event states at once as an `NxT` matrix from a binomial distribution with $p=\alpha$, and independently draw all of the news states as an `NxT` matrix from a binomial with $p=\delta$. 
        
        An information event occurs for stock i on day t if `events[i][t]=1`, and zero otherwise. The news is good if `news[i][t]=1` and bad if `news[i][t]=-1`. 

        The element-wise product of `events` with `news` gives a complete description of the states for the sequential trade model, where the state variable can take the values (-1,0,1) for bad news, no news, and good news respectively.

        self : EOSequentialTradeModel instance which contains parameter definitions
        """
        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.binomial.html
        events = np.random.binomial(1, self.a, (self.N,self.T)) 
        news = np.random.binomial(1, self.d, (self.N,self.T))
        news[news == 0] = -1

        states = events*news
        return states

# function that represents the Poisson log-likelihood which is common to each of the three states: good, bad, and no news
def _lf(eb, es, n_buys, n_sells):
    return -eb+n_buys*log(eb)-lfact(n_buys)-es+n_sells*log(es)-lfact(n_sells)

# function that represents the full vector of log-likelihoods for the PIN model (EHO likelihood)
def _ll(a, d, eb, es, u, n_buys, n_sells):
    return np.array([log(a*(1-d))+_lf(eb,es+u,n_buys,n_sells), 
                   log(a*d)+_lf(eb+u,es,n_buys,n_sells), 
                   log(1-a)+_lf(eb,es,n_buys,n_sells)])

# CPIE: function that will compute CPIEs for real or simulated data. The computation of the CPIE depends on the likelihood function definitions          
def compute_alpha(a, d, eb, es, u, n_buys, n_sells):
    '''Compute the conditional alpha given parameters, buys, and sells.

    '''
    ll = _ll(a, d, eb, es, u, n_buys, n_sells) # (3, N, T) 
    llmax = ll.max(axis=0)
    y = exp(ll-llmax)
    alpha = y[:-1].sum(axis=0)/y.sum(axis=0) # Duarte Computation
    # alpha = ll[:-1].sum(axis=0)/ll.sum(axis=0) # Guillaume's computation
    
    return alpha

# the total likelihood that will be used in the optimization
def loglik(theta, n_buys, n_sells):
    a,d,eb,es,u = theta
    ll = _ll(a, d, eb, es, u, n_buys, n_sells)
    
    return sum(logsumexp(ll,axis=0))
            
def fit(n_buys, n_sells, starts=10, maxiter=100, 
        a=None, d=None, eb=None, es=None, u=None,
        se=None, **kwargs):
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    """_summary_

    Args:
        n_buys (_type_): _description_
        n_sells (_type_): _description_
        starts (int, optional): _description_. Defaults to 10.
        maxiter (int, optional): _description_. Defaults to 100.
        a (_type_, optional): _description_. Defaults to None.
        d (_type_, optional): _description_. Defaults to None.
        eb (_type_, optional): _description_. Defaults to None.
        es (_type_, optional): _description_. Defaults to None.
        u (_type_, optional): _description_. Defaults to None.
        se (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    nll = lambda *args: -loglik(*args) # define the negative log likelihood that we will minimize
    bounds = [(0.00001,0.99999)]*2+[(0.00001,np.inf)]*3 # we will do a constrained optimization
    ranges = [(0.00001,0.99999)]*2 # we will define the min-max range for our random guesses

    # if we do not have a prior on what the estimates are, we compute them here
    a0,d0 = [x or 0.5 for x in (a,d)] # 50% chance of information/news
    eb0,es0 = eb or np.mean(n_buys), es or np.mean(n_sells) # expected buys/sells = mean of observed buy/sells
    oib = n_buys - n_sells # Turnover / Order imbalance
    u0 = u or np.mean(abs(oib)) # expected order imbalance = mean of absolute order imbalance

    res_final = [a0,d0,eb0,es0,u0] # define the vector that will hold all the parameters
    stderr = np.zeros_like(res_final) # define the vector that will hold our standard errors
    f = nll(res_final,n_buys,n_sells) # initialize the log likelihood function with the buys/sells data
    for i in range(starts):
        # rc is going to be our return code
        rc = -1
        j = 0
        # somtimes bug
        while (rc != 0) & (j <= maxiter):
            if (None in (res_final)) or i:
                # guess parameters
                a0,d0 = [np.random.uniform(l,np.nan_to_num(h)) for (l,h) in ranges]
                eb0,es0,u0 = np.random.poisson([eb,es,u])

            # do actual optimization here
            res = op.minimize(nll, [a0,d0,eb0,es0,u0], method=None,
                              bounds=bounds, args=(n_buys,n_sells))
            rc = res['status']
            # see if the optimization step violated any constraints
            check_bounds = list(imap(lambda x,y: x in y, res['x'], bounds))
            if any(check_bounds):
                rc = 3
            j+=1
        if (res['success']) & (res['fun'] <= f):
            # if everything worked fine and we have a 
            # smaller (negative) likelihood then store these parameters
            f,rc = res['fun'],res['status']
            res_final = res['x'].tolist()
            # and compute standard errors
            stderr = 1/np.sqrt(inv(res['hess_inv'].todense()).diagonal())
    
    # output the final parameter estimates
    param_names = ['a','d','eb','es','mu']
    output = dict(zip(param_names+['f','rc'],
                    res_final+[f,rc]))
    if se:
        output = {'params': dict(zip(param_names,res_final)),
                  'se': dict(zip(param_names,stderr)),
                  'stats':{'f': f,'rc': rc}
                 }
    return output

def cpie_mech(turn):
    mech = np.zeros_like(turn)
    mech[turn > turn.mean()] = 1
    return mech

def compute_pin(res):
    PIN = (res['a']*res['mu'])/((res['a']*res['mu'])+res['eb']+res['es'])
    return PIN

def simulation(numb_simu):
    ## Daily simulation ##
    
    ## Hidden factor ##
    a = np.random.uniform(0,0.9,1)[0] # [0,1]
    d = np.random.uniform(0,0.9,1)[0] # [0,1]
    es = int(np.random.uniform(200,300,1)[0]) # create cluster (frequent (2300), infrequent (150), heavy (5600) => mean) 
    eb = int(np.random.uniform(200,300,1)[0])
    mu = int(np.random.uniform(200,300,1)[0])
    # number of firm
    N = 1
    T = 1 # yearly => Monthly, weekly

    model = PINModel(a,d,es,eb,mu,n=N,t=T)

    ## Factor ##
    buys = to_series(model.buys)
    sells = to_series(model.sells)
        
    

    array_MLE = _ll(a,d,eb,es,mu,buys,sells)
    MLE = logsumexp(array_MLE,axis=0)[0]
    # print(logsumexp(array_MLE,axis=0))
    # print(sum(logsumexp(array_MLE,axis=0)))
    # print(est_tab(res.results, est=['params','tvalues'], stats=['rsquared','rsquared_sp']))
    #print(buys, sells)
    # compute PIN 
        
    # resultat = fit(buys[:], sells[:],1, max_iter)
    # PIN = compute_pin(resultat)

        # problem with this function
        # CPIE = compute_alpha(resultat['a'], resultat['d'], resultat['eb'], resultat['es'], resultat['mu'], buys, sells)
        # print(fit(buys, sells, 1))

        ### Initial parameters ###
    f = open("./data/simulation_output_MLE.txt", "a")
    f.write(f"{a},{d},{es},{eb},{mu},{buys.values[0]},{sells.values[0]},{MLE}\n")
    f.close()
        # if i % 10 == 0:
        #     output = f"""
        #     alpha: {a}
        #     delta: {d}
        #     epsilon sell: {es}
        #     epsilon buy: {eb}
        #     mu: {mu}

        #     ==========
        #     PIN: {PIN}
        #     """

    # if numb_simu % 10 == 0:
    #     output = f"""
    #     buys: {buys.values}
    #     sells: {sells.values}

    #     ==========
    #     PIN: {PIN}
    #     """

    #     print(output)

if __name__ == '__main__':
    
    import pandas as pd
    from regressions import *
    # number of simulation
    if os.path.isfile("./data/simulation_output_MLE.txt") == False:
        print("=== creating simulation file ===")
        f = open("./data/simulation_output_MLE.txt", "a")
        f.write("alpha,delta,epsilon_b,epsilon_s,mu,buy,sell,MLE\n")
        f.close()

    sim = 100000
    max_iter = 10
    num_cores = multiprocessing.cpu_count()
    print(f"== number of CPU: {num_cores} ==")

    Parallel(n_jobs=num_cores)(delayed(simulation)(i) for i in tqdm(range(sim)))
       