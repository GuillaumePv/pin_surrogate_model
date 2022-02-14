"""
PIN model created by Guillaume Pavé

reproduce paper Easley, O'Hara (2002)
"""

# Scientific libraries
import numpy as np
from numpy import log, exp
from numpy.math import factorial

import pandas as pd
from scipy import optimize

# utility librairies
import os
from tqdm import tqdm

class PINModel:
    def __init__(self, a, d, es, eb, u, n, t, is_simulated=False):
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

    def EHOlikelihood(self, a, d, es, eb, u, B, S):
        part1 = 

    def fit():
        # create optimizer
        pass
if __name__ == '__main__':
    print("test code")

