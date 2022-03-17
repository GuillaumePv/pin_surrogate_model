import itertools
from enum import Enum
import numpy as np
import pandas as pd

import sys


##################
# params Enum
##################

class Optimizer(Enum):
    SGD = 1
    SGS_DECAY = 2
    ADAM = 3
    RMS_PROP = 4
    ADAMAX = 5
    NADAM = 5
    ADAGRAD = 5

class models(Enum):
    PIN = 1
    APIN = 2

class Sampling(Enum):
    RANDOM = 1
    LATIN_HYPERCUBE = 2
    SPARSE_GRID = 3
    GRID_10 = 4
    GIRD_20 = 5
    GRID_LOCALP = 6
    GRID_ZERO = 7
    GRID_15 = 8
    GRID_20_LOCALP = 9
    GRID_20_ZERO = 10
    GRID_30 = 30
    GRID_50 = 12
    GRID_50_bis = 13

class Loss(Enum):
    MSE = 1
    MAE = 2

## add our models

class ParamsModels:
    def __init__(self):
        self.save_dir = './model_save/'
        self.res_dir = './res/'

        self.normalize = True
        self.layers = [64,32,16]
        self.batch_size = 512
        self.activation = "swish"
        self.opti = Optimizer.ADAM
        self.loss = Loss.MSE
        self.learning_rate = 0.001

class ParamsData:
    def __init__(self):
        self.path_sim_save = './data/'
        self.train_size = 3
        self.test_size = 10000
        self.cross_vary_list = ["alpha,delta,epsilon_b,epsilon_s,u,buy,sell"]
        self.parallel = False

class Params:
    def __init__(self):
        self.name = ''
        self.seed = 12345
        self.model = ParamsModels()
        self.data =  ParamsData()

    def update_model_name():
        """
        change model name
        """
        # In constrcution
        pass

    def print_values(self):
        """
        Print all parameters used in the model
        """

        for key, v in self.__dict__.items():
            try:
                print("#######",key,'#######')
                for key2, vv in v.__dict__.items():
                    print(key2, ":", vv)
            except:
                print(v)


    def save(self, save_dir, filename="./parameters.p"):
        """
        save parameters of the model
        """
        #In construction
        pass

    def load(self, load_dir, filename="./parameters.p"):
        """
        load parameters of the model
        """
        # In construction
        pass

if __name__ == "__main__":
    params = Params()
    params.print_values()