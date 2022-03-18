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

class Process(Enum):
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
class ParamsProcess:
    def draw_values(self, nb=1, smaller_range=False, mean_value=False):
        d = self.__dict__.copy()
        for k in d.keys():
            if smaller_range:
                min_ = d[k][0]
                max_ = d[k][1]
                delt = max_ - min_
                min_ = min_ + delt * 0.1
                max_ = max_ - delt * 0.1
            else:
                min_ = d[k][0]
                max_ = d[k][1]

            if (type(d[k][0]) == int) & (type(d[k][1]) == int):
                d[k] = np.random.randint(int(np.ceil(min_)), int(np.ceil(max_)) + 1, nb)
            else:
                if mean_value:
                    d[k] = np.array([(min_ + max_) / 2])
                else:
                    d[k] = np.random.uniform(min_, max_, nb)

        return d


class ParamsPin(ParamsProcess):
    def __init__(self):
        self.alpha = [0.0, 1.0]
        self.delta = [0.0, 1.0]
        self.epsilon_b = [0, 300]
        self.epsilon_s = [0, 300]
        self.mu = [0, 200]
        self.buy = [0, 600]
        self.sell = [0, 600]

class ParamsApin(ParamsProcess):
    def __init__(self) -> None:
        pass
    #In construction

class ParamsOption:
    def __init__(self):
        self.process = Process.PIN    

class ParamsModels:
    def __init__(self):
        self.save_dir = './model_save/'
        self.res_dir = './res/'

        self.name = Process.PIN
        self.normalize = True
        self.layers = [64,32,16]
        self.batch_size = 512
        self.activation = "swish"
        self.opti = Optimizer.ADAM
        self.loss = Loss.MSE
        self.learning_rate = 0.001
        self.E = 10

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
        self.opt = ParamsOption()

        self.process = None
        self.update_process()
        self.update_model_name()

    def update_process(self, process=None):
        if process is not None:
            self.opt.process = process

        if self.opt.process.name == Process.PIN.name:
            self.process = ParamsPin()

        if self.opt.process.name == Process.APIN.name:
            self.process = ParamsApin()


    def update_model_name(self):
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
    # params.print_values()

    process = Process
    