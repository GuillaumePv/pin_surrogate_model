1# ML_model
# Created by Guillaume Pavé, at xx.xx.xx

import pickle
import socket
from numpy import dtype
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from scipy.stats import norm
from parameters import *
import pandas as pd
import numpy as np

class FirstLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs, par):
        super(FirstLayer, self).__init__(dtype='float64')
        self.num_outputs = num_outputs
        self.par = par

class NetworkModel:
    def __init__(self, par: Params()):
        self.par = par
        self.model = None

        self.m = None
        self.std = None
        self.m_y = None
        self.std_y = None
        
        if socket.gethostname() in ['Guillaume']:
            print("personal computer")

        self.save_dir = self.par.model.save_dir + '/' + self.par.name

        def normalize(self, X=None, y=None):
            if self.par.model.normalize:
                if X is not None:
                    X = (X - self.m) / self.std

                if y is not None:
                    pass

            return X, y

        def unnormalize(self, X=None, y=None):
            if self.par.model.normalize:
                if X is not None:
                    if self.par.model.normalize_range:
                        X = X * self.x_range
                    else:
                        X = (X * self.std) + self.m

                if y is not None:
                    if type(y) == np.ndarray:
                        y = y * self.std_y.values + self.m_y.values + self
                    else:
                        y = (y * self.std_y) + self.m_y
            
            return X, y

        def train(self): # In Construction
            #################
            # first get the data_plit col size
            #################

            c = []
            y = ['y'] # LE estimation
            opt_data = self.par.data.cross_vary_list

            for cc in self.par.process.__dict__.keys():
                if (cc not in opt_data):
                    c.append(cc)

            c1 = len(c)
            c2 = len(c) + len(opt_data)

            d = self.par.process.__dict__
            m = {}
            std = {}

            for i, k in enumerate(d):
                m[k] = np.mean(d[k])
                std[k] = (((max(d[k]) - min(d[k])) ** 2) / 12) ** (1/2)
            
            self.m = pd.Series(m)
            self.std = pd.Series(std)

            ###################
            # prepare data sets
            ###################

            def tr(x): # check code
                y = (((x[:c1])))

    def get_pin(self, X):
        """
        function that generate probability of informed trading
        """
        # In construction

        pass
    def create_nnet_model(self):
        L = []

        for i, l in enumerate(self.par.model.layers):
            if i == 0:
                L.append(FirstLayer(l,self.par))
            else:
                L.append(layers.Dense(l, activation= self.par.model.activation, dtype=tf.float64))

            L.append(layers.Dense(1,dtype=tf.float64))
            self.model = keras.Sequential(L)

            # optimizer = tf.keras.optimizers.RMSprop(0.05)
        if self.par.model.opti == Optimizer.SGD:
            optimizer = tf.keras.optimizers.SGD(self.par.model.learning_rate)
        if self.par.model.opti == Optimizer.RMS_PROP:
            optimizer = tf.keras.optimizers.RMSprop(self.par.model.learning_rate)
        if self.par.model.opti == Optimizer.ADAM:
            optimizer = tf.keras.optimizers.Adam(self.par.model.learning_rate)
        if self.par.model.opti == Optimizer.NADAM:
            optimizer = tf.keras.optimizers.Nadam(self.par.model.learning_rate)
        if self.par.model.opti == Optimizer.ADAMAX:
            optimizer = tf.keras.optimizers.Adamax(self.par.model.learning_rate)
        if self.par.model.opti == Optimizer.ADAGRAD:
            optimizer = tf.keras.optimizers.Adamax(self.par.model.learning_rate)

        # optimizer = tf.keras.optimizers.Adam(0.00005/2)

        def r_square(y_true, y_pred):
            SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
            SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
            return (1 - SS_res / (SS_tot + tf.keras.backend.epsilon()))

        if self.par.model.loss == Loss.MAE:
            self.model.compile(loss='mae', optimizer=optimizer, metrics=['mae', 'mse', r_square])
        if self.par.model.loss == Loss.MSE:
            self.model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse', r_square])

if __name__ == "__main__":
    par = Params()
    model = NetworkModel(par)