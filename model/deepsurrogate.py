import os
from numpy import float64
import tensorflow as tf
import tensorflow_probability as tfp
# https://www.tensorflow.org/probability?hl=fr
from ml_model import NetworkModel
from parameters import *
import scipy.optimize as op

import pandas as pd
import time
from tqdm import tqdm

# add params args
class Optimizer:
    def __init__(self):
        #print('Loading model', call_model_name)
        par_c = Params()
        print('par.model.save_dir')
        #par_c.load(self.par.model.save_dir + call_model_name)
        self.par_c = par_c
        self.c_model = NetworkModel(par_c)
        self.c_model.load(par_c.name)

        # à trouver le moyen de mettre dans une classe pour éviter les erreurs
        if self.par_c.opt.process.name == Process.PIN.name:
            data_dir = self.par_c.data.path_sim_save + 'PIN_MLE_new.txt'
        else:
            data_dir = self.par_c.data.path_sim_save + 'APIN_MLE.txt'
        self.X = pd.read_csv(data_dir)
        self.Y = self.X['MLE']

        r = pd.Series(self.par_c.process.__dict__).apply(lambda x: (x[1]))
        m = self.c_model.m[r.index]
        s = self.c_model.std[r.index]
        pivot = (r - m) / s
        self.pivot = pivot[0]
        
        indexes = pd.Series(self.par_c.process.__dict__).index[:-2]
        
        self.means = self.c_model.m[indexes]
        self.std= self.c_model.std[indexes]
        

    def estimate_par_lbfgs(self,num=5):
        COL = ["alpha","delta","epsilon_b","epsilon_s","mu"]
        COL_PLUS = COL + self.par_c.data.cross_vary_list
        
        data = self.X.iloc[:num,:]

        init_x = data[COL].mean().values

        def func_g(x_params):
            # génial pour faire sur plusieurs colonnes la même valeur
            df[COL] = x_params.numpy()
            # print("=== df ===")
            # print(df)
            grad, mle = self.c_model.get_grad_and_mle(df,True)
            loss_value = np.mean(np.square(mle-y))
            g = grad.mean()[COL].values

            g = tf.convert_to_tensor(g)
            loss_value = tf.convert_to_tensor(loss_value)

            # print('---',loss_value,flush=True)
            return loss_value, g
        s = time.time()
        for i in tqdm(range(data.shape[0])):
            df = data.loc[i].to_frame().transpose()
            # init_x = df[COL].values
            y = data["MLE"].loc[i]
            
            soln = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=func_g, initial_position=init_x, max_iterations=50, tolerance=1e-60)
            pred_par = soln.position.numpy()
            obj_value = soln.objective_value.numpy()
            #print(soln,flush=True)
            
            print(pred_par)
            print(obj_value)
            
        
        soln_time = np.round((time.time() - s) / 60, 2)
        print(soln_time)
        print(data)
        

        

    def get_pin(self):
        COL = ["alpha","delta","epsilon_b","epsilon_s","mu"]
        COL_PLUS = COL + self.par_c.data.cross_vary_list
        
        test = self.X.iloc[:1,:]
        df = test.iloc[:,:-1]
        y = test.iloc[:,-1:]
        print(df)
        print(y)
        def func_g(x_params):
            # génial pour faire sur plusieurs colonnes la même valeur
            df[COL] = x_params.numpy()
            # print("=== df ===")
            # print(df)
            grad, mle = self.c_model.get_grad_and_mle(df,True)
            g = grad.mean()[COL].values

            g = tf.convert_to_tensor(g)
            loss_value = tf.convert_to_tensor(-mle[0][0])

            #print('---',loss_value,flush=True)
            return loss_value, g

        # init_x =  self.means.values
        init_x = df[COL].mean().values
        
        s = time.time()
        soln = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=func_g, initial_position=init_x, max_iterations=50, tolerance=1e-60)
        soln_time = np.round((time.time() - s) / 60, 2)
        pred_par = soln.position.numpy()
        obj_value = soln.objective_value.numpy()
        PIN = (pred_par[0]*pred_par[4])/((pred_par[0]*pred_par[4])+pred_par[2]+pred_par[3])
        print("PIN")
        print(PIN)
        print(obj_value)
        print(pred_par)


optimizer = Optimizer()
optimizer.get_pin()


        
