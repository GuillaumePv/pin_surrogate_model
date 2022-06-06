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
class Deepsurrogate:
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
        
    def pre_process_X(self, X):
        # in construction
        X = X.copy()
        col = []
        for cc in self.par_c.process.__dict__.keys():
            col.append(cc)
        if X.shape[1] == len(col):
            X = pd.DataFrame(X, columns=col)
        else:
            assert False, 'wrong number of columns'
        
        return X
        

    # réussi à deboguer
    def estimate_par_lbfgs(self,num=100):
        print(f"=== inverse modelling for {num} rows ===")
        COL = ["alpha","delta","epsilon_b","epsilon_s","mu"]
        COL_PLUS = COL + self.par_c.data.cross_vary_list
        tf_loss = tf.keras.losses.MSE
        data = self.X.iloc[:num,:]
        data_y = self.Y.head(num)
        print("=== score of data ===")
        print(self.c_model.score(data,data_y))
        init_x = data[COL].mean().values

        def func_g(x_params):
            # génial pour faire sur plusieurs colonnes la même valeur
            df[COL] = x_params.numpy()
            # print("=== df ===")
            # print(df)
            grad, mle = self.c_model.get_grad_and_mle(df[COL_PLUS],True)

            loss_value = np.mean(np.square(mle-y)**2)
            g = grad.mean()[COL].values

            g = tf.convert_to_tensor(g)
            loss_value = tf.convert_to_tensor(loss_value)

            #print('---',loss_value,flush=True)
            return loss_value, g
        s = time.time()
        list_of_ei = []
        for i in tqdm(range(data.shape[0])):
            df = data.loc[i].to_frame().transpose()
            #init_x = df[COL].values[0]+0.001
            y = data["MLE"].loc[i]
            
            soln = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=func_g, initial_position=init_x, max_iterations=50, tolerance=1e-60)
            pred_par = soln.position.numpy()
            obj_value = soln.objective_value.numpy()
            #print(soln,flush=True)
            
            # print(pred_par)
            e_i = np.abs(df[COL].values[0]-pred_par)/np.abs(self.X[COL].max()-self.X[COL].min())
            list_of_ei.append(e_i)
            #print(e_i)
            
        soln_time = np.round((time.time() - s) / 60, 2)
        print(soln_time)
        df_ei = pd.DataFrame(list_of_ei)
        df_ei.to_csv("./results/table/ei_results.csv",index=False)
        print(df_ei)
        

    def get_pin(X):
        pass
    # my innovation
    def pin_estimation(self):
        COL = ["alpha","delta","epsilon_b","epsilon_s","mu"]
        COL_PLUS = COL + self.par_c.data.cross_vary_list
        
        bound_cost = tf.constant(100.0, dtype=tf.float64)

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
            grad, mle = self.c_model.get_grad_and_mle(df[COL_PLUS],True)
            loss_value = tf.reduce_sum(-mle)
            g = grad.mean()[COL].values

            g = tf.convert_to_tensor(g)
            
            loss_value = tf.convert_to_tensor(loss_value)

            print('---',loss_value,flush=True)
            return loss_value, g

        init_x = self.means.values
        bounds = tf.fill((1, len(init_x)), self.pivot)
        
        
        s = time.time()
        soln = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=func_g, initial_position=init_x, max_iterations=50, tolerance=1e-60)
        soln_time = np.round((time.time() - s) / 60, 2)
        pred_par = soln.position.numpy()
        obj_value = soln.objective_value.numpy()
        print(y)
        print(obj_value)
        print(init_x)
        print(pred_par)

if __name__ == '__main__':
    deepsurrogate = Deepsurrogate()
    deepsurrogate.estimate_par_lbfgs()


        
