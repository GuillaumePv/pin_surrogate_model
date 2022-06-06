import os
from numpy import float64
import scipy.optimize as op
import pandas as pd
import time
from tqdm import tqdm
import tensorflow as tf
import tensorflow_probability as tfp
# https://www.tensorflow.org/probability?hl=fr
try:
    from ml_model import NetworkModel
except ModuleNotFoundError:
    from model.ml_model import NetworkModel

try:
    from parameters import *
except ModuleNotFoundError:
    from model.parameters import *



# add params args
class DeepSurrogate:
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
    
    # obtain gradients for each parameters
    def get_derivative(self, X):
        X = self.pre_process_X(X)
        return self.c_model.get_grad_and_mle(X)

    def get_model_score(self):
        print("=== score of the model ===")
        score = self.c_model.score(self.X.head(1000),self.Y.head(1000))
        score.to_latex("./results/table/result_model.tex",index=False)


    def get_pin(X):
        pass
    
    # my innovation
    def pin_estimation(self,X=None):
        COL = ["alpha","delta","epsilon_b","epsilon_s","mu"]
        COL_PLUS = COL + self.par_c.data.cross_vary_list
        init_x = self.means.values

        data = self.X.iloc[:100,:]
        df = data.iloc[:,:-1]
       
        def func_g(x_params):
            # génial pour faire sur plusieurs colonnes la même valeur
            df[COL] = x_params.numpy()
            # print("=== df ===")
            # print(df)
            grad, mle = self.c_model.get_grad_and_mle(df[COL_PLUS],True)
            # add possibilities to use it on several period
            # bug here
            loss_value = np.abs(np.sum(mle))
            g = grad.mean()[COL].values

            g = tf.convert_to_tensor(g)
            
            loss_value = tf.convert_to_tensor(loss_value)

            #print('---',loss_value,flush=True)
            return loss_value

       

        s = time.time()
        for i in tqdm(range(data.shape[0])):
            df = data.loc[i].to_frame().transpose()
            y = data["MLE"].loc[i]
            soln = tfp.optimizer.nelder_mead_minimize(objective_function=func_g, initial_vertex=init_x, max_iterations=50)
            soln_time = np.round((time.time() - s) / 60, 2)
            pred_par = soln.position.numpy()
            obj_value = soln.objective_value.numpy()
       
            # print(obj_value)
            # print(pred_par)
            # create a table with PIN, buys and sells
            PIN = np.round((pred_par[0]*pred_par[4])/((pred_par[0]*pred_par[4])+pred_par[2]+pred_par[3]),4)
            # print("=== Deep PIN value ===")
            # print(PIN)

if __name__ == '__main__':
    deepsurrogate = DeepSurrogate()
    deepsurrogate.pin_estimation()


        
