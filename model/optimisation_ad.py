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
            data_dir = self.par_c.data.path_sim_save + 'PIN_MLE.txt'
        else:
            data_dir = self.par_c.data.path_sim_save + 'APIN_MLE.txt'
        self.X = pd.read_csv(data_dir)
        self.Y = self.X['MLE']

        r = pd.Series(self.par_c.process.__dict__).apply(lambda x: (x[1]))
        m = self.c_model.m[r.index]
        s = self.c_model.std[r.index]
        pivot = (r - m) / s
        self.pivot = pivot[0]
        
        indexes = pd.Series(self.par_c.process.__dict__).index[-2:]
        self.means = self.c_model.m[indexes]
        self.std= self.c_model.std[indexes]
        # print(self.c_model.m[test])
        # params_df, true_opt = self.c_model.split_state_data_par(X)

        # print(self.c_model.model([params_df+true_opt]))
        # print(self.X.iloc[:1].values)
        # print(self.c_model.predict(self.X.iloc[:1])[0][0])
        # print(self.c_model.model.predict(self.X.iloc[:1,:-1].values))

    def test_optimize(self):
        tf_loss = tf.keras.losses.MSE
        x_init = pd.DataFrame([0.5,0.5,250,250,250,250,250]).transpose()
        bounds = tf.fill((1, len(x_init)), self.pivot)
        bound_cost = tf.constant(100.0, dtype=tf.float64)
        number_echantillon = 200

        test_value = self.X.iloc[:1,:]
        Y = test_value["MLE"]
        x_params = test_value.iloc[:,:-1]
        
        buy_and_sell = test_value.iloc[:,-3:-1]
        buy_and_sell = (buy_and_sell - self.means)/self.std
        buy_and_sell = tf.convert_to_tensor(buy_and_sell.values)
        buy_and_sell = tf.cast(buy_and_sell,tf.double)
        print(buy_and_sell)
        params = test_value.iloc[:,:-3]
        # print(params)
        print("initial value")
        print(x_params)
        print(-Y)
        
        def func(x_params):
            #print(x_params)
            test_value = tf.concat([x_params, buy_and_sell],axis=1)
            # print(test_value)
            pred = self.c_model.model(test_value)
            
        
            v_call = tf_loss(pred,Y)
            # print(v_call)
            # print(tf.reduce_mean(v_call))
            bnd = tf.reduce_sum(tf.nn.relu(x_params - bounds) + tf.nn.relu(-(x_params + bounds))) * bound_cost
            # return tf.reduce_mean(v_call) + tf.reduce_mean(bnd) # converge ok 
            # return tf.reduce_mean(v_call) # not converge !!!
            ## Use that to minimize MLE
            return tf.reduce_mean(-pred[0][0]) + tf.reduce_mean(bnd)

        ## need to convert to tensor ##
        def func_g(x_params):
            with tf.GradientTape() as tape:
                tape.watch(x_params)
                loss_value = func(x_params)
            
            grads = tape.gradient(loss_value,[x_params])
            return loss_value, grads[0]

        x_init = pd.DataFrame([0.5,0.5,250,250,250]).transpose()
        # x_init = tf.convert_to_tensor(x_init, tf.double)
        x_init = tf.convert_to_tensor(x_init)
        print(x_init)
        s = time.time()
        soln = tfp.optimizer.lbfgs_minimize(func_g, x_init, max_iterations=50, tolerance=1e-60)
        print(soln.converged)
        soln_time = np.round((time.time() - s) / 60, 2)
        pred_par = soln.position.numpy()
        obj_value = soln.objective_value.numpy()
        print(pred_par[0])
        # print(buy_and_sell.numpy()[0])
        final_value = np.concatenate([pred_par[0],buy_and_sell.numpy()[0]],axis=0)
        print("=== final value ===")
        print(final_value)
        
        print(obj_value)
        # print(self.c_model.unnormalize(pred_par[0])[0])
        res = self.c_model.unnormalize(final_value)[0]
        print(res)
        res = res.T
        PIN = (res['alpha']*res['mu'])/((res['alpha']*res['mu'])+res['epsilon_b']+res['epsilon_s'])
        print(PIN)
        # score=self.c_model.score(self.c_model.unnormalize(pred_par)[0],Y)
        


optimizer = Optimizer()
optimizer.test_optimize()

        
