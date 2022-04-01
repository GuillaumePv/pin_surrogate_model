import tensorflow as tf
import tensorflow_probability as tfp
# https://www.tensorflow.org/probability?hl=fr

from ml_model import NetworkModel
from parameters import *

import pandas as pd
import time


class Optimizer:
    def __init__(self):
        self.c_model = NetworkModel()

        # à trouver le moyen de mettre dans une classe pour éviter les erreurs
        if self.par.opt.process.name == Process.PIN.name:
            data_dir = self.par.data.path_sim_save + 'PIN_MLE.txt'
        else:
            data_dir = self.par.data.path_sim_save + 'APIN_MLE.txt'
        X = pd.read_csv(data_dir)
        Y = X['MLE']
        params_df, true_opt = self.c_model.split_state_data_par(X)

        bounds = tf.fill((1,))
        bound_cost = tf.constant(100.0, dtype=tf.float64)


        def tf_loss(self):
            pass

        def Optimize(self, x_params):
            @tf.function
            def func( x_params):
                x_params = tf.reshape(x_params,(1,-1))
                par_est = x_params[:,:-1]
                pred = self.c_model.model([par_est,true_opt])
                v_call = tf_loss(pred,Y)
                bnd = tf.reduce_sum(tf.nn.relu(x_params - bounds) + tf.nn.relu(-(x_params + bounds))) * bound_cost
                return tf.reduce_mean(v_call) + tf.reduce_mean(bnd)

            @tf.function
            def func_g( x_params):
                with tf.GradientTape() as tape:
                    tape.watch(x_params)
                    loss_value = func(x_params)
                grads = tape.gradient(loss_value, [x_params])
                return loss_value, grads[0]

            s = time.time()
            # https://www.tensorflow.org/probability/api_docs/python/tfp/optimizer
            soln = tfp.optimizer.lbfgs_minimize(func_g, init_x, max_iterations=50, tolerance=1e-60)
            soln_time = np.roudn((time.time() - s)/60, 2)
            pred_par = soln.position.numpy()
            obj_value = soln.objective_value.numpy()

            for ii in range(params_df.shape[1]):
                X.loc[:,params_df.columns[ii]] = pred_par[ii]

            ## checker à qui ça sert
            # X['v0'] = pred_par[ii + 1]
            score = self.c_model.score(self.c_model.unonrmalize(X)[0], Y)

            perf = pd.Series(list(score) + [obj_value, soln_time], index=['r2','mae','mse','obj','time'])

            pr = self.c_model.unnormalize(X)[0].iloc[0,:]
            res=pr[pd.Series(pr.index.tolist())[~pd.Series(pr.index.tolist()).isin(self.par_c.data.cross_vary_list)]]

        
