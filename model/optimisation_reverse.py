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
        # params_df, true_opt = self.c_model.split_state_data_par(X)

        # print(self.c_model.model([params_df+true_opt]))
        # print(self.X.iloc[:1].values)
        # print(self.c_model.predict(self.X.iloc[:1])[0][0])
        # print(self.c_model.model.predict(self.X.iloc[:1,:-1].values))

    def test_optimize(self):
        tf_loss = tf.keras.losses.MSE
        number_echantillon = 200

        def func(x_params):
            pred = self.c_model.model.predict(x_params)
            Y = self.X['MLE'].iloc[:number_echantillon]
        
            v_call = tf_loss(pred,Y)
            print(v_call)
            print(tf.reduce_mean(v_call))
            return tf.reduce_mean(v_call)

        ## need to convert to tensor ##
        def func_g(x_params):
            with tf.GradientTape() as tape:
                tape.watch(x_params)
                loss_value = func(x_params)
            
            grads = tape.gradient(loss_value,[x_params])
            return loss_value, grads[0]

        def pin_opt(x_params_est,buy_sell):

            x_params = np.concatenate((x_params_est,buy_sell),axis=None)
            df_test = pd.DataFrame(x_params)
            # print(df_test.shape)
            df_test_normalized, y = self.c_model.normalize(x_params)
            # print(df_test_normalized.values)
            df_test_normalized = df_test_normalized.values.reshape(1,7)
            # if x_params.shape == (7,):
            #     # print("find solution")
            #     x_params = x_params.reshape(1,x_params.shape[0])
            return -self.c_model.model.predict(df_test_normalized)[0][0]

        Bounds = []
        d = self.par_c.process.__dict__
        for i,k in enumerate(d):
            Bounds.append(d[k]) #need to be no normalize
            # print(d[k])
        for i in tqdm(range(self.X.iloc[:,:-1].shape[0])):
            x_params = self.X.iloc[i:i+1,:-1].values
            x_params_est = x_params[0][:-2]
            buy_and_sell = x_params[0][-2:]
            # print(x_params)
            print("MLE intial")
            print(pin_opt(x_params_est,buy_and_sell))
            print("optimizer")
            # add bounds quand j'ai réussi à normaliser les bounds
            res = op.minimize(pin_opt, x0=x_params_est,method="SLSQP",args=(buy_and_sell),bounds=Bounds[:-2],options={'disp': True})
            # res = op.minimize(pin_opt, x0=x_params_est,method="SLSQP",args=(buy_and_sell),options={'disp': False})
            print("value of our optimizer")
            fin_res = res.x
            print(res.x)
            print("=== buy and sell === value")
            print(buy_and_sell)
            print("MLE result")
            print(pin_opt(res.x,buy_and_sell))
            
            PIN = (fin_res[0]*fin_res[4])/((fin_res[0]*fin_res[4])+fin_res[2]+fin_res[3])
            print(f"PIN value: {PIN}")
            print("=============","\n")

    def fct_test(self):

        columns = ["alpha","delta","epsilon_b","epsilon_s","mu"]
        x_init = [0.5,0.5,250,250,250]
        # x_init = np.array([0,0,1,1,1])+0.000001
        Bounds = []
        d = self.par_c.process.__dict__
        for i,k in enumerate(d):
            Bounds.append(d[k]) #need to be no normalize
            # print(d[k])

        def pin_opt(x_params_est,buy_sell):

            data_used = np.array([x_params_est,x_params_est,x_params_est,x_params_est, x_params_est])
            data_used = pd.DataFrame(data_used,columns=columns)
            buy_sell.index = data_used.index
            data_concat = pd.concat([data_used,buy_sell],axis=1)
            # print(data_concat)
            data_norm,y = self.c_model.normalize(data_concat)
            # print(data_norm)
            value_model_sum = -self.c_model.model.predict(data_norm).sum()
            
            return value_model_sum

        print(f"Number of split possible: {np.trunc(len(self.X)/5)}")
        number_split = int(np.trunc(len(self.X)/5))
        print(f"Number of split possible: {len(self.X)/5}")

        for i in tqdm(range(number_split)):

            data_test = self.X.iloc[i*5:(i+1)*5,:-1]
            # print(np.array(data_test.iloc[0,:-2]))
            # x_init = np.array(data_test.iloc[0,:-2])
            # print(x_init)
            
            data_buy_sell = data_test.iloc[:,-2:]
            eb0,es0 = np.mean(data_buy_sell['buy']), np.mean(data_buy_sell['sell'])
            oib = data_buy_sell['buy'] - data_buy_sell['sell'] # Turnover / Order imbalance
            u0 = np.mean(abs(oib)) # expected order imbalance = mean of absolute order imbalance
            x_init = [0.5,0.5,eb0,es0,u0]
            print("== new init ==")
            print(x_init)
            print("=====")
            ## find a better way to do it
            print(pin_opt(x_init, data_buy_sell))
            print("optimizer")
                # add bounds quand j'ai réussi à normaliser les bounds
            res = op.minimize(pin_opt, x0=x_init,method="SLSQP",args=(data_buy_sell),bounds=Bounds[:-2],options={'disp': False})
                # res = op.minimize(pin_opt, x0=x_params_est,method="SLSQP",args=(buy_and_sell),options={'disp': False})
            print("value of our optimizer")
            fin_res = res.x
            print(res.x)
            print("=== buy and sell === value")
            # print(data_buy_sell)
            print("MLE result")
            print(res.fun)
                
            PIN = (fin_res[0]*fin_res[4])/((fin_res[0]*fin_res[4])+fin_res[2]+fin_res[3])
            print(f"PIN value: {PIN}")
            print("=============","\n")
        

    def Optimize(self, init_x, x_params):
        tf_loss = tf.keras.losses.MSE
        bounds = tf.fill((1,))
        bound_cost = tf.constant(100.0, dtype=tf.float64)
        @tf.function
        def func(x_params):
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
        soln_time = np.round((time.time() - s)/60, 2)
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
        return res, perf, pred_par

optimizer = Optimizer()
optimizer.fct_test()

        
