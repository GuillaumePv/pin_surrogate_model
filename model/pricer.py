import socket
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from setuptools.command.saveopts import saveopts
from sklearn import metrics
from data import *
from ml_model import NetworkModel
from parameters import *
import time
from matplotlib import pyplot as plt


class Pricer:
    def __init__(self, param: Params, NB_DAYS=1, model_names=[]):
        self.par = param
        self.NB_DAYS = NB_DAYS
        self.BENCHMARK = False
        if socket.gethostname() in ['work', 'workstation']:
            self.save_dir = 'res/real' + str(self.NB_DAYS) + '/' + self.par.pricer.save_name + self.par.pricer.fit_process.name + 'COST_' + self.par.pricer.cost_function.name
        else:
            if os.path.exists('/scratch/snx3000/adidishe/'):
                self.save_dir = '/scratch/snx3000/adidishe/fop/res/real' + str(self.NB_DAYS) + '/' + self.par.pricer.save_name + self.par.pricer.fit_process.name + 'COST_' + self.par.pricer.cost_function.name
            else:
                self.save_dir = 'res' + str(self.NB_DAYS) + '/' + self.par.pricer.save_name + self.par.pricer.fit_process.name + 'COST_' + self.par.pricer.cost_function.name

        if self.par.pricer.cost_function == CostFunction.VOL_LASSO:
            self.save_dir = self.save_dir + '_' + str(self.par.pricer.lbda).replace('.', '')

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if self.par.pricer.fit_process == Process.HESTON_MODEL:
            call_model_name = self.par.pricer.best_heston

        if self.par.pricer.fit_process == Process.DOUBLE_EXP:
            # NP1
            call_model_name = self.par.pricer.best_bate

        if len(model_names) == 1:
            # NP
            print('load pre-specified model',flush=True)
            call_model_name = model_names[0]


        print('Loading model', call_model_name)
        par_c = Params()
        print('par.model.save_dir')
        par_c.load(self.par.model.save_dir + call_model_name)
        self.par_c = par_c

        self.c_model = NetworkModel(par_c)
        self.c_model.load(par_c.name)

        # compute the bounding value
        r = pd.Series(self.par_c.process.__dict__).apply(lambda x: (x[1]))
        m = self.c_model.m[r.index]
        s = self.c_model.std[r.index]
        pivot = (r - m) / s
        self.pivot = pivot[0]

    def load_data_test(self):
        SAMPLE_SIZE = 1000
        data_c = DataTrain(self.par_c)
        data_p = DataTrain(self.par_p)
        d = data_c.generate_sample(sample_size=SAMPLE_SIZE * 2, smaller_range=True, one_params_set=True)

        x_call = d.drop(columns='y').iloc[:SAMPLE_SIZE, :]
        y_call = d[['y']].iloc[:SAMPLE_SIZE, :]
        x_call_test = d.drop(columns='y').iloc[SAMPLE_SIZE:, :]
        y_call_test = d[['y']].iloc[SAMPLE_SIZE:, :]
        x_call, y_call = self.c_model.normalize(x_call, y_call)
        x_call_test, y_call_test = self.c_model.normalize(x_call_test, y_call_test)

        d = data_p.generate_sample(sample_size=SAMPLE_SIZE * 2, smaller_range=True, one_params_set=True, given_param_set=d)
        x_put = d.drop(columns='y').iloc[:SAMPLE_SIZE, :]
        y_put = d[['y']].iloc[:SAMPLE_SIZE, :]
        x_put_test = d.drop(columns='y').iloc[SAMPLE_SIZE:, :]
        y_put_test = d[['y']].iloc[SAMPLE_SIZE:, :]
        x_put, y_put = self.c_model.normalize(x_put, y_put)
        x_put_test, y_put_test = self.c_model.normalize(x_put_test, y_put_test)
        return x_put, y_put, x_call, y_call

    def pre_process_day(self, day, return_id=False):


        col = []
        for i, v in enumerate(self.par_c.process.__dict__):
            if v not in day.columns:
                day.loc[:,v] = np.mean(self.par_c.process.__dict__[v])
            col.append(v)
        col.append('y')
        call = day.loc[:,col].reset_index(drop=True)
        call_id = day.loc[:,['optionid', 'S', 'mid_p', 'strike_un', 'strike', 'T']].reset_index(drop=True)
        ind = np.arange(0, call.shape[0])
        np.random.shuffle(ind)
        call = call.iloc[ind, :]
        call_id = call_id.iloc[ind, :]

        x_call = call.drop(columns='y').iloc[:, :]
        y_call = call[['y']].iloc[:, :]
        x_call, y_call = self.c_model.normalize(x_call, y_call)

        if return_id:
            return x_call, y_call, call_id
        else:
            return x_call, y_call

    def estimate_opt_parameters(self, DAY=[], check_neighboor=False):
        data = DataReal()
        data = data.load_all()
        ID = -1
        time_start_abs = time.time()
        data = data.rename(columns={'iv':'y'})

        if len(DAY) == 0:
            T_IND = np.sort(data['t_ind'].unique())[self.NB_DAYS:]
        else:
            T_IND = np.array(DAY)

        done_list = os.listdir(self.save_dir)

        ft = '{:<20}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}'
        f = '{:<20}{:>12d}{:>12d}{:>12f}{:>12f}{:>12f}{:>12f}'
        print(ft.format(' Days to fit: ' + str(len(T_IND)), 'ID', 'T_IND', 'T. time', 'R. Time', 'Avg. Time', 'Rest (H)'), flush=True)
        print('-' * 12 * 8, flush=True)
        nb_change = 0
        for d in T_IND:
            # loading the old versions for additional init points
            save_dir = self.save_dir + '/'
            n = 'perf_' + str(d) + '_best.csv'
            if n in done_list:
                perf_old = pd.read_csv(save_dir + 'perf_' + str(d) + '.csv', index_col=0)
                if 'obj' not in perf_old.columns:
                    # compatibiltiy with old code without the obj columns
                    perf_old['obj'] = perf['mse']
                res_old = pd.read_csv(save_dir + 'res_' + str(d) + '.csv', index_col=0)
                old_best_x = np.load(save_dir + 'best_x_' + str(d) + '.npy')
                comp_list = perf_old.index.tolist()
            else:
                comp_list = []

            time_start_round = time.time()
            ID += 1




            day = data.loc[data['t_ind'] == d,:]
            X, Y = self.pre_process_day(day)

            r = self.par_c.process.draw_values(smaller_range=True)
            for k in (self.par_c.data.cross_vary_list):
                r.pop(k)

            par_list = list(r.keys())

            iDim = len(par_list) + self.NB_DAYS - 1  # one v0 per day


            P = []
            R = []
            BEST_PAR = [old_best_x for x in comp_list]

            if check_neighboor:
                # print(save_dir + 'perf_' + str(d) + '_best.csv')
                # print('start best', pd.read_csv(save_dir + 'perf_' + str(d) + '_best.csv',header=0,index_col=0).loc['obj',:].iloc[0])
                for dD in [-1, 1]:
                    file = save_dir+'best_x_'+str(d-dD)+'.npy'
                    if os.path.exists(file):
                        init_x = np.load(file)
                        res, perf, best_par = self.solve_for_single_day(init_x, X, Y, est_gmm=False)
                        R.append(res)
                        P.append(perf)
                        BEST_PAR.append(best_par)
            else:
                for i in range(self.par.pricer.nb_init):
                    if not i in comp_list:
                        init_x = np.random.uniform(-self.pivot, self.pivot, size=iDim) * (i != 0)
                        res, perf, best_par = self.solve_for_single_day(init_x, X, Y, est_gmm=False)
                        R.append(res)
                        P.append(perf)
                        BEST_PAR.append(best_par)

            p = pd.concat(P, axis=1).T.reset_index(drop=True)
            r = pd.concat(R, axis=1).T.reset_index(drop=True)
            if len(comp_list) > 0:
                p.index = [x + len(comp_list) for x in p.index]
                r.index = [x + len(comp_list) for x in r.index]

                r = res_old.append(r)
                p = perf_old.append(p)

            best_id = p['obj'].idxmin()

            # r.loc[p['r2']>0.95,:].agg(['mean','std'])
            # saving the results
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            r.to_csv(save_dir + '/res_' + str(d) + '.csv')
            p.to_csv(save_dir + '/perf_' + str(d) + '.csv')


            if best_id not in comp_list:
                nb_change+=1
                print('New best!')
                # init_x = BEST_PAR[best_id];est_gmm = True
                # res, perf, pr_list = self.find_opt_params(par_list, BEST_PAR[best_id], X_put, Y_put, X_call, Y_call, vol, est_gmm=True)

                perf = p.loc[best_id,:]
                perf.name = 0
                pr_list = BEST_PAR[best_id]
                res = r.iloc[best_id,:]
                res.name=0

                perf.to_csv(save_dir + '/perf_' + str(d) + '_best' + '.csv')
                res.to_csv(save_dir + '/res_' + str(d) + '_best' + '.csv')
                np.save(save_dir + '/best_x_' + str(d) + '.npy', pr_list)

            tot_time = np.round((time.time() - time_start_abs) / 60, 2)
            rnd_time = np.round((time.time() - time_start_round) / 60, 2)
            av_time = np.round(tot_time / (1 + ID), 2)
            e_time = ((len(T_IND) - (ID + 1)) * av_time) / 60
            print(f.format(' ', ID, d, tot_time, rnd_time, av_time, e_time), flush=True)
            print('Changed', nb_change, 'out of', ID)

    def get_anderson(self, DAY=[]):
        data = DataReal()
        data = data.load_all()
        ID = -1
        time_start_abs = time.time()
        data = data.rename(columns={'iv':'y'})

        if len(DAY) == 0:
            T_IND = np.sort(data['t_ind'].unique())[self.NB_DAYS:]
        else:
            T_IND = np.array(DAY)

        done_list = os.listdir(self.save_dir)

        ft = '{:<20}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}'
        f = '{:<20}{:>12d}{:>12d}{:>12f}{:>12f}{:>12f}{:>12f}'
        print(ft.format(' Days to fit: ' + str(len(T_IND)), 'ID', 'T_IND', 'T. time', 'R. Time', 'Avg. Time', 'Rest (H)'), flush=True)
        print('-' * 12 * 8, flush=True)
        for d in T_IND:
            # loading the old versions for additional init points
            save_dir = self.save_dir + '/'

            # load the current best init x
            file = save_dir + 'best_x_' + str(d) + '.npy'
            if os.path.exists(file):
                init_x = np.load(file)
                time_start_round = time.time()
                ID += 1


                day = data.loc[data['t_ind'] == d,:].copy()
                X, Y = self.pre_process_day(day)

                params_df, true_s, true_opt = self.c_model.split_state_data_par(X)
                true_opt = tf.convert_to_tensor(true_opt.values)

                init_x = tf.reshape(init_x, (1, -1))

                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(init_x)
                    par_est = init_x[:, :-1]
                    state = init_x[:, -1:]
                    p = self.c_model.model([par_est, state, true_opt])
                    p_split = tf.split(p, num_or_size_splits=p.shape[0])
                    # pred_call = tf.split(pred_call, num_or_size_splits=pred_call.shape[0])
                    # pred_put = tf.split(pred_put, num_or_size_splits=pred_put.shape[0])

                grads = []
                for i in range(len(p_split)):
                    grads.append(tape.gradient(p_split[i], [init_x]))

                g = [gg[0] for gg in grads]
                g = tf.concat(g, 0).numpy()
                y =Y.values

                # for now this system simply plug g p y into a list so that the code can easily expand to multiple days
                gg = [g]
                pp = [p]
                yy = [y]
                # Use this to expand on a multiple day version
                # for j in range(self.NB_DAYS):
                #     g, p, y = ander_g(opt_values, j)
                #     gg.append(g)
                #     pp.append(p)
                #     yy.append(y)


                par_est_len = max(init_x.shape)-1
                ## creating the H building blocks
                Htt = []
                Omega_tt = []
                for j in range(self.NB_DAYS):
                    dS = gg[j][:, par_est_len + j]
                    t = np.sum(dS ** 2) / dS.shape[0]
                    Htt.append(t)
                    e = (yy[j][:, 0] - pp[j][:, 0]) ** 2
                    Omega_tt.append(np.sum(e * (dS ** 2)) / dS.shape[0])

                HTpTp = 0
                Omega_TpTp = 0
                N = []
                for j in range(self.NB_DAYS):
                    dTheta = gg[j][:, :par_est_len]
                    temp_omega = 0
                    for i in range(dTheta.shape[0]):
                        t = dTheta[i, :].reshape(-1, 1)
                        t = (t @ t.T)
                        HTpTp = HTpTp + t
                        temp_omega += ((yy[j][i, 0] - pp[j][i, 0]) ** 2) * t
                    Omega_TpTp += temp_omega / (dTheta.shape[0] ** 2)
                    HTpTp = HTpTp / dTheta.shape[0]
                    N.append(dTheta.shape[0])
                Omega_TpTp = Omega_TpTp * (np.sum(N) / self.NB_DAYS)

                Hside = []
                Omega_side = []
                for j in range(self.NB_DAYS):
                    dS = gg[j][:, par_est_len + j]
                    dTheta = gg[j][:, :par_est_len]
                    temp = 0
                    temp_omega = 0
                    for i in range(dTheta.shape[0]):
                        t = dTheta[i, :].reshape(-1, 1)
                        temp += dS[i] * t.T
                        temp_omega += ((yy[j][i, 0] - pp[j][i, 0]) ** 2) * dS[i] * t.T
                    temp = temp / dTheta.shape[0]
                    temp_omega = temp_omega * np.sqrt(np.sum(N)) / np.sqrt(self.NB_DAYS * N[j] ** 3)
                    Hside.append(temp)
                    Omega_side.append(temp_omega)
                Hside = np.concatenate(Hside, axis=0)
                Omega_side = np.concatenate(Omega_side, axis=0)

                # creating the big H and Omega matrix
                top_left = np.zeros(shape=(self.NB_DAYS, self.NB_DAYS))
                top_left_omega = np.zeros(shape=(self.NB_DAYS, self.NB_DAYS))
                for j in range(self.NB_DAYS):
                    top_left[j, j] = Htt[j]
                    top_left_omega[j, j] = Omega_tt[j]
                left = np.concatenate([top_left, Hside.T], axis=0)
                right = np.concatenate([Hside, HTpTp], axis=0)

                left_omega = np.concatenate([top_left_omega, Omega_side.T], axis=0)
                right_omega = np.concatenate([Omega_side, Omega_TpTp], axis=0)
                H = np.concatenate([left, right], axis=1)
                Omega = np.concatenate([left_omega, right_omega], axis=1)
                # finally we can estimate the standard errors
                s = N + [np.sum(N) / self.NB_DAYS for x in range(par_est_len)]
                # s = np.sqrt(s)
                H_inv = np.linalg.inv(H)
                sig = H_inv @ Omega @ H_inv.T
                sig = (np.diag(sig) / s) ** (1 / 2)

                # sig=[np.mean(np.abs(dS))]+np.mean(np.abs(dTheta),0).tolist()



                ind = ['v' + str(i) for i in range(self.NB_DAYS)] + list(params_df.columns)
                sig = pd.DataFrame(data={'sig': sig}, index=ind)
                sig.to_csv(save_dir+f'anderson_std_{d}.csv')
            else:
                print('Missing day', d, flush=True)
            # print(sig)
            tot_time = np.round((time.time() - time_start_abs) / 60, 2)
            rnd_time = np.round((time.time() - time_start_round) / 60, 2)
            av_time = np.round(tot_time / (1 + ID), 2)
            e_time = ((len(T_IND) - (ID + 1)) * av_time) / 60
            print(f.format(' ', ID, d, tot_time, rnd_time, av_time, e_time), flush=True)


    def solve_for_single_day(self, init_x, X, Y, est_gmm=False):
        params_df, true_s, true_opt = self.c_model.split_state_data_par(X)
        true_opt = tf.convert_to_tensor(true_opt.values)
        tf_loss = tf.keras.losses.MSE

        bounds = tf.fill((1, len(init_x)), self.pivot)
        bound_cost = tf.constant(100.0, dtype=tf.float64)

        @tf.function
        def func(x_params):
            x_params = tf.reshape(x_params, (1, -1))
            par_est = x_params[:, :-1]
            state = x_params[:, -1:]
            pred = self.c_model.model([par_est, state, true_opt])
            v_call = tf_loss(pred, Y)
            bnd = tf.reduce_sum(tf.nn.relu(x_params - bounds) + tf.nn.relu(-(x_params + bounds))) * bound_cost
            return tf.reduce_mean(v_call) + tf.reduce_mean(bnd)

        @tf.function
        def func_g(x_params):
            with tf.GradientTape() as tape:
                tape.watch(x_params)
                loss_value = func(x_params)
            grads = tape.gradient(loss_value, [x_params])
            return loss_value, grads[0]

        s = time.time()
        soln = tfp.optimizer.lbfgs_minimize(func_g, init_x, max_iterations=50, tolerance=1e-60)
        soln_time = np.round((time.time() - s) / 60, 2)
        pred_par = soln.position.numpy()
        obj_value = soln.objective_value.numpy()

        for ii in range(params_df.shape[1]):
            X.loc[:, params_df.columns[ii]] = pred_par[ii]
        X['v0'] = pred_par[ii + 1]

        score=self.c_model.score(self.c_model.unnormalize(X)[0],Y)

        perf=pd.Series(list(score)+[obj_value, soln_time], index=['r2','mae','mse','obj','time'])
        pr = self.c_model.unnormalize(X)[0].iloc[0, :]
        res=pr[pd.Series(pr.index.tolist())[~pd.Series(pr.index.tolist()).isin(self.par_c.data.cross_vary_list)]]
        return res, perf, pred_par





    def find_opt_params(self, par_list, init_x, X_put, Y_put, X_call, Y_call, vol, est_gmm=False, perf_only=False, return_pred=False, granular_return=False):
        true_rf = [x['rf'].iloc[0] for x in X_call]
        true_rf = [tf.convert_to_tensor(np.array(x).reshape(1, 1)) for x in true_rf]
        opt_data_call_day = []
        y_call_day = []
        opt_data_put_day = []
        y_put_day = []
        days_len_call = []
        days_len_put = []
        # creating the concatenated batches of days and y
        for i in range(self.NB_DAYS):
            X = self.c_model.split_state_data_par(X_call[i])
            par_est = tf.convert_to_tensor(X[0].iloc[0:1, :].values)
            par_est_len = par_est.shape[1]
            state = tf.convert_to_tensor(X[1].iloc[0:1, :])
            opt_data_call_day.append(tf.convert_to_tensor(X[2]))

            y_call_day.append(tf.convert_to_tensor(Y_call[i].values))
            days_len_call.append(X_call[i].shape[0])

            X = self.c_model.split_state_data_par(X_put[i])
            opt_data_put_day.append(tf.convert_to_tensor(X[2]))
            y_put_day.append(tf.convert_to_tensor(Y_put[i].values))
            days_len_put.append(X_put[i].shape[0])

        y_call = tf.concat(y_call_day, axis=0)
        opt_data_call = tf.concat(opt_data_call_day, axis=0)
        y_put = tf.concat(y_put_day, axis=0)
        opt_data_put = tf.concat(opt_data_put_day, axis=0)

        # taking care of the templates
        v_temp_call = []
        v_temp_put = []
        rf_temp_call = []
        rf_temp_put = []

        for i in range(self.NB_DAYS):
            t0 = tf.fill(dims=(X_call[i].shape[0], state.shape[1]), value=0.0)
            t0 = tf.cast(t0, tf.float64)
            t1 = tf.fill(dims=(X_call[i].shape[0], state.shape[1]), value=1.0)
            t1 = tf.cast(t1, tf.float64)
            left = tf.concat((t1[:, 0:1], t0[:, 0:1]), 1)
            right = tf.concat((t0[:, 0:1], t1[:, 0:1]), 1)
            # the two above are when we have one additional state
            # v_temp_call.append(left)
            # rf_temp_call.append(right)
            v_temp_call.append(t1)
            rf_temp_call.append(t1)

            t0 = tf.fill(dims=(X_put[i].shape[0], state.shape[1]), value=0.0)
            t0 = tf.cast(t0, tf.float64)
            t1 = tf.fill(dims=(X_put[i].shape[0], state.shape[1]), value=1.0)
            t1 = tf.cast(t1, tf.float64)
            left = tf.concat((t1[:, 0:1], t0[:, 0:1]), 1)
            right = tf.concat((t0[:, 0:1], t1[:, 0:1]), 1)
            #  same as up
            # v_temp_put.append(left)
            # rf_temp_put.append(right)
            v_temp_put.append(t1)
            rf_temp_put.append(t1)

        vol_call = []
        rf_call = []
        vol_put = []
        rf_put = []
        for i in range(self.NB_DAYS):
            vc = []
            rc = []
            vp = []
            rp = []
            for j in range(self.NB_DAYS):
                if i == j:
                    vc.append(v_temp_call[i])
                    rc.append(rf_temp_call[i])
                    vp.append(v_temp_put[i])
                    rp.append(rf_temp_put[i])
                else:
                    vc.append(v_temp_call[j] * 0)
                    rc.append(rf_temp_call[j] * 0)
                    vp.append(v_temp_put[j] * 0)
                    rp.append(rf_temp_put[j] * 0)
            vol_call.append(tf.concat(vc, 0))
            rf_call.append(tf.concat(rc, 0))
            vol_put.append(tf.concat(vp, 0))
            rf_put.append(tf.concat(rp, 0))

        # add the len vector
        l_c = tf.cast(tf.constant(y_call.shape[0]), tf.float64)
        l_p = tf.cast(tf.constant(y_put.shape[0]), tf.float64)
        l_tot = l_c + l_p

        ### create the par_name_vector

        par_name = X[0].columns.tolist() + X[1].columns.tolist()


        tf_loss = tf.keras.losses.MAE

        s = state
        x_params = init_x  # DEBUG

        vol_tf = tf.convert_to_tensor(vol)
        vol_m = tf.convert_to_tensor(self.c_model.m['v0'])
        vol_s = tf.convert_to_tensor(self.c_model.std['v0'])
        lbda_cost = tf.cast(tf.convert_to_tensor(self.par.pricer.lbda), dtype=tf.float64)

        bounds = tf.fill((1, len(init_x)), self.pivot)
        bound_cost = tf.constant(1.0, dtype=tf.float64)

        @tf.function
        def func(x_params, return_mean=True, return_error=True):
            x_params = tf.reshape(x_params, (1, -1))
            par_est = x_params[:, :par_est_len]
            state_call = None
            state_put = None
            for j in range(self.NB_DAYS):
                s = x_params[:, (par_est_len + j):(par_est_len + j + 1)]
                if state_call is None:
                    state_call = vol_call[j] * s # + rf_call[j] * true_rf[j]
                    state_put = vol_put[j] * s # + rf_put[j] * true_rf[j]
                else:
                    state_call = state_call + vol_call[j] * s # + rf_call[j] * true_rf[j]
                    state_put = state_put + vol_put[j] * s # + rf_put[j] * true_rf[j]

            pred_call = self.c_model.model([par_est, state_call, opt_data_call])
            v_call = tf_loss(pred_call, y_call)
            pred_put = self.p_model.model([par_est, state_put, opt_data_put])
            v_put = tf_loss(pred_put, y_put)

            if return_mean:
                bnd = tf.reduce_sum(tf.nn.relu(x_params - bounds) + tf.nn.relu(-(x_params + bounds))) * bound_cost
                v = ((tf.reduce_mean(v_call) * l_c + tf.reduce_mean(v_put) * l_p) / l_tot) + bnd
                if self.par.pricer.cost_function == CostFunction.VOL_LASSO:
                    s = x_params[:, (par_est_len):(par_est_len + self.NB_DAYS + 1)] * vol_s + vol_m
                    s = tf.reduce_sum(tf.abs(s - vol_tf)) * lbda_cost
                    # s = tf.reduce_sum(tf.square(tf.sqrt(s)-tf.sqrt(vol_tf)))*lbda_cost
                    v = v + s
                return v
            else:
                if return_error:
                    return tf.concat([v_call, v_put], 0)
                else:
                    return tf.concat([pred_call, pred_put], 0), tf.concat([y_call, y_put], 0)

        # @tf.function
        def gmm_g(x_params):
            x_params = tf.convert_to_tensor(x_params)
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x_params)
                u = func(x_params, return_mean=False)
                v = tf.split(u, num_or_size_splits=u.shape[0])
            grads = []
            for i in range(u.shape[0]):
                grads.append(tape.gradient(v[i], [x_params]))
            del tape
            return grads, u

        @tf.function
        def func_g(x_params):
            with tf.GradientTape() as tape:
                tape.watch(x_params)
                loss_value = func(x_params)
            grads = tape.gradient(loss_value, [x_params])
            return loss_value, grads[0]

        if (not est_gmm) & (not perf_only):
            s = time.time()
            # soln = tfp.optimizer.bfgs_minimize(func_g, init_x, tolerance=1e-8, parallel_iterations=1, max_iterations=500)
            # soln = tfp.optimizer.bfgs_minimize(func_g, init_x, tolerance=1e-60, parallel_iterations=1, max_iterations=50)
            # soln = tfp.optimizer.lbfgs_minimize(func_g, init_x)
            # soln = tfp.optimizer.bfgs_minimize(func_g, init_x,tolerance=1e-60)
            soln = tfp.optimizer.bfgs_minimize(func_g, init_x)
            soln_time = np.round((time.time() - s) / 60, 2)
            opt_values = soln.position.numpy()
            obj_value = soln.objective_value.numpy()
            print('finish running, obj,',obj_value, 'time,', soln_time,flush=True)
        else:
            opt_values = init_x
            soln_time = -1
            obj_value = func(opt_values).numpy()

        pred_ = []
        true_ = []
        par_ = []
        for j in range(self.NB_DAYS):
            for i, p in enumerate(par_name):
                if p == 'v0':
                    X_call[j][p] = opt_values[par_est_len + j]
                    X_put[j][p] = opt_values[par_est_len + j]
                else:
                    X_call[j][p] = opt_values[i]
                    X_put[j][p] = opt_values[i]


            call = self.c_model.unnormalize(X_call[j])[0]
            put = self.p_model.unnormalize(X_put[j])[0]

            # for c in call.columns:
            #     if c not in ['strike','T']:
            #         call[c] = true_par[c]

            self.c_model.score(call,Y_call[j])

            c_pred = self.c_model.predict(call)
            pred_.append(c_pred)
            true_.append(Y_call[j])

            # ((c_pred - Y_call[j]) ** 2).mean()
            # self.c_model.score(call, Y_call[j])
            #
            # ((p_pred - Y_put[j]) ** 2).mean()
            # self.p_model.score(put, Y_call[j])

            p_pred = self.p_model.predict(put)
            pred_.append(p_pred)
            true_.append(Y_put[j])
            par_.append(call.iloc[0, :].drop(index=self.par_c.data.cross_vary_list).rename(index={'v0': 'v' + str(j)}))

        true_ = pd.concat(true_, axis=0).values
        pred_ = np.concatenate(pred_, axis=0)
        res = pd.concat(par_).drop_duplicates()  # .sort_index()

        perf = pd.Series({
            'mae': metrics.mean_absolute_error(true_, pred_),
            'mse': metrics.mean_squared_error(true_, pred_),
            'time': soln_time,
            'r2': metrics.r2_score(true_, pred_),
            'obj': obj_value
        })
        res.name = 'pred'
        res = pd.DataFrame(res)

        ##################
        # Add anderson std
        ##################

        if est_gmm:
            def ander_g(x_params, j):
                x_params = tf.convert_to_tensor(x_params)
                x_params = tf.reshape(x_params, (1, -1))

                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(x_params)
                    par_est = x_params[:, :par_est_len]
                    state = x_params[:, (par_est_len + j):(par_est_len + j + 1)]
                    state = tf.concat([state, true_rf[j]], axis=1)
                    pred_call = self.c_model.model([par_est, state, opt_data_call_day[j]])
                    pred_put = self.p_model.model([par_est, state, opt_data_put_day[j]])
                    p = tf.concat([pred_call, pred_put], 0).numpy()
                    y = tf.concat([y_call_day[j], y_put_day[j]], 0).numpy()
                    pred_call = tf.split(pred_call, num_or_size_splits=pred_call.shape[0])
                    pred_put = tf.split(pred_put, num_or_size_splits=pred_put.shape[0])

                grads = []
                for i in range(len(pred_call)):
                    grads.append(tape.gradient(pred_call[i], [x_params]))
                for i in range(len(pred_put)):
                    grads.append(tape.gradient(pred_put[i], [x_params]))

                g = [gg[0] for gg in grads]
                g = tf.concat(g, 0).numpy()
                del tape
                return g, p, y

            gg = []
            pp = []
            yy = []
            for j in range(self.NB_DAYS):
                g, p, y = ander_g(opt_values, j)
                gg.append(g)
                pp.append(p)
                yy.append(y)

            ## creating the H building blocks
            Htt = []
            Omega_tt = []
            for j in range(self.NB_DAYS):
                dS = gg[j][:, par_est_len + j]
                t = np.sum(dS ** 2) / dS.shape[0]
                Htt.append(t)
                e = (yy[j][:, 0] - pp[j][:, 0]) ** 2
                Omega_tt.append(np.sum(e * (dS ** 2)) / dS.shape[0])

            HTpTp = 0
            Omega_TpTp = 0
            N = []
            for j in range(self.NB_DAYS):
                dTheta = gg[j][:, :par_est_len]
                temp_omega = 0
                for i in range(dTheta.shape[0]):
                    t = dTheta[i, :].reshape(-1, 1)
                    t = (t @ t.T)
                    HTpTp = HTpTp + t
                    temp_omega += ((yy[j][i, 0] - pp[j][i, 0]) ** 2) * t
                Omega_TpTp += temp_omega / (dTheta.shape[0] ** 2)
                HTpTp = HTpTp / dTheta.shape[0]
                N.append(dTheta.shape[0])
            Omega_TpTp = Omega_TpTp * (np.sum(N) / self.NB_DAYS)

            Hside = []
            Omega_side = []
            for j in range(self.NB_DAYS):
                dS = gg[j][:, par_est_len + j]
                dTheta = gg[j][:, :par_est_len]
                temp = 0
                temp_omega = 0
                for i in range(dTheta.shape[0]):
                    t = dTheta[i, :].reshape(-1, 1)
                    temp += dS[i] * t.T
                    temp_omega += ((yy[j][i, 0] - pp[j][i, 0]) ** 2) * dS[i] * t.T
                temp = temp / dTheta.shape[0]
                temp_omega = temp_omega * np.sqrt(np.sum(N)) / np.sqrt(self.NB_DAYS * N[j] ** 3)
                Hside.append(temp)
                Omega_side.append(temp_omega)
            Hside = np.concatenate(Hside, axis=0)
            Omega_side = np.concatenate(Omega_side, axis=0)

            # creating the big H and Omega matrix
            top_left = np.zeros(shape=(self.NB_DAYS, self.NB_DAYS))
            top_left_omega = np.zeros(shape=(self.NB_DAYS, self.NB_DAYS))
            for j in range(self.NB_DAYS):
                top_left[j, j] = Htt[j]
                top_left_omega[j, j] = Omega_tt[j]
            left = np.concatenate([top_left, Hside.T], axis=0)
            right = np.concatenate([Hside, HTpTp], axis=0)

            left_omega = np.concatenate([top_left_omega, Omega_side.T], axis=0)
            right_omega = np.concatenate([Omega_side, Omega_TpTp], axis=0)
            H = np.concatenate([left, right], axis=1)
            Omega = np.concatenate([left_omega, right_omega], axis=1)
            # finally we can estimate the standard errors
            s = N + [np.sum(N) / self.NB_DAYS for x in range(par_est_len)]
            # s = np.sqrt(s)
            H_inv = np.linalg.inv(H)
            sig = H_inv @ Omega @ H_inv.T
            sig = (np.diag(sig) / s) ** (1 / 2)

            # sig=[np.mean(np.abs(dS))]+np.mean(np.abs(dTheta),0).tolist()

            ind = ['v' + str(i) for i in range(self.NB_DAYS)] + par_name[:par_est_len]
            sig = pd.DataFrame(data={'sig': sig}, index=ind)
            res = res.merge(sig, right_index=True, left_index=True)

            ###############
            #  ADD GMM
            ###############
            # d, u = gmm_g(opt_values)
            # d = np.concatenate(d)
            # u = u.numpy()
            # W = np.identity(d.shape[0])
            # a = d.T @ W
            # ad = a @ d
            # ad_inv = np.linalg.inv(ad)
            #
            # U = u.reshape(1, -1)
            # S = U.T @ U
            # std_err = ad_inv @ a @ S @ a.T @ ad_inv.T
            #
            # # add
            # ind = par_name[:par_est_len] + ['v' + str(x) for x in range(self.NB_DAYS)]
            # res = res.loc[ind, :]
            # res['gmm'] = np.diag(std_err)

        if return_pred:
            return res, perf, opt_values, c_pred, p_pred
        else:
            return res, perf, opt_values



    def get_oss_perf(self, DAY=[], best_par_list=[], get_gradient=True):
        data = DataReal()
        vol_data = data.load_cv()
        data = data.load_all()
        ID = -1
        RES = []
        if len(DAY) == 0:
            T_IND = np.sort(data['t_ind'].unique())[self.NB_DAYS:]
        else:
            T_IND = np.array(DAY)

        ft = '{:<20}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}'
        f = '{:<20}{:>12d}{:>12d}{:>12f}{:>12f}{:>12f}{:>12f}'
        print(ft.format(' Days to fit: ' + str(len(T_IND)), 'ID', 'T_IND', 'T. time', 'R. Time', 'Avg. Time', 'Rest (H)'), flush=True)

        print('-' * 15 * 8, flush=True)

        time_start = time.time()
        for d in T_IND:
            time_round_start = time.time()
            ID += 1

            # single day
            # x_put, y_put, x_call, y_call = self.pre_process_day(day)
            X_put, Y_put, X_call, Y_call = [], [], [], []
            vol = []
            for i in range(self.NB_DAYS):
                ind = (data['t_ind'] == d - i)
                day = data.loc[ind, :]

                try:
                    vol.append(vol_data.loc[vol_data['t_ind'] == d - i + 1, 'sv'].iloc[0] / 100)
                except:
                    vol.append(-1)
                if get_gradient:
                    x_put_, y_put_, x_call_, y_call_, put_id_, call_id_ = self.pre_process_day(day, True)
                else:
                    x_put_, y_put_, x_call_, y_call_ = self.pre_process_day(day)
                X_put.append(x_put_)
                Y_put.append(y_put_)
                X_call.append(x_call_)
                Y_call.append(y_call_)

            for m in [self.c_model, self.p_model]:
                for l in m.model.layers:
                    l.trainable = False

            r = self.par_c.process.draw_values(smaller_range=True)
            for k in (self.par_c.data.cross_vary_list + ['rf']):
                r.pop(k)
            # TODO add the renaming of v0s into v1, v2... in the data directly, and add them to par_list
            par_list = list(r.keys())

            # saving the results

            # init_x = BEST_PAR[best_id];est_gmm = True
            try:
                if get_gradient:
                    res, perf, best_par, c_call, c_put = self.find_opt_params(par_list, best_par_list[ID], X_put, Y_put, X_call, Y_call, vol, est_gmm=False, perf_only=True, return_pred=True)
                    d = call_id_[['S', 'strike_un', 'T', 'mid_p']].rename(columns={'strike_un': 'strike'})
                    d['sigma'] = c_call
                    d['rf'] = day['rf'].iloc[0]
                    from ql_pricing import BSPricer
                    par = Params()
                    par.opt.process = Process.GBM

                    par.opt.option_type = OptionType.CALL_EUR
                    par.update_process()
                    pp = BSPricer(par)
                    d['p'] = pp.BlackScholes_price(d)
                    d.describe()
                    (d['p'] - d['mid_p']).abs().mean()

                    temp = d
                    d['strike'] = d['strike'].round(1)
                    d.groupby('strike').mean()['p'].plot()
                    plt.show()

                else:
                    # res, perf, best_par = self.find_opt_params(par_list, best_par_list[ID], X_put, Y_put, X_call, Y_call, vol, est_gmm=False, perf_only=True, return_pred =False, granular_return = True)
                    res, perf, best_par, c_pred, p_pred = self.find_opt_params(par_list, best_par_list[ID], X_put, Y_put, X_call, Y_call, vol, est_gmm=False, perf_only=True, return_pred=True, granular_return=True)
                    t = Y_put[0].values - p_pred
                    tt = Y_call[0] - c_pred
                    PUT = Y_put[0]
                    PUT['y_pred'] = p_pred
                    CALL = Y_call[0]
                    CALL['y_pred'] = c_pred
                    dir_ = self.save_dir + '/granular/'
                    if not os.path.exists(dir_):
                        os.makedirs(dir_)
                    CALL.to_pickle(dir_ + str(d) + '_call.p')
                    PUT.to_pickle(dir_ + str(d) + '_put.p')

                    t = np.concatenate([t, tt])
                    np.mean(np.abs(t))

                    perf.name = d
                    RES.append(perf)

                r_time = np.round((time.time() - time_round_start) / 60, 10)
                t_time = np.round((time.time() - time_start) / 60, 10)
                a_time = np.round(t_time / (ID + 1), 10)
                rest_time = np.round((len(T_IND) - ID - 1) * a_time, 2)

                print(f.format(' ', ID, d, t_time, r_time, a_time, rest_time), flush=True)
            except:
                print('Exception', ID, flush=True)

        return RES


# param = Params()
# param.opt.process = Process.DOUBLE_EXP
# param.pricer.fit_process = Process.DOUBLE_EXP
# param.update_process()
# param.update_model_name()
# self = Pricer(param)
# NB_DAYS=1; model_names=[]
# DAY = []; check_neighboor = False