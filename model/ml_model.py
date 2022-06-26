# ML_modelpyhon -m venv 
# Created by Guillaume Pavé, at xx.xx.xx
import os
import pickle
import socket
from numpy import dtype
import tensorflow as tf
# remove tensorflow warnings
tf.get_logger().setLevel('ERROR')
tf.config.optimizer.set_jit(True)
tf.compat.v1.OptimizerOptions(cpu_global_jit=True)
tf.keras.mixed_precision.set_global_policy('mixed_float16')
print(f"Number of GPUs available: {tf.test.is_gpu_available()}")

from tensorflow import keras
from tensorflow.keras import layers
import tensorboard
import datetime
try:
    from parameters import *
except ModuleNotFoundError:
    from model.parameters import *
    
import pandas as pd
import numpy as np

tf.random.set_seed(12345)
class FirstLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs, par):
        super(FirstLayer, self).__init__(dtype='float64')
        self.num_outputs = num_outputs
        self.par = par

        c = []
        y = ['MLE']
        opt_data = self.par.data.cross_vary_list
        for cc in self.par.process.__dict__.keys():
            if (cc not in opt_data):
                c.append(cc)
        
        self.l1 = len(c)
        self.l2 = len(opt_data)

    def build(self,input_shape):
        self.kernel_par = self.add_weight("kernel_par",
                                          shape=[self.l1,
                                                 self.num_outputs], dtype=tf.float64)
        self.kernel_state = self.add_weight("kernel_data",
                                                shape=[self.l2,
                                                    self.num_outputs], dtype=tf.float64)

    def call(self,input):
        
        r = tf.matmul(input[0], self.kernel_par) + tf.matmul(tf.cast(input[1],tf.float64), self.kernel_state)
        
        r = tf.nn.swish(r)
        return r



class NetworkModel:
    def __init__(self, par: Params()):
        self.par = par
        self.model = None

        self.m = None
        self.std = None
        self.m_y = None
        self.std_y = None
        if socket.gethostname() in ['MBP-de-admin']:
            print("Guillaume's computer")

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

    def split_state_data_par(self, df):

        opt_data = df[['buy', 'sell']]
        c = []
        for cc in self.par.process.__dict__.keys():
            # for cc in df.columns:
            if (cc not in opt_data.columns):
                c.append(cc)

        par_est = df[c]

        return [par_est, opt_data]

    def train(self): 

        c = []
        y = ['MLE'] # LE estimation
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
                
        self.m = pd.Series(m) #column in index
        self.std = pd.Series(std)
    
        ##################
        # prepare data sets
        ##################
        def tr(x):
            y = (x[:,:c1], x[:,c1:])
            
            return y

        print("=== Loading data ===")
        if self.par.opt.process.name == Process.PIN.name:
            data_dir = self.par.data.path_sim_save + 'simulation_data_PIN.txt'
        else:
            data_dir = self.par.data.path_sim_save + 'APIN_MLE.txt'
        
        data = pd.read_csv(data_dir,on_bad_lines='skip')
        print(f"shape of the data: {data.shape}")

        if self.model is None:
            self.create_nnet_model()


        # create splitting data
       
        y_data = data[y]
        x_data = data[c + opt_data]
        
        x_data, y_none = self.normalize(x_data)
        final_data = self.split_state_data_par(x_data)
        #Create a callback that saves the model's weights
        log_dir = "./model/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.save_dir + '/', save_weights_only=True, verbose=0, save_best_only=True)
        print('start training for', self.par.model.E, 'epochs', flush=True)

        self.history_training = self.model.fit(x=final_data, y=y_data, validation_split=0.01,batch_size=self.par.model.batch_size, validation_batch_size=self.par.model.batch_size,epochs=self.par.model.E ,callbacks=[tensorboard_callback,cp_callback], verbose=1,use_multiprocessing=True)  # Pass callback to training
        

        self.history_training = pd.DataFrame(self.history_training.history)
        self.save()

    def predict(self, X):
        X, y = self.normalize(X, y=None)
        X = self.split_state_data_par(X)
        pred = self.model.predict(X)
        return pred

    def score(self,X,y):
        X, y = self.normalize(X, y)
        X = self.split_state_data_par(X)
        loss, mae, mse, r2 = self.model.evaluate(X, y, verbose=0)
        df = pd.DataFrame(np.array([mae,mse,r2]))
        df = df.T
        df.columns = ["mae","mse","r2"]
        return df

    def get_grad_and_mle(self, X, return_pred=False):
        """
        function that generate gradient and MLE
        """
        X, y = self.normalize(X, y=None)
        X = self.split_state_data_par(X)

        xx = [tf.convert_to_tensor(x.values) for x in X]
        
        with tf.GradientTape(persistent=True) as g:
            g.watch(xx[0])
            g.watch(xx[1])
            pred = self.model(xx)
            
        d = g.gradient(pred, xx[0])
        d1 = g.gradient(pred, xx[1])
        del g
        d = np.concatenate([d.numpy(), d1.numpy()], axis=1)
        col = list(X[0].columns) + list(X[1].columns)

        d = pd.DataFrame(d, columns=col, index=X[0].index)
        d = d*(1/self.std)

        if return_pred:
            return d, pred
        else:
            return d

    def save(self, other_save_dir=None):
        """
        function to save a file
        """
        self.par.save(save_dir=self.save_dir)

        with open(self.save_dir + '/m' + '.p', 'wb') as handle:
            pickle.dump(self.m, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.save_dir + '/std' + '.p', 'wb') as handle:
            pickle.dump(self.std, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.save_dir + '/m_y' + '.p', 'wb') as handle:
            pickle.dump(self.m_y, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.save_dir + '/std_y' + '.p', 'wb') as handle:
            pickle.dump(self.std_y, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.history_training.to_pickle(self.save_dir + '/history.p')

    def load(self, n, other_save_dir=None):

        self.par.name = n
        if other_save_dir is None:
            temp_dir = self.par.model.save_dir + '' + self.par.name
        else:
            temp_dir = other_save_dir + '' + self.par.name

        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        par = Params()
        # par.load(load_dir=temp_dir)
        self.par = par

        with open(temp_dir + '/m' + '.p', 'rb') as handle:
            self.m = pickle.load(handle)

        with open(temp_dir + '/std' + '.p', 'rb') as handle:
            self.std = pickle.load(handle)

        with open(temp_dir + '/m_y' + '.p', 'rb') as handle:
            self.m_y = pickle.load(handle)
        with open(temp_dir + '/std_y' + '.p', 'rb') as handle:
            self.std_y = pickle.load(handle)

        self.history_training = pd.read_pickle(self.save_dir + '/history.p')

        if self.model is None:
            self.create_nnet_model()
        self.model.load_weights(self.save_dir + '/')


    def create_nnet_model(self):
        L = []
        for i, l in enumerate(self.par.model.layers):
            if i == 0:
                L.append(FirstLayer(l,self.par))
            else:
                L.append(layers.Dense(l, activation= self.par.model.activation, dtype=tf.float64))
                L.append(tf.keras.layers.BatchNormalization())

            L.append(layers.Dense(1,dtype=tf.float64))
            self.model = keras.Sequential(L)

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
    model.train()