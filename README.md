# Master thesis: Deep Structural estimation With an Application to PIN estimation

Create surrogate model for PIN based models

## Authors

- Antoine Didisheim (Swiss Finance Institute, antoine.didisheim@unil.ch)
- Guillaume Pavé (guillaumepave@gmail.com)

## Abstract


## Model

- PIN
- APIN
- GPIN

self.layers = [400,200,100] 0.93 R2

self.layers = [400,200,100,50] # 0.9416 R2

[400,400,200,100] => 0.9898 R2
10 layer of 400 neurons
20mio (5 epoch)
0.485 / 0.77 / 0.8453 / 0.8669 /0.8782
r2: 0.8782 => 20mio points
30mio (5 epoch) => a little bit better
0.57 / 0.8167 / 0.8492 / 0.883 / 
r2: 0.89

0.5840 / 0.84 / 0.8716 / bug ....
9 layers of 400 neurons
20mio


10mio

 self.layers = [400,400,400,400,400,400,400,400,400,400,400] # 6 hidden layer
        #self.layers = [400,400,200,100] # 0.98
        self.batch_size = 256*2
        self.activation = "swish"
        self.opti = Optimizer.ADAM # use this
        self.loss = Loss.MSE

        self.learning_rate = 0.1e-2

        self.E = 5
## TO-DO

- [x] see each MLE is equal tp sum of MLE
- [x] it is possible to normalize data in order to have one model for all
see to contact professor
- [ ] chekc how work optimizer
- [ ] use First layer to optimize data
- [x] trouver le moyen de changer l'input



## Instruction

In construction

## Parameter range

Surrogate model are defined inside some specific range of parameter. PIN model in this surrogate library have been trained inside the range defined the table below.
The surroate can not estimate PIN probability with parameters outside of this range of parameters.

| Parameter | Min | Max
| ------------- | ------------- | ------------- 
| alpha  | 0  | 1
| delta  | 0  | 1
| u  | 0  | 200
| epsilon buys  | 0  | 300
| epsilon sells  | 0  | 300

## Prerequisitres / Installation

In construction

## Acknoledgements

=> put citation here

- Duarte & Young
- Young
- Deep Structural 

optimization pour trouver les parameters
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