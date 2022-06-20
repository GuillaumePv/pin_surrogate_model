# Master thesis: Deep Structural estimation: with an application to market microstructure modelling

This package proposes an easy application of the master thesis: "Deep Structural estimation: with an application to market microstructure modelling"

![alt text](https://github.com/GuillaumePv/pin_surrogate_model/blob/main/results/graphs/3d_comparison_model_surrogate.png)
The figure above shows the log-likelihood value of the PIN model (left) and the Deep-Surrogate (right)

## Authors

- Guillaume Pavé (guillaumepave@gmail.com)

## Supervisors

- Simon Scheidegger (Department of Economics, HEC Lausanne, simon.scheidegger@unil.ch)
- Antoine Didisheim (Swiss Finance Institute, antoine.didisheim@unil.ch)

## Deep surrogate (architecture)

| Hyparameter | Value 
| ------------- | -------------  
| architecture  | [400,400,200,100]
| activation function  | Swish
| optimizer  | ADAM
| loss function | MSE
| learning rate  | 0.5e-3
| # of epoch | 15

## Instruction

1) Cloning project

```bash
git clone https://github.com/GuillaumePv/pin_surrogate_model.git
```

2) Going into project folder

```bash
cd pin_surrogate_model
```

3) Creating your virtual environment (optional)

```bash
python3 -m venv venv
```

4) Entering in your virtual environment (optional)

* Mac OS / linux
```bash
source venv/bin/activate venv venv
```

* Windows
```bash
.\venv\Scripts\activate
```

5) Installing libraries

* Python 3
```bash
pip3 install -r requirements.txt
```
6) Using this library for your project
* Create a folder called "data", download data from this link: https://drive.google.com/drive/folders/1zNTgVVaENjVuB2ORKXpEsXcdSQbWgkDx?usp=sharing
* Put datasets inside "data" folder
* Instantiate a surrogate object with:  *surrogate = DeepSurrogate()*
* Use *get_derivative* to get the first derivative of the log-likelihood function's for each input: 
    * *surrogate.get_derivative(X)*
* Use *get_pin* to get the PIN value with the number of buy and sell trades computed thanks to the Lee and ready algorithm
    * *surogate.get_pin(X) -> X should be a pandas Dataframe containing 'Buy' and "sell colmuns. Or a numpy array with the colmuns in the following order: 'buy', 'sell']
* The Input X should be a pandas DataFrame containing the name of the models parameters. Or a numpy with the columns in the order below:
    * PIN | ['alpha', 'delta', 'epsilon_b', 'epsilon_s', 'mu', 'buy', 'sell']

## Example 

- To see a demonstration of inverse modelling procedure: see estimate_par_lbfgs.py
- To see how to determine the PIN value: see demo.py and demo.ipynb
## Parameter range

Surrogate model are defined inside some specific range of parameter. PIN model in this surrogate library have been trained inside the range defined the table below.
The surroate can not estimate PIN probability with parameters outside of this range of parameters.

| Parameter | Min | Max
| ------------- | ------------- | ------------- 
| a  | 0  | 0.99
| &delta;  | 0  | 0.99
| &mu;  | 100  | 300
| &epsilon;_buy  | 100  | 300
| &epsilon;_sell  | 100  | 300
| # of buy trades  | 55  | 700
| # of sell trades  | 55  | 700



## Python library (under development)
pip install DeepSurrogatepin

link of the pypl library: https://pypi.org/project/DeepSurrogatepin/
