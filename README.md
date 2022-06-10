# Master thesis: Deep Structural estimation With an Application to a likelihood fnction (PIN model)

This package proposes an easy application of the master thesis: "Deep Structural estimation: with an application to market microstructure modelling"

## Installation

pip install -i https://test.pypi.org/simple/ DeepSurrogate-pin

link of the pypl library: https://test.pypi.org/project/DeepSurrogate-pin/

## Authors

- Guillaume Pavé (guillaumepave@gmail.com)

## Supervisors

- Simon Scheidegger (simon.scheidegger@unil.ch)
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

1) Clone project

```bash
git clone https://github.com/GuillaumePv/pin_surrogate_model.git
```

2) Go into project folder

```bash
cd pin_surrogate_model
```

3) Create your virtual environment (optional)

```bash
python3 -m venv venv
```

4) Enter in your virtual environment (optional)

* Mac OS / linux
```bash
source venv/bin/activate venv venv
```

* Windows
```bash
.\venv\Scripts\activate
```

5) Install libraries

* Python 3
```bash
pip3 install -r requirements.txt
```

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

## Demo 

- To see demo of inverse modelling: see estimate_par_lbfgs.py
- to see how to determine the PIN value: demo.ipynb
