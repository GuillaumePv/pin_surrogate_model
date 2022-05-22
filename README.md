# Master thesis: Deep Structural estimation With an Application to PIN estimation

This package proposes an easy application of the master thesis: "Deep Structural estimation With an Application to PIN estimation"

## Abstract

Create surrogate model for PIN based models

## Installation

pip install DeepSurrogate_pin

## Authors

- Antoine Didisheim (Swiss Finance Institute, antoine.didisheim@unil.ch)
- Guillaume Pavé (guillaumepave@gmail.com)

## Model


self.layers = [400,200,100] 0.93 R2

self.layers = [400,200,100,50] # 0.9416 R2

[400,400,200,100] => 0.9898 R2


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
| alpha  | 0  | 1
| delta  | 0  | 1
| u  | 0  | 200
| epsilon buys  | 0  | 300
| epsilon sells  | 0  | 300

## Prerequisitres / Installation

In construction
