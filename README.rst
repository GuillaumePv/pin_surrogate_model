# Master thesis: Deep Structural estimation: with an application to market microstructure modelling

This package proposes an easy application of the master thesis: "Deep Structural estimation: with an application to market microstructure modelling"

## Authors

- Guillaume Pavé (HEC Lausanne,guillaumepave@gmail.com)

## Supervisors

- Simon Scheidegger (Department of Economics, HEC Lausanne, simon.scheidegger@unil.ch)
- Antoine Didisheim (Swiss Finance Institute, antoine.didisheim@unil.ch)

## Instructions

1) Download parameters of the surrogate (https://drive.google.com/drive/folders/1RTtYqOipJ-OJpveLu9Ui9NbYGvCDJtNL?usp=sharing)
2) Create a folder "model_save" and put parameters inside
3) Download training datatset "simulation_data_PIN.txt" from https://drive.google.com/file/d/1iUR-Zsd_UAo8bnZEMh5hpQ0SjYtpmtQA/view?usp=sharing
4) Create a folder "data" and put the dataset inside.
5) Now, you can use the train dataset or you could generate your own dataset (https://github.com/edwinhu/pin-code)

* Instantiate a surrogate object with:  *surrogate = DeepSurrogate()*
* Use *get_derivative* to get the first derivative of the log-likelihood function's for each input: 
    * *surrogate.get_derivative(X)*
* Use *get_pin* to get the PIN value with the number of buy and sell trades computed thanks to the Lee and ready algorithm
    * *surogate.get_pin(X) -> X should be a pandas Dataframe containing 'Buy' and "sell colmuns. Or a numpy array with the colmuns in the following order: ['buy', 'sell']
* The Input X should be a pandas DataFrame containing the name of the models parameters. Or a numpy with the columns in the order below:
    * PIN | ['alpha', 'delta', 'epsilon_b', 'epsilon_s', 'mu', 'buy', 'sell']

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

## Contact

The Github repository is available at: https://github.com/GuillaumePv/pin_surrogate_model.

If you find bugs, do not hesitate to create Issues in the repository.