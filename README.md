# bayesian_neural_networks

This repository implements a one layer Bayesian Neural network model
in pytorch using EMCEE for the MCMC sampling. The model is then used to train the network to predict 
time series forecasts with statistical uncertainty bands.  


In addition, the neural network class also implements a sequential learning method,
that allows the posterior distributions of the model weights to be updated once new 
data is made available, without having to resample all of the weights.  

# Contents  

### Code  
* Contains all BNN codes


# Getting Started  
Install the conda enviroment using  

```
$ conda env create -f py35.yml
``` 

# Results



# Development Notes  
In order to save the conda enviroment to file
```
$ conda env export > pyBNN.yml
```
