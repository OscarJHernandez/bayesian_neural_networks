# bayesian_neural_networks

This repository implements a one layer Bayesian Neural network model
in pytorch using EMCEE for the MCMC sampling. The model is then used to train the network to predict 
time series forecasts with statistical uncertainty bands.  


In addition, the neural network class also implements a sequential learning method,
that allows the posterior distributions of the model weights to be updated once new 
data is made available, without having to resample all of the weights.  

# Repository Contents 

### [Code/]  
* Contains all BNN codes

### [Data/]
* Contains all of the data used in the examples

### [Notes/]
* Contains a pdf of the mathematical details of the Bayesian Neural Network 


# Getting Started  
Assuming that you have anaconda installed on your machine, you can install the conda virtual enviroment
containing all needed packages using  
```
$ conda env create -f pyBNN.yml
``` 
Once that is installed, then activate the enviroment
```
$ source activate pyBNN
``` 



# Results



# Development Notes  
In order to save the conda enviroment to file
```
$ conda env export > pyBNN.yml
```
