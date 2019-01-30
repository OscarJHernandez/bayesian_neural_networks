#
# The implementation of the Bayesian Feed forward neural network
#
import emcee 
import torch
import torch.nn.functional as F
import numpy as np

class BayesianNN:
	
	
	def __init__(self, n_feature,n_hidden,n_output):
		
		# Initialize the neural network
		self.net = Net(n_feature=n_feature,n_hidden=n_hidden,n_output=n_output)
		
		# Initiliaze the least squares Loss function
		self.loss_function = torch.nn.MSELoss()
		
		# Here we choose the optimizer
		self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.2) # Learning rate parameter
		
		
		# Calculate the number of parameters in the model
		self.Nparam = self.return_number_of_parameters()
		
		print("Number of Model Parameters: ", self.Nparam)
		
	def return_number_of_parameters(self):
		"""
		Computes the number of parameters in the model
		"""
		Nparam = 0
		
			# Here we convert all of the weights into a vector
		for name,param in self.net.named_parameters():
			for w_vec in param.data.numpy():
				
				if(w_vec.shape!=()):
					for wi in w_vec:
						Nparam+=1
				else:
					Nparam+=1	
		
		return Nparam
		
	def print_all_parameters(self):
		"""
		This function prints all parameters of the NN model
		"""
		print(self.net.hidden.weight.data)
		print(self.net.hidden.bias.data)
		print(self.net.predict.weight.data)
		print(self.net.predict.bias.data)
		
		return None
		
		
	def maximum_likelyhood(self,x_data,y_data, n_iterations=100):
		"""
		This computes the maximum likelyhood estimates of the parameters
		
		x_data: numpy array
		y_data: numpy array
		
		"""
		x_data = torch.from_numpy(x_data)
		y_data = torch.from_numpy(y_data)
		
		for i in range(n_iterations):
			
			# Generate predictions
			y_predictions = self.net(x_data)
			
			# Compute the loss functions
			self.loss = self.loss_function(y_predictions,y_data)
			self.optimizer.zero_grad()
			self.loss.backward()
			self.optimizer.step()
			
	def return_prediction(self, x_data):
		"""
		Return the predictions of the model
		"""
		
		y_predictions = self.net(x_data)
		
		return y_predictions

		
	def convert_model_weights_into_vector(self):
		"""
		This function will convert all of the weights of the model and flattens them into a 1D-array
		
		Also stores the length of the individual weight components
		"""
		w_vector = np.zeros(self.Nparam)
		
		weights_hidden = self.net.hidden.weight.data.numpy().flatten()
		weights_hidden_bias = self.net.hidden.bias.data.numpy().flatten()
		
		weights_predict = self.net.predict.weight.data.numpy().flatten()
		weights_predict_bias = self.net.predict.bias.data.numpy().flatten()
		
		# Store the length of each array vector
		self.N_weights_hidden = len(self.net.hidden.weight.data.numpy().flatten())
		self.N_weights_hidden_bias = len(self.net.hidden.bias.data.numpy().flatten())
		self.N_weights_predict = len(self.net.predict.weight.data.numpy().flatten())
		self.N_weights_predict_bias = len(self.net.predict.bias.data.numpy().flatten())
		
		
		w_vector = np.append(weights_hidden,weights_hidden_bias)
		w_vector = np.append(w_vector,weights_predict)
		w_vector = np.append(w_vector,weights_predict_bias)
		
		return w_vector
		
	def convert_vector_into_model_weights(self,vector):
		"""
		Converts the flattened weight vector into the model weights 
		"""
		N1 = self.N_weights_hidden
		N2 = N1+self.N_weights_hidden_bias
		N3 = N2+self.N_weights_predict
		N4 = N3+self.N_weights_predict_bias
		
		w1 = vector[0:N1]
		w2 = vector[N1:N2]
		w3 = vector[N2:N3]
		w4 = vector[N3:N4]
		
		# Use the weights of the vector for the model
		self.net.hidden.weight.data[:,0] = torch.from_numpy(w1)
		self.net.hidden.bias.data[:] = torch.from_numpy(w2)
		
		self.net.predict.weight.data[0,:] = torch.from_numpy(w3)
		self.net.predict.bias.data[:] = torch.from_numpy(w4)
		
		return None
		
		
	
	def ln_likelyhood(self,theta,x_data,y_data):
		"""
		The Log likelyhood of the Neural Network
		
		x_data: numpy array
		y-data: numpy array
		"""
		vec_W = theta
		#vec_WpSigma= theta
		#sigma = vec_WpSigma[-1]
		#vec_W =  vec_WpSigma[0:-1]
		sigma = 1.0

		inv_sigma2 = 1.0/sigma**2
		
		# 
		self.convert_vector_into_model_weights(vec_W)
		
		y_predictions = self.net(torch.from_numpy(x_data))
		y_predictions = y_predictions.data.numpy()
		
		#y_data = torch.from_numpy(y_data)
		
		#ln = -0.5*inv_sigma2*self.loss_function(y_predictions,y_data)-0.5*np.log(2.0*np.pi*sigma**2)
		ln = -0.5*(np.sum((y_predictions-y_data)**2*inv_sigma2 - np.log(inv_sigma2)))
		#ln = ln.data.numpy()
		
		if not np.isfinite(ln):
			return -np.inf
		
		return ln
		
	def generate_posteriors_weights(self,x_data,y_data,n_iterations,N_mcmc_walkers=100,N_mcmc_runs=500, N_mcmc_burn=100,deps=1e-3):
		
		
		# First we optimize the network to generate the Maxlikelyhood estimate V0
		self.maximum_likelyhood(x_data,y_data,n_iterations=n_iterations)
		
		ndim, nwalkers = self.Nparam, N_mcmc_walkers
		
		v0 = self.convert_model_weights_into_vector()
		
		# Initialize the sigma value
		#sigma = np.asarray([1.0])
		
		#vp1 = np.append(v0,sigma)
		
		# The initial position of all of the walkers
		pos = [v0 + deps*np.random.randn(ndim) for i in range(nwalkers)]
		
		sampler = emcee.EnsembleSampler(nwalkers, ndim, self.ln_likelyhood, args=(x_data,y_data))
		
		sampler.run_mcmc(pos, N_mcmc_runs)
		samples = sampler.chain[:, N_mcmc_burn:, :].reshape((-1, ndim))
		
		# Reset the model weights to the maximum likelyhood
		self.convert_vector_into_model_weights(v0)
		
		# Store the samples, this will be used for the uncertainty quantification
		self.samples = samples
		
		
		return samples
			
			
			
		
		
		



class Net(torch.nn.Module):
	
	def __init__(self, n_feature, n_hidden, n_output):
		super(Net, self).__init__()
		self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
		self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

	def forward(self, x):
		"""
		The forward pass of the neural network
		"""
		x = F.relu(self.hidden(x))      # activation function for hidden layer
		x = self.predict(x)             # linear output
		return x
	
