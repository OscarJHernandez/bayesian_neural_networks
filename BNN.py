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
		
		
	def maximum_likelyhood(self,x_data,y_data, n_iterations=100):
		"""
		This computes the maximum likelyhood estimates of the parameters
		"""
		
		for i in range(n_iterations):
			
			# Generate predictions
			y_predictions = self.net(x_data)
			
			# Compute the loss functions
			self.loss = self.loss_function(y_predictions,y_data)
			self.optimizer.zero_grad()
			self.loss.backward()
			self.optimizer.step()
			
	def return_prediction(self, x_data):
		
		y_predictions = self.net(x_data)
		
		return y_predictions
		
	def return_model_weights(self):
		
		return self.net.parameters().data.numpy()
		
	def convert_model_weights_into_vector(self):
		"""
		This function will convert all of the weights of the model into a 1D-array
		"""
		w_vector = np.zeros(self.Nparam)
		i=0
		
		# Here we convert all of the weights into a vector
		for name,param in self.net.named_parameters():
			for w_vec in param.data.numpy():
				
				if(w_vec.shape!=()):
					for wi in w_vec:
						w_vector[i]=wi
						i+=1
				else:
					w_vector[i]=w_vec
					i+=1	
		
		return w_vector
		
	def convert_vector_into_model_weights(self,vector):
		
		for name,param in self.net.named_parameters():
			for w_vec in param.data.numpy():
				
				if(w_vec.shape!=()):
					for wi in w_vec:
						w_vector[i]=wi
						i+=1
				else:
					w_vector[i]=w_vec
					i+=1
	
	def ln_likelyhood(self,theta,x_data,y_data):
		"""
		The Log likelyhood of the model
		"""
		
		W, sigma = theta
		
		inv_sigma2 = 1.0/sigma**2
		
		y_predictions = self.net(x_data)
		
		ln = -0.5*inv_sigma2*self.loss_function(y_predictions,y_data)-0.5*np.log(2.0*np.pi*sigma**2)
		
		
		return ln
			
			
			
		
		
		



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
	
