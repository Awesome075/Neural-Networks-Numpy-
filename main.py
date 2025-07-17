import numpy as np
from activations import *

'''
class neuron:
	def __init__(self, inputs : np.ndarray, weights : np.ndarray, bias : float):
		self.inputs=inputs
		self.weights=weights
		self.bias=bias

	def forward(self) -> float:
		return np.dot(self.inputs,self.weights) + self.bias
'''

class Dense:
	def __init__(self,n_dimension : int , n_inputs : int,  activation : str):
		self.weights 		= np.random.rand(n_dimension,n_inputs)
		self.bias			= np.zeros(n_dimension)
		self.activation 	= self.get_activation(activation)

	def get_activation(self,name):
		if name == 'relu':
			return relu
		elif name == 'sigmoid':
			return sigmoid
		elif name == 'softmax':
			return softmax
		elif name == 'linear':
			return linear
		else:
			raise ValueError('Unsupported Activation')

	def forward(self, x :np.ndarray):
		z = x @ self.weights.T + self.bias    # '@' Does matrix multiplication
		return self.activation(z)

a= Dense(8, n_inputs = 4, activation='linear')
input_data=np.array([1,2,3,4])
output=a.forward(input_data)

print(output)
print('weights ', a.weights )
print('bias ', a.bias )




