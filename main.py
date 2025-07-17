import numpy as np
from activations import *
np.random.seed(42)
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
	def __init__(self,n_dimension : int , n_inputs : int,  activation):
		
		self.weights = np.random.rand(n_dimension,n_inputs)
		self.bias = np.zeros(n_dimension)

		if isinstance(activation, str):
			self.activation = ACTIVATION_MAP[activation.lower()]()
		else:
			self.activation = activation
		
	def forward(self, x :np.ndarray):
		z = x @ self.weights.T + self.bias    # '@' Does matrix multiplication
		return self.activation.forward(z)

a= Dense(8, n_inputs = 4, activation='relu')
input_data=np.array([1,2,3,4])
output=a.forward(input_data)

print(output)
print('weights ', a.weights )
print('bias ', a.bias )




