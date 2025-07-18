import numpy as np
from activations import *
from losses import *
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

a = Dense(4, n_inputs = 4, activation='relu')
input_data=np.array([1,2,3,4])
output=a.forward(input_data)

print("Network output (y_pred):" , output)
print('weights ', a.weights )
print('bias ', a.bias )

y_true=np.array([0,1,0,1])
mse = mse()
loss_value = mse.forward(output,y_true)

print(loss_value)

