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
		self.input = x
		z = x @ self.weights.T + self.bias    # '@' Does matrix multiplication
		return self.activation.forward(z)


	def backward(self , grad_output : np.ndarray):
		grad_activation =  self.activation.backward(grad_output)

		self.weights_gradient = grad_activation.T @ self.input

		self.bias_gradient = np.sum(grad_activation, axis = 0)

		grad_input = grad_activation @ self.weights

		return grad_input


a = Dense(8, n_inputs = 8, activation='softmax')
input_data=np.array([1,2,3,4,5,6,7,8])
output=a.forward(input_data)

print("Network output (y_pred) : " , output)
print('weights ', a.weights )
print('bias ', a.bias )

y_true=np.array([0,1,0,1,0,1,0,1])
mse = mse()
loss_value = mse.forward(output,y_true)
print("Network Loss : ",loss_value)

loss_grad = mse.backward()
final_grad = a.backward(loss_grad)

print(final_grad)