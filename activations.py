import numpy as np

class activation:
	def forward(self, x : float):
		raise NotImplementedError

	def backward(self, grad_output : float):
		raise NotImplementedError

class linear(activation):
	def forward(self, x : float):
		return x

	def backward(self, grad_output : float):
		pass

class relu(activation):
	def forward(self, x : float):
		return np.maximum(0,x)

	def backward(self, grad_output : float):
		pass

class sigmoid(activation):
	def forward(self, x : float):
		return 1 / ( 1 + np.exp(-x) )

	def backward(self, grad_output : float):
		pass

class softmax(activation):
	def forward(self, x : float):
	
	# Original Softmax Formula
#		return np.exp(x) / np.sum(np.exp(x))   
			
	# Stable Softmax Formula	
		e_x = np.exp(x-np.max(x))
		return  e_x / np.sum(e_x)

	def backward(self, grad_output : float):
		pass



ACTIVATION_MAP = {
	'linear' : linear,
	'relu' : relu,
	'sigmoid' : sigmoid,
	'softmax' : softmax
}
