import numpy as np

class activation:
	def forward(self, x : np.ndarray):
		raise NotImplementedError

	def backward(self, grad_output : np.ndarray):
		raise NotImplementedError

class linear(activation):
	def forward(self, x : np.ndarray):
		return x

	def backward(self, grad_output : np.ndarray):
		pass

class relu(activation):
	def forward(self, x : np.ndarray):
		return np.maximum(0,x)

	def backward(self, grad_output : np.ndarray):
		pass

class sigmoid(activation):
	def forward(self, x : np.ndarray):
		return 1 / ( 1 + np.exp(-x) )

	def backward(self, grad_output : np.ndarray):
		pass

class softmax(activation):
	def forward(self, x : np.ndarray):
	
	# Original Softmax Formula
#		return np.exp(x) / np.sum(np.exp(x))   
			
	# Stable Softmax Formula	
		e_x = np.exp(x-np.max(x))
		return  e_x / np.sum(e_x)

	def backward(self, grad_output : np.ndarray):
		pass



ACTIVATION_MAP = {
	'linear' : linear,
	'relu' : relu,
	'sigmoid' : sigmoid,
	'softmax' : softmax
}
