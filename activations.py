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
		return grad_output

class relu(activation):
	def forward(self, x : np.ndarray):
		self.input=x

		return np.maximum(0,x)

	def backward(self, grad_output : np.ndarray):
		relu_grad = self.input > 0

		return grad_output * relu_grad

class sigmoid(activation):
	def forward(self, x : np.ndarray):

		self.output = 1 / ( 1 + np.exp(-x) )
		return self.output

	def backward(self, grad_output : np.ndarray):
		return grad_output * (self.output * (1 - self.output) )


class softmax(activation):
	def forward(self, x : np.ndarray):
	
	# Original Softmax Formula
	# 	return np.exp(x) / np.sum(np.exp(x))   
			
	# Stable Softmax Formula	
		max_x = np.max(x, axis=1, keepdims=True)
		e_x = np.exp(x - max_x)
		sum_e_x = np.sum(e_x, axis=1, keepdims=True)

		return e_x / sum_e_x

	def backward(self, grad_output : np.ndarray):
	#	Mathematically Correct way

	#	It is very computationally expensive & complex 
	# 	return grad_output * Jacobian_matrix

	# 	Standard Way (Common Case) : because softmax is pairwed with cross_entropy loss almost all the times.

	# 	The gradient for softmax is paired with cross_entropy loss 
	# 	for simplicity and numerical stability, which gets cancelled out.

	#	The grad_output that we receive from the cross_entropy loss 
	#	is already the gradient we need. So we just pass through it.

		return grad_output



ACTIVATION_MAP = {
	'linear' : linear,
	'relu' : relu,
	'sigmoid' : sigmoid,
	'softmax' : softmax
}
