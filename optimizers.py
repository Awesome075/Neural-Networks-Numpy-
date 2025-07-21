import numpy as np

class Optimizer:
	def __init__(self, learning_rate=0.01):
		self.learning_rate = learning_rate

	def update(self, layer):
		raise NotImplementedError

class SGD(Optimizer):
	def update(self, layer):
		layer.weights = layer.weights - (self.learning_rate * layer.weights_gradient)
		layer.bias = layer.bias - (self.learning_rate * layer.bias_gradient)