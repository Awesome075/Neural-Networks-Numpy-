import numpy as np
from losses import LOSSES_MAP


class Model:
	def __init__(self, layers=[]):
		self.layers = layers

	def add(self, layer):
		self.layers.append(layer)

	def compile(self, optimizer, loss):
		self.loss = LOSSES_MAP[loss]()
		self.optimizer = optimizer

	def fit(self, x_train, y_train, epochs, batch_size=None):

		for epoch in range(epochs):

			output = x_train
			#1. FORWARD PASS 
			for layer in self.layers:
				output = layer.forward(output)

			#2. CALCULATE LOSS(ERROR)
			loss = self.loss.forward(output,y_train)
			print(f"Epochs = {epoch+1}, Loss = {loss}")

			#3. BACKWARD PASS
			gradient = self.loss.backward()

			for layer in reversed(self.layers):
				gradient=layer.backward(gradient)

			#4. UPDATE PARAMETERS
			for layer in self.layers:
				self.optimizer.update(layer)


