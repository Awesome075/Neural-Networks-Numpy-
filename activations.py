import numpy as np

def linear(x):
	return x

def relu(x):
	return np.maximum(0,x)

def sigmoid(x):
	return 1/(1+np.exp(-x))

# Original Softmax
'''
def softmax(x):
	return np.exp(x) / np.sum(np.exp(x))
'''

# Stable Softmax
def softmax(x):
	e_x = np.exp(x-np.max(x))
	return  e_x / np.sum(e_x)
