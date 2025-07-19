import numpy as np

class loss:
	
	def forward(self, y_pred : np.ndarray, y_org : np.ndarray):
		raise NotImplementedError

	def backward(self):
		raise NotImplementedError

class mse(loss):
	
	def forward(self, y_pred : np.ndarray, y_org : np.ndarray):
		self.y_pred = y_pred
		self.y_org = y_org

		return np.mean(np.square(y_org-y_pred))
	
	def backward(self):
		return 2*(self.y_pred - self.y_org)/self.y_org.size

class mae(loss):
	
	def forward(self, y_pred : np.ndarray, y_org : np.ndarray):
		self.y_pred = y_pred
		self.y_org = y_org

		return np.mean(np.abs(y_pred-y_org))
	
	def backward(self):
		return np.sign(self.y_pred - self.y_org)/self.y_org.size

class sse(loss):
	
	def forward(self, y_pred : np.ndarray, y_org : np.ndarray):
		self.y_pred = y_pred
		self.y_org = y_org

		return np.sum(np.square(y_pred-y_org))
	
	def backward(self):
		return 2*(self.y_pred-self.y_org)

class categorical_cross_entropy(loss):
	
	def forward(self, y_pred : np.ndarray, y_org : np.ndarray):
		self.y_pred = y_pred
		self.y_org = y_org

	# Original Formula
	#	return -np.sum(y_org * np.log(y_pred))
	
	# Stable Formula
		y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
		return -np.sum(y_org * np.log(y_pred_clipped))

	def backward(self):
		return (self.y_pred - self.y_org) / self.y_pred.size
		

class sparse_cross_entropy(loss):
	
	def forward(self, y_pred : np.ndarray, y_org : np.ndarray):
		self.y_pred = y_pred
		self.y_org = y_org

	# Original Formula
	#	correct_confidences = y_pred[range(len(y_pred)),y_org]
	#	return np.mean(-np.log(correct_confidences))

	# Stable Formula
		y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
		correct_confidences = y_pred_clipped[range(len(y_pred)),y_org]
		return np.mean(-np.log(correct_confidences))

	def backward(self):
		samples = len(self.y_org)

		if len(self.y_org.shape) == 1:
			y_org_one_hot = np.zeros_like(self.y_pred)
			y_org_one_hot[np.arange(samples),self.y_org] = 1

		else:
			y_org_one_hot = self.y_org

		return (self.y_pred - y_org_one_hot) / samples


LOSSES_MAP = {
	'mae' : mae,
	'mse' : mse,
	'sse' : sse,
	'categorical_cross_entropy' : categorical_cross_entropy,
	'sparse_cross_entropy' : sparse_cross_entropy
}