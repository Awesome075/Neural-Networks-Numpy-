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
		pass

class mae(loss):
	
	def forward(self, y_pred : np.ndarray, y_org : np.ndarray):
		self.y_pred = y_pred
		self.y_org = y_org

		return np.mean(np.abs(y_pred-y_org))
	
	def backward(self):
		pass

class sse(loss):
	
	def forward(self, y_pred : np.ndarray, y_org : np.ndarray):
		self.y_pred = y_pred
		self.y_org = y_org

		return np.sum(np.square(y_pred-y_org))
	
	def backward(self):
		pass

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
		pass

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
		pass


LOSSES_MAP = {
	'mae' : mae,
	'mse' : mse,
	'sse' : sse,
	'categorical_cross_entropy' : categorical_cross_entropy,
	'sparse_cross_entropy' : sparse_cross_entropy
}