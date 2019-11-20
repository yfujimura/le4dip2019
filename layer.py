import numpy as np

class Layer:

	def forward():
		pass

class AffineLayer(Layer):

	def __init__(self, in_d, out_d):
		self.in_d = in_d
		self.out_d = out_d

		scale = 1 / np.sqrt(in_d)
		self.W = np.random.normal(0, scale, (self.in_d, self.out_d))
		self.b = np.random.normal(0, scale, self.out_d)

	def forward(self, x):
		# x: samples x in_d
		# W: in_d x out_d
		# b: out_d 
		return np.dot(x, self.W) + self.b[np.newaxis,:]


class SigmoidLayer(Layer):

	def forward(self, x):
		return 1 / (1 + np.exp(-x))

class SoftmaxLayer(Layer):

	def forward(self, x):
		# x: samples x d
		# alpha: samples
		# np.exp(x - alpha[:,np.newaxis]): samples x d
		# np.sum(np.exp(x - alpha[:,np.newaxis]), axis=1) : samples
		alpha = np.max(x, axis=1)
		return np.exp(x - alpha[:,np.newaxis]) / np.sum(np.exp(x - alpha[:,np.newaxis]), axis=1)[:,np.newaxis]


class SoftmaxWithCrossEntropy(Layer):

	def forward(self, x, target):
		alpha = np.max(x, axis=1)
		x = np.exp(x - alpha[:,np.newaxis]) / np.sum(np.exp(x - alpha[:,np.newaxis]), axis=1)[:,np.newaxis]
		return np.sum(-target * np.log(x))/ x.shape[0]