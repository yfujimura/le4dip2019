import numpy as np

class Layer:

	def forward():
		pass

class AffineLayer(Layer):

	def __init__(self, in_d, out_d):
		np.random.seed(0)

		self.in_d = in_d
		self.out_d = out_d

		scale = 1 / np.sqrt(in_d)
		self.W = np.random.normal(0, scale, (self.out_d, self.in_d))
		self.b = np.random.normal(0, scale, self.out_d)

	def forward(self, x):
		return np.dot(self.W, x) + self.b


class SigmoidLayer(Layer):

	def forward(self, x):
		return 1 / (1 + np.exp(-x))

class SoftmaxLayer(Layer):

	def forward(self, x):
		alpha = np.max(x)
		return np.exp(x - alpha) / np.sum(np.exp(x - alpha))
