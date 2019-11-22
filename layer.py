import numpy as np
from optimizer import *

class Layer:

	def forward(self):
		pass

	def backward(self):
		pass

	def updateParams(self, optimizer):
		pass

class AffineLayer(Layer):

	def __init__(self, in_d, out_d):
		self.in_d = in_d
		self.out_d = out_d

		scale = 1 / np.sqrt(in_d)
		self.W = np.random.normal(0, scale, (self.in_d, self.out_d))
		self.b = np.random.normal(0, scale, self.out_d)

		self.is_learnable = True


	def forward(self, x):
		# x: samples x in_d
		# W: in_d x out_d
		# b: out_d 
		self.input = x
		return np.dot(x, self.W) + self.b[np.newaxis,:]

	def backward(self, loss):
		self.grad_W = np.dot((self.input).T, loss)
		self.grad_b = np.sum(loss, axis=0)
		return np.dot(loss, (self.W).T)

	def updateParams(self, optimizer):
		self.W = optimizer.update(self.W, self.grad_W)
		self.b = optimizer.update(self.b, self.grad_b)

	def getParams(self):
		return [self.W, self.b]

	def setParams(self, params):
		self.W = params[0]
		self.b = params[1]




class SigmoidLayer(Layer):

	def __init__(self):
		self.is_learnable = False

	def forward(self, x):
		self.out = 1 / (1 + np.exp(-x))
		return self.out

	def backward(self, loss):
		return loss * (1 - self.out) * self.out


class SoftmaxLayer(Layer):

	def __init__(self):
		self.is_learnable = False

	def forward(self, x):
		# x: samples x d
		# alpha: samples
		# np.exp(x - alpha[:,np.newaxis]): samples x d
		# np.sum(np.exp(x - alpha[:,np.newaxis]), axis=1) : samples
		alpha = np.max(x, axis=1)
		return np.exp(x - alpha[:,np.newaxis]) / np.sum(np.exp(x - alpha[:,np.newaxis]), axis=1)[:,np.newaxis]


class SoftmaxWithCrossEntropy(Layer):

	def __init__(self):
		self.is_learnable = False

	def forward(self, x, target):
		self.target = target.copy()

		alpha = np.max(x, axis=1)
		self.softmax_out = np.exp(x - alpha[:,np.newaxis]) / np.sum(np.exp(x - alpha[:,np.newaxis]), axis=1)[:,np.newaxis]
		return np.sum(-target * np.log(self.softmax_out))/ x.shape[0]


	def backward(self, loss):
		# return: samples x d
		return (self.softmax_out - self.target) / self.target.shape[0]

