import numpy as np
from layer import *

class Optimizer:

	def step():
		pass

class SGD(Optimizer):

	def __init__(self, layers, lr=1.0e-2):
		self.layers = layers
		self.lr = lr

	def step(self):
		for l in self.layers:
			if l.is_learnable:
				params = l.getLearnableParams()
				grads = l.getGrads()
				for i in range(len(params)):
					params[i] += -self.lr * grads[i]

class MomentumSGD(Optimizer):

	def __init__(self, layers, lr=1.0e-2, alpha=0.9):
		self.layers = layers
		self.lr = lr
		self.alpha = alpha
		self.delta = []

		for l in self.layers:
			if l.is_learnable:
				_delta = []
				grads = l.getGrads()
				for g in grads:
					_delta.append(np.zeros(g.shape))
				self.delta.append(_delta)

	def step(self):
		k = 0
		for l in self.layers:
			if l.is_learnable:
				params = l.getLearnableParams()
				grads = l.getGrads()
				for i in range(len(params)):
					self.delta[k][i] *= self.alpha
					self.delta[k][i] += -self.lr * grads[i]
					params[i] += self.delta[k][i]
				k += 1


