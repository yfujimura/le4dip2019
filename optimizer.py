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

