import numpy as np
from layer import *

class Optimizer:

	def update():
		pass

class SGD(Optimizer):

	def __init__(self, layers, lr=1.0e-2):
		self.layers = layers
		self.lr = lr

	def step(self):
		for l in layers:
			if l.is_learnable:
				params = l.getParams()
				grads = l.getGrads()
				for i in range(len(params)):
					params[i] = params[i] - self.lr * grads[i]

