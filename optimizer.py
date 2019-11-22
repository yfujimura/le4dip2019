import numpy as np

class Optimizer:

	def update():
		pass

class SGD(Optimizer):

	def __init__(self, lr=1.0e-3):
		self.lr = lr

	def update(self, param, grad):
		return param - self.lr * grad