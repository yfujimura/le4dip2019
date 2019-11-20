import numpy as np
import layer

class Sequential():

	def __init__(self):
		self.layers = []

	def __call__(self, x):
		for l in self.layers:
			x = l.forward(x)
		return x

	def __call__(self, x, target):
		for i, l in enumerate(self.layers):
			if i != len(self.layers)-1:
				x = l.forward(x)
			else:
				x = l.forward(x, target)
		return x

	def add(self, layer):
		self.layers.append(layer)

