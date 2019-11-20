import numpy as np
import layer

class Sequential():

	def __init__(self):
		self.layers = []

	def __call__(self, x):
		for l in self.layers:
			x = l.forward(x)
		return x

	def add(self, layer):
		self.layers.append(layer)

