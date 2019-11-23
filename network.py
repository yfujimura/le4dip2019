import numpy as np
import layer

class Sequential():

	def __init__(self, *layers):
		self.layers = []

		for l in layers:
			self.layers.append(l)


	def __call__(self, x, target=None):

		if target is None:
			for l in self.layers:
				x = l.forward(x)
				
		else:
			for i, l in enumerate(self.layers):
				if i != len(self.layers)-1:
					x = l.forward(x)
				else:
					x = l.forward(x, target)
		return x

	def add(self, layer):
		self.layers.append(layer)

	def backprop(self, loss):
		for l in self.layers[::-1]:
			loss = l.backward(loss)
			

	def getParams(self):
		parameters = []
		for l in self.layers:
			if l.is_learnable:
				parameters.append(l.getParams())
		return parameters

	def saveParams(self, fn):
		parameters = self.getParams()
		np.save(fn, np.array(parameters))

	def loadParams(self, fn):
		parameters = np.load(fn)
		i = 0
		for l in self.layers:
			if l.is_learnable:
				l.setParams(parameters[i])
				i+=1

	def getLayers(self):
		return self.layers

	def train(self):
		for l in self.layers:
			if hasattr(l, "mode"):
				l.mode = "train"

	def eval(self):
		for l in self.layers:
			if hasattr(l, "mode"):
				l.mode = "eval"



