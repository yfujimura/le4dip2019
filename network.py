import numpy as np
import layer

class Sequential():

	def __init__(self, *layers):
		self.layers = []

		for l in layers:
			self.layers.append(l)

	#def __call__(self, x):
	#	for l in self.layers:
	#		x = l.forward(x)
	#	return x

	def __call__(self, x, target = None):

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

	def step(self, optimizer):
		for l in self.layers:
			l.updateParams(optimizer)

	def saveParams(self, fn):
		parameters = []
		for l in self.layers:
			if l.is_learnable:
				parameters.append(l.getParams())

		np.save(fn, np.array(parameters))

	def loadParams(self, fn):
		parameters = np.load(fn)
		i = 0
		for l in self.layers:
			if l.is_learnable:
				l.setParams(parameters[i])
				i+=1



