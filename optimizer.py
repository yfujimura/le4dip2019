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

class AdaGrad(Optimizer):

	def __init__(self, layers, lr=1.0e-3, h0=1.0e-8):
		self.layers = layers
		self.lr = lr
		self.h = []

		for l in self.layers:
			if l.is_learnable:
				grads = l.getGrads()
				self.h.append([h0]*len(grads))

	def step(self):
		k = 0
		for l in self.layers:
			if l.is_learnable:
				params = l.getLearnableParams()
				grads = l.getGrads()
				for i in range(len(params)):
					self.h[k][i] += grads[i]*grads[i]
					params[i] += -self.lr * grads[i] / np.sqrt(self.h[k][i])
				k+=1

class RMSProp(Optimizer):

	def __init__(self, layers, lr=1.0e-3, rho=0.9, epsilon=1.0e-8):
		self.layers = layers
		self.lr = lr
		self.rho = rho
		self.epsilon = epsilon
		self.h = []

		for l in self.layers:
			if l.is_learnable:
				grads = l.getGrads()
				self.h.append([0]*len(grads))

	def step(self):
		k = 0
		for l in self.layers:
			if l.is_learnable:
				params = l.getLearnableParams()
				grads = l.getGrads()
				for i in range(len(params)):
					self.h[k][i] *= self.rho
					self.h[k][i] += (1 - self.rho) * grads[i]*grads[i]
					params[i] += -self.lr * grads[i] / np.sqrt(self.h[k][i] + self.epsilon)
				k+=1

class AdaDelta(Optimizer):

	def __init__(self, layers, rho=0.95, epsilon=1.0e-6):
		self.layers = layers
		self.rho = rho
		self.epsilon = epsilon
		self.h = []
		self.s = []

		for l in self.layers:
			if l.is_learnable:
				grads = l.getGrads()
				self.h.append([0]*len(grads))
				self.s.append([0]*len(grads))

	def step(self):
		k = 0
		for l in self.layers:
			if l.is_learnable:
				params = l.getLearnableParams()
				grads = l.getGrads()
				for i in range(len(params)):
					self.h[k][i] *= self.rho
					self.h[k][i] += (1 - self.rho) * grads[i]*grads[i]
					delta = -grads[i] * (np.sqrt(self.s[k][i] + self.epsilon)) / (np.sqrt(self.h[k][i] + self.epsilon)) 
					self.s[k][i] *= self.rho
					self.s[k][i] += (1 - self.rho) * delta*delta
					params[i] += delta
				k+=1

class Adam(Optimizer):

	def __init__(self, layers, lr=1.0e-3, beta1=0.9, beta2=0.999, epsilon=1.0e-8):
		self.layers = layers
		self.lr = lr
		self.beta1 = beta1
		self.beta2 = beta2
		self.epsilon = epsilon
		self.t = 0
		self.m = []
		self.v = []

		for l in self.layers:
			if l.is_learnable:
				grads = l.getGrads()
				self.m.append([0]*len(grads))
				self.v.append([0]*len(grads))

	def step(self):
		k = 0
		for l in self.layers:
			if l.is_learnable:
				params = l.getLearnableParams()
				grads = l.getGrads()
				for i in range(len(params)):
					self.t += 1
					self.m[k][i] *= self.beta1
					self.m[k][i] += (1 - self.beta1) * grads[i]
					self.v[k][i] *= self.beta2
					self.v[k][i] += (1 - self.beta2) * grads[i]*grads[i]
					m_hat = self.m[k][i] / (1 - pow(self.beta1, self.t))
					v_hat = self.v[k][i] / (1 - pow(self.beta2, self.t))
					params[i] += -self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
				k+=1






