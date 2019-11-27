import numpy as np
from optimizer import *

class Layer:

	def forward(self):
		pass

	def backward(self):
		pass

class Affine(Layer):

	def __init__(self, in_d, out_d):
		self.in_d = in_d
		self.out_d = out_d

		scale = 1 / np.sqrt(in_d)
		self.W = np.random.normal(0, scale, (self.in_d, self.out_d))
		self.b = np.random.normal(0, scale, self.out_d)
		self.grad_W = np.zeros(self.W.shape)
		self.grad_b = np.zeros(self.b.shape)

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

	def getParams(self):
		return [self.W, self.b]

	def getLearnableParams(self):
		return [self.W, self.b]

	def setParams(self, params):
		self.W = params[0]
		self.b = params[1]

	def getGrads(self):
		return [self.grad_W, self.grad_b]




class Sigmoid(Layer):

	def __init__(self):
		self.is_learnable = False

	def forward(self, x):
		self.out = 1 / (1 + np.exp(-x))
		return self.out

	def backward(self, loss):
		return loss * (1 - self.out) * self.out


class Softmax(Layer):

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

class ReLU(Layer):

	def __init__(self):
		self.is_learnable = False

	def forward(self, x):
		self.input = x.copy()
		out = x.copy()
		out[x < 0] = 0
		return out

	def backward(self, loss):
		loss[self.input < 0] = 0
		return loss

class Dropout(Layer):

	def __init__(self, rho):
		self.is_learnable = False
		self.rho = rho
		self.mode = "train"

	def forward(self, x):
		if self.mode == "train":
			self.mask = np.random.random_sample(x.shape)
			self.mask[self.mask < self.rho] = 0
			self.mask[self.mask >= self.rho] = 1
			return x * self.mask
		else:
			return x * (1 - self.rho)

	def backward(self, loss):
		return loss * self.mask


class BatchNorm(Layer):

	def __init__(self, dim, epsilon=1.0e-6):
		self.is_learnable = True
		self.gamma = np.ones((1, dim))
		self.beta = np.zeros((1, dim))
		self.grad_gamma = np.zeros(self.gamma.shape)
		self.grad_beta = np.zeros(self.beta.shape)
		self.epsilon = epsilon
		self.mode = "train"

		self.E_x = np.zeros((1, dim))
		self.Var_x = np.zeros((1, dim))
		self.forward_num = 0

	def forward(self, x):
		if self.mode == "train":
			self.x = x
			self.mu = np.mean(x, axis=0)[np.newaxis,:]
			self.sigma = np.mean(np.power(x - self.mu, 2), axis=0)[np.newaxis,:]
			self.xh = (x - self.mu) / np.sqrt(self.sigma + self.epsilon)

			self.E_x = self.E_x + (self.mu - self.E_x) / (self.forward_num + 1)
			self.Var_x = self.Var_x + (self.sigma - self.Var_x) / (self.forward_num + 1)
			self.forward_num += 1

			return self.gamma * self.xh + self.beta
		else:
			return (self.gamma / np.sqrt(self.Var_x + self.epsilon)) * x + (self.beta - self.gamma * self.E_x / np.sqrt(self.Var_x + self.epsilon))


	def backward(self, loss):
		dxh = loss * self.gamma
		dsigma = np.sum(dxh * (self.x - self.mu) * (-1/2) * np.power(self.sigma + self.epsilon, -3/2), axis=0)[np.newaxis,:]
		dmu = np.sum(dxh * (-1) / np.sqrt(self.sigma + self.epsilon), axis=0)[np.newaxis,:] + dsigma * np.sum(-2*(self.x - self.mu), axis=0)[np.newaxis,:] / loss.shape[0]
		dx = dxh / np.sqrt(self.sigma + self.epsilon) + dsigma * (2*(self.x - self.mu)) / loss.shape[0] + dmu / loss.shape[0]
		self.grad_gamma = np.sum(loss * self.xh, axis=0)[np.newaxis,:]
		self.grad_beta = np.sum(loss, axis=0)[np.newaxis,:]

		return dx

	def getParams(self):
		return [self.gamma, self.beta, self.E_x, self.Var_x, self.forward_num]

	def getLearnableParams(self):
		return [self.gamma, self.beta]

	def setParams(self, params):
		self.gamma = params[0]
		self.beta = params[1]
		self.E_x = params[2]
		self.Var_x = params[3]
		self.forward_num = params[4]

	def getGrads(self):
		return [self.grad_gamma, self.grad_beta]








