import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
from pylab import cm
import tqdm
import sys

from layer import *
from network import *
from dataloader import *
from utils import *
from optimizer import *


def train(net, X, Y, optimizer, batch_size=100, epochs=10):
	loader = DataLoader((X,Y), batch_size)
	losses = []

	for i in range(epochs):
		running_loss = 0

		for j in tqdm.tqdm(range(int(X.shape[0] / batch_size))):
			_X,_Y = loader.load()
			loss = net(_X, target=_Y)
			running_loss += loss

			net.backprop(loss)
			optimizer.step()

		running_loss /= int(X.shape[0] / batch_size)
		losses.append(running_loss)
		print(i, running_loss)

	return losses

def eval(net, X, Y):
	accuracy = 0

	for i in range(X.shape[0]):
		_X = np.reshape(X[i], (1,784))
		_Y = np.reshape(Y[i], (1,10))
		predict = net(_X)
		predict[predict == np.max(predict)] = 1
		predict[predict < 1] = 0
		accuracy += np.sum(predict * _Y)

	return accuracy / X.shape[0]


def main(fn):

	mndata = MNIST("./")
	
	X, Y = mndata.load_training()
	X = np.array(X) / 255. # 60000 x 784
	Y = np.array(Y)
	Y = convertOneHotVector(Y, 10)

	net = Sequential(
		Affine(784,100),
		BatchNorm(100),
		ReLU(),
		Affine(100,10),
		SoftmaxWithCrossEntropy()
		)

	#optimizer = SGD(net.getLayers(), lr=0.01)
	#optimizer = MomentumSGD(net.getLayers())
	optimizer = AdaGrad(net.getLayers())
	batch_size = 100
	epochs = 50
	training_loss = train(net, X, Y, optimizer, batch_size=batch_size, epochs=epochs)

	net.saveParams("params/" + fn)
	
	plt.plot(training_loss)
	plt.savefig("training_loss/" + fn + ".png")
	plt.show()
	
	X_test, Y_test = mndata.load_testing()
	X_test = np.array(X_test) / 255. 
	Y_test = np.array(Y_test)
	Y_test = convertOneHotVector(Y_test, 10)
	
	net = Sequential(
		Affine(784,100),
		BatchNorm(100),
		ReLU(),
		Affine(100,10),
		Softmax()
		)
	net.loadParams("params/" + fn + ".npy")
	net.eval()

	accuracy = eval(net, X_test, Y_test)
	print("accuracy:", accuracy)

	print(">", end="")
	idx = int(input())
	_X = np.reshape(X_test[idx], (1,784))
	predict = net(_X)
	print("predict:", np.argmax(predict))
	plt.imshow(np.reshape(X_test[idx],(28,28)))
	plt.show()
	
	
	
	
if __name__ == "__main__":
	args = sys.argv
	main(args[1])