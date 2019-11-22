import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
from pylab import cm
import tqdm

from layer import *
from network import *
from dataloader import *
from utils import *
from optimizer import *

def train(net, X, Y, optimizer=SGD(), batch_size=32, epochs=10):
	losses = []

	for i in range(epochs):
		running_loss = 0

		for j in tqdm.tqdm(range(int(X.shape[0] / batch_size))):
			loader = DataLoader((X,Y), batch_size)
			_X,_Y = loader.load()
			_X = np.reshape(_X, (_X.shape[0], 784))
			_Y = convertOneHotVector(_Y, 10)
			loss = net(_X, target=_Y)
			running_loss += loss

			net.backprop(loss)
			net.step(optimizer)

		running_loss /= int(X.shape[0] / batch_size)
		losses.append(running_loss)
		print(i, running_loss)

	return losses


def main():
	
	mndata = MNIST("./")
	X, Y = mndata.load_training()
	X = np.array(X)
	X = X.reshape((X.shape[0],28,28)) / 255.
	Y = np.array(Y)

	net = Sequential(
		AffineLayer(784,100),
		SigmoidLayer(),
		AffineLayer(100,10),
		SoftmaxWithCrossEntropy()
		)

	optimizer = SGD(lr=0.01)
	batch_size = 100
	epochs = 20
	training_loss = train(net, X, Y, optimizer=optimizer, batch_size=batch_size, epochs=epochs)
	net.saveParams("params")

	net = Sequential(
		AffineLayer(784,100),
		SigmoidLayer(),
		AffineLayer(100,10),
		SoftmaxLayer()
		)
	net.loadParams("params.npy")

	print(">", end="")
	idx = int(input())
	_X = X[idx]
	_Y = Y[idx]
	_X = np.reshape(_X, (1,784))
	_Y = np.reshape(convertOneHotVector(_Y, 10), (1,10))
	predict = net(_X)
	print(predict)
	plt.imshow(X[idx])
	plt.show()
	
	
	
if __name__ == "__main__":
	main()