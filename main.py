import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
from pylab import cm

from layer import *
from network import *

def main():
	mndata = MNIST("./")
	X, Y = mndata.load_training()
	X = np.array(X)
	X = X.reshape((X.shape[0],28,28))
	Y = np.array(Y)

	net = Sequential()
	net.add(AffineLayer(784,100))
	net.add(SigmoidLayer())
	net.add(AffineLayer(100,10))
	net.add(SoftmaxLayer())

	print(">", end="")
	idx = int(input())
	if 0 <= idx < 10000:
		x = X[idx,:,:]
		x = np.reshape(x, 784)
		print(np.argmax(net(x)))
	else:
		print("invalid input")
	
	
if __name__ == "__main__":
	main()