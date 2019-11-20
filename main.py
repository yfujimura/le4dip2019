import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
from pylab import cm

from layer import *
from network import *
from dataloader import *
from utils import *

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
	net.add(SoftmaxWithCrossEntropy())


	loader = DataLoader((X,Y), 100)
	X,Y = loader.load()
	X = np.reshape(X, (X.shape[0], 784))
	Y = convertOneHotVector(Y, 10)
	loss = net(X, Y)
	print(loss)
	
	
if __name__ == "__main__":
	main()