import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
from pylab import cm

mndata = MNIST("./")

X, Y = mndata.load_training()
X = np.array(X)
X = X.reshape((X.shape[0],28,28))
Y = np.array(Y)

idx = 100
plt.imshow(X[idx], cmap=cm.gray)
plt.show()
print(Y[idx])