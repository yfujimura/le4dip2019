import numpy as np
import pickle


def convertOneHotVector(labels, max_n):
	return np.eye(max_n)[labels]

def unpickle(file):
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	X = np.array(dict[b'data'])
	Y = np.array(dict[b'labels'])
	return X, Y


