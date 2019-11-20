import numpy as np


def convertOneHotVector(labels, max_n):
	return np.eye(max_n)[labels]


