import numpy as np

class DataLoader:

	def __init__(self, dataset, batch_size):
		# dataset: 二次元のtuple
		# dataset[0]: 入力のndarray
		# dataset[1]: ラベルのndarray
		self.dataset = dataset
		self.idx_list = [i for i in range(dataset[0].shape[0])]

		self.batch_size = batch_size

	def load(self):
		idx_batch = np.random.choice(self.idx_list, self.batch_size, replace=False) 
		return self.dataset[0][idx_batch], self.dataset[1][idx_batch]
