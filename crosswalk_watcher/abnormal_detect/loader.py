import torch
from torch.utils.data import TensorDataset, DataLoader

def load_dataset( 
	batch_size,
	train_datas, train_labels,
	val_datas, val_labels,
	test_datas = None, test_labels = None
) :
	train_dataset = TensorDataset(torch.from_numpy(train_datas), torch.from_numpy(train_labels))
	val_dataset = TensorDataset(torch.from_numpy(val_datas), torch.from_numpy(val_labels))
	
	test_loader = None
	if(test_datas is not None and test_datas is not None):
		test_dataset = TensorDataset(torch.from_numpy(test_datas), torch.from_numpy(test_labels))
		test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

	train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
	val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size)
	
	return train_loader, val_loader, test_loader

import numpy as np

train_datas = np.array(
	[11, 12, 13, 14, 15]
)

train_labels = np.array([
	[1, 2, 3, 4, 5],
	[1, 2, 3, 4, 5],
	[1, 2, 3, 4, 5],
	[1, 2, 3, 4, 5],
	[1, 2, 3, 4, 5],
])

val_datas = np.array(
	[21, 22]
)

val_labels = np.array([
	[1, 2, 3, 4, 5],
	[1, 2, 3, 4, 5],
])

a, b, c = load_dataset( 1, train_datas, train_labels, val_datas, val_labels )
print(a)