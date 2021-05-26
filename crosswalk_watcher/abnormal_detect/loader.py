import torch
from torch.utils.data import TensorDataset, DataLoader

def load_dataset( 
	batch_size,
	train_datas, train_labels,
	val_datas = None, val_labels = None,
	test_datas = None, test_labels = None
) :
	# train_dataset = TensorDataset(torch.from_numpy(train_datas), torch.from_numpy(train_labels))
	train_dataset = TensorDataset(train_datas, train_labels)
	train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
	
	val_loader = None
	if(val_datas is not None and val_datas is not None):
		val_dataset = TensorDataset(torch.from_numpy(val_datas), torch.from_numpy(val_labels))
		val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size)

	test_loader = None
	if(test_datas is not None and test_datas is not None):
		test_dataset = TensorDataset(torch.from_numpy(test_datas), torch.from_numpy(test_labels))
		test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)
	
	return train_loader, val_loader, test_loader