import torch
from torch.utils.data import TensorDataset, DataLoader

from util import dataset_loader as dl

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

image_list, label_list = dl.dataset_folder_open("../../../../project/train_dataset/tracked/")

# answer_label_list = torch.empty((191, 7, 1))

img_tf = torch.stack(image_list, dim=0)
lbs_tf = torch.stack(label_list, dim=0)

print("img_tf :", img_tf.shape)
print("lbs_tf :", lbs_tf.shape)

train, _, _ = load_dataset(1, img_tf, lbs_tf)

for xb, yb in train:
	xb = torch.squeeze(xb)
	yb = torch.squeeze(yb)

	print("image tensor : ", xb.shape)
	print("label tensor : ", yb.shape)
