from logging import critical

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary

import numpy as np

from tqdm import tqdm
import time

from util import dataset_loader as dl
import loader
import model

import math

# < Is available cuda >
is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

PRINT_PER 	= 100

MAX_GRAD_NORM = 5
LEARNING_RATE = 0.01

CROSS_VAL_RATE = (9, 1)

datas = []

def train(
	model, 
	train_data_list, val_data_list, 
	criterion_func, optim, clip=5
):
	global device

	# [num video][0~3][datas]
	# second : 1~3 : 
	#	0. image data, 1. yolo label data, 2. output lable, 3. batch size

	train_idx 	= [ x for x in range(len(train_data_list)) ]
	val_idx		= [ x for x in range(len(val_data_list)) ]

	train_idx_ft = torch.tensor(train_idx);
	val_idx_ft = torch.tensor(val_idx);

	train_dataloader, _, _ = loader.load_dataset(1, train_idx_ft, train_idx_ft)
	val_dataloader, _, _ = loader.load_dataset(1, val_idx_ft, val_idx_ft)

	'''
	---------------------------------------------------------------------------
	< Training Part >
	---------------------------------------------------------------------------
	'''

	model.train()
	
	train_losses = []

	pbar = enumerate(train_dataloader)
	for i, (vid_idx, _) in pbar:
		iter_dataset = train_data_list[vid_idx]

		mini_batch = iter_dataset[-1]
		hidden = model.init_hidden(mini_batch)

		# vid_pbar = enumerate()

		vid_img  = iter_dataset[1].float().to(device)
		vid_lbs  = iter_dataset[2].float().to(device)
		lbs 	 = iter_dataset[3].float().to(device)

		for frame in tqdm(range(iter_dataset[0]), desc='train progress ', unit=" frame"):
			hidden = tuple([e.data for e in hidden])

			img_into_model, lbs_into_model, model_label = (
				vid_img[frame].clone().detach().requires_grad_(True), 
				vid_lbs[frame].clone().detach().requires_grad_(True), 
				lbs[frame].clone().clone().detach().requires_grad_(True), 
			)
			
			img_into_model.to(device); lbs_into_model.to(device); model_label.to(device)

			model.zero_grad()
			output, hidden = model((img_into_model, lbs_into_model), hidden)

			loss = criterion_func(output, model_label)

			img_into_model.retain_grad()
			lbs_into_model.retain_grad()
			model_label.retain_grad()
			
			loss.backward(retain_graph=True)
			nn.utils.clip_grad_norm_(model.parameters(), clip)
			optim.step()

			train_losses.append( loss.item() )

	'''
	---------------------------------------------------------------------------
	< Validation part >
	---------------------------------------------------------------------------
	'''
	
	model.eval()
	
	val_losses = []
		
	pbar = enumerate(train_dataloader)
	for i, (vid_idx, _) in pbar:
		iter_dataset = train_data_list[vid_idx]

		mini_batch = iter_dataset[-1]
		hidden = model.init_hidden(mini_batch)

		vid_img  = iter_dataset[1].float().to(device)
		vid_lbs  = iter_dataset[2].float().to(device)
		lbs 	 = iter_dataset[3].float().to(device)

		for i, frame in tqdm(range(iter_dataset[0]),  desc='valication progress ', unit=" frame"):
			hidden = tuple([e.data for e in hidden])

			img_into_model, lbs_into_model, model_label = (
				vid_img[frame].clone().detach().requires_grad_(True), 
				vid_lbs[frame].clone().detach().requires_grad_(True), 
				lbs[frame].clone().clone().detach().requires_grad_(True), 
			)
			
			img_into_model.to(device); lbs_into_model.to(device); model_label.to(device)

			output, hidden = model((img_into_model, lbs_into_model), hidden)

			loss = criterion_func(output, model_label)
			loss.item()
			val_losses.append(loss.item())

	'''
	---------------------------------------------------------------------------
	'''

	return train_losses, val_losses

def iter_train(
	model, epochs, batch_size, dataset, optim,
	max_grad_norm = MAX_GRAD_NORM,
	learning_rate = LEARNING_RATE,
	criterion_func = nn.MSELoss(),
):
	global device

	valid_loss_min = np.Inf
	
	start_t = time.time()

	# Cross Validation : Calculate Rate
	n_data = len(dataset)

	data_rate = 0
	for rate in CROSS_VAL_RATE:
		data_rate += rate

	val_range = int( math.floor(n_data / data_rate) ) * CROSS_VAL_RATE[1]
	if (val_range is 0) and (CROSS_VAL_RATE[1] is not 0):
		val_range = 1

	train_range = n_data - val_range

	for i in range(0, epochs):

		# Cross Validation : Divide Datset
		val_fold 	= i % n_data
		val_start	= val_range * val_fold
		val_end		= val_start + val_range

		train_dataset = dataset[0:val_start]
		train_dataset.extend(dataset[val_end:])
		val_dataset = dataset[val_start:val_end]

		train_losses, val_losses = train(
			model, 
			train_dataset, val_dataset, 
			criterion_func, optimizer
		)

		print("Epoch: {}/{}...".format(i+1, epochs),
				"Loss: {:.6f}...".format(np.mean(val_losses)),
				"Val Loss: {:.6f}".format(np.mean(val_losses)))

		if np.mean(val_losses) <= valid_loss_min:
			torch.save(model.state_dict(), './state_dict.pt')
			print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
			valid_loss_min = np.mean(val_losses)

'''
---------------------------------------------------------------------------
 For Test
---------------------------------------------------------------------------
'''
test_label = [[0] for _ in range(7)]
test_label = [test_label for _ in range(191)]
test_label = torch.tensor(test_label)

dataset = dl.dataset_folder_open("../../../../project/train_dataset/tracked_test/")

# < Create Model >
lstm_hidden_size = 128
model = model.AbnormalDetector(
	device, 
	8, 1, 				# label input, output size
	lstm_hidden_size, 	# lstm hidden layer size
	5, 					# label input hidden size
	4, 					# concat_size size
	max_obj_size=128
)

# < Training >
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
torch.autograd.set_detect_anomaly(True)
iter_train(model, 250, 1, dataset, optimizer)
'''
---------------------------------------------------------------------------
'''