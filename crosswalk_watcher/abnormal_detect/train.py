import os
import sys
sys.path.insert(0, '../../external/yolo_v5_deepsort/yolov5')
sys.path.insert(0, '../../external/yolo_v5_deepsort')

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

MAX_OBJ = 64

datas = []

def train(
	model, data_path,
	train_idx, val_idx, 
	criterion_func, optim, clip=5
):
	global device

	# [num video][0~3][datas]
	# second : 1~3 : 
	#	0. image data, 1. yolo label data, 2. output lable, 3. batch size


	train_idx_ft	= torch.tensor([ x for x in range(len(train_idx)) ]);
	val_idx_ft		= torch.tensor([ x for x in range(len(val_idx)) ]);

	train_dataloader, _, _ 	= loader.load_dataset(1, train_idx_ft, train_idx_ft)
	val_dataloader, _, _ 	= loader.load_dataset(1, val_idx_ft, val_idx_ft)

	'''
	---------------------------------------------------------------------------
	< Training Part >
	---------------------------------------------------------------------------
	'''

	print()
	print(" Train ")
	print("--------------------------------------------------")

	model.train()
	train_losses = []

	pbar = enumerate(train_dataloader)
	for i, (vid_idx, _) in pbar:

		print(train_idx[vid_idx], " data loding ... ", end='')
		loaded_data = dl.dataset_folder_open(data_path, train_idx[vid_idx], MAX_OBJ)
		if loaded_data == None: continue
		
		n_frame, vid_img, vid_lbs, lbs, mini_batch = loaded_data

		hidden = model.init_hidden(mini_batch)

		for frame in tqdm(range(n_frame), desc='train progress ', unit=" frame"):
			hidden = tuple([e.data for e in hidden])

			img_into_model, lbs_into_model, model_label = (
				vid_img[frame].clone().detach().requires_grad_(True),
				vid_lbs[frame].clone().detach().requires_grad_(True),
				lbs[frame].clone().clone().detach().requires_grad_(True) 
			)
			
			img_into_model 	= img_into_model.to(device); 
			lbs_into_model 	= lbs_into_model.to(device); 
			model_label		= model_label.to(device)

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

		del vid_img; del vid_lbs; del lbs; del mini_batch;

	'''
	---------------------------------------------------------------------------
	< Validation part >
	---------------------------------------------------------------------------
	'''
	
	print()
	print(" Valication ")
	print("--------------------------------------------------")

	model.eval()
	val_losses = []

	pbar = enumerate(val_dataloader)
	for i, (vid_idx, _) in pbar:
		print(train_idx[vid_idx], " data loding ... ", end='')
		loaded_data = dl.dataset_folder_open(data_path, train_idx[vid_idx], MAX_OBJ)
		if loaded_data == None: continue
		
		n_frame, vid_img, vid_lbs, lbs, mini_batch = loaded_data

		hidden = model.init_hidden(mini_batch)

		for frame in tqdm(range(n_frame), desc='valication progress ', unit=" frame"):
			hidden = tuple([e.data for e in hidden])

			img_into_model, lbs_into_model, model_label = (
				vid_img[frame].clone().detach().requires_grad_(True),
				vid_lbs[frame].clone().detach().requires_grad_(True),
				lbs[frame].clone().clone().detach().requires_grad_(True) 
			)
			
			img_into_model 	= img_into_model.to(device); 
			lbs_into_model 	= lbs_into_model.to(device); 
			model_label		= model_label.to(device)

			output, hidden = model((img_into_model, lbs_into_model), hidden)

			loss = criterion_func(output, model_label)
			loss.item()
			val_losses.append(loss.item())

		del vid_img; del vid_lbs; del lbs; del mini_batch;

	'''
	---------------------------------------------------------------------------
	'''

	return train_losses, val_losses

def iter_train(
	model, epochs, batch_size, data_path, optim,
	max_grad_norm = MAX_GRAD_NORM,
	learning_rate = LEARNING_RATE,
	criterion_func = nn.MSELoss(),
):
	global device

	valid_loss_min = np.Inf
	
	start_t = time.time()

	# Cross Validation : Calculate Rate
	dataset_list = os.listdir(data_path)
	n_data = len(dataset_list)

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

		train_idx = dataset_list[0:val_start]
		train_idx.extend(dataset_list[val_end:])
		val_idx = dataset_list[val_start:val_end]

		train_losses, val_losses = train(
			model, 
			data_path,
			train_idx, val_idx, 
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

dataset_path = "../../../../project/train_dataset/tracked_tes/"

# dataset = dl.dataset_folder_open("../../../../project/train_dataset/tracked_test/")

# < Create Model >
lstm_hidden_size = 128
model = model.AbnormalDetector(
	device, 
	6, 1, 				# label input, output size
	lstm_hidden_size, 	# lstm hidden layer size
	96, 					# label input hidden size
	24, 					# concat_size size
	max_obj_size=MAX_OBJ
)

# < Training >
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
torch.autograd.set_detect_anomaly(True)
iter_train(model, 100, 1, dataset_path, optimizer)
'''
---------------------------------------------------------------------------
'''