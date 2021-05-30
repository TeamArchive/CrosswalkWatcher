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
	train_data, train_label,
	val_data, val_label,
	mini_batch, 
	criterion_func, optim, clip=5
):
	global device

	train_dataloader, _, _ = loader.load_dataset(1, train_data, train_label)
	val_dataloader, _, _ = loader.load_dataset(1, val_data, val_label)

	model.train()

	'''
	---------------------------------------------------------------------------
	< Training Part >
	---------------------------------------------------------------------------
	'''

	hidden = model.init_hidden(mini_batch)
	
	losses = []
	running_loss = 0.0

	pbar = enumerate(train_dataloader)
	for i, (videos, labels) in pbar:

		# videos, labels = videos.to(device), labels.to(device)
		vid_num = int(videos[0][0])

		vid_pbar = enumerate(range(datas[vid_num]['n_frame']))

		vid_img  = datas[vid_num]['images']
		vid_lbs  = datas[vid_num]['yolo_lbs'].float()
		lbs 	 = datas[vid_num]['labels'].float()

		for i, frame in vid_pbar:
			print("frame : ", frame)
			hidden = tuple([e.data for e in hidden])

			img_into_model, lbs_into_model, model_label = (
				vid_img[frame].clone().detach().requires_grad_(True), 
				vid_lbs[frame].clone().detach().requires_grad_(True), 
				lbs[frame].clone().clone().detach().requires_grad_(True), 
			)
			
			img_into_model.to(device); lbs_into_model.to(device); model_label.to(device)

			model.zero_grad()
			output, hidden = model((img_into_model, lbs_into_model), hidden)
			
			# print("result : ", output.clone().detach().half())

			loss = criterion_func(output, model_label)

			img_into_model.retain_grad()
			lbs_into_model.retain_grad()
			model_label.retain_grad()
			
			loss.backward(retain_graph=True)
			nn.utils.clip_grad_norm_(model.parameters(), clip)
			optim.step()

			print("running_loss : ", loss.item())

	'''
	---------------------------------------------------------------------------
	< Validation part >
	---------------------------------------------------------------------------
	'''

	hidden = model.init_hidden(mini_batch)
	val_losses = []
	
	model.eval()
		
	pbar = enumerate(train_dataloader)
	for i, (videos, labels) in pbar:

		vid_num = int(videos[0][0])

		vid_pbar = enumerate(range(datas[vid_num]['n_frame']))

		vid_img  = datas[vid_num]['images']
		vid_lbs  = datas[vid_num]['yolo_lbs'].float()
		lbs 	 = datas[vid_num]['labels'].float()

		for i, frame in vid_pbar:
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
			print(loss.item())

	model.train()

	'''
	---------------------------------------------------------------------------
	'''

def iter_train(
	model, epochs, batch_size, dataset, optim,
	max_grad_norm = MAX_GRAD_NORM,
	learning_rate = LEARNING_RATE,
	criterion_func = nn.MSELoss(),
):
	global device
	
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

		print(len(train_dataset[:][0]))

		# train(
		# 	model, 
		# 	train_dataset[:][0], train_dataset[:][1], 
		# 	val_dataset[:][0], val_dataset[:][1], 
		# 	criterion_func, optimizer
		# )

# 	counter = 0
# 	criterion = critical_func()
# 	optimizer = optim_func(model.parameters(), lr=learning_rate)

# 	model.to(device)
# 	model.train()

# 	for i in range(0, epochs):
# 		train(model, train_loader, val_loader, batch_size, criterion_func, optimizer)
	
# 		# print("Epoch: {}/{}...".format(i+1, epochs),
# 		# 		"Step: {}...".format(counter),
# 		# 		"Loss: {:.6f}...".format(loss.item()),
# 		# 		"Val Loss: {:.6f}".format(np.mean(val_losses)))

# 		# if np.mean(val_losses) <= valid_loss_min:
# 		# 	torch.save(model.state_dict(), './state_dict.pt')
# 		# 	print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
# 		# 	valid_loss_min = np.mean(val_losses)

'''
---------------------------------------------------------------------------
 For Test
---------------------------------------------------------------------------
'''
test_label = [[0] for _ in range(7)]
test_label = [test_label for _ in range(191)]
test_label = torch.tensor(test_label)

dataset = dl.dataset_folder_open("../../../../project/train_dataset/tracked/")

# datas.append({
# 	'n_frame'	: 191,
# 	'images' 	: torch.stack(dataset[0][0], dim=0),
# 	'yolo_lbs' 	: torch.stack(dataset[0][1], dim=0),
# 	'labels' 	: test_label
# })

# video_tf = torch.tensor([0]).view([-1, 1])
# label_tf = torch.tensor([0]).view([-1, 1])

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

# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# < Training >
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
torch.autograd.set_detect_anomaly(True)
# train(model, video_tf, label_tf, video_tf, label_tf, 7, nn.MSELoss(), optimizer)
iter_train(model, 250, 1, dataset,optimizer)
'''
---------------------------------------------------------------------------
'''