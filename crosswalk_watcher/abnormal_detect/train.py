from logging import critical

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim as optim
from torch.utils.data import dataloader
from torchsummary import summary

import numpy as np

from tqdm import tqdm
import time

from . import loader
from . import model

# < Is available cuda >
is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

PRINT_PER 	= 100

MAX_GRAD_NORM = 5
LEARNING_RATE = 0.01

def train(
	model, train_data, val_data, mini_batch, 
	criterion_func, optim, clip=5
):
	'''
	---------------------------------------------------------------------------
	< Training Part >
	---------------------------------------------------------------------------
	'''

	hidden = model.init_hidden(mini_batch)
	
	losses = []
	running_loss = 0.0

	pbar = enumerate(train_data)
	for i, (videos, labels) in pbar:
		videos, labels = videos.to(device), labels.to(device)

		model.zero_grad()

		pre_stage_result = None # TODO : YOLO v5 + Deep SORT execution
		output, hidden = model(pre_stage_result, hidden)

		loss = criterion_func(output, labels)
		loss.backward()
		running_loss + loss.item()

		nn.utils.clip_grad_norm_(model.parameters(), clip)
		optim.step()

	'''
	---------------------------------------------------------------------------
	< Validation part >
	---------------------------------------------------------------------------
	'''

	hidden = model.init_hidden(mini_batch)
	val_losses = []
	
	model.eval()
		
	pbar = enumerate(val_data)
	for i, (videos, labels) in pbar:
		videos, labels = videos.to(device), labels.to(device)
		
		pre_stage_result = None # TODO : YOLO v5 + Deep SORT execution
		output, val_h = model(pre_stage_result, hidden)
		
		loss = criterion_func(output, labels)
		val_losses.append(loss.item())

	model.train()

	'''
	---------------------------------------------------------------------------
	'''

	# # 매개변수의 경사도에 학습률을 곱해서 그 매개변수의 값에 더합니다.
	# for p in model.parameters():
	# 	p.data.add_(p.grad.data, alpha=-learning_rate)

	# return output, loss.item()

def iter_train(
	model, device,
	epochs, batch_size, 
	train_loader, val_loader,
	max_grad_norm = MAX_GRAD_NORM,
	learning_rate = LEARNING_RATE,
	optim_func = optim.Adam,
	criterion_func = nn.CrossEntropyLoss
):
	start_t = time.time()

	counter = 0
	criterion = critical_func()
	optimizer = optim_func(model.parameters(), lr=learning_rate)

	model.to(device)
	model.train()

	for i in range(0, epochs):
		train(model, train_loader, val_loader, batch_size, criterion_func, optimizer)
	
		# print("Epoch: {}/{}...".format(i+1, epochs),
		# 		"Step: {}...".format(counter),
		# 		"Loss: {:.6f}...".format(loss.item()),
		# 		"Val Loss: {:.6f}".format(np.mean(val_losses)))

		# if np.mean(val_losses) <= valid_loss_min:
		# 	torch.save(model.state_dict(), './state_dict.pt')
		# 	print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
		# 	valid_loss_min = np.mean(val_losses)


# Load