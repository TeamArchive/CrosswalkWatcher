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

# < Is available cuda >
is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

PRINT_PER 	= 100

MAX_GRAD_NORM = 5
LEARNING_RATE = 0.01

datas = []

def train(
	model, 
	train_data, train_label,
	val_data, val_label,
	mini_batch, 
	criterion_func, optim, clip=5
):
	train_dataloader, _, _ = loader.load_dataset(1, train_data, train_label)
	# for xb, yb in train:
	# 	print("image tensor : ", xb.shape)
	# 	print("label tensor : ", yb.shape)
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

		# vid_img.to(device); vid_lbs.to(device); vid_lbs.to(device)

		# optim.zero_grad()

		for i, frame in vid_pbar:
			print("frame : ", frame)
			hidden = tuple([e.data for e in hidden])

			img_into_model, lbs_into_model, model_label = (
				vid_img[frame].clone().detach().requires_grad_(True), 
				vid_lbs[frame].clone().detach().requires_grad_(True), 
				lbs[frame].clone().clone().detach().requires_grad_(True), 
			)
			
			img_into_model.to(device)
			lbs_into_model.to(device)
			model_label.to(device)

			model.zero_grad()
			output, hidden = model((img_into_model, lbs_into_model), hidden)
			
			print("result : ", output)

			loss = criterion_func(output, model_label)

			img_into_model.retain_grad()
			lbs_into_model.retain_grad()
			model_label.retain_grad()
			
			loss.backward(retain_graph=True)
			running_loss += loss.item()
			nn.utils.clip_grad_norm_(model.parameters(), clip)
			optim.step()

			print("running_loss : ", running_loss)

	'''
	---------------------------------------------------------------------------
	< Validation part >
	---------------------------------------------------------------------------
	'''

	# hidden = model.init_hidden(mini_batch)
	# val_losses = []
	
	# model.eval()
		
	# pbar = enumerate(val_data)
	# for i, (videos, labels) in pbar:
	# 	videos, labels = videos.to(device), labels.to(device)
		
	# 	pre_stage_result = None # TODO : YOLO v5 + Deep SORT execution
	# 	output, val_h = model(pre_stage_result, hidden)
		
	# 	loss = criterion_func(output, labels)
	# 	val_losses.append(loss.item())

	# model.train()

	'''
	---------------------------------------------------------------------------
	'''

	# # 매개변수의 경사도에 학습률을 곱해서 그 매개변수의 값에 더합니다.
	# for p in model.parameters():
	# 	p.data.add_(p.grad.data, alpha=-learning_rate)

	# return output, loss.item()

# def iter_train(
# 	model, device,
# 	epochs, batch_size, 
# 	train_loader, val_loader,
# 	max_grad_norm = MAX_GRAD_NORM,
# 	learning_rate = LEARNING_RATE,
# 	optim_func = optim.Adam,
# 	criterion_func = nn.CrossEntropyLoss
# ):
# 	start_t = time.time()

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

image_list, label_list = dl.dataset_folder_open("../../../../project/train_dataset/tracked/")

datas.append({
	'n_frame'	: 191,
	'images' 	: torch.stack(image_list, dim=0),
	'yolo_lbs' 	: torch.stack(label_list, dim=0),
	'labels' 	: test_label
})

video_tf = torch.tensor([0]).view([-1, 1])
label_tf = torch.tensor([0]).view([-1, 1])

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
train(model, video_tf, label_tf, None, None, 7, nn.MSELoss(), optimizer)

'''
---------------------------------------------------------------------------
'''