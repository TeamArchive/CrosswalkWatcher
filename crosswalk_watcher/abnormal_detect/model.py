from logging import critical

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import dropout

import numpy as np
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTM

from util import init_models as models

"""

< Yolo v5 DeepSort - Result Shape >

		┌──────────────────────────────────────── Frame Number 
		│ ┌───────────────────────────────────── Identity Number
  		│ │	   ┌─────────────────────────────── AnchorBox - ( Left Top, Right Top, Width, Height )
		│ │	   │			┌───────────────── Yolo Label; What kind of object
		│ │	   │			│	  ┌────────── Fixed Value ( Nothing )
		│ │	 ┌─┴─────────┐  │  ┌──┴──┐
		2 1 409 317 499 383 3 -1 -1 -1 

< INPUT >
	- conv	Images
	- YOLO v5 + DeepSORT : [ Id, Pc, Bx, By, Bh, Bw]
			
			Id 	: Identity
			Pc 	: Yolo Label ( 0. Person, 1. Bike, 2. Bus, 3. Car, 4. Tractor)
			B	: Anchor Box

< Result >
	- Object Label ( DeepSort Label )
	- Warning Label

< Model >

		( Cropped Image ) -→ Inception v3
				↑				  ↓ 	   ⤺
( YOLO v5 + DeepSORT )	  	  FC Layer -→ LSTM -→ FC Layer -→ Activation -→ ( Output )
				↓				  ↑		 
		( Label, Anchor Box ) -→ MLP 

"""

# < INPUT SIZE >
LABEL_BATCH_SIZE = 6

class AbnormalDetector(nn.Module):

	def __init__(
		self, 
		device, 
		input_size, output_size, hidden_size, label_hidden_size, concat_size,
		n_layer=3, drop_prob=0.2, feature_extract=True
	):
		super(AbnormalDetector, self).__init__()
		
		self.device = device

		self.input_size  = input_size
		self.hidden_size = hidden_size
		self.n_layer 	 = n_layer
		self.concat_size = concat_size
		half_concat_size = int(concat_size/2);

		self.dropout 	 = nn.Dropout(drop_prob)

		# < Image Input : Inception v3 >
		self.image_net, self.image_input_size = models.initialize_model(
			"inception", half_concat_size, feature_extract, use_pretrained=True)
		
		# < Label Inputs and MLP >
		self.in_fc = nn.Sequential(
			nn.Linear(input_size, label_hidden_size), 		nn.ReLU(), self.dropout,
			nn.Linear(label_hidden_size, half_concat_size), nn.ReLU(),
		)
		
		# < Video Input >
		self.lstm =nn.LSTM(
			input_size=input_size, 
			hidden_size=hidden_size, 
			dropout=drop_prob,
			num_layers=n_layer,
			bidirectional=True,
			batch_first=True,
		)

		# < Output : FC Layer for Labelling >
		self.out_fc = nn.Sequential(
			nn.Linear(hidden_size, 256), 		nn.ReLU(), self.dropout,
			nn.Sequential(nn.Linear(256, 128)), nn.ReLU(), self.dropout,
			nn.Linear(128, output_size)
		)

	def forwoard(self, images, labels, hidden_and_cell):
		
		# < Image input >
		image_result = []
		for img in images:
			image_result.append(self.image_net.forward(img))
		
		image_result = torch.stack(image_result, dim=0)
		
		# < Label input and label MLP >
		label_result = self.in_fc_(labels.flatten())
		label_result = label_result.view(self.concat_size, -1)

		# < Concatenate >
		concat_result = torch.cat([image_result, label_result], dim=1)

		# < LSTM layer >
		lstm_result, hidden_and_cell = self.lstm(concat_result.flatten(), hidden_and_cell)
		lstm_result = self.dropout(lstm_result)

		# < FC Layer >
		fc_result = self.out_fc(lstm_result)

		return fc_result, hidden_and_cell

	def init_hidden(self, batch_size):
		weight = next(self.parameters()).data
		hidden = ( 
			weight.new(self.n_layers, batch_size, self.hidden_size).zero_().to(self.device), 
			weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device) 
		)
		
		return hidden


is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = AbnormalDetector(device, 6, 2, 4, 6, 6)
print(model)