from logging import critical

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import dropout
from torch.autograd import Variable

import numpy as np
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTM

from util.init_models import initialize_model

"""

< Yolo v5 DeepSort - Result Shape >

		┌──────────────────────────────────────── Frame Number 
		│ ┌───────────────────────────────────── Identity Number
  		│ │	   ┌─────────────────────────────── AnchorBox - ( Left Top, Right Top, Width, Height )
		│ │	   │			┌───────────────── Yolo Label; What kind of object
		│ │	   │			│	  ┌────────── Fixed Value ( Nothing )
		│ │	 ┌─┴─────────┐  │  ┌──┴──┐
		2 1 409 317 499 383 3 -1 -1 -1 
		└─────────────────────────┴──┴─────── Skip

< INPUT >
	- conv	Images
	- YOLO v5 + DeepSORT : [ Id, Pc, Bx, By, Bh, Bw, -1 ]
			
			Id 	: Identity
			Pc 	: Yolo Label ( 0. Person, 1. Bike, 2. Bus, 3. Car, 4. Tractor)
			B	: Anchor Box

< Result >
	- Tracked Object Label List

< Model >

		( Cropped Image ) -→ Inception v3
				↑				  ↓ 	   ⤺
( YOLO v5 + DeepSORT )	  	  FC Layer -→ LSTM -→ FC Layer -→ Activation -→ ( Output )
				↓				  ↑		 
		( Label, Anchor Box ) -→ MLP 

"""

# < INPUT SIZE >

class AbnormalDetector(nn.Module):
	def __init__(
		self, device, 
		input_size, output_size, lstm_hidden_size, label_hidden_size, concat_size,
		n_layer=3, drop_prob=0.2, feature_extract=True, max_obj_size=128
	):
		super(AbnormalDetector, self).__init__()
		
		self.device = device

		self.max_obj_size = max_obj_size

		self.input_size 	  = input_size
		self.lstm_hidden_size = lstm_hidden_size
		self.n_layer 		  = n_layer
		self.concat_size	  = concat_size
		half_concat_size	  = int(concat_size/2);

		self.dropout 	 = nn.Dropout(drop_prob)

		# < Image Input : ImageNet >
		self.image_net, self.image_input_size = initialize_model(
			"squeezenet", half_concat_size, feature_extract, use_pretrained=False)
		self.image_net.to(device)

		# < Label Inputs and MLP >
		self.in_fc = nn.Sequential(
			nn.Linear(input_size, label_hidden_size), 		nn.ReLU(), self.dropout,
			nn.Linear(label_hidden_size, half_concat_size), nn.ReLU(),
		)
		
		# < Video Input >
		self.lstm =nn.LSTM(
			input_size=concat_size, 
			hidden_size=lstm_hidden_size, 
			dropout=drop_prob,
			num_layers=n_layer,
			# bidirectional=True,
			# batch_first=True,
		)

		# < Output : FC Layer for Labelling >
		self.out_fc = nn.Sequential(
			nn.Linear(lstm_hidden_size, 128), nn.ReLU(), self.dropout,
			nn.Sequential(nn.Linear(128, 64)), nn.ReLU(), self.dropout,
			nn.Linear(64, output_size)
		)

		# < Activation function >
		self.sigmoid = nn.Sigmoid()

	def forward(self, x, hidden):

		images = x[0]; labels = x[1]

		# < Image inputs >
		if not self.training:
			self.image_net.eval()
		
		image_result = self.image_net(images)

		# < Label input >
		label_result = self.in_fc(labels)

		# < Concatenate >
		image_result = image_result.repeat(self.max_obj_size, 1)
		stacked_result = torch.cat([image_result, label_result], dim=1)
		stacked_result = torch.unsqueeze(stacked_result, 0)

		# < LSTM layer >
		lstm_result, hidden = self.lstm(stacked_result, hidden)
		lstm_result = self.dropout(lstm_result)
		lstm_result = lstm_result.squeeze()

		# < FC Layer >
		fc_result = self.out_fc(lstm_result)

		# < Activation >
		result = self.sigmoid(fc_result)
		
		return result, hidden

	def init_hidden(self, batch_size):
		weight = next(self.parameters()).data
		hidden = ( 
			weight.new(self.n_layer, batch_size, self.lstm_hidden_size).zero_().to(self.device), 
			weight.new(self.n_layer, batch_size, self.lstm_hidden_size).zero_().to(self.device) 
		)
		
		return hidden