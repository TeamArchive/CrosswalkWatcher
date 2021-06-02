import os
from glob import glob
import cv2
import re

import torch
from torch import tensor
import numpy as np

from tqdm import tqdm

from functools import cmp_to_key

LABEL			= ["person", "bike", "bus", "car", "tractor"]

IGNORE_FOLDER 	= re.compile('\..')

IMAGE_FOLDER  	= '/images/*'
LABEL_FILE	  	= '/results.txt'
OUTPUT_FILE		= '/output/vid_label.txt'

IMG_NET_SIZE 	= 224

# < Dummy Datas >
DUMMY_LABEL 	= torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1])
DUMMY_IMG_W 	= [-1 for _ in range(IMG_NET_SIZE)]
DUMMY_IMG_WH	= [DUMMY_IMG_W for _ in range(IMG_NET_SIZE)]
DUMMY_IMG 		= torch.tensor([DUMMY_IMG_WH for _ in range(3)])
DUMMY_OUTPUT	= torch.tensor([0])

def dataset_folder_open(folder_name):

	path = folder_name
	dataset_list = os.listdir(path)
	dataset = []

	for data in tqdm(dataset_list, desc='Dataset loading'):

		if IGNORE_FOLDER.match(data):
			continue

		frame_num = 0

		img_path_list = glob(path+data+IMAGE_FOLDER)
		temp_label_list = []
		image_list = []
		largest_batch = 0;

		'''
		Label Data Loading
		------------------------------------------------------------
		'''

		with open(path+data+LABEL_FILE, 'r') as label_f:
			while True:
				line = label_f.readline()
				if not line: break

				line = [ int(x) for x in line.split(' ')[:8] ]
				temp_label_list.append(line)

			frame_num = temp_label_list[-1][0]
			label_f.close()

		# < Label Classification >
		label_list = [[] for i in range(frame_num)]
		for label in temp_label_list:
			current_idx = label[0]-1
			label_list[current_idx].append(torch.tensor(label))
		
		# < Get Largest Batch >
		for label in label_list:
			if len(label) > largest_batch:
				largest_batch = len(label)
 
		# < Stacking >
		for i, label in enumerate(label_list):

			# Append Dummy Label
			if len(label) < largest_batch:
				n_dummy = largest_batch - len(label) 
				for _ in range(0, n_dummy):
					label.append(DUMMY_LABEL)

			# Stacking
			label_list[i] = torch.stack(label)

		'''
		Image Data Loading
		------------------------------------------------------------
		'''
		
		image_list = [[] for i in range(frame_num)]
		image_id_sec = [[] for i in range(frame_num)]

		for img in img_path_list:
			img_info 	= img.split('/')[-1].split('_')
			img_frame 	= int(img_info[0][1:])-1
			img_id 		= int(img_info[1][2:])
			img_label	= int(LABEL.index(img_info[-1].split('.')[0]))

			# check is image's label exist
			is_exist = False
			for elem_label in label_list[img_frame]:
				if int(elem_label[1]) is img_id and int(elem_label[6]) is img_label:
					is_exist = True
					break;

			if is_exist:
				cv_img = cv2.resize(cv2.imread(img), dsize=(IMG_NET_SIZE, IMG_NET_SIZE), interpolation=cv2.INTER_LINEAR)
				cv_img = np.rollaxis(cv_img, 2, 0)

				image_list[img_frame].append(torch.Tensor(cv_img))
				image_id_sec[img_frame].append((img_id, img_label))

		# < Sequence Sort >
		sorted_seq = [[] for i in range(frame_num)]
		for i, seq in enumerate(image_id_sec):
			if len(seq) is 0 :
				continue
			
			temp = sorted(seq, key=lambda x: (x[1], x[0]))
			for elem in temp:
				sorted_seq[i].append(seq.index(elem))

		# < Image Matching >
		for i, img in enumerate(image_list):
			sorted_img = [DUMMY_IMG for _ in range(largest_batch)]

			# Insert images in sorted order
			for enum_i, idx in enumerate(sorted_seq[i]):
				sorted_img[enum_i] = img[idx]

			image_list[i] = torch.stack(sorted_img, dim=0)

		output = []
		output_list = torch.tensor([DUMMY_OUTPUT for _ in range(largest_batch)])
		output_list = torch.stack([ output_list for _ in range(frame_num) ])

		with open(path+data+OUTPUT_FILE, 'r') as output_f:
			while True:
				line = output_f.readline()
				if not line: break

				line = [ int(x) for x in line.split(' ')[:-1] ]
				output.append(line)

		for i in range(frame_num):
			
			# Frame Check
			for out in output:
				if out[2] <= i <= out[3]:

					# Check if objects are the same
					for j, elem in enumerate(label_list[i]):	
						if (int(elem[1]) is out[0]) and (int(elem[6]) is out[1]):
							output_list[i][j] = tensor([1])		# Set object label to accident occurred

			# output_list[i] = torch.stack(output_list[i], dim=0)

		image_list = torch.stack(image_list, dim=0)
		label_list = torch.stack(label_list, dim=0)

		dataset.append( (frame_num, image_list, label_list, output_list, largest_batch) )

	print("dataset load done.")

	return dataset

