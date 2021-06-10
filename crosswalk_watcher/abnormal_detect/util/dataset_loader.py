import os
import sys
sys.path.insert(0, '../../../external/yolo_v5_deepsort/yolov5')
sys.path.insert(0, '../../../external/yolo_v5_deepsort')

from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device, time_synchronized


from glob import glob
import cv2
import re

import torch
from torch import tensor
import numpy as np

from tqdm import tqdm

from functools import cmp_to_key

import pickle

LABEL			= ["person", "bike", "bus", "car", "tractor"]

IGNORE_FOLDER 	= re.compile('\..')

# IMAGE_FOLDER  	= '/images/*'
LABEL_FILE	  	= '/results.txt'
OUTPUT_FILE		= '/output/vid_label.txt'

IMG_NET_SIZE 	= 224

# < Dummy Datas >
DUMMY_LABEL 	= torch.tensor([-1, -1, -1, -1, -1, -1]).float()
DUMMY_OUTPUT	= torch.tensor([0]).float()

#for data in tqdm(dataset_list, desc='Dataset loading', unit=" video"):

def dataset_folder_open(path, data, max_obj):
	dataset = []

	if IGNORE_FOLDER.match(data):
		return None

	if os.path.isfile(path+'/'+data+"/dataset_cache.pickle"):
		with open(path+'/'+data+"/data.pickle", 'rb') as f:
			return pickle.load(f)

		return None

	frame_num = 0

	# img_path_list = glob(path+data+IMAGE_FOLDER)
	temp_label_list = []
	image_list = []
	largest_batch = 0;

	'''
	Image Data Loading
	------------------------------------------------------------
	'''
	image_dataset = LoadImages(path+'/'+data+'/'+data+".mp4", img_size=IMG_NET_SIZE)
	vid_w, vid_h = None, None

	# Blocking Print Anything
	sys.stdout = open(os.devnull, 'w')
	for (_, img, _, vid_cap) in tqdm(image_dataset, desc='Video Loading', unit=" Frame"):
		
		# Enable to Print Anything
		sys.stdout = sys.__stdout__
		
		img = torch.from_numpy(img)
		img = img.float()
		img /= 255.0 					# 0 - 255 to 0.0 - 1.0
		if img.ndimension() == 3:
			img = img.unsqueeze(0)

		image_list.append(img)
		frame_num += 1

		# Blocking Print Anything
		sys.stdout = open(os.devnull, 'w')

		vid_w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		vid_h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	
	# Enable to Print Anything
	sys.stdout = sys.__stdout__

	'''
	Label Data Loading
	------------------------------------------------------------
	'''

	label_list = [[] for i in range(frame_num)]

	with open(path+data+LABEL_FILE, 'r') as label_f:
		while True:
			line = label_f.readline()
			if not line: break

			line = [ int(x) for x in line.split(' ')[:7] ]

			line[2] /= vid_w; line[3] /= vid_h; line[4] /= vid_w; line[5] /= vid_h;

			label_list[line[0]].append(torch.tensor(line[1:]))

		label_f.close()

	# < Append Dummy Labels >
	for i in range(len(label_list)):
		n_dummy = max_obj - len(label_list[i])
		for _ in range(0, n_dummy):
			label_list[i].append(DUMMY_LABEL)

		label_list[i] = torch.stack(label_list[i], dim=0)

	'''
	Abnormal Label Loading
	------------------------------------------------------------
	'''

	output = []
	output_list = torch.tensor([DUMMY_OUTPUT for _ in range(max_obj)])
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

					if (int(elem[0]) == out[0]) and (int(elem[-1]) == out[1]):

						# Set object label to accident occurred
						output_list[i][j] = tensor([1]).float()

	max_obj = float(max_obj)
	label_size = float(len(LABEL))

	for i in range( len(label_list) ):
		for j in range( len(label_list[i]) ):
			if label_list[i][j][0] < 0:  continue

			label_list[i][j][0]  = float(label_list[i][j][0]) / max_obj
			label_list[i][j][-1] = float(label_list[i][j][-1]) / label_size

	result = (frame_num, image_list, label_list, output_list, int(max_obj))
	with open(path+data+"/data.pickle", 'wb') as f:
		pickle.dump(result, f, protocol=4)

	print(" Done . ")

	return result
