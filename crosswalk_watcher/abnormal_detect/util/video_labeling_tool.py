import sys

from numpy.lib.function_base import select
sys.path.insert(0, '../../../external/yolo_v5_deepsort/yolov5')
sys.path.insert(0, '../../../external/yolo_v5_deepsort')

import cv2
from PIL import ImageFont, ImageDraw, Image

import numpy as np
from yolov5.utils.datasets import LoadImages, LoadStreams		

import os
import shutil

WINDOW_NAME = "abnomal detection model labeling tool"

RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
BLACK_COLOR = (255, 255, 255)

INFO_FONT_SIZE = 2
LABEL_FONT_SIZE = 2
LABEL_FONT_THICK = 3

YOLO_NAME = ['person', 'bike', 'bus', 'car']

video_f = "../../../../../project/train_dataset/videos/a22.mp4"
label_f = "../../../../../project/train_dataset/tracked/a22/results.txt"
project_path = "../../../../../project/train_dataset/tracked"
video_name = "a22"

# < Init Folder >
save_path = project_path+"/"+video_name+"/output"

if os.path.exists(save_path):
	shutil.rmtree(save_path)  # delete output folder
os.makedirs(save_path)  # make new output folder

imgsz = 640

dataset = LoadImages(video_f, img_size=imgsz)
dataset = list(enumerate(dataset))

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.createTrackbar("Frame", WINDOW_NAME, 0, len(dataset)-1, lambda x: x)

label = [[] for _ in range(len(dataset))]
with open(label_f, "r") as f:
	while True:
		line = [ int(x) for x in f.readline().split(' ')[:-1] ]
		if not line: break

		label[line[0]].append(line)

idx = 0

# form is { id: ?, lb: ?, start_frame: ?, end_frame: ? }
#	* lb : yolo v5 label
selected = []

def mouse_callback(event, x, y, flags, param): 
	global label, idx, selected

	if event == cv2.EVENT_LBUTTONDOWN:
		for box in label[idx]:
			if (box[2] <= x <= box[4]) and (box[3] <= y <= box[5]):		# Check Mouse in Box
				is_modify = False;

				for sel in selected:									# Check is Selection exist
					if sel['id'] is box[1] and sel['lb'] is box[6]:		# Check which is same Box
						is_modify = True
					
						if sel['start_frame'] >= idx:
							if not 'end_frame' in sel:
								sel['end_frame'] = sel['start_frame']

							sel['start_frame'] = idx
						
						else:
							sel['end_frame'] = idx

				if not is_modify:
					selected.append({'id': box[1], 'lb': box[6], 'start_frame': idx})

		print(selected)

	if event == cv2.EVENT_RBUTTONDOWN:
		for box in label[idx]:
			if (box[2] <= x <= box[4]) and (box[3] <= y <= box[5]):		# Check Mouse in Box
				for sel in selected:									# Check is Selection exist
					if sel['id'] is box[1] and sel['lb'] is box[6]:		# Check which is same Box
						del sel

		print(selected)

cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

while True:
	idx = cv2.getTrackbarPos("Frame", WINDOW_NAME)
	img = dataset[idx][1][2]
	shape = img.shape

	# < Draw Box and Label >
	for box in label[idx]:

		# Check is Selected
		is_selected = False
		for sel in selected:
			is_same_frame = False
			if sel.get('start_frame') <= idx:
				if not 'end_frame' in sel:
					is_same_frame = True
				else:
					if sel['end_frame'] >= idx:
						is_same_frame = True

			if is_same_frame:
				if sel['id'] == box[1] and sel['lb'] == box[6]:
					is_selected = True

		# Color Select
		box_color = RED_COLOR if is_selected else GREEN_COLOR

		# Box
		cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), box_color, 2)

		# Label
		label_txt = '{} {:d}'.format(YOLO_NAME[box[6]], box[1])
		t_size = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_PLAIN, LABEL_FONT_SIZE, LABEL_FONT_THICK)[0]

		cv2.rectangle(img, (box[2], box[3]-t_size[1]), (box[2]+t_size[0], box[3]+1), box_color, -1)
		cv2.putText(img, label_txt, (box[2], box[3]), cv2.FONT_HERSHEY_PLAIN, LABEL_FONT_SIZE, [255, 255, 255], LABEL_FONT_THICK)

	# < Infomation >
	info_path = "video : " + dataset[idx][1][0] 
	info_frame = "current frame : " + str(idx)
	info_selected = "selected id (label - id) : "
	for x in selected:
		info_selected += str(x["lb"])+"-"+str(x["id"])+", "

	font_scale = 1.0
	text_back = np.array(Image.fromarray(np.zeros((200,shape[1],3),np.uint8)))
	cv2.putText(text_back, info_path, (10,50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, BLACK_COLOR, 1, cv2.LINE_AA)
	cv2.putText(text_back, info_frame, (10,100), cv2.FONT_HERSHEY_SIMPLEX, font_scale, BLACK_COLOR, 1, cv2.LINE_AA)
	cv2.putText(text_back, info_selected, (10,150), cv2.FONT_HERSHEY_SIMPLEX, font_scale, BLACK_COLOR, 1, cv2.LINE_AA)
	
	# < Concatenate Image >
	result = np.array(Image.fromarray(np.zeros((shape[0]+200,shape[1],3),np.uint8)))
	result[:shape[0], :shape[1]] = img
	result[shape[0]:shape[0]+200, :shape[1]] = text_back

	# < Resizing & Draw >
	shape = result.shape
	win_h = int(shape[0] * (imgsz / shape[1]))
	cv2.imshow(WINDOW_NAME, result)
	cv2.resizeWindow(WINDOW_NAME, imgsz, win_h+25)

	# < Save >

	with open(save_path+"/vid_label.txt", "w") as f:
		for sel in selected:
			data = (
				str(sel["id"])+" "+str(sel["lb"])+" "+str(sel["start_frame"])+" "+
				str(sel.get("end_frame") if sel.get("end_frame") else len(dataset)-1) +" \n"
			)
			f.write(data)

	# < Exit Process >
	key = cv2.waitKey(1)
	if key == 27:
		break;

	# < Key Process >
	elif key == 93 or key == 3: # right
		print("right")
		idx+=1
	elif key == 91 or key == 2: # left
		print("left")
		idx-=(1 if idx>0 else 0)

	cv2.setTrackbarPos("Frame", WINDOW_NAME, idx)

cv2.destroyAllWindows()