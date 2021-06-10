import sys
sys.path.insert(0, '../external/yolo_v5_deepsort/yolov5')
sys.path.insert(0, '../external/yolo_v5_deepsort')

sys.path.insert(0, './abnormal_detect')
# sys.path.insert(0, './abnormal/util')

from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device, time_synchronized

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

from abnormal_detect.model import AbnormalDetector
from abnormal_detect.util.init_models import initialize_model

import torch
import torch.backends.cudnn as cudnn

import numpy as np
from pathlib import Path
import cv2
from PIL import ImageFont, ImageDraw, Image
import yaml

import argparse
import os
import platform
import shutil
import time

SERVER_URL 		 = "http://localhost:3000/"
ABNORMAL_EPSILON = 0.7
MAX_OBJECT       = 64

IMG_NET_SIZE = 224

DUMMY_LABEL 	= torch.tensor([-1, -1, -1, -1, -1, -1]).float()
DUMMY_OUTPUT	= torch.tensor([0]).float()

RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
WHITE_COLOR = (255, 255, 255)

img_save_path = ""

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

label_names = []
default_classes = []

# < Set CCTV Infomation >
CCTV_NUM        = 1
CCTV_LOCATION   = 1   
CCTV_STATE      = "warning"
CCTV_URL        = "url"

def upload_data():
	command = (
		'curl -L -v -d ' +
		'\'{{ "cctv_number":"{0}","cctv_location":"{1}","cctv_state":"{2}","cctv_url":"{3}" }}\''+
		' -H "Accept: application/json" -H "Content-Type: application/json" -X POST {4}'
	).format(CCTV_NUM, CCTV_LOCATION, CCTV_STATE, CCTV_URL, SERVER_URL)

	os.system(command)

def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

font_size = None
def draw_boxes(frame, img, bbox, identities=None, offset=(0, 0), yolo_label=None):

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        name = "" if yolo_label is None else label_names[yolo_label]
        color = compute_color_for_labels(id)
        label = '{} {:d}'.format(name, id)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        if font_size is not None:
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, font_size, 2)[0]

            cv2.rectangle(
                img, (x1, y1), (x1 + t_size[0]+2, y1 + t_size[1]+2), color, -1)
            cv2.putText(
                img, label, (x1, y1 + t_size[1] + 1), cv2.FONT_HERSHEY_PLAIN, font_size, [255, 255, 255], 1)

    return img

GRAPH_UNIT = 100
val_record = [] 
def draw_indicator(frame, img, val, epsilon = 0.7):
    global val_record
    shape = img.shape

    val_record.append(val)

    # < Infomation >
    font_scale = 1.0
    text_back = np.array(Image.fromarray(np.zeros((200,shape[1],3),np.uint8)))
    cv2.putText(text_back, str(frame), (10,50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, WHITE_COLOR, 1, cv2.LINE_AA)

    start = 0
    if len(val_record) > GRAPH_UNIT:
        start = len(val_record)-GRAPH_UNIT

    pos = 0
    start_h = 40

    graph_w = int(shape[1] / GRAPH_UNIT)
    graph_h = 125

    text_back = cv2.line( text_back, 
        (0, int(graph_h - epsilon*graph_h + start_h)), 
        (shape[1], int(graph_h - epsilon*graph_h + start_h)), 
        RED_COLOR, 4
    )

    for i in range(start, len(val_record)-1):
        text_back = cv2.line( text_back, 
            (pos*graph_w, graph_h - int(val_record[i] * graph_h) + start_h), 
            ((pos+1)*graph_w, graph_h - int(val_record[i+1] * graph_h) + start_h), 
            WHITE_COLOR, 4
        )
        pos += 1

    # < Concatenate Image >
    result = np.array(Image.fromarray(np.zeros((shape[0]+200,shape[1],3),np.uint8)))
    result[:shape[0], :shape[1]] = img
    result[shape[0]:shape[0]+200, :shape[1]] = text_back

    return result

def detect(opt, save_img=True):
    global label_names
    global font_size
    global img_save_path

    with open(opt.config_yolo) as f:
        loaded = yaml.load(f, Loader=yaml.FullLoader)
        label_names = loaded['names']
    
    classes = opt.classes

    if len(classes) is 0:
        classes = [ i for i in range(len(label_names)) ]

    opt.img_size = check_img_size(opt.img_size)

    img_save_path = opt.output

    font_size = opt.font_size

    # get options var
    out, source, yolo_weights, abnormal_weight, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.yolo_weights, opt.abnormal_weight, opt.view_img, opt.save_txt, opt.img_size

    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    pre_img = None

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)

    vid_output = []

    deepsort = []
    for i in range(len(classes)):
        deepsort.append(
            DeepSort(
                cfg.DEEPSORT.REID_CKPT,
                max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                use_cuda=True
            )
        )

    # Initialize
    device = None
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    os.makedirs(out+"/images/")

    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    print ("Load Yolo Model And Weight ... ", end='')
    yolo_model = torch.load(yolo_weights, map_location=device)[
        'model'].float()  # load to FP32
    yolo_model.to(device).eval()
    if half:
        yolo_model.half()  # to FP16
    print ("Done.")

    print ("Load Abnormal Detection Model And Weight ... ", end='')
    lstm_hidden_size = 128
    abnormal_model = AbnormalDetector(
        device, 
        6, 1, 				# label input, output size
        lstm_hidden_size, 	# lstm hidden layer size
        96, 					# label input hidden size
        24, 					# concat_size size
        max_obj_size=MAX_OBJECT
    )
    abnormal_model.load_state_dict( torch.load(abnormal_weight, map_location=device) )
    abnormal_model.to(device).eval()

    abnormal_h = abnormal_model.init_hidden(MAX_OBJECT)
    print("Done.")

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        # view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        # view_img = True
        # save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = yolo_model.module.names if hasattr(yolo_model, 'module') else yolo_model.names

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # run once
    _ = yolo_model(img.half() if half else img) if device.type != 'cpu' else None

    save_path = str(Path(out))
    txt_path = str(Path(out)) + '/results.txt'

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        vid_w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Inference
        t1 = time_synchronized()
        pred = yolo_model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=classes, agnostic=opt.agnostic_nms)

        # Process detections
        labels = []
        for i, det in enumerate(pred):  # detections per image

            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            idx = 0
            t2 = 0
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # for c in det[:, -1].unique():
                #     n = (det[:, -1] == c).sum()  # detections per class
                #     s += '%g %ss, ' % (n, names[int(c)])  # add to string

                det_by_label = [[] for i in range(len(classes))]
                for it in det:
                    det_by_label[int(it[-1].item())].append(it)

                for idx, elem in enumerate(det_by_label):

                    if len(elem) == 0: 
                        continue

                    cur_det = torch.stack(elem, dim=0)

                    bbox_xywh = []
                    confs = []
                    sliced_img = None

                    # Adapt detections to deep sort input format
                    for *xyxy, conf, cls in cur_det:
                        x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                        obj = [x_c, y_c, bbox_w, bbox_h]
                        bbox_xywh.append(obj)
                        confs.append([conf.item()])

                    xywhs = torch.Tensor(bbox_xywh)
                    confss = torch.Tensor(confs)

                    # Pass detections to deepsort
                    outputs = deepsort[idx].update(xywhs, confss, im0)

                    # draw boxes for visualization
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        draw_boxes(frame_idx, im0, bbox_xyxy, identities, yolo_label=idx)

                        for output in outputs:
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2]
                            bbox_h = output[3]
                            identity = output[-1]

                            label = torch.tensor(
                                (identity, bbox_left/vid_w, bbox_top/vid_h, bbox_w/vid_w, bbox_h/vid_h, idx)
                            )
                            labels.append(label)

                    # Write MOT compliant results to file
                    if save_txt and len(outputs) != 0:
                        for j, output in enumerate(outputs):
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2]
                            bbox_h = output[3]
                            identity = output[-1]
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_left,
                                                            bbox_top, bbox_w, bbox_h, idx, -1, -1, -1))  # label format
                                                            
                if len(labels) < MAX_OBJECT:
                    n_dummy = MAX_OBJECT - len(labels)
                    for _ in range(0, n_dummy):
                        labels.append(DUMMY_LABEL)

                stacked_label = torch.stack(labels, dim = 0).float()

                if frame_idx % 10 == 0:
                    print("reset!!!!!!!!")
                    abnormal_h = abnormal_model.init_hidden(MAX_OBJECT)

                abnormal_h  = tuple([e.data for e in abnormal_h])

                stacked_label   = stacked_label.to(device)
                img             = img.to(device)

                result, abnormal_h = abnormal_model((img, stacked_label), abnormal_h)
                labels = []

                max_result = 0.0
                for r in result:
                    print(float(r[0]), " - ", end='')
                    if float(r[0]) > max_result:
                        max_result = float(r[0])
                print(max_result)

                # abnormal situation detected
                if max_result >= ABNORMAL_EPSILON:
                    upload_data()

                im0 = draw_indicator(frame_idx, im0, max_result)

            else:
                deepsort[idx].increment_ages()

            t2 = time_synchronized()

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (vid_w, vid_h+200))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', type=str,
                        default='checkpoint/yolov5s.pt', help='yolo_model.pt path')
    parser.add_argument('--abnormal-weight', type=str,
                        default='checkpoint/abnormal.pt', help='abnormal_detection_model.pt path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='inference/images', help='source')
    parser.add_argument('--output', type=str, default='inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int,
                        default=default_classes, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument("--config-deepsort", type=str,
                        default="configs/deep_sort.yaml")
    parser.add_argument("--config-yolo", type=str,
                        default="configs/data.yaml")
    parser.add_argument("--font-size", type=int,
                        default=None)

    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)

    with torch.no_grad():
        detect(args)