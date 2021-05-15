import sys
sys.path.insert(0, '../external/yolo_v5_deepsort/yolov5')
sys.path.insert(0, '../external/yolo_v5_deepsort')

from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device, time_synchronized

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

import torch
import torch.backends.cudnn as cudnn

import argparse
import os
import platform
import shutil
import time

from pathlib import Path
import cv2
import yaml

img_save_path = ""
is_save_cropped = False

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

label_names = []
default_classes = []

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
def draw_boxes(frame, img, bbox, wh, identities=None, offset=(0, 0), yolo_label=None):
    no_colored = None
    stacked_img = []

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        
        cropped_img = img[y1:y2, x1:x2]
        if(cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0): # wrong coord
            continue

        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        name = "" if yolo_label is None else label_names[yolo_label]
        color = compute_color_for_labels(id)
        label = '{} {:d}'.format(name, id)

        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

        if is_save_cropped:
            cv2.imwrite(
                img_save_path+"/images/f"+str(frame)+"_id"+str(id)+"_label_is_"+str(name)+".jpg", 
                cropped_img
            )

        no_colored = torch.Tensor( 
            cv2.resize(cropped_img, dsize=wh, interpolation=cv2.INTER_CUBIC)
        )

        stacked_img.append(torch.tensor(no_colored))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        if font_size is not None:
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, font_size, 2)[0]

            cv2.rectangle(
                img, (x1, y1), (x1 + t_size[0]+2, y1 + t_size[1]+2), color, -1)
            cv2.putText(
                img, label, (x1, y1 + t_size[1] + 1), cv2.FONT_HERSHEY_PLAIN, font_size, [255, 255, 255], 1)

    test = torch.stack(stacked_img, dim=0)
    print(test.shape)
    return img, test


def detect(opt, save_img=False):
    global label_names
    global is_save_cropped
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
    is_save_cropped = opt.save_cropped_img

    font_size = opt.font_size

    # get options var
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size

    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    pre_img = None

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)

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
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    os.makedirs(out+"/images/")

    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = torch.load(weights, map_location=device)[
        'model'].float()  # load to FP32
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        # view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        # view_img = True
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # run once
    _ = model(img.half() if half else img) if device.type != 'cpu' else None

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
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

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
                        _, pre_img = draw_boxes(
                            frame_idx, im0, bbox_xyxy, (vid_w, vid_h), identities, yolo_label=idx)

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

            else:
                deepsort[idx].increment_ages()

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                print('saving img!')
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    print('saving video!')
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        # w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        # h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (vid_w, vid_h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='checkpoint/yolov5s.pt', help='model.pt path')
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
    parser.add_argument("--save-cropped-img", action='store_true',
                        help="save detected object's cropped images")

    args = parser.parse_args()

    with torch.no_grad():
        detect(args)