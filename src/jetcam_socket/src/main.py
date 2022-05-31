#! /home/drcl_yang/anaconda3/envs/py36/bin/python

from pathlib import Path
import sys
import os

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add code to path

path = str(FILE.parents[0])
sys.path.append(path + '/lib')  # add code to path

import socket

import numpy as np
import time
import cv2

import torch
import torch.backends.cudnn as cudnn

from tools import *

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, time_sync

from yolov5.utils.augmentations import letterbox

import argparse

parser = argparse.ArgumentParser(description = 'predict_tennis_ball_landing_point')

parser.add_argument('--video_path', type = str, default='videos/tennis_video_2/1.mov', help = 'input your video path')
parser.add_argument('--record', type = bool, default=False, help = 'set record video')
parser.add_argument('--debug', type = bool, default=False, help = 'set debug mod')


args = parser.parse_args()

device = 0
weights = path + "/lib/yolov5/weights/yolov5m6.pt"
imgsz = 640
conf_thres = 0.25
iou_thres = 0.45
classes = None
agnostic_nms = False
max_det = 1000
half=False
dnn = False

device = select_device(device)
model = DetectMultiBackend(weights, device=device, dnn=dnn)
stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
imgsz = check_img_size(imgsz, s=stride)  # check image size

half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA

if pt:
    model.model.half() if half else model.model.float()

cudnn.benchmark = True  # set True to speed up constant image size inference


color = tuple(np.random.randint(low=200, high = 255, size = 3).tolist())
color = tuple([0,125,255])

start_frame = 0

def object_tracking(model, img, img_ori, device):

    img_in = torch.from_numpy(img).to(device)
    img_in = img_in.float()
    img_in /= 255.0

    if img_in.ndimension() == 3:
        img_in = img_in.unsqueeze(0)
    
    pred = model(img_in, augment=False, visualize=False)

    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    for i, det in enumerate(pred):  # detections per image
        
        im0 = img_ori.copy()

        if len(det):
            det[:, :4] = scale_coords(img_in.shape[2:], det[:, :4], im0.shape).round()

            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s = f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class

                label = names[c] #None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')

                x0, y0, x1, y1 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                plot_one_box([x0, y0, x1, y1], im0, label=label, color=colors(c, True), line_thickness=3)

    return im0



def recvall(sock, count):
    # 바이트 문자열
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def main():
    #"-----------------------------------------------------------------------------"

    HOST = '192.168.79.16'
    PORT = 8888

    # TCP 사용
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    print('Socket created')

    # 서버의 아이피와 포트번호 지정
    s.bind((HOST, PORT))
    print('Socket bind complete')
    # 클라이언트의 접속을 기다린다. (클라이언트 연결을 10개까지 받는다)
    s.listen(10)
    print('Socket now listening')

    # 연결, conn에는 소켓 객체, addr은 소켓에 바인드 된 주소
    conn, addr = s.accept()

    #"-----------------------------------------------------------------------------"
    (x00,y00), (x01,y01), (x10,y10), (x11,y11) = (250, 100), (390,100), (100,350), (540, 350)
    iou = np.array([[(x00,y00), (x01,y01), (x10,y10), (x11,y11)]],dtype=np.int32)


    if args.record:
        codec = cv2.VideoWriter_fourcc(*'X264')
        out = cv2.VideoWriter("yolo_test_video.mp4", codec, fps, (1280,720))

    while True:

        print("-----------------------------------------------------------------")
        t1 = time.time()

        length = recvall(conn, 16)
        stringData = recvall(conn, int(length))
        data = np.fromstring(stringData, dtype='uint8')
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

        frame_k_mean_main = frame.copy()
        frame_yolo_main = frame.copy()


        K_mean_img = K_mean_img_preprocessing(frame_k_mean_main)
        img, img_ori = yolo_img_preprocessing(frame_yolo_main, imgsz, stride, pt)


        #K_means_clustering
        object_check_flag = count_object_using_k_means(K_mean_img, iou)

        #Yolo object detect

        if object_check_flag > 2:
            try:
                object_tracking_img = object_tracking(model, img, img_ori, device)

            except:
                continue

        t2 = time.time()

        print("FPS : " , 1/(t2-t1))
       
        # cv2.imshow('person_tracking_img',cv2.resize(person_tracking_img,(800,600)))
        if object_tracking_img:
            cv2.imshow('object_tracking_img',object_tracking_img)

        cv2.imshow('main_frame',frame)

        if args.record:
            out.write(object_tracking_img)

        key = cv2.waitKey(1)

        if key == 27 : 
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":

    main()