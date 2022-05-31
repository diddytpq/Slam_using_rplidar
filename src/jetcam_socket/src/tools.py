#! /home/drcl_yang/anaconda3/envs/py36/bin/python

from pathlib import Path
import sys
import os

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add code to path

path = str(FILE.parents[0])
sys.path.insert(0, './yolov5')


import numpy as np
import time
import cv2

from yolov5.utils.augmentations import letterbox


# ball_tracking setup
fgbg = cv2.createBackgroundSubtractorMOG2(20, 25, False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

kernel_erosion_1 = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
kernel_erosion_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))

kernel_dilation_1 = cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))
kernel_dilation_2 = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))



def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def yolo_img_preprocessing(img0, imgsz, stride, pt):
    img = letterbox(img0, new_shape = imgsz, stride= stride, auto= pt)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    return img, img0

def K_mean_img_preprocessing(img0):

    img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    img = img.reshape((-1,3))
    img = np.float32(img)

    return img

def count_object_using_k_means(img0, iou):

    mask=np.zeros_like(segment_img)  


    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1.0)
    K = 2   
    ret,label,center=cv2.kmeans(img0,K,None,criteria,5,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    # test_0 = frame_main[label==0]
    res = center[label.flatten()]
    segment_img = res.reshape((img0.shape))

    mask_img = cv2.fillPoly(mask,iou,(255,255,255))
    bg_removed_img = np.where((mask_img), segment_img,(0))
    segment_img_iou = cv2.cvtColor(bg_removed_img, cv2.COLOR_BGR2GRAY)

    # NO object == 2, detect other object > 2
    object_cnt = len(set(segment_img_iou.flatten().tolist()))

    return object_cnt