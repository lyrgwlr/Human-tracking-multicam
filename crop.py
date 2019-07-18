import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
import random

def main(yolo):
    pics = os.listdir('pictures')
    for g in pics:
        os.remove(os.path.join('pictures/',g))

    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    nms_max_overlap = 1.0

    video_capture = cv2.VideoCapture('videos/walk1.mp4')
    # frame1 = cv2.imread('combine.jpg')
    # video_capture.set(cv2.CAP_PROP_POS_FRAMES, 189)
    # ret, frame2 = video_capture.read()
    # cv2.imwrite('189.jpg', frame2)
    # combine_pic(frame1, frame2, yolo, encoder, 'combine2.jpg')
    frame_cnt = 0   
    w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # init_w = w
    init_w = 0
    add_cnt = 0 
    all_boxes = []
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        boxs = yolo.detect_image(image)
        features = encoder(frame,boxs)
        
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        out_boxes = []
        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
            out_boxes.append([int(bbox[0]-25),int(bbox[1]-25),int(bbox[2]+25),int(bbox[3]+25)])
        all_boxes.append(out_boxes)

        if len(out_boxes) == 0:
            continue

        # if out_boxes[0][0] < 0:
        #     break
        # if out_boxes[0][2] < init_w:
        #     init_w = out_boxes[0][0]
        #     if add_cnt == 0:
        #         output_frame = frame.copy()
        #     else:
        #         output_frame = combine_pic(output_frame, frame, out_boxes)
        #     add_cnt += 1
        #     print(add_cnt)

        if out_boxes[0][2] > w:
            break
        if out_boxes[0][0] >= init_w:
            init_w = out_boxes[0][2]
            if add_cnt == 0:
                output_frame = frame.copy()
            else:
                output_frame = combine_pic(output_frame, frame, out_boxes)
            add_cnt += 1
            print(add_cnt)
            
    all_boxes = np.array(all_boxes)
    np.save('boxes.npy', all_boxes)
    # cv2.imwrite('1.jpg', output_frame)

def combine_pic(bg, add, box):
    img2 = add[box[0][1]:box[0][3], box[0][0]:box[0][2]]
    roi = bg[box[0][1]:box[0][3], box[0][0]:box[0][2]]
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img2gray, 200, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask)
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask_inv)
    dst = cv2.add(img1_bg,img2_fg)
    bg[box[0][1]:box[0][3], box[0][0]:box[0][2]] = dst
    return bg

def get_boxes(frame, yolo, encoder, nms_max_overlap = 1.0):
    out_boxes = []
    image = Image.fromarray(frame[...,::-1]) #bgr to rgb
    boxs = yolo.detect_image(image)
    # print("box_num",len(boxs))
    features = encoder(frame,boxs)
    
    # score to 1.0 here).
    detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
    
    # Run non-maxima suppression.
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]
    

    for det in detections:
        bbox = det.to_tlbr()
        out_boxes.append([int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])])
    return out_boxes
if __name__ == '__main__':
    main(YOLO())
