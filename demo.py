#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
import tensorflow as tf
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
import random
warnings.filterwarnings('ignore')

def main(yolo):

   # Definition of the parameters
    max_cosine_distance = 0.2
    nn_budget = None
    nms_max_overlap = 0.4
    
   # deep_sort 
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True 
    
    video_capture = cv2.VideoCapture('videos/0701/1_1.mp4')
    frame_nums = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('videos/3_3_out.avi', fourcc, frame_rate, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1 
        
    fps = 0.0
    frame_cnt = 0
    track_cnt = dict()
    while True:

        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        t1 = time.time()

       # image = Image.fromarray(frame)
        image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        boxs = yolo.detect_image(image)
       # print("box_num",len(boxs))
        features = encoder(frame,boxs)
        
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.delete_overlap_box(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 

            bbox = track.to_tlbr()

            area = (int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1]))
            other_people_num = len(tracker.tracks)-1

            if track.track_id not in track_cnt:
                track_cnt[track.track_id] = [(frame_cnt, other_people_num, area)]
            else:
                track_cnt[track.track_id].append((frame_cnt, other_people_num, area))

            if track.track_id == 1:
                cv2.imwrite('images/' + str(frame_cnt) + '_human.png', frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(0,255,0), 2)
            cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 1.2, (0,255,0),2)


        # for det in detections:
        #     bbox = det.to_tlbr()
        #     cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
            
        #cv2.imshow('', frame)
        
        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')
            
        frame_cnt += 1
        print(frame_cnt, '/', frame_nums)
        
        # Press Q to stop!
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
    t2 = time.time()

    # first_num = 3 
    # ''' get first three indexs of the most occur '''
    # ''' make sure the number of people in the video is much than valuable: first_num '''
    # occur_frames_num = dict()
    # for index in track_cnt:
    #     occur_frames_num[index] = len(track_cnt[index])
    # sorted_frames_num = sorted(occur_frames_num.values())
    # sorted_frames_num.reverse()
    # if len(sorted_frames_num) > first_num:
    #     sorted_frames_num = sorted_frames_num[:first_num] #get first 3 occur frames
    # sorted_indexs = []
    # for index in sorted_frames_num:
    #     for key,value in occur_frames_num.items():
    #         if value == index and index not in sorted_indexs:
    #             sorted_indexs.append(key)

    # first_num = 50
    # second_frames = dict()
    # ''' choose the frames based on area'''
    # area_sorted = list()
    # for index in sorted_indexs:
    #     area_sorted.append(frame[2] for frame in track_cnt[index])
    # for i in range(len(area_sorted)):
    #     area_sorted[i] = sorted(area_sorted[i])
    #     area_sorted[i].reverse()
    #     if len(area_sorted[i]) > first_num:
    #         area_sorted[i] = area_sorted[i][:first_num]
    # for i in range(len(sorted_indexs)):
    #     second_frames[sorted_indexs[i]] = list()
    #     for area in area_sorted[i]:
    #         for frame in track_cnt[sorted_indexs[i]]:
    #             if frame[2] == area and frame[0] not in second_frames[sorted_indexs[i]]:
    #                 second_frames[sorted_indexs[i]].append(frame[0])
    #                 break

    # first_num = 10
    # last_frames = dict()
    # # choose the frames based on other_people
    # other_people_sorted = list()
    # for index in sorted_indexs:
    #     tmp = list()
    #     for frame in track_cnt[index]:
    #         if frame[0] in second_frames[index]:
    #             tmp.append(frame[1])
    #     other_people_sorted.append(tmp)
    # for i in range(len(other_people_sorted)):
    #     other_people_sorted[i] = sorted(other_people_sorted[i])
    #     if len(other_people_sorted[i]) > first_num:
    #         other_people_sorted[i] = other_people_sorted[i][:first_num]
    # for i in range(len(sorted_indexs)):
    #     last_frames[sorted_indexs[i]] = list()
    #     for other_people in other_people_sorted[i]:
    #         for frame in track_cnt[sorted_indexs[i]]:  
    #             if frame[1] == other_people and frame[0] not in last_frames[sorted_indexs[i]]:
    #                 last_frames[sorted_indexs[i]].append(frame[0])

    ''' save frames'''
    for key in track_cnt:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, random.sample(track_cnt[key], 1)[0][0])
        _, f = video_capture.read()
        cv2.imwrite('output/'+str(key)+'.jpg', f)

    t3 = time.time()
    fps  = (t3-t1) / frame_nums * 1000  
    select_time = t3-t2
    network_fps = (t2-t1) / frame_nums * 1000
    print("select time: ", select_time)
    print("network fps: ", network_fps)
    print("all fps: ", fps)

    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    main(YOLO())
