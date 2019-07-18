#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from timeit import time
import warnings
import sys
import cv2
import numpy as np
import base64
import requests
import urllib
from urllib import parse
import json
import random
import time
from PIL import Image
from collections import Counter
import operator

from yolo import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

from myengine import REID

warnings.filterwarnings('ignore')

def main(yolo):
         
   # Definition of the parameters
    max_cosine_distance = 0.2
    nn_budget = None
    nms_max_overlap = 0.4
    
   # deep_sort 
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1) # use to get feature
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_age=10)

    output_frames = []
    output_rectanger = []
    output_areas = []
    output_wh_ratio = []

    is_vis = True
    videos = ['videos/0701/1_1.mp4']

    all_frames = []
    for video in videos:
        video_capture = cv2.VideoCapture(video)
        w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
        while True:
            ret, frame = video_capture.read() 
            if ret != True:
                video_capture.release()
                break
            all_frames.append(frame)

    frame_nums = len(all_frames)
    if is_vis:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        output_path = video.split('.')[0]+'_output'+'.avi'
        out = cv2.VideoWriter(output_path, fourcc, frame_rate, (w, h))

    fps = 0.0
    frame_cnt = 0
    t1 = time.time()

    track_cnt = dict()
    images_by_id = dict()
    ids_per_frame = []
    for frame in all_frames:
        image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        boxs = yolo.detect_image(image) # n * [topleft_x, topleft_y, w, h]
        features = encoder(frame,boxs) # n * 128
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)] # length = n
        
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        #indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        indices = preprocessing.delete_overlap_box(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices] # length = len(indices)

        # Call the tracker 
        tracker.predict()
        tracker.update(detections)
        tmp_ids = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            
            bbox = track.to_tlbr()
            area = (int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1]))
            if bbox[0] >= 0 and bbox[1] >= 0 and bbox[3] < h and bbox[2] < w:
                tmp_ids.append(track.track_id)
                if track.track_id not in track_cnt:
                    track_cnt[track.track_id] = [[frame_cnt, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), area]]
                    images_by_id[track.track_id] = [frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]]
                else:
                    track_cnt[track.track_id].append([frame_cnt, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), area])
                    images_by_id[track.track_id].append(frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 2, (0,255,0),2)
        ids_per_frame.append(set(tmp_ids))
        
        if is_vis:
        # save a frame
            out.write(frame)
        t2 = time.time()
        
        frame_cnt += 1
        print(frame_cnt, '/', frame_nums)

    if is_vis:
        out.release()

    for i in images_by_id:
        print(i, len(images_by_id[i]))

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    reid = REID()
    threshold = 250
    exist_ids = set()
    final_fuse_id = dict()
    for f in ids_per_frame:
        if f:
            if len(exist_ids) == 0:
                for i in f:
                    final_fuse_id[i] = [i]
                exist_ids = exist_ids or f
            else:
                new_ids = f-exist_ids
                for nid in new_ids:
                    dis = []
                    if len(images_by_id[nid])<10:
                        exist_ids.add(nid)
                        continue
                    unpickable = []
                    for i in f:
                        for key,item in final_fuse_id.items():
                            if i in item:
                                unpickable += final_fuse_id[key]

                    for oid in (exist_ids-set(unpickable))&set(final_fuse_id.keys()): 
                        tmp = np.mean(reid.get_features(images_by_id[nid], images_by_id[oid]))
                        print(nid, oid, tmp)
                        dis.append([oid, tmp])
                    exist_ids.add(nid)
                    if not dis:
                        final_fuse_id[nid] = [nid]
                        continue
                    dis.sort(key=operator.itemgetter(1))
                    if dis[0][1] < threshold:
                        combined_id = dis[0][0]
                        images_by_id[combined_id] += images_by_id[nid]
                        final_fuse_id[combined_id].append(nid)
                    else:
                        final_fuse_id[nid] = [nid]
    print(final_fuse_id)

    if is_vis:
        print('writing video...')
        output_dir = 'videos/' + videos[0].split('/')[-1].split('.')[0]
        print(output_dir)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        video_capture = cv2.VideoCapture(videos[0])
        w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        for idx in final_fuse_id:
            output_path = os.path.join(output_dir, str(idx)+'.avi')
            out = cv2.VideoWriter(output_path, fourcc, frame_rate, (w, h))
            for i in final_fuse_id[idx]:
                print(idx, i)
                for f in track_cnt[i]:
                    video_capture.set(cv2.CAP_PROP_POS_FRAMES, f[0])
                    _, frame = video_capture.read()
                    cv2.rectangle(frame, (f[1], f[2]), (f[3], f[4]),(255,0,0), 2)
                    out.write(frame)
            out.release()
        video_capture.release()
    #    sorted_index = get_first_n_indexs(track_cnt, 5)

    #     print(final_select_ids)
   
    #     for i in final_select_ids:
    #         for item in track_cnt[i]:
    #             if item[0] not in output_frames:
    #                 output_frames.append(str(cam_id) + '_' + str(item[0]))
    #                 output_rectanger.append(item[1:5])
    #                 output_areas.append(item[-1])
    #                 output_wh_ratio.append((item[3]-item[1])/(item[4]-item[2]))

    #     t3 = time.time()
    #     fps  =  frame_nums / (t3-t1)
    #     select_time = frame_nums / (t3-t2)
    #     network_fps = frame_nums / (t2-t1)
    #     print("select fps: ", select_time)
    #     print("network fps: ", network_fps)
    #     print("all fps: ", fps)

    #     video_capture.release()
    #     cv2.destroyAllWindows()

    # biggest_area = max(output_areas)
    # biggest_whratio = max(output_wh_ratio)
    # area_score = []
    # whratio_score = []
    # for area in output_areas:
    #     area_score.append(area/biggest_area)
    # for whratio in output_wh_ratio:
    #     whratio_score.append(whratio/biggest_whratio)
    # all_score = np.array(area_score) + np.array(whratio_score)
    # sorted_all_score = list(np.argsort(all_score))
    # sorted_all_score.reverse()
    # picked_frames = []
    # gap = len(sorted_all_score) // 20
    # # gap = 1
    # for i in range(10):
    #     picked_frames.append(output_frames[sorted_all_score[gap * i]])
    # # for i in sorted_all_score[:10]:
    # #     picked_frames.append(output_frames[i])
    # print(picked_frames)
    # save_images(videos, picked_frames)
        
    



def save_images(videos, picked_frames):
    for i in picked_frames:
        video = videos[int(i.split('_')[0])-1]
        video_capture = cv2.VideoCapture(video)
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, int(i.split('_')[-1]))
        _, f = video_capture.read()
        cv2.imwrite('picked/'+str(i)+'.jpg', f)
        video_capture.release()
    return


def get_first_n_indexs(track_cnt, n):
    # get first n indexs of the most occur
    occur_frames_num = dict()
    for index in track_cnt:
        if len(track_cnt[index]) > n:
            occur_frames_num[index] = len(track_cnt[index])
    sorted_frames_num = sorted(occur_frames_num.values())
    sorted_frames_num.reverse()
    # if len(sorted_frames_num) > n:
    #     sorted_frames_num = sorted_frames_num[:n] #get first 3 occur frames
    sorted_indexs = []
    for index in sorted_frames_num:
        for key,value in occur_frames_num.items():
            if value == index and index not in sorted_indexs:
                sorted_indexs.append(key)
    return sorted_indexs

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    main(YOLO())
