#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
config = tf.ConfigProto()  
config.gpu_options.allow_growth=True  
session = tf.Session(config=config)
KTF.set_session(session)
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

from yolo import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

import torchreid
from myreid import NewDataset

warnings.filterwarnings('ignore')

def main(yolo):

   # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    
   # deep_sort 
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    output_frames = []
    output_rectanger = []
    output_areas = []
    output_wh_ratio = []
    is_vis = True

    videos = ['videos/2_1.mp4']
    cam_id = 0
    for video in videos:
        cam_id += 1
        video_capture = cv2.VideoCapture(video)
        frame_nums = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if is_vis:
            w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            output_path = video.split('.')[0]+'_output'+'.avi'
            out = cv2.VideoWriter(output_path, fourcc, frame_rate, (w, h))

        fps = 0.0
        frame_cnt = 0
        track_cnt = dict()
        t1 = time.time()
        while True:

            ret, frame = video_capture.read()  # frame shape 640*480*3
            if ret != True:
                break

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
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]
            
            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                
                bbox = track.to_tlbr()
                area = (int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1]))
                if track.track_id not in track_cnt:
                    track_cnt[track.track_id] = [[frame_cnt, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), area]]
                else:
                    track_cnt[track.track_id].append([frame_cnt, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), area])
                    
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 2, (0,255,0),2)

            if is_vis:
            # save a frame
                out.write(frame)

            frame_cnt += 1
            print(frame_cnt, '/', frame_nums)
            
            # Press Q to stop!
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
        t2 = time.time()
        if is_vis:
            out.release()

        sorted_index = get_first_n_indexs(track_cnt, 5)
        #body_info = get_body_info(track_cnt, sorted_index, video_capture, 20)
        #merged_track_cnt = get_merged_track_cnt(track_cnt, body_info)
        gallerys = os.listdir('output/gallery')
        for g in gallerys:
            os.remove(os.path.join('output/gallery',g))
        ''' save frames'''
        w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        for index in sorted_index:
            # detect_frame = track_cnt[index][len(track_cnt[index]) // 2]
            if len(track_cnt[index]) > 5:
                detect_frames = random.sample(track_cnt[index], 5)
            else:
                detect_frames = track_cnt[index]
            for item in detect_frames:
                detect_frame = item
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, detect_frame[0])
                _, f = video_capture.read()
                if detect_frame[2] >= 0 and detect_frame[1] >= 0 and detect_frame[4] < h and detect_frame[3] < w:
                    f = f[detect_frame[2]:detect_frame[4], detect_frame[1]:detect_frame[3]]
                    cv2.imwrite('output/gallery/'+str(index)+'_'+str(detect_frame[0])+'.jpg', f)

        final_select_ids = run_reid(cam_id)
        print(final_select_ids)
        """
        select higher similar one index!!!
        """
        #final_select_ids = [1,37,44]
        #final_select_ids = final_select_ids[:1]
        

        
        for i in final_select_ids:
            for item in track_cnt[i]:
                if item[0] not in output_frames:
                    output_frames.append(str(cam_id) + '_' + str(item[0]))
                    output_rectanger.append(item[1:5])
                    output_areas.append(item[-1])
                    output_wh_ratio.append((item[3]-item[1])/(item[4]-item[2]))

        t3 = time.time()
        fps  =  frame_nums / (t3-t1)
        select_time = frame_nums / (t3-t2)
        network_fps = frame_nums / (t2-t1)
        print("select fps: ", select_time)
        print("network fps: ", network_fps)
        print("all fps: ", fps)

        video_capture.release()
        cv2.destroyAllWindows()

    biggest_area = max(output_areas)
    biggest_whratio = max(output_wh_ratio)
    area_score = []
    whratio_score = []
    for area in output_areas:
        area_score.append(area/biggest_area)
    for whratio in output_wh_ratio:
        whratio_score.append(whratio/biggest_whratio)
    all_score = np.array(area_score) + np.array(whratio_score)
    sorted_all_score = list(np.argsort(all_score))
    sorted_all_score.reverse()
    picked_frames = []
    gap = len(sorted_all_score) // 20
    # gap = 1
    for i in range(10):
        picked_frames.append(output_frames[sorted_all_score[gap * i]])
    # for i in sorted_all_score[:10]:
    #     picked_frames.append(output_frames[i])
    print(picked_frames)
    save_images(videos, picked_frames)
        
    if is_vis:
        print('writing video...')
        video_capture = cv2.VideoCapture(videos[0])
        w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
        video_capture.release()

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        output_path = videos[0].split('.')[0] + '_all_combine.avi'
        out = cv2.VideoWriter(output_path, fourcc, frame_rate, (w, h))
        for v in range(len(videos)):
            video_capture = cv2.VideoCapture(videos[v])
            cur_frames = []
            cur_retangle = []
            for j in range(len(output_frames)):
                if int(output_frames[j].split('_')[0])-1 == v:
                    cur_frames.append(int(output_frames[j].split('_')[1]))
                    cur_retangle.append(output_rectanger[j])
            cur_index = list(np.argsort(cur_frames))
            for i in cur_index:
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, cur_frames[i])
                _, f = video_capture.read()
                cv2.rectangle(f, (cur_retangle[i][0], cur_retangle[i][1]), (cur_retangle[i][2], cur_retangle[i][3]),(255,0,0), 2)
                #f = cv2.resize(f, (1920,1080))
                out.write(f)
            video_capture.release()
        out.release()



def save_images(videos, picked_frames):
    for i in picked_frames:
        video = videos[int(i.split('_')[0])-1]
        video_capture = cv2.VideoCapture(video)
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, int(i.split('_')[-1]))
        _, f = video_capture.read()
        cv2.imwrite('picked/'+str(i)+'.jpg', f)
        video_capture.release()
    return

def run_reid(cam_id):
    data_name = 'skicapture' + str(cam_id)
    torchreid.data.register_image_dataset(data_name, NewDataset)
        # Load data manager
    datamanager = torchreid.data.ImageDataManager(
        root='reid-data',
        sources=data_name
    )
    # Build model
    model = torchreid.models.build_model(
        name='resnet50',
        num_classes=datamanager.num_train_pids,
        loss='softmax',
        pretrained=True,
        use_gpu = True
    )
    torchreid.utils.load_pretrained_weights(model, '/home/dell/wlr/deep-person-reid/log/resnet50/model.pth.tar-60')
    model = model.cuda()
    # Build optimizer
    optimizer = torchreid.optim.build_optimizer(
        model,
        optim='adam',
        lr=0.0003
    )
    # Build lr_scheduler
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler='single_step',
        stepsize=20
    )
    # Build engine
    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )

    final_select_ids = engine.run(
        save_dir='log/resnet50',
        max_epoch=60,
        eval_freq=10,
        print_freq=10,
        test_only=True,
        visrank=True,
        visrank_topk=5
    )

    return final_select_ids

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

def get_body_info(track_cnt, indexs, cap, n):
    # for each index, get n body information by BaiduAI API.
    # then count them to get the upper&lower for each index people.
    body_info = {}
    request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/body_attr"
    for index in indexs:
        if len(track_cnt[index]) > n:
            random_pick = random.sample(list(range(len(track_cnt[index]))), n)
        else:
            random_pick = list(range(len(track_cnt[index])))
        single_orient = list()
        single_upper = list()
        single_lower = list()
        for i in random_pick:
            item = track_cnt[index][i]
            cap.set(cv2.CAP_PROP_POS_FRAMES, item[0])
            _, f = cap.read()
            f = f[item[2]:item[4], item[1]:item[3]]
            imgRGB = cv2.cvtColor(f, cv2.IMREAD_COLOR)
            _, buf = cv2.imencode (".jpg", imgRGB)
            bytes_image = Image.fromarray(np.uint8(buf)).tobytes()
            img = base64.b64encode(bytes_image)
            params = {"image":img}
            params = parse.urlencode(params).encode('utf-8')
            header = {'Content-Type': 'application/x-www-form-urlencoded'}
            access_token = '24.b6082cecf7b400621ea94b95b4cb7b0d.2592000.1558848915.282335-16117436'
            request_url = request_url + "?access_token=" + access_token
            res = requests.post(url=request_url, data=params, headers=header)
            new_dict = json.loads(str(res.content,encoding='utf-8'))
            if 'person_info' in new_dict.keys():
                attri = new_dict['person_info'][0]['attributes']
                #print(attri)
                orient = attri['orientation']['name']#正反面
                upper = ""
                lower = ""
                upper += attri['upper_color']['name']
                #upper += attri['upper_wear_texture']['name']
                upper += attri['upper_wear']['name']    
                upper += attri['upper_wear_fg']['name'] 
                lower += attri['lower_color']['name']
                lower += attri['lower_wear']['name']
            else:
                orient = ""
                upper = ""
                lower = ""
            single_orient.append(orient)
            single_upper.append(upper)
            single_lower.append(lower)
        ocnt = 0
        ucnt = 0
        lcnt = 0
        for i in range(len(random_pick)):
            if single_orient.count(single_orient[i]) > ocnt:
                ocnt = single_orient.count(single_orient[i])
                orient = single_orient[i]
            if single_upper.count(single_upper[i]) > ucnt:
                ucnt = single_upper.count(single_upper[i])
                upper = single_upper[i]
            if single_lower.count(single_lower[i]) > lcnt:
                lcnt = single_lower.count(single_lower[i])
                lower = single_lower[i]

        body_info[index] = {'orient':orient, 'upper':upper, 'lower':lower}
        print(index, body_info[index])
    return body_info

def get_merged_track_cnt(track_cnt, body_info):
    # merge different indexs together if is same person
    # by compare with body information.
    merged_track_cnt = dict()
    indexs = list(body_info.keys())
    same_id = [[] for _ in range(len(indexs))]
    for i in range(len(indexs)):
        if i != 0:
            is_continue = False
            for befor in range(0,i):
                if indexs[i] in same_id[befor]:
                    is_continue = True
            if is_continue:
                continue
        for j in range(i+1, len(indexs)):
            if body_info[indexs[i]]['upper'] == body_info[indexs[j]]['upper'] and body_info[indexs[i]]['lower'] == body_info[indexs[j]]['lower']:
                if indexs[i] not in same_id[i]:
                    same_id[i].append(indexs[i])
                same_id[i].append(indexs[j])
                if indexs[i] not in merged_track_cnt.keys():
                    merged_track_cnt[indexs[i]] = track_cnt[indexs[i]] + track_cnt[indexs[j]]
                else:
                    merged_track_cnt[indexs[i]] += track_cnt[indexs[j]]
    all_indexs = []
    for i in same_id:
        all_indexs += i
    for i in indexs:
        if i not in all_indexs:
            merged_track_cnt[i] = track_cnt[i]
    return merged_track_cnt

if __name__ == '__main__':
    main(YOLO())
