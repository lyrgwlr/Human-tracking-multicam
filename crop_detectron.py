from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import numpy as np
from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default='model_data/e2e_mask_rcnn_R-101-FPN_2x.yaml',
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default='model_data/model_final.pkl',
        type=str
    )
    parser.add_argument(
        '--input',
        default='videos/walk1.mp4',
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        default='/home/dell/wlr/deep_sort_yolov3-master/mask_images/',
        type=str
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='Threshold for visualizing detections',
        default=0.7,
        type=float
    )
    parser.add_argument(
        '--num_gpu',
        default=1,
        type=int
    )
    parser.add_argument(
        '--left_in',
        default=False,
        type=bool
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def combine_pic(bg, add, box, mask):
    box[0] = int(box[0])
    box[1] = int(box[1])
    box[2] = int(box[2])
    box[3] = int(box[3])
    mask = mask[box[1]:box[3],box[0]:box[2]]
    img2gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img2gray, 200,255, cv2.THRESH_BINARY) 
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(bg[box[1]:box[3],box[0]:box[2]],bg[box[1]:box[3],box[0]:box[2]],mask = mask_inv)
    img2_fg = cv2.bitwise_and(add[box[1]:box[3],box[0]:box[2]],add[box[1]:box[3],box[0]:box[2]],mask = mask)
    dst = cv2.add(img1_bg,img2_fg)
    bg[box[1]:box[3],box[0]:box[2]] = dst

    # img2 = add[box[1]:box[3], box[0]:box[2]]
    # roi = bg[box[1]:box[3], box[0]:box[2]]
    # img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    # _, mask = cv2.threshold(img2gray, 200, 255, cv2.THRESH_BINARY)
    # mask_inv = cv2.bitwise_not(mask)
    # img1_bg = cv2.bitwise_and(roi,roi,mask = mask)
    # img2_fg = cv2.bitwise_and(img2,img2,mask = mask_inv)
    # dst = cv2.add(img1_bg,img2_fg)
    # bg[box[1]:box[3],box[0]:box[2]] = dst
    return bg

workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
args = parse_args()
merge_cfg_from_file(args.cfg)
cfg.NUM_GPUS = args.num_gpu
args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
assert_and_infer_cfg(cache_urls=False)

assert not cfg.MODEL.RPN_ONLY, \
    'RPN models are not supported'
assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
    'Models that require precomputed proposals are not supported'

model = infer_engine.initialize_model_from_cfg(args.weights)
dummy_coco_dataset = dummy_datasets.get_coco_dataset()

video_capture = cv2.VideoCapture(args.input)
w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_num = 0

if args.left_in:
    init_w = 0
else:
    init_w = w

add_cnt = 0
print(args.left_in)
while True:
    ret, im = video_capture.read()  # frame shape 640*480*3
    if ret != True:
        break

    with c2_utils.NamedCudaScope(0):
        cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(model, im, None)

    output_path = os.path.join(args.output_dir, str(frame_num)+".jpg")
    box, mask = vis_utils.get_box_mask(im[:, :, ::-1], cls_boxes, cls_segms, cls_keyps, 
                                        args.thresh, output_dir=output_path)

    if args.left_in:
        if mask is not None:
            if box[2] > w:
                break
            if box[0] >= init_w:
                print(box[0])
                init_w = box[2]
                if add_cnt == 0:
                    output_frame = im.copy()
                else:
                    output_frame = combine_pic(output_frame, im, box, mask)
                add_cnt += 1
    else:
        if mask is not None:
            if box[0] < 0:
                break
            if box[2] < init_w:
                init_w = box[0]
                if add_cnt == 0:
                    output_frame = im.copy()
                else:
                    output_frame = combine_pic(output_frame, im, box, mask)
                add_cnt += 1
    frame_num += 1
cv2.imwrite('picked/combine_jump3.jpg', output_frame)

