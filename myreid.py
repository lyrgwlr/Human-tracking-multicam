from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp

import torchreid
from torchreid.data import ImageDataset

class NewDataset(ImageDataset):
    dataset_dir = 'skicapture'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        root_path = '/home/dell/wlr/deep_sort_yolov3-master/output'
        query_path = os.path.join(root_path, 'query')
        gallery_path = os.path.join(root_path, 'gallery')

        # All you need to do here is to generate three lists,
        # which are train, query and gallery.
        # Each list contains tuples of (img_path, pid, camid),
        # where
        # - img_path (str): absolute path to an image.
        # - pid (int): person ID, e.g. 0, 1.
        # - camid (int): camera ID, e.g. 0, 1.
        # Note that
        # - pid and camid should be 0-based.
        # - query and gallery should share the same pid scope (e.g.
        #   pid=0 in query refers to the same person as pid=0 in gallery).
        # - train, query and gallery share the same camid scope (e.g.
        #   camid=0 in train refers to the same camera as camid=0
        #   in query/gallery).
        train = []
        query = []
        gallery = []

        query_imgs = os.listdir(query_path)
        for img in query_imgs:
            img_path = os.path.join(query_path, img)
            query.append((img_path, 0, 0)) 
        
        gallery_imgs = os.listdir(gallery_path)
        idxs = []
        for img in gallery_imgs:
            if int(img.split('_')[0]) not in idxs:
                idxs.append(int(img.split('_')[0]))
        print(idxs)
        for i, img in enumerate(gallery_imgs):
            img_path = os.path.join(gallery_path, img)
            gallery.append((img_path, idxs.index(int(img.split('_')[0])), 1))
            train.append((img_path, idxs.index(int(img.split('_')[0])), 1))
        super(NewDataset, self).__init__(train, query, gallery, **kwargs)

        