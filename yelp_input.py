"""Code for reading Yelp's input data"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import consts


class LBPDict(object):
    def __init__(self):
        self.p2b_file = consts.P2B_TRAIN
        self.label_file = consts.LABELS_TRAIN
        self.p2b_dict = None
        self.labels_dict = None
        self._read_dicts()

    def _read_dicts(self):
        p2b = {}
        with open(self.p2b_file, 'rb') as f:
            _ = f.next()  # Skip header
            for line in f:
                p, b = line.split(',')
                p2b[int(p)] = int(b)
        labels = {}
        with open(self.label_file, 'rb') as f:
            _ = f.next()  # Skip header
            for line in f:
                b, l_str = line.split(',')
                l_str = l_str.strip()
                if len(l_str) == 0:  # Skip lines with unknown labels
                    continue
                l_array = np.zeros((9, ), dtype=int)
                for l in l_str.split(' '):
                    l_array[int(l)] = 1
                labels[int(b)] = l_array
        self.p2b_dict = p2b
        self.labels_dict = labels

    def get_label(self, photo_id):
        #if photo_id in [148777, 457176, 250188]:
        #    print('====== Skipping small image ======')
        #    return None
        return self.labels_dict.get(self.p2b_dict.get(photo_id, -1), None)

    def get_business(self, photo_id):
        return self.p2b_dict.get(photo_id, -1)

    def get_label_str(self, photo_id):
        r = self.get_label(photo_id)
        return ','.join(map(lambda x: str(int(x)), list(r)))

    def get_train_val_split_point(self, train_ratio=0.8):
        return int(len(self.p2b_dict.keys()) * train_ratio)

    def get_train_photo_ids(self):
        return [i for i in self.p2b_dict.keys()[:self.get_train_val_split_point()]
                if self.get_label(i) is not None]

    def get_val_photo_ids(self):
        return [i for i in self.p2b_dict.keys()[self.get_train_val_split_point():]
                if self.get_label(i) is not None]

    def get_all_train_photo_ids(self):
        return [i for i in self.p2b_dict.keys() if self.get_label(i) is not None]


def read_yelp(filename_queue):
    class YelpRecord(object):
        pass

    result = YelpRecord()

    reader = tf.IdentityReader()
    key, file_contents = reader.read(filename_queue)
    r = tf.decode_csv(file_contents, [[''], [-1], [0], [0], [0], [0], [0], [0], [0], [0], [0]], field_delim=',')
    img_contents = tf.read_file(r[0])
    result.uint8image = tf.image.decode_jpeg(img_contents, channels=3)
    result.label = tf.cast(tf.pack(r[2:]), tf.float32)
    result.business = r[1]

    return result