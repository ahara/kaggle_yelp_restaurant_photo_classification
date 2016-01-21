import numpy as np

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
        return self.labels_dict.get(self.p2b_dict.get(photo_id, -1), None)

    def get_business(self, photo_id):
        return self.p2b_dict.get(photo_id, -1)

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