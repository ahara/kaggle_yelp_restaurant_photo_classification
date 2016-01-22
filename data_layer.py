import numpy as np


class LBPDict(object):
    def __init__(self, p2b_file_train, p2b_file_test, label_file):
        self.p2b_file_train = p2b_file_train
        self.p2b_file_test = p2b_file_test
        self.label_file = label_file
        self.p2b_dict_train = None
        self.p2b_dict_test = None
        self.labels_dict = None
        self._read_dicts()

    def _read_dicts(self):
        p2b_train, p2b_test = {}, {}
        # Train images
        with open(self.p2b_file_train, 'rb') as f:
            _ = f.next()  # Skip header
            for line in f:
                p, b = line.split(',')
                p2b_train[int(p)] = int(b)
        # Test images
        with open(self.p2b_file_test, 'rb') as f:
            _ = f.next()  # Skip header
            for line in f:
                p, b = line.split(',')
                p2b_test[int(p)] = int(b)
        # Train labels
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
        self.p2b_dict_train = p2b_train
        self.p2b_dict_test = p2b_test
        self.labels_dict = labels

    def get_label(self, photo_id):
        return self.labels_dict.get(self.p2b_dict_train.get(photo_id, -1), None)

    def get_business(self, photo_id):
        business_id = self.p2b_dict_train.get(photo_id, -1)
        if business_id == -1:
            business_id = self.p2b_dict_test.get(photo_id, -1)
        return business_id

    def get_all_train_photo_ids(self):
        return [i for i in self.p2b_dict_train.keys() if self.get_label(i) is not None]

    def get_all_test_photo_ids(self):
        return self.p2b_dict_test.keys()