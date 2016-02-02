import numpy as np
from collections import defaultdict


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
        p2b_train, p2b_test = defaultdict(list), defaultdict(list)
        # Train images
        with open(self.p2b_file_train, 'rb') as f:
            _ = f.next()  # Skip header
            for line in f:
                p, b = line.split(',')
                p2b_train[int(p)].append(int(b))
        # Test images
        with open(self.p2b_file_test, 'rb') as f:
            _ = f.next()  # Skip header
            for line in f:
                p, b = line.split(',')
                p2b_test[int(p)].append(b)
        # Train labels
        labels = {}
        with open(self.label_file, 'rb') as f:
            _ = f.next()  # Skip header
            for line in f:
                b, l_str = line.split(',')
                l_str = l_str.strip()
                l_array = np.zeros((9, ), dtype=int)
                if len(l_str) > 0:
                    for l in l_str.split(' '):
                        l_array[int(l)] = 1
                labels[int(b)] = l_array
        self.p2b_dict_train = p2b_train
        self.p2b_dict_test = p2b_test
        self.labels_dict = labels

    # Probably will be only needed for training on separated photos
    #def get_label(self, photo_id):
    #    return self.labels_dict.get(self.p2b_dict_train.get(photo_id, -1), None)

    def get_business_label(self, business_id):
        return self.labels_dict[business_id]

    def get_business_ids(self, photo_id):
        business_ids = self.p2b_dict_train[photo_id]
        if len(business_ids) == 0:
            business_ids = self.p2b_dict_test[photo_id]
        return business_ids

    def get_all_train_photo_ids(self):
        return self.p2b_dict_train.keys()

    def get_all_test_photo_ids(self):
        return self.p2b_dict_test.keys()


if __name__ == '__main__':
    import consts
    lbp = LBPDict(consts.P2B_TRAIN, consts.P2B_TEST, consts.LABELS_TRAIN)
    print len(lbp.get_all_train_photo_ids())
    print len(lbp.get_all_test_photo_ids())
    print len(lbp.labels_dict.keys())