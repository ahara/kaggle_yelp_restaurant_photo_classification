import cPickle
import logging
import mxnet as mx
import numpy as np
import os
from collections import defaultdict
from skimage import io, transform

import consts
import data_layer


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Params
top_n = [5, 10, 20]
aggregation = ['sum', 'max']
model_name = 'ibn'  # iv3|ibn

inception_bn = {'prefix': 'pretrained/InceptionBN/Inception_BN',
                'synset': 'pretrained/InceptionBN/synset.txt',
                'mean_img': 'pretrained/InceptionBN/mean_224.nd',
                'size': 224,
                'round': 39}
inception_v3 = {'prefix': 'pretrained/Inception-v3/Inception-7',
                'synset': 'pretrained/Inception-v3/synset.txt',
                'mean_img': None,
                'size': 299,
                'round': 1}
model_conf = {'ibn': inception_bn, 'iv3': inception_v3}

# Load the pre-trained model
prefix = model_conf[model_name]['prefix']
img_size = model_conf[model_name]['size']
num_round = model_conf[model_name]['round']
model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.gpu(), numpy_batch_size=1)

# Load mean image
mean_img = None
if model_conf[model_name]['mean_img'] is not None:
    mean_img = mx.nd.load(model_conf[model_name]['mean_img'])["mean_img"]

# Load synset (text label)
synset = [l.strip() for l in open(model_conf[model_name]['synset']).readlines()]


def preprocess_image(path):
    img = io.imread(path)
    # We crop image from center
    short_egde = min(img.shape[:2])
    yy = int((img.shape[0] - short_egde) / 2)
    xx = int((img.shape[1] - short_egde) / 2)
    crop_img = img[yy:(yy + short_egde), xx:(xx + short_egde)]
    # Resize to 224, 224
    resized_img = transform.resize(crop_img, (img_size, img_size))
    # Convert to numpy.ndarray
    sample = np.asarray(resized_img) * 256
    # Swap axes to make image from (img_size, img_size, 4) to (3, img_size, img_size)
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)
    # Sub mean
    if mean_img is None:
        normed_img = (sample - 128.) / 128.
        normed_img = np.reshape(normed_img, (1, 3, img_size, img_size))
    else:
        normed_img = sample - mean_img.asnumpy()
        normed_img.resize(1, 3, img_size, img_size)
    return normed_img


def img2words(photo_ids, photo_dir, use_labels):
    x, y = defaultdict(dict), {}
    cnt = 0

    for i in photo_ids:
        cnt += 1
        if cnt % 100 == 0:
            print photo_dir, cnt / float(len(photo_ids))
        business_ids = lbp.get_business_ids(i)

        if use_labels:
            for b in business_ids:
                y[b] = lbp.get_business_label(b)

        p = os.path.join(photo_dir, '%s.jpg' % str(i))

        # Get preprocessed batch (single image batch)
        batch = preprocess_image(p)
        # Get prediction probability of 1000 classes from model
        prob = model.predict(batch)[0]
        # Argsort, get prediction index from largest prob to lowest
        pred = np.argsort(prob)[::-1]

        for b in business_ids:
            for n in top_n:
                for agg in aggregation:
                    data_name = '%d_%s' % (n, agg)
                    features = x[data_name].get(b, {})

                    # Get topN labels
                    for j in xrange(n):
                        k = synset[pred[j]]
                        p = prob[pred[j]]
                        if agg == 'max':
                            features[k] = max(features.get(k, 0.0), p)
                        elif agg == 'sum':
                            features[k] = features.get(k, 0.0) + p
                        else:
                            print 'Not supported'
                            exit(0)

                    x[data_name][b] = features

    return x, y


def save_data(data_dict, data_type, n, agg, model_name):
    file_name = '%s_top%d_%s_%s.obj' % (data_type, n, agg, model_name)
    cPickle.dump(data_dict, open(os.path.join(consts.STORAGE_DATA_PATH, file_name), 'wb'))


if __name__ == '__main__':
    lbp = data_layer.LBPDict(consts.P2B_TRAIN, consts.P2B_TEST, consts.LABELS_TRAIN)
    all_train_photos = lbp.get_all_train_photo_ids()
    all_test_photos = lbp.get_all_test_photo_ids()

    # Get words and serialize objects with features and labels

    # Train
    x_train, y_train = img2words(all_train_photos, consts.PHOTOS_TRAIN, True)
    for n in top_n:
        for agg in aggregation:
            data_name = '%d_%s' % (n, agg)
            save_data(x_train[data_name], 'x_train', n, agg, model_name)
            save_data(y_train, 'y_train', n, agg, model_name)

    # Test
    x_test, _ = img2words(all_test_photos, consts.PHOTOS_TEST, False)
    for n in top_n:
        for agg in aggregation:
            data_name = '%d_%s' % (n, agg)
            save_data(x_test[data_name], 'x_test', n, agg, model_name)
