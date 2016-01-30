import cPickle
import logging
import mxnet as mx
import numpy as np
import os
from skimage import io, transform

import consts
import data_layer


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Params
top_n = 10
aggregation = ''
model_name = 'iv3'

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
    x, y = {}, {}
    cnt = 0

    for i in photo_ids:
        cnt += 1
        if cnt % 100 == 0:
            print photo_dir, cnt / float(len(photo_ids))
        b = lbp.get_business(i)

        if use_labels:
            y[b] = lbp.get_label(i)

        p = os.path.join(photo_dir, '%s.jpg' % str(i))

        features = x.get(b, {})

        # Get preprocessed batch (single image batch)
        batch = preprocess_image(p)
        # Get prediction probability of 1000 classes from model
        prob = model.predict(batch)[0]
        # Argsort, get prediction index from largest prob to lowest
        pred = np.argsort(prob)[::-1]
        # Get topN labels
        for j in xrange(top_n):
            k = synset[pred[j]]
            p = prob[pred[j]]
            if aggregation == 'max':
                features[k] = max(features.get(k, 0.0), p)
            else:
                features[k] = features.get(k, 0.0) + p

        x[b] = features

    return x, y


if __name__ == '__main__':
    x_train, y_train, x_test = {}, {}, {}

    lbp = data_layer.LBPDict(consts.P2B_TRAIN, consts.P2B_TEST, consts.LABELS_TRAIN)
    all_train_photos = lbp.get_all_train_photo_ids()
    all_test_photos = lbp.get_all_test_photo_ids()

    # Get words and serialize objects with features and labels
    suffix = 'top%d_%s_%s' % (top_n, aggregation, model_name)

    # Train
    x_train, y_train = img2words(all_train_photos, consts.PHOTOS_TRAIN, True)
    cPickle.dump(x_train, open(os.path.join(consts.STORAGE_DATA_PATH, 'x_train_%s.obj' % suffix), 'wb'))
    cPickle.dump(y_train, open(os.path.join(consts.STORAGE_DATA_PATH, 'y_train_%s.obj' % suffix), 'wb'))

    # Test
    x_test, _ = img2words(all_test_photos, consts.PHOTOS_TEST, False)
    cPickle.dump(x_test, open(os.path.join(consts.STORAGE_DATA_PATH, 'x_test_%s.obj' % suffix), 'wb'))
