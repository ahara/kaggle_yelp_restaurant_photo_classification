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

# Load the pre-trained model
prefix = "Inception/Inception_BN"
num_round = 39
model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.gpu(), numpy_batch_size=1)

# Load mean image
mean_img = mx.nd.load("Inception/mean_224.nd")["mean_img"]

# Load synset (text label)
synset = [l.strip() for l in open('Inception/synset.txt').readlines()]


def PreprocessImage(path):
    img = io.imread(path)
    # We crop image from center
    short_egde = min(img.shape[:2])
    yy = int((img.shape[0] - short_egde) / 2)
    xx = int((img.shape[1] - short_egde) / 2)
    crop_img = img[yy:(yy + short_egde), xx:(xx + short_egde)]
    # Resize to 224, 224
    resized_img = transform.resize(crop_img, (224, 224))
    # Convert to numpy.ndarray
    sample = np.asarray(resized_img) * 256
    # Swap axes to make image from (224, 224, 4) to (3, 224, 224)
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)
    # Sub mean
    normed_img = sample - mean_img.asnumpy()
    normed_img.resize(1, 3, 224, 224)
    return normed_img

x = {}
y = {}

lbp = data_layer.LBPDict()
atp = lbp.get_all_train_photo_ids()
cnt = 0

for i in atp:
    cnt += 1
    if cnt % 100 == 0:
        print cnt / float(len(atp))
    b = lbp.get_business(i)
    y[b] = lbp.get_label(i)

    p = os.path.join(consts.PHOTOS_TRAIN, '%d.jpg') % i

    features = x.get(b, {})

    # Get preprocessed batch (single image batch)
    batch = PreprocessImage(p)
    # Get prediction probability of 1000 classes from model
    prob = model.predict(batch)[0]
    # Argsort, get prediction index from largest prob to lowest
    pred = np.argsort(prob)[::-1]
    # Get topN label
    for j in xrange(20):
        k = synset[pred[j]]
        p = prob[pred[j]]
        features[k] = features.get(k, 0.0) + p

    x[b] = features

# Serialize objects
cPickle.dump(x, open(os.path.join(consts.STORAGE_PATH, 'x.obj'), 'wb'))
cPickle.dump(y, open(os.path.join(consts.STORAGE_PATH, 'y.obj'), 'wb'))
