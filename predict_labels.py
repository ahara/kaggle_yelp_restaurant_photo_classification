import cPickle
import logging
import mxnet as mx
import numpy as np
import os
from skimage import io, transform

import consts
import yelp_input


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Load the pre-trained model
prefix = "Inception/Inception_BN"
num_round = 39
model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.gpu(), numpy_batch_size=1)

# load mean image
mean_img = mx.nd.load("Inception/mean_224.nd")["mean_img"]

# if you like, you can plot the network
# mx.viz.plot_network(model.symbol, shape={"data" : (1, 3, 224, 224)})

# load synset (text label)
synset = [l.strip() for l in open('Inception/synset.txt').readlines()]


def PreprocessImage(path, show_img=False):
    # load image
    img = io.imread(path)
    #print("Original Image Shape: ", img.shape)
    # we crop image from center
    short_egde = min(img.shape[:2])
    yy = int((img.shape[0] - short_egde) / 2)
    xx = int((img.shape[1] - short_egde) / 2)
    crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
    # resize to 224, 224
    resized_img = transform.resize(crop_img, (224, 224))
    if show_img:
        io.imshow(resized_img)
    # convert to numpy.ndarray
    sample = np.asarray(resized_img) * 256
    # swap axes to make image from (224, 224, 4) to (3, 224, 224)
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)
    # sub mean
    normed_img = sample - mean_img.asnumpy()
    normed_img.resize(1, 3, 224, 224)
    return normed_img

x = {}
y = {}

lbp = yelp_input.LBPDict()
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
    batch = PreprocessImage(p, False)
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