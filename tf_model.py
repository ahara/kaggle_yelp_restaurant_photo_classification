import cPickle
import numpy as np
import os
import tensorflow as tf
from sklearn.cross_validation import KFold
from sklearn.metrics import f1_score

import consts


def soft_f1_loss(preds, labels):
    tp = tf.reduce_sum(tf.cast(preds * labels, tf.float32))
    fp = tf.reduce_sum(tf.cast(tf.maximum(preds - labels, 0), tf.float32))
    fn = tf.reduce_sum(tf.cast(tf.maximum(labels - preds, 0), tf.float32))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_op = 2. * (precision * recall) / tf.maximum(precision + recall, 1.)
    return 1 - f1_op

# Load data
for suffix in ['mean_ibn']:
    x = cPickle.load(open(os.path.join(consts.STORAGE_DATA_PATH, 'x_train_weights_%s.obj' % suffix), 'rb'))
    y = cPickle.load(open(os.path.join(consts.STORAGE_DATA_PATH, 'y_train_weights_%s.obj' % suffix), 'rb'))
    x_test = cPickle.load(open(os.path.join(consts.STORAGE_DATA_PATH, 'x_test_weights_%s.obj' % suffix), 'rb'))
    print len(x_test)

    # Extract features and labels from dictionaries in the same order
    xm, ym = [], []

    for k in y.keys():
        xm.append(x[k])
        ym.append(y[k])

    xm = np.array(xm)
    ym = np.array(ym)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cross_validation import cross_val_predict
    from sklearn.svm import SVC, LinearSVC
    from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
    from sklearn.preprocessing import StandardScaler
    print 'Standardization'
    pp = StandardScaler()
    xm = pp.fit_transform(xm)
    #clf = RandomForestClassifier(n_estimators=600, bootstrap=False, min_samples_leaf=3, random_state=0, n_jobs=4)
    clf = OneVsRestClassifier(SVC(C=150., kernel='rbf', random_state=0))
    #clf = OneVsOneClassifier(LinearSVC(random_state=0))
    p = cross_val_predict(clf, xm, ym, 5, n_jobs=5)
    print f1_score(ym, p, average='samples')
    exit(0)

    folds = KFold(ym.shape[0], n_folds=5, shuffle=True, random_state=0)
    preds = np.zeros(ym.shape, dtype='float')
    for train_index, test_index in folds:
        xtr, ytr = xm[train_index], ym[train_index]

        tf_x = tf.placeholder(tf.float32, [None, 1024])
        # Layer 1
        W_1 = tf.Variable(tf.zeros([1024, 1000]))
        b_1 = tf.Variable(tf.zeros([1000]))
        o_1 = tf.nn.relu_layer(tf_x, W_1, b_1)
        # Layer 2
        W = tf.Variable(tf.zeros([1000, 9]))
        b = tf.Variable(tf.zeros([9]))
        tf_y = tf.nn.sigmoid(tf.nn.xw_plus_b(o_1, W, b))
        tf_y_ = tf.placeholder(tf.float32, [None, 9])
        #cross_entropy = -tf.reduce_sum(tf_y_*tf.log(tf_y))
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(soft_f1_loss(tf_y, tf_y_))
        #train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)

        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)

        for i in range(1000):
            print 'Epoch', i
            rti = np.random.permutation(train_index)
            bsize = 5
            for j in xrange(len(rti / bsize)):
                batch_range = rti[(j * bsize):((j + 1) * bsize)]
                batch_xs, batch_ys = xm[batch_range, :], ym[batch_range, :]
                sess.run(train_step, feed_dict={tf_x: batch_xs, tf_y_: batch_ys})

        preds[test_index, :] = sess.run(tf_y, feed_dict={tf_x: xm[test_index, :]})

        tmp = soft_f1_loss(tf_y, tf_y_)
        print(sess.run(tmp, feed_dict={tf_x: xm, tf_y_: ym}))

    print 'Meta', f1_score(ym, np.array(preds > 0.5, dtype=int), average='samples')




