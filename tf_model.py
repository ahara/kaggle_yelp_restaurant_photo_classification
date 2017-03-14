import cPickle
import numpy as np
import os
from sklearn.cross_validation import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

import consts
import utils


use_test = False

# Load data
suffix = 'instances_ibn'
x = cPickle.load(open(os.path.join(consts.STORAGE_DATA_PATH, 'x_train_weights_%s.obj' % suffix), 'rb'))
y = cPickle.load(open(os.path.join(consts.STORAGE_DATA_PATH, 'y_train_weights_%s.obj' % suffix), 'rb'))
x_test = cPickle.load(open(os.path.join(consts.STORAGE_DATA_PATH, 'x_test_weights_%s.obj' % suffix), 'rb')) if use_test else None

# Extract features and labels from dictionaries in the same order
xm, ym, yk, yorig = [], [], [], []

for k in y.keys():
    for item in x[k]:
        xm.append(item)
        ym.append(y[k])
        yk.append(k)
    yorig.append(y[k])

xm = np.array(xm)
ym = np.array(ym)

x = None

clf = RandomForestClassifier(n_estimators=200, bootstrap=False, random_state=0, n_jobs=7)
#clf = OneVsRestClassifier(LogisticRegression(C=1, penalty='l1', fit_intercept=True, random_state=23), n_jobs=1)
p = utils.cross_val_proba(clf, xm, ym, 5, 0, n_jobs=1)
print f1_score(ym, np.array(p >= 0.5, dtype='l'), average='samples')


idt, xt = [], [],
if use_test:
    for k, v in x_test.items():
        for item in v:
            idt.append(k)
            xt.append(item)
    xt = np.array(xt)
    clf.fit(xm, ym)
    pt = clf.predict_proba(xt)
    pt = np.transpose(np.array(pt)[:, :, 1])

key_old = None
porig = []
b = 0
for i, key in enumerate(yk):
    if (key_old != key and key_old is not None) or i == len(yk) - 1:
        porig.append(np.hstack((np.mean(p[b:i, :], axis=0), np.amax(p[b:i, :], axis=0), np.sum(p[b:i, :], axis=0),
                                np.log(np.sum(xm[b:i, :], axis=0) + 1))))
        b = i
    key_old = key

ym2 = np.array(yorig)
xm2 = np.array(porig)
clf = OneVsRestClassifier(SVC(C=.005, kernel='linear', random_state=4, probability=True), n_jobs=9)
print 'Train SVC - CV'
p2 = utils.cross_val_proba(clf, xm2, ym2, 5, 0, n_jobs=1)
print f1_score(ym2, np.array(p2 >= 0.5, dtype='l'), average='samples')

key_old = None
pt_orig, idt_orig = [], []
b = 0
for i, key in enumerate(idt):
    if (key_old != key and key_old is not None) or i == len(idt) - 1:
        #pt_orig.append(np.hstack((np.mean(pt[b:i, :], axis=0), np.amax(pt[b:i, :], axis=0), np.sum(pt[b:i, :], axis=0),
        #                          np.log(np.sum(xt[b:i, :], axis=0) + 1))))
        pt_orig.append(np.hstack(np.sum(pt[b:i, :], axis=0),
                                  np.log(np.sum(xt[b:i, :], axis=0) + 1)))
        idt_orig.append(key_old)
        b = i
    key_old = key

idt_orig = np.array(idt_orig)
xt2 = np.array(pt_orig)
print 'Train SVC'
clf.fit(xm2, ym2)
pt2 = clf.predict_proba(xt2)
#pt2 = np.transpose(np.array(pt2)[:, :, 1])
utils.save_predictions(idt_orig, np.array(pt2 >= 0.5, dtype='l'))
