import cPickle
import numpy as np
import os
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import cross_val_predict
from sklearn.metrics import f1_score
from sklearn.preprocessing import Normalizer, StandardScaler

import consts
import utils
from multilabel import ClassifierChain


def load_data(suffix):
    x = cPickle.load(open(os.path.join(consts.STORAGE_DATA_PATH, 'x_train_%s.obj' % suffix), 'rb'))
    y = cPickle.load(open(os.path.join(consts.STORAGE_DATA_PATH, 'y_train_%s.obj' % suffix), 'rb'))
    x_test = cPickle.load(open(os.path.join(consts.STORAGE_DATA_PATH, 'x_test_%s.obj' % suffix), 'rb'))

    return x, y, x_test


classifiers = {
    #'rf600': RandomForestClassifier(n_estimators=600, bootstrap=False, min_samples_leaf=3,
    #                                random_state=0, n_jobs=7),
    #'svc': OneVsRestClassifier(SVC(C=.06, kernel='linear', random_state=0, probability=True), n_jobs=9),
    #'svc2': OneVsRestClassifier(SVC(C=.2, kernel='linear', random_state=1, probability=True)),
    #'svc3': OneVsRestClassifier(SVC(C=.5, kernel='linear', random_state=2, probability=True)),
    #'svc4': OneVsRestClassifier(SVC(C=.03, kernel='linear', random_state=3, probability=True)),
    #'svc5': OneVsRestClassifier(SVC(C=.01, kernel='linear', random_state=4, probability=True)),
    #'svc6': OneVsRestClassifier(SVC(C=.005, kernel='linear', random_state=4, probability=True), n_jobs=9),
    'svc6chain1': ClassifierChain(SVC(C=.005, kernel='linear', random_state=4, probability=True)),
    'svc6chain2': ClassifierChain(SVC(C=.005, kernel='linear', random_state=4, probability=True)),
    'svc6chain3': ClassifierChain(SVC(C=.005, kernel='linear', random_state=4, probability=True)),
    'svc6chain4': ClassifierChain(SVC(C=.005, kernel='linear', random_state=4, probability=True)),
    'svc6chain5': ClassifierChain(SVC(C=.005, kernel='linear', random_state=4, probability=True)),
    'svc6chain6': ClassifierChain(SVC(C=.005, kernel='linear', random_state=4, probability=True)),
    'svc6chain7': ClassifierChain(SVC(C=.005, kernel='linear', random_state=4, probability=True)),
    #'svc_rbf': OneVsRestClassifier(SVC(C=150., kernel='rbf', random_state=0, probability=True)),
#    'baglr': OneVsRestClassifier(BaggingClassifier(
#        base_estimator=LogisticRegression(C=1, penalty='l1', fit_intercept=True, random_state=23),
#        n_estimators=40, max_samples=1., max_features=1.,
#        bootstrap=True, random_state=33), n_jobs=9),
#    'gbm': OneVsRestClassifier(
#        GradientBoostingClassifier(n_estimators=125, learning_rate=0.1, random_state=1), n_jobs=9)
}

np.random.seed(1410)
predicted = None
train_preds = None
test_preds = None
train_preds = cPickle.load(open(os.path.join(consts.STORAGE_PARTIAL_PATH, '20160407', 'train_preds_24.obj'), 'rb'))
test_preds = cPickle.load(open(os.path.join(consts.STORAGE_PARTIAL_PATH, '20160407', 'test_preds_24.obj'), 'rb'))
ym = None
idt = None

use_test = True
print_params = False

suffixes = [#('weights_sum_iv3', 'weights_max_iv3', 'weights_mean_iv3'),
            ('weights_sum_ibn', 'weights_max_ibn', 'weights_mean_ibn'),
            ('weights_sum_iv3',), ('weights_max_iv3',), ('weights_mean_iv3',),
            ('weights_sum_ibn',), ('weights_max_ibn',), ('weights_mean_ibn',)]


xm_trans, xt_trans = None, None
iter_cntr = 25
for data in suffixes:
    for transformation in ['log']:
        # Load data
        for i, suffix in enumerate(data):
            x, y, x_test = load_data(suffix)

            # Extract features and labels from dictionaries in the same order
            xm, ym = [], []
            for k in y.keys():
                xm.append(x[k])
                ym.append(y[k])

            if use_test:
                idt, xt = zip(*x_test.items())

            # Encode features (dict to sparse matrix) or standardize weights
            if 'weights' in suffix:
                if len(data) > 1:
                    pp = Normalizer()
                    xm = pp.fit_transform(xm)
                    xt = pp.transform(xt) if use_test else xt
            else:
                pp = DictVectorizer(sparse=False)
                xm = pp.fit_transform(xm)
                xt = pp.transform(xt) if use_test else xt

            if xm_trans is None:
                xm_trans = xm
                xt_trans = xt
            else:
                xm_trans = np.hstack((xm_trans, xm))
                xt_trans = np.hstack((xt_trans, xt))

        ym = np.array(ym)
        xm = xm_trans
        xt = xt_trans

        xm = np.array(xm)
        xt = np.array(xt) if use_test else xt
        if transformation == 'log':
            xm = np.log1p(xm)
            xt = np.log1p(xt) if use_test else xt

        for clf_name, clf in classifiers.items():
            try:
                #p = cross_val_predict(clf, xm, ym, 5, n_jobs=5)
                p = utils.cross_val_proba(clf, xm, ym, 5, hash(clf_name + str(data)))
                # Print partial results
                partial_result = f1_score(ym, np.array(p >= 0.5, dtype='l'), average='samples')
                print data, transformation, clf_name, partial_result
                if partial_result < 0.8:
                    continue
                if use_test:
                    clf.fit(xm, ym)
                    pt = clf.predict_proba(xt)
                    pt = np.transpose(np.array(pt)[:,:,1]) if type(pt) == type([]) else pt
                if print_params:
                    print clf.get_params()
                # Aggregate predictions
                predicted = p if predicted is None else np.array(predicted, dtype='float') + p
                train_preds = p if train_preds is None else np.hstack((train_preds, p))
                cPickle.dump(train_preds, open(os.path.join(consts.STORAGE_PARTIAL_PATH, 'train_preds_%d.obj' % iter_cntr), 'wb'))
                if use_test:
                    test_preds = pt if test_preds is None else np.hstack((test_preds, pt))
                    cPickle.dump(test_preds, open(os.path.join(consts.STORAGE_PARTIAL_PATH, 'test_preds_%d.obj' % iter_cntr), 'wb'))
            except Exception as e:
                print e
                print 'Cannot train', clf_name
            iter_cntr += 1

scores = map(lambda j: f1_score(ym, np.array(predicted >= j, dtype='l'), average='samples'),
             np.arange(1, len(classifiers) * len(suffixes) * 2, 0.2))

print 'Best score', max(scores)

meta_train = None
meta_test = None
for name, meta in classifiers.items():
    p = utils.cross_val_proba(meta, train_preds, ym, 5, hash(name + str(meta)))
    print 'Meta', name, f1_score(ym, np.array(p >= 0.5, dtype='l'), average='samples')
    meta_train = p if meta_train is None else np.hstack((meta_train, p))
    # Test set
    meta.fit(train_preds, ym)
    pt = meta.predict_proba(test_preds)
    pt = np.transpose(np.array(pt)[:,:,1]) if type(pt) == type([]) else pt
    meta_test = pt if meta_test is None else np.hstack((meta_test, pt))

meta = classifiers['svc6']
print 'Meta', f1_score(ym, cross_val_predict(meta, meta_train, ym, 5, n_jobs=5), average='samples')

if use_test:
    meta.fit(meta_train, ym)
    pt = meta.predict(meta_test)
    utils.save_predictions(idt, pt)
