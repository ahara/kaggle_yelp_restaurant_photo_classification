import cPickle
import numpy as np
import os
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import cross_val_predict
from sklearn.metrics import f1_score
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.pipeline import Pipeline

import consts
import utils


def load_data(suffix):
    x = cPickle.load(open(os.path.join(consts.STORAGE_DATA_PATH, 'x_train_%s.obj' % suffix), 'rb'))
    y = cPickle.load(open(os.path.join(consts.STORAGE_DATA_PATH, 'y_train_%s.obj' % suffix), 'rb'))
    x_test = None #cPickle.load(open(os.path.join(consts.STORAGE_DATA_PATH, 'x_test_%s.obj' % suffix), 'rb'))

    return x, y, x_test


classifiers = {
    'knn33': KNeighborsClassifier(n_neighbors=33, weights='distance'),  # 5-fold CV: 0.7552 / 33
    'knn21': KNeighborsClassifier(n_neighbors=21, weights='distance'),  # 5-fold CV: 0.7506 / 21
    'knn15': KNeighborsClassifier(n_neighbors=15, weights='distance'),
    'knn9': KNeighborsClassifier(n_neighbors=9, weights='distance'),
    'mnb': OneVsRestClassifier(MultinomialNB()),  # 0.7537
    'ada200': OneVsRestClassifier(AdaBoostClassifier(n_estimators=200, random_state=0)),  # 5-fold CV: 0.7629
    'rf600': RandomForestClassifier(n_estimators=600, bootstrap=False, min_samples_leaf=3,
                                    random_state=0, n_jobs=4),  # 5-fold CV: 0.7720
    'ext': ExtraTreesClassifier(n_estimators=600, bootstrap=False, min_samples_leaf=3,
                                random_state=10, n_jobs=4),
    #'svc': OneVsRestClassifier(SVC(C=.06, kernel='linear', random_state=0, probability=True)),  # 5-fold CV: 0.7666
    #'svc2': OneVsRestClassifier(SVC(C=.2, kernel='linear', random_state=1, probability=True)),
    #'svc3': OneVsRestClassifier(SVC(C=.5, kernel='linear', random_state=2, probability=True)),
    #'svc4': OneVsRestClassifier(SVC(C=.03, kernel='linear', random_state=3, probability=True)),
    'svc5': OneVsRestClassifier(SVC(C=.01, kernel='linear', random_state=4, probability=True)),
    #'svc6': OneVsRestClassifier(SVC(C=.005, kernel='linear', random_state=4, probability=True)),
    'svc_rbf': OneVsRestClassifier(SVC(C=150., kernel='rbf', random_state=0, probability=True)),
    'baglr': OneVsRestClassifier(BaggingClassifier(
        base_estimator=LogisticRegression(C=1, penalty='l1', fit_intercept=True, random_state=23),
        n_estimators=40, max_samples=1., max_features=1.,
        bootstrap=True, random_state=33)),  # 5-fold CV: 0.7742 (top20)
    'gbm': OneVsRestClassifier(
        GradientBoostingClassifier(n_estimators=125, learning_rate=0.1, random_state=1))  # 5-fold CV: 7889 (top20)
}
#classifiers = {'knn9': classifiers['knn9'],
#               'knn21': classifiers['knn21']}

np.random.seed(1410)
predicted = None
train_preds = None
test_preds = None
ym = None
idt = None

use_test = False
print_params = False

suffixes = [('top10_max_ibn',), ('top20_sum_ibn',), ('top10_sum_ibn',), ('top10_sum_iv3',),
            ('weights_sum_ibn', 'weights_max_ibn', 'weights_mean_ibn'),
            ('weights_sum_ibn',), ('weights_max_ibn',), ('weights_mean_ibn',)]

suffixes = [('weights_sum_iv3', 'weights_max_iv3', 'weights_mean_iv3'),
            ('weights_sum_iv3',), ('weights_max_iv3',), ('weights_mean_iv3',),
            ('weights_sum_ibn', 'weights_max_ibn', 'weights_mean_ibn'),
            ('weights_sum_ibn',), ('weights_max_ibn',), ('weights_mean_ibn',),
            ('top10_max_ibn',), ('top20_sum_ibn',), ('top10_sum_ibn',), ('top10_sum_iv3',)]
suffixes = [('top10_mean_ibn',), ('top20_mean_ibn',)]

xm_trans, xt_trans = None, None
for data in suffixes:

    # Load data
    for i, suffix in enumerate(data):
        x, y, x_test = load_data(suffix)

        # Extract features and labels from dictionaries in the same order
        xm, ym = [], []
        for k in y.keys():
            xm.append(x[k])
            ym.append(y[k])

        idt, xt = zip(*x_test.items()) if use_test else (None, None)

        # Encode features (dict to sparse matrix) or standardize weights
        if 'weights' in suffix:
            if len(data) > 1:
                pp = Normalizer()
                xm = pp.fit_transform(xm)
                xt = pp.transform(xt) if use_test else xt
            else:
                xm = np.array(xm)
#                xm = np.log1p(np.array(xm))
                xt = np.log1p(np.array(xt)) if use_test else xt
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

    for clf_name, clf in classifiers.items():
        try:
            #p = cross_val_predict(clf, xm, ym, 5, n_jobs=5)
            p = utils.cross_val_proba(clf, xm, ym, 5, hash(clf_name + str(data)))
            # Print partial results
            partial_result = f1_score(ym, np.array(p >= 0.5, dtype='l'), average='samples')
            print data, clf_name, partial_result
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
            if use_test:
                test_preds = pt if test_preds is None else np.hstack((test_preds, pt))
        except Exception as e:
            print 'Cannot train', clf_name

scores = map(lambda j: f1_score(ym, np.array(predicted >= j, dtype='l'), average='samples'),
             np.arange(1, len(classifiers) * len(suffixes), 0.2))

print 'Best score', max(scores)

meta_train = None
meta_test = None
for name, meta in classifiers.items():
    if name == 'mnb':
        continue
    p = utils.cross_val_proba(meta, train_preds, ym, 5, hash(name + str(meta)))
    print 'Meta', name, f1_score(ym, np.array(p >= 0.5, dtype='l'), average='samples')
    meta_train = p if meta_train is None else np.hstack((meta_train, p))
    # Test set
    if use_test:
        meta.fit(train_preds, ym)
        pt = meta.predict_proba(test_preds)
        pt = np.transpose(np.array(pt)[:,:,1]) if type(pt) == type([]) else pt
        meta_test = pt if meta_test is None else np.hstack((meta_test, pt))

#meta = classifiers['svc']
#print 'Meta', f1_score(ym, cross_val_predict(meta, meta_train, ym, 5, n_jobs=5), average='samples')

#if use_test:
#    meta.fit(meta_train, ym)
#    pt = meta.predict(meta_test)
#    utils.save_predictions(idt, pt)

meta = classifiers['svc6']
print 'Meta', f1_score(ym, cross_val_predict(meta, train_preds[:, 9:], ym, 5, n_jobs=5), average='samples')
if use_test:
    meta.fit(train_preds[:, 9:], ym)
    pt = meta.predict(test_preds[:, 9:])
    utils.save_predictions(idt, pt)
