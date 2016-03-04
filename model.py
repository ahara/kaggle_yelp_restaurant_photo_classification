import cPickle
import numpy as np
import os
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import cross_val_predict
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

import consts
import utils
import model_definition


classifiers = {
    'knn33': KNeighborsClassifier(n_neighbors=33, weights='distance'),  # 5-fold CV: 0.7552 / 33
    'knn21': KNeighborsClassifier(n_neighbors=21, weights='distance'),  # 5-fold CV: 0.7506 / 21
    'mnb': OneVsRestClassifier(MultinomialNB()),  # 0.7537
    'ada200': OneVsRestClassifier(AdaBoostClassifier(n_estimators=200, random_state=0)),  # 5-fold CV: 0.7629
    'rf600': RandomForestClassifier(n_estimators=600, bootstrap=False, min_samples_leaf=3,
                                    random_state=0, n_jobs=4),  # 5-fold CV: 0.7720
    'svc': OneVsRestClassifier(SVC(C=.06, kernel='linear', random_state=0, probability=True)),  # 5-fold CV: 0.7666
    'svc_rbf': OneVsRestClassifier(SVC(C=150., kernel='rbf', random_state=0, probability=True)),
    'baglr': OneVsRestClassifier(BaggingClassifier(
        base_estimator=LogisticRegression(C=1, penalty='l1', fit_intercept=True, random_state=23),
        n_estimators=40, max_samples=1., max_features=1.,
        bootstrap=True, random_state=23)),  # 5-fold CV: 0.7742 (top20)
    'gbm': OneVsRestClassifier(
        GradientBoostingClassifier(n_estimators=125, learning_rate=0.1, random_state=1))  # 5-fold CV: 7889 (top20)
}

np.random.seed(1410)
predicted = None
train_preds = None
test_preds = None
ym = None
idt = None

use_test = True
print_params = False

suffixes = ['top20_max_ibn', 'top10_max_ibn', 'top5_max_ibn',
            'top20_sum_ibn', 'top10_sum_ibn', 'top5_sum_ibn',
            'top20_max_iv3', 'top10_max_iv3', 'top5_max_iv3',
            'top20_sum_iv3', 'top10_sum_iv3', 'top5_sum_iv3']
suffixes = ['top10_max_ibn', 'top20_sum_ibn', 'top10_sum_ibn', 'top10_sum_iv3',
            'weights_max_ibn', 'weights_sum_ibn', 'weights_mean_ibn']

for suffix in suffixes:
    # Load data
    x = cPickle.load(open(os.path.join(consts.STORAGE_DATA_PATH, 'x_train_%s.obj' % suffix), 'rb'))
    y = cPickle.load(open(os.path.join(consts.STORAGE_DATA_PATH, 'y_train_%s.obj' % suffix), 'rb'))
    x_test = cPickle.load(open(os.path.join(consts.STORAGE_DATA_PATH, 'x_test_%s.obj' % suffix), 'rb'))
    print len(x_test)

    # Extract features and labels from dictionaries in the same order
    xm, ym = [], []

    for k in y.keys():
        xm.append(x[k])
        ym.append(y[k])

    if use_test:
        idt, xt = zip(*x_test.items())

    # Encode features (dict to sparse matrix) or standardize weights
    if 'weights' in suffix:
        pp = StandardScaler()
    else:
        pp = DictVectorizer()
    xm = pp.fit_transform(xm)
    if use_test:
        xt = pp.transform(xt)

    ym = np.array(ym)

    for clf_name, clf in classifiers.items():
        print 'Train', clf_name
        try:
            xm2 = xm.toarray() if clf_name in ['baglr', 'mnb', 'gbm'] and 'weights' not in suffix else xm
            #p = cross_val_predict(clf, xm2, ym, 5, n_jobs=5)
            p = utils.cross_val_proba(clf, xm2, ym, 5)
            if use_test:
                xt2 = xt.toarray() if clf_name in ['baglr', 'mnb', 'gbm'] and 'weights' not in suffix else xt
                clf.fit(xm2, ym)
                pt = clf.predict_proba(xt2)
                pt = np.transpose(np.array(pt)[:,:,1]) if type(pt) == type([]) else pt
            # Print partial results
            print suffix, clf_name, f1_score(ym, np.array(p >= 0.5, dtype='l'), average='samples')
            if print_params:
                print clf.get_params()
            # Aggregate predictions
            predicted = p if predicted is None else np.array(predicted, dtype='float') + p
            train_preds = p if train_preds is None else np.hstack((train_preds, p))
            if use_test:
                test_preds = pt if test_preds is None else np.hstack((test_preds, pt))
        except Exception as e:
            print 'Cannot train', clf_name
            print e

scores = map(lambda j: f1_score(ym, np.array(predicted >= j, dtype='l'), average='samples'),
             np.arange(1, len(classifiers) * len(suffixes) + 1, 0.2))

print 'Best score', max(scores)

# Meta-learning
meta = RandomForestClassifier(n_estimators=500, bootstrap=True, min_samples_leaf=2, max_features=300,
                              random_state=0)  # 0.8288 500 T 2 300
print 'Meta', f1_score(ym, cross_val_predict(meta, train_preds, ym, 5, n_jobs=5), average='samples')
if use_test:
    meta.fit(train_preds, ym)
    pt = meta.predict(test_preds)
    utils.save_predictions(idt, pt)