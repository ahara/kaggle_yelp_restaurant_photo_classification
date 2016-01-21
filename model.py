import cPickle
import numpy as np
import os
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import cross_val_predict
from sklearn.metrics import f1_score

import consts


np.random.seed(1410)
predicted = None

for suffix in ['', '_top10']:
    # Load data
    x = cPickle.load(open(os.path.join(consts.STORAGE_PATH, 'x%s.obj' % suffix), 'rb'))
    y = cPickle.load(open(os.path.join(consts.STORAGE_PATH, 'y%s.obj' % suffix), 'rb'))

    # Extract features and labels from dictionaries in the same order
    xm, ym = [], []

    for k in y.keys():
        xm.append(x[k])
        ym.append(y[k])

    # Encode features (dict to sparse matrix)
    dv = DictVectorizer()
    xm = dv.fit_transform(xm)

    ym = np.array(ym)

    # linear_clf = LogisticRegression(C=0.1, penalty='l1', fit_intercept=True, random_state=23)
    # clf = OneVsRestClassifier(BaggingClassifier(base_estimator=linear_clf, n_estimators=40,
    #                                             max_samples=1., max_features=1., bootstrap=True,
    #                                             random_state=23))
    # predicted = cross_val_predict(clf, xm.toarray(), ym, 5, n_jobs=5)
    # print f1_score(ym, predicted, average='samples')
    # exit(0)

    clf = KNeighborsClassifier(n_neighbors=33, weights='distance', random_state=0)  # 5-fold CV: 0.7552 / 33
    if predicted is None:
        predicted = cross_val_predict(clf, xm, ym, 5, n_jobs=5)
    else:
        predicted += cross_val_predict(clf, xm, ym, 5, n_jobs=5)
    clf = KNeighborsClassifier(n_neighbors=21, weights='distance', random_state=0)  # 5-fold CV: 0.7506 / 31
    predicted += cross_val_predict(clf, xm, ym, 5, n_jobs=5)
    # KNN(33) + KNN(21) 0.7575
    clf = OneVsRestClassifier(AdaBoostClassifier(n_estimators=200, random_state=0))  # 5-fold CV: 0.7629
    predicted += cross_val_predict(clf, xm, ym, 5, n_jobs=5)
    # 0.7723
    clf = RandomForestClassifier(n_estimators=600, bootstrap=False, min_samples_leaf=3, random_state=0, n_jobs=4)  # 5-fold CV: 0.7720
    predicted = np.array(predicted, dtype='float') + cross_val_predict(clf, xm, ym, 5, n_jobs=5)
    # 0.7819
    clf = OneVsRestClassifier(SVC(C=.06, kernel='linear', random_state=0))  # 5-fold CV: 0.7666
    predicted += cross_val_predict(clf, xm, ym, 5, n_jobs=5)
    # 0.7879 top10
    # 0.7887 top20
    linear_clf = LogisticRegression(C=1, penalty='l1', fit_intercept=True, random_state=23)

    clf = OneVsRestClassifier(BaggingClassifier(base_estimator=linear_clf, n_estimators=40,
                                                max_samples=1., max_features=1., bootstrap=True,
                                                random_state=23))  # 5-fold CV: 0.7742 (top20)
    predicted += cross_val_predict(clf, xm.toarray(), ym, 5, n_jobs=5)
    # 0.7964 top20
    # 0.7980 top20 + top10

for i in range(1, 15):
    voted = np.array(predicted >= i, dtype='l')
    print f1_score(ym, voted, average='samples')