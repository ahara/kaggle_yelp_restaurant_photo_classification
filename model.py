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

import consts
import utils


classifiers = {
    'knn33': KNeighborsClassifier(n_neighbors=33, weights='distance', random_state=0),  # 5-fold CV: 0.7552 / 33
    'knn21': KNeighborsClassifier(n_neighbors=21, weights='distance', random_state=0),  # 5-fold CV: 0.7506 / 21
    #'mnb': OneVsRestClassifier(MultinomialNB()),  # 0.7537
    'ada200': OneVsRestClassifier(AdaBoostClassifier(n_estimators=200, random_state=0)),  # 5-fold CV: 0.7629
    'rf600': RandomForestClassifier(n_estimators=600, bootstrap=False, min_samples_leaf=3,
                                    random_state=0, n_jobs=4),  # 5-fold CV: 0.7720
    'svc': OneVsRestClassifier(SVC(C=.06, kernel='linear', random_state=0, probability=True)),  # 5-fold CV: 0.7666
    'baglr': OneVsRestClassifier(BaggingClassifier(
        base_estimator=LogisticRegression(C=1, penalty='l1', fit_intercept=True, random_state=23),
        n_estimators=40, max_samples=1., max_features=1.,
        bootstrap=True, random_state=23)),  # 5-fold CV: 0.7742 (top20)
    'gbm': OneVsRestClassifier(
    GradientBoostingClassifier(n_estimators=125, learning_rate=0.1, random_state=1))  # 5-fold CV: 7989 (top20)
}
# 0.8022 lables
# 0.8025 proba
#classifiers = {}
np.random.seed(1410)
predicted = None

suffixes = ['_top20', '_top10']

for suffix in suffixes:
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

    for clf_name, clf in classifiers.items():
        print 'Train', clf_name
        xm2 = xm.toarray() if clf_name in ['baglr', 'mnb', 'gbm'] else xm
        if predicted is None:
            #predicted = cross_val_predict(clf, xm2, ym, 5, n_jobs=5)
            predicted = utils.cross_val_proba(clf, xm2, ym, 5)
        else:
            #predicted = np.array(predicted, dtype='float') + cross_val_predict(clf, xm2, ym, 5, n_jobs=5)
            predicted = np.array(predicted, dtype='float') + utils.cross_val_proba(clf, xm2, ym, 5)


scores = map(lambda j: f1_score(ym, np.array(predicted >= j, dtype='l'), average='samples'),
             np.arange(1, len(classifiers) * len(suffixes) + 1, 0.2))

print 'Best score', max(scores)