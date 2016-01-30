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
import model_definition


classifiers = {
    'knn33': KNeighborsClassifier(n_neighbors=33, weights='distance', random_state=0),  # 5-fold CV: 0.7552 / 33
    'knn21': KNeighborsClassifier(n_neighbors=21, weights='distance', random_state=0),  # 5-fold CV: 0.7506 / 21
    'mnb': OneVsRestClassifier(MultinomialNB()),  # 0.7537
    'ada200': OneVsRestClassifier(AdaBoostClassifier(n_estimators=200, random_state=0)),  # 5-fold CV: 0.7629
    'rf600': RandomForestClassifier(n_estimators=600, bootstrap=False, min_samples_leaf=3,
                                    random_state=0, n_jobs=4),  # 5-fold CV: 0.7720
    'svc': OneVsRestClassifier(SVC(C=.06, kernel='linear', random_state=0, probability=True)),  # 5-fold CV: 0.7666
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

use_test = False
print_params = False

suffixes = ['top20', 'top10', 'top10_max', 'top10__iv3']

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

    # Encode features (dict to sparse matrix)
    dv = DictVectorizer()
    xm = dv.fit_transform(xm)
    if use_test:
        xt = dv.transform(xt)

    ym = np.array(ym)

    for clf_name, clf in classifiers.items():
        print 'Train', clf_name
        xm2 = xm.toarray() if clf_name in ['baglr', 'mnb', 'gbm'] else xm
        #p = cross_val_predict(clf, xm2, ym, 5, n_jobs=5)
        p = utils.cross_val_proba(clf, xm2, ym, 5)
        if use_test:
            xt2 = xt.toarray() if clf_name in ['baglr', 'mnb', 'gbm'] else xt
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

scores = map(lambda j: f1_score(ym, np.array(predicted >= j, dtype='l'), average='samples'),
             np.arange(1, len(classifiers) * len(suffixes) + 1, 0.2))

print 'Best score', max(scores)

# Meta-learning
meta = RandomForestClassifier(n_estimators=230, bootstrap=False, min_samples_leaf=2, random_state=0)  # 0.8138
p = cross_val_predict(meta, train_preds, ym, 5, n_jobs=5)
print 'Meta', f1_score(ym, p, average='samples')
if use_test:
    meta.fit(train_preds, ym)
    pt = meta.predict(test_preds)

#ap = np.concatenate(np.array(np.split(all_preds, 24, axis=1))[np.array([i for i in range(24) if i not in [16, 17, 20, 21, 22, 23]]),:,:], axis=1)
#meta = RandomForestClassifier(n_estimators=230, bootstrap=False, min_samples_leaf=2, random_state=0)  # 0.8107
#p = cross_val_predict(meta, ap, ym, 5, n_jobs=5)
#print 'Meta', f1_score(ym, p, average='samples')
