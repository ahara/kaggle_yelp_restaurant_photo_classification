import numpy as np
from sklearn.base import BaseEstimator, clone

import utils


class ClassifierChain(BaseEstimator):
    def __init__(self, estimator):
        self.base_estimator = estimator
        self.estimators = None
        self.n_labels = None
        self.order = None

    def fit(self, X, y):
        self.n_labels = y.shape[1]
        self.estimators = [None] * self.n_labels
        self.order = np.random.permutation(self.n_labels)
        self._fit_or_predict(X, y, True)

        return self

    def predict(self, X):
        return self._fit_or_predict(X, None, False)

    def predict_proba(self, X):
        return self._fit_or_predict(X, None, True)

    def _fit_or_predict(self, X, y, proba=False):
        data = np.copy(X)
        clf_order = self.order
        for i in clf_order:
            p = None
            if y is not None:
                clf = clone(self.base_estimator)
                clf.fit(data, y[:, i])
                p = utils.cross_val_proba(clf, data, y[:, i], 5)
                self.estimators[i] = clf
            else:
                clf = self.estimators[i]
                if proba:
                    p = clf.predict_proba(data)[:, 1]
                else:
                    p = clf.predict(data)
            data = np.hstack((data, p.reshape(p.shape[0], 1)))

        labels = data[:, -len(clf_order):]

        return labels[:, np.argsort(clf_order)]