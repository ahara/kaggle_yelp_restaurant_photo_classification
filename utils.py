import numpy as np
from sklearn.cross_validation import StratifiedKFold, KFold


def cross_val_proba(clf, X, y, cv, random_state=0):
    """
    Simplified equivalent of cross_val_predict from sklearn
    but instead labels it predicts probabilities
    """
    folds = KFold(y.shape[0], n_folds=cv, shuffle=False, random_state=random_state)
    preds = np.zeros(y.shape, dtype='float')
    for train_index, test_index in folds:
        clf.fit(X[train_index], y[train_index])
        p = clf.predict_proba(X[test_index])
        preds[test_index, :] = np.transpose(np.array(p)[:,:,1]) if type(p) == type([]) else p
    return preds