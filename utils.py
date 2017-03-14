import cPickle
import csv
import numpy as np
import os
from joblib import Parallel, delayed
from sklearn.cross_validation import KFold

import consts


def train_and_predict(clf, x, y, t):
    clf.fit(x, y)
    return clf.predict_proba(t)


def cross_val_proba(clf, X, y, cv, random_state=0, n_jobs=5):
    """
    Simplified equivalent of cross_val_predict from sklearn
    but instead labels it predicts probabilities
    """
    folds = KFold(y.shape[0], n_folds=cv, shuffle=True, random_state=random_state % 10**6)

    if n_jobs > 1:
        results = Parallel(n_jobs=n_jobs)(delayed(train_and_predict)
                                          (clf, X[train_index], y[train_index], X[test_index])
                                          for train_index, test_index in folds)
    else:
        results = [train_and_predict(clf, X[train_index], y[train_index], X[test_index])
                   for train_index, test_index in folds]

    preds = np.zeros(y.shape, dtype='float')
    for i, indices in enumerate(folds):
        _, test_index = indices
        p = results[i]
        preds[test_index] = np.transpose(np.array(p)[:, :, 1]) if type(p) == type([]) else p
        #preds[test_index, :] = np.transpose(np.array(p)[:, :, 1]) if type(p) == type([]) else p
    return preds


def fit_or_restore_model(clf, clf_name, data_name, x, y, save_model=True):
    """
    Try to restore already trained model from disk if classifier
    parameters and data name have not change. If proper model
    cannot be found then new classifier is trained and saved on disk
    (until save_model flag is set on True).
    """
    model_path = os.path.join(consts.STORAGE_MODELS_PATH, clf_name, data_name, 'clf.obj')
    if os.path.exists(model_path):
        loaded_clf = cPickle.load(model_path)
        # Compare arguments - convert to string to compare internal estimators
        loaded_clf_params = params_to_str(loaded_clf.get_params())
        current_clf_params = params_to_str(clf.get_params())
        if loaded_clf_params == current_clf_params:
            return loaded_clf
    clf.fit(x, y)
    if save_model:
        cPickle.dump(clf, model_path)
    return clf


def params_to_str(params):
    """
    Converts values in dictionary to strings.
    """
    return dict(map(lambda x: (x[0], str(x[1])), params.items()))


def save_predictions(businesses, predictions):
    with open(consts.SUBMISSION_PATH, 'wb') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['business_id', 'labels'])
        for i in xrange(len(businesses)):
            label_str = ' '.join(map(str, np.nonzero(predictions[i, :])[0]))
            businesses_id = businesses[i].replace('\n', '')
            writer.writerow([businesses_id, label_str])
