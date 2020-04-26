import threading

import numpy as np
from joblib import Parallel
from joblib import delayed
from joblib import effective_n_jobs
from sklearn.base import ClassifierMixin
from sklearn.base import clone

__all__ = ['MakeEnsemble']


def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    n_jobs = min(effective_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = np.full(n_jobs, n_estimators // n_jobs,
                                   dtype=np.int)
    n_estimators_per_job[:n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()


def _fit_with_resample(clf, resamplers, X, y, clf_idx, n_clfs, verbose=0):
    if verbose > 0:
        print(f"Fitting classifier {clf_idx} of {n_clfs}")
    x_resample, y_resample = np.array(X), np.array(y)
    if verbose > 0:
        print(x_resample.shape, y_resample.shape)
    for resampler in resamplers:
        x_resample, y_resample = resampler.fit_resample(x_resample, y_resample)
    clf.fit(x_resample, y_resample)
    return clf


def _accumulate_prediction(predict, X, out, lock):
    prediction = predict(X)
    with lock:
        if len(out) == 1:
            out[0] += prediction
        else:
            for i in range(len(out)):
                out[i] += prediction[i]


class MakeEnsemble(ClassifierMixin):
    def __init__(self, base_classifier, resamplers, n_estimators=10, n_jobs=1, verbose=0):
        self.estimators_ = []
        self.base_classifier = base_classifier
        if not isinstance(resamplers, list):
            resamplers = [resamplers]
        self.resamplers = resamplers
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.n_classes_ = 2

    def fit(self, X, y):
        clfs = [clone(self.base_classifier) for _ in range(self.n_estimators)]
        clfs = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, prefer='processes')(
            delayed(_fit_with_resample)(clf, self.resamplers,
                                        X, y, i, self.n_estimators, self.verbose)
            for i, clf in enumerate(clfs)
        )
        self.estimators_ = clfs
        return self

    def predict_proba(self, X):
        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        all_proba = [np.zeros((X.shape[0], j), dtype=np.float64)
                     for j in np.atleast_1d(self.n_classes_)]
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_accumulate_prediction)(e.predict_proba, X, all_proba, lock)
            for e in self.estimators_)

        for proba in all_proba:
            proba /= len(self.estimators_)

        if len(all_proba) == 1:
            return all_proba[0]
        else:
            return all_proba
