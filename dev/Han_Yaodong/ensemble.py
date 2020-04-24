from copy import copy

import numpy as np
from sklearn.base import ClassifierMixin

__all__ = ['MakeEnsemble']


class MakeEnsemble(ClassifierMixin):
    def __init__(self, base_classifier, resamplers, n_estimators=10):
        self.clfs = [copy(base_classifier) for _ in range(n_estimators)]
        self.base_classifier = base_classifier
        self.n_estimators = n_estimators
        if not isinstance(resamplers, list):
            resamplers = [resamplers]
        self.resamplers = resamplers

    def fit(self, X, y):
        for clf in self.clfs:
            x_resample, y_resample = X, y
            for resampler in self.resamplers:
                x_resample, y_resample = resampler.fit_resample(x_resample, y_resample)
            clf.fit(x_resample, y_resample)
        return self

    def predict_proba(self, X):
        return np.average([c.predict_proba(X) for c in self.clfs], axis=0)
