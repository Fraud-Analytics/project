import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

__all__ = ['load_data', 'plot_report', 'fdr', 'fdr_prob']


def load_data(filename, test_size=0.2, oot_size=0.15):
    """Loads data from file and split into train and test data.
    Parameters
    ==========
    filename : str
    test_size : float
    """
    df = pd.read_csv(filename)
    y = df['fraud_label']
    df.drop(columns=['fraud_label'], inplace=True)
    x_train, x_oot, y_train, y_oot = train_test_split(
            df, y, test_size=oot_size, shuffle=False)
    if test_size == 0:
        x_train, y_train = shuffle(x_train, y_train)
        return x_train, None, x_oot, y_train, None, y_oot
    x_train, x_test, y_train, y_test = train_test_split(
            x_train, y_train, test_size=test_size)
    return x_train, x_test, x_oot, y_train, y_test, y_oot


def load_raw_data(filename, n_feature=25, test_size=0.2):
    """Loads data from file and split into train and test data.
    Parameters
    ==========
    filename : str
    n_feature: int
    test_size : float
    """
    df = pd.read_csv(filename)
    features = pd.read_csv('wrapper_1.csv',
                           index_col=False,
                           header=None).to_numpy()
    y = df['fraud_label']
    df.drop(columns=['fraud_label'], inplace=True)
    dr.filter(items=features[0:n_feature,0], inplcae=True)
    x_train, x_test, y_train, y_test = train_test_split(
        df, y, test_size=test_size, shuffle=False)
    x_train, y_train = shuffle(x_train, y_train)
    return x_train, x_test, y_train, y_test


def plot_report(classifier, x, y):
    """Plots a report of the given classifier, including
    FDR score, confusion matrix and ROC curve.
    """
    print(f'FDR: {fdr(classifier, x, y)}\n')
    print(classification_report(y, classifier.predict(x)))
    plot_confusion_matrix(classifier, x, y,
                          values_format='', cmap=plt.cm.Blues)
    plot_roc_curve(classifier, x, y)


def fdr(classifier, x, y, cutoff=0.03):
    """Calculates FDR score for the given classifier
    on dataset x and y with cutoff value."""
    return fdr_prob(y, classifier.predict_proba(x), cutoff)


def fdr_prob(y, y_prob, cutoff=0.03):
    if len(y_prob.shape) != 1:
        y_prob = y_prob[:, -1:]
    num_fraud = len(y[y == 1])
    sorted_prob = np.asarray(sorted(zip(y_prob, y), key=lambda x: x[0], reverse=True))
    cutoff_bin = sorted_prob[0:int(len(y) * cutoff), 1:]
    return len(cutoff_bin[cutoff_bin == 1]) / num_fraud

