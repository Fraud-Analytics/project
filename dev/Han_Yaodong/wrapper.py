import operator

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def load_data(file):
    df = pd.read_csv(file)
    std_scale = StandardScaler().fit(df)
    df_std = pd.DataFrame(std_scale.transform(df), columns=df.columns)

    with open('ks_stat.csv', 'r') as f:
        lines = f.read().split('\n')
        ks_stat = [i.split(',') for i in lines]

    y = df['fraud_label'].to_numpy()
    df_std.drop(columns=['record', 'fraud_label'], inplace=True)

    ks_col = [i[0] for i in ks_stat[:102]]
    ks_df = df_std.filter(items=ks_col)
    return ks_df, y


def fdr(y, cutoff=0.03, *, prob=None, classifier=None, x=None):
    if prob is None:
        assert classifier is not None
        prob = classifier.predict_proba(x)
    fraud_num = len(y[y == 1])
    total_num = len(y)
    fraud_prob = [(i[1], j) for i, j in zip(prob, y)]
    sorted_prob = sorted(fraud_prob, key=lambda x: x[0], reverse=True)
    cutoff_bin = sorted_prob[0:int(total_num * cutoff)]
    return len(cutoff_bin[cutoff_bin == 1]) / fraud_num


def ensemble_fdr(x, y, num_model=5):
    prob = np.zeros((len(y), 2))
    for _ in range(num_model):
        x_sample, y_sample = RandomUnderSampler().fit_resample(x, y)
        clf_ = LogisticRegression(penalty='l2', C=0.01, multi_class='ovr').fit(x_sample, y_sample)
        prob += clf_.predict_proba(x)
    prob /= num_model
    return fdr(x, y, prob=prob)


if __name__ == '__main__':
    print('loading data')
    df, y = load_data('vars_308.csv')
    # df, y = load_data('sample.csv')
    print('Done!')

    print("select features")
    best_feature = set()
    all_feature = set(df.columns)
    while len(best_feature) < 20:
        print("Current best features:")
        print(best_feature)
        # get candidate variables
        candidate = all_feature - best_feature
        scores = dict()
        for col in candidate:
            print('Trying', col)
            features = list(best_feature) + [col]
            x = df.filter(items=features).to_numpy()
            if len(features) == 1:  # only one feature, reshape needed
                x = x.reshape(-1, 1)
            s = ensemble_fdr(x, y)
            print('fdr', s)
            scores[col] = s

        # get the best feature
        best = max(scores.items(), key=operator.itemgetter(1))
        print(best)
        best_feature.add(best[0])

    with open('wrap.csv', 'w') as f:
        f.write('\n'.join(best_feature))
