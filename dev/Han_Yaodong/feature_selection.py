import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    print('Reading data')
    df = pd.read_csv('vars_308.csv')

    # z-scaling
    # TODO: is this necessary?
    std_scale = StandardScaler().fit(df)
    df_std = pd.DataFrame(std_scale.transform(df), columns=df.columns)
    print(df_std.mean(axis=0))
    print(df_std.std(axis=0))

    # KS
    print('calculating KS')
    y = df_std['fraud_label']
    ks_stat = {}
    for col in df_std.columns:
        ks = ks_2samp(y, df_std[col])
        ks_stat[col] = ks
    sorted_ks_pvalue = sorted(ks_stat.items(), key=lambda x: x[1].pvalue, reverse=True)
    sorted_ks_stat = sorted(ks_stat.items(), key=lambda x: x[1].statistic)
    # store
    with open('ks_stat.csv', 'w') as f:
        f.write('\n'.join([f'{i[0]},{i[1].statistic},{i[1].pvalue}' for i in sorted_ks_stat]))

    # MI
    print('calculating MI')
    mi = {}
    for col in df_std.columns:
        mi[col] = normalized_mutual_info_score(y, df_std[col])
    sorted_mi = sorted(mi.items(), key=lambda x: x[1], reverse=True)
    with open('mi_stat.csv', 'w') as f:
        f.write('\n'.join([f'{i[0]},{i[1]}' for i in sorted_mi]))
