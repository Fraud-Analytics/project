import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv('vars_308.csv')
    sample = df[0:10000].copy()
    sample.to_csv('sample.csv', index=False)
