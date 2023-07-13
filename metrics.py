import pandas as pd
import numpy as np

OUT_DIR = "./out/"

def get_path(s_space, fn, dataset, predictor, seed):
    if predictor is None:
        path = f"{OUT_DIR}{fn}/{s_space}/{dataset}"
    else:
        path = f"{OUT_DIR}{fn}/{s_space}/{predictor}/{dataset}"

    models_path = f"{path}/seeds/{seed}.csv"
    
    return models_path


def csv_to_df(df_path):
    return pd.read_csv(df_path)


def get_df(s_space, fn, dataset, predictor, seed):
    path = get_path(s_space, fn, dataset, predictor, seed)
    
    return csv_to_df(path)


def get_col(df, col):
    return df[col]


def get_best(df, col):
    best_acc = df[df[col] == df[col].max()]
    
    return best_acc


def get_metrics(df, col):
    return df[col].mean(), df[col].std()


if __name__ == '__main__':
    df = get_df('nasbench', 'reinforce', 'cifar10', None, '10')
    best_valid = get_best(df, 'acc_valid')
    print(best_valid)
    time_mean, time_std = get_metrics(df, 'time')
    print(time_mean, time_std)
    