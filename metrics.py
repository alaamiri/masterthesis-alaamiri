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



if __name__ == '__main__':
    path = get_path('nasbench', 'reinforce', 'cifar10', None, '10')
    df = csv_to_df(path)
    
    print(df)
    