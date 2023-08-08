import pandas as pd
import plot
import numpy as np
from nats_bench import create

OUT_DIR = "./out/"
NATS_BENCH_TSS_PATH = "nats_bench_data/NATS-tss-v1_0-3ffb9-simple"
seeds = ['1', '10', '100', '1000', '10000']


api = create(NATS_BENCH_TSS_PATH, 'tss', fast_mode=True, verbose=False)
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

def get_training_time(df, dataset):
    ids = df['id'].tolist()
    print(api.get_more_info(ids[0], 'cifar10'))
    print(api.get_cost_info(ids[0], 'cifar10-valid'))
    input()
    if dataset == 'cifar10':
        l = [api.get_more_info(id, 'cifar10-valid')["train-all-time"] for id in ids]
    else:
        l = [api.get_more_info(id, dataset)["train-all-time"] for id in ids]

    return l
def get_str_metric(metrics):
    return f"{metrics[0]*100:.2f}+-{metrics[1]*100:.2f}"

def get_metrics(df, col):
    return df[col].mean(), df[col].std()


def generate_df_metrics(ss, fn, dataset):
    nasnet_l = []

    dfs = [get_df(ss, fn, dataset, None, seed) for seed in seeds]
    nasnet_df = pd.concat(dfs, axis=0, ignore_index=True).sort_values(by='acc_valid', ascending=False)
    nasnet_l.append(get_str_metric(get_metrics(nasnet_df, 'acc_valid')))

    nasnet_25 = nasnet_df.iloc[:int(len(nasnet_df) * 0.25)]
    nasnet_10 = nasnet_df.iloc[:int(len(nasnet_df) * 0.1)]
    nasnet_5 = nasnet_df.iloc[:int(len(nasnet_df) * 0.05)]
    nasnet_1 = nasnet_df['acc_valid'].max()

    nasnet_l.append(get_str_metric(get_metrics(nasnet_25, 'acc_valid')))
    nasnet_l.append(get_str_metric(get_metrics(nasnet_10, 'acc_valid')))
    nasnet_l.append(get_str_metric(get_metrics(nasnet_5, 'acc_valid')))
    nasnet_l.append(f"{nasnet_1*100:.2f}")
    #mean of access time
    nasnet_l.append(f"{get_metrics(nasnet_df, 'time')[0]:.4f}")

    t_time = np.array(get_training_time(nasnet_df, dataset))
    avg_time, std_time = np.mean(t_time), np.std(t_time)
    print(avg_time,std_time)
    nasnet_l.append(f"{avg_time:.2f}+-{std_time:.2f}")

    t_time = np.array(get_training_time(nasnet_5, dataset))
    avg_time, std_time = np.mean(t_time), np.std(t_time)
    nasnet_l.append(f"{avg_time:.2f}+-{std_time:.2f}")

    print(nasnet_df)
    print(nasnet_l)

    return nasnet_l

def generate_acc_table(column, col_name):
    dict ={}
    rows_name = ["Acc(%), Top25(%), Top10(%), Top5(%), Best(%), QTime(s), TrainTime(min), TrainTimeTop5(min)"]
    for i in range(len(column)):
        dict[col_name[i]] = column[i]

    df = pd.DataFrame(dict)
    print(df)

def generate_cost_table(columns, col_name):
    dict = {}
    rows_name = ["FLOPS(M)", "FLOPS5(M)", "FLOPSbest(M)",  "#Param", "Latency"]
    for i in range(len(column)):
        dict[col_name[i]] = column[i]

    df = pd.DataFrame(dict)
    print(df)



if __name__ == '__main__':
    df = get_df('nasbench', 'reinforce', 'cifar10', None, '10')
    best_valid = get_best(df, 'acc_valid')
    print(best_valid)
    val = get_metrics(df, 'acc_valid')
    print(val)
    nasnet_nasbench = generate_df_metrics('nasbench', 'reinforce', 'cifar10')
    random_nasbench = generate_df_metrics('nasbench', 'random', 'cifar10')

    generate_acc_table([nasnet_nasbench, random_nasbench], ["NASNET", "Random Search"])
