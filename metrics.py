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

def get_str_metric(metrics):
    return f"{metrics[0]:.2f}+-{metrics[1]:.2f}"

def get_str_metric_acc(metrics):
    return f"{metrics[0]*100:.2f}+-{metrics[1]*100:.2f}"

def get_metrics(df, col):
    return df[col].mean(), df[col].std()


def generate_df_metrics(ss, fn, dataset, predictor=None):
    nasnet_l = []

    dfs = [get_df(ss, fn, dataset, predictor, seed) for seed in seeds]
    nasnet_df = pd.concat(dfs, axis=0, ignore_index=True).sort_values(by='acc_valid', ascending=False)
    if predictor is not None:
        nasnet_df['acc_valid'] = nasnet_df['acc_valid'] / 100
    nasnet_l.append(get_str_metric_acc(get_metrics(nasnet_df, 'acc_valid')))

    nasnet_25 = nasnet_df.iloc[:int(len(nasnet_df) * 0.25)]
    nasnet_10 = nasnet_df.iloc[:int(len(nasnet_df) * 0.1)]
    nasnet_5 = nasnet_df.iloc[:int(len(nasnet_df) * 0.05)]
    nasnet_1 = nasnet_df['acc_valid'].max()

    nasnet_l.append(get_str_metric_acc(get_metrics(nasnet_25, 'acc_valid')))
    nasnet_l.append(get_str_metric_acc(get_metrics(nasnet_10, 'acc_valid')))
    nasnet_l.append(get_str_metric_acc(get_metrics(nasnet_5, 'acc_valid')))
    nasnet_l.append(f"{nasnet_1*100:.2f}")
    #mean of access time
    nasnet_l.append(f"{get_metrics(nasnet_df, 'time')[0]:.4f}")

    t_time = get_col(nasnet_df, 'train')
    avg_time, std_time = t_time.mean(), t_time.std()
    nasnet_l.append(f"{avg_time:.2f}+-{std_time:.2f}")

    t_time = get_col(nasnet_5, 'train')
    avg_time, std_time = t_time.mean(), t_time.std()
    nasnet_l.append(f"{avg_time:.2f}+-{std_time:.2f}")


    return nasnet_l

def generate_acc_table(column, col_name):
    dict ={}
    rows_name = ["Acc(%)", "Top25(%)", "Top10(%)", "Top5(%)",
                 "Best(%)", "Query Time(s)", "Train Time(min)", "Train Time Top5(min)"]
    for i in range(len(column)):
        dict[col_name[i]] = column[i]

    df = pd.DataFrame(dict, index=rows_name)
    print(df)

def generate_df_cost(ss, fn, dataset, predictor=None):
    nasnet_l = []

    dfs = [get_df(ss, fn, dataset, predictor, seed) for seed in seeds]
    nasnet_df = pd.concat(dfs, axis=0, ignore_index=True).sort_values(by='acc_valid', ascending=False)

    nasnet_5 = nasnet_df.iloc[:int(len(nasnet_df) * 0.05)]
    nasnet_1 = get_best(nasnet_df, 'acc_valid')


    nasnet_l.append(get_str_metric(get_metrics(nasnet_df, 'flops')))
    nasnet_l.append(get_str_metric(get_metrics(nasnet_5, 'flops')))
    nasnet_l.append(f"{nasnet_1['flops'].tolist()[0]:.2f}")

    nasnet_l.append(get_str_metric(get_metrics(nasnet_df, 'params')))
    nasnet_l.append(get_str_metric(get_metrics(nasnet_5, 'params')))
    nasnet_l.append(f"{nasnet_1['params'].tolist()[0]:.2f}")

    """latency = np.array(get_metrics(nasnet_df, 'latency'))
    avg_latency, std_latency = np.mean(latency), np.std(latency)
    nasnet_l.append(f"{avg_latency:.4f}+-{std_latency:.4f}")"""

    latency = get_col(nasnet_df, 'latency')
    avg_time, std_time = latency.mean(), latency.std()
    nasnet_l.append(f"{avg_time:.4f}+-{std_time:.4f}")

    return nasnet_l


def generate_cost_table(columns, col_name):
    dict = {}
    rows_name = ["FLOPS(x10^6)", "FLOPS Top5(x10^6)", "FLOPS Best(x10^6)", "#Params(x10^6)",
                 "#Params Top5(x10^6)", "#Params Best(x10^6)", "Latency(ms)"]

    for i in range(len(columns)):
        dict[col_name[i]] = columns[i]

    df = pd.DataFrame(dict, index=rows_name)
    print(df)

def get_5_best(ss, fn, dataset, predictor=None, col = 'acc_valid'):

    dfs = [get_df(ss, fn, dataset, predictor, seed).sort_values(by=col, ascending=False) for seed in seeds]

    concatenated_first_rows = pd.concat([df.iloc[:1] for df in dfs])

    if predictor is not None:
        concatenated_first_rows['acc_valid'] = concatenated_first_rows['acc_valid'] / 100

    return concatenated_first_rows

def get_all_metrics(df):
    l = []
    """valid = get_str_metric_acc(get_metrics(df,'acc_valid'))
    train = get_str_metric(get_metrics(df,'train'))
    eval = f"{get_metrics(df,'time')[0]:.4f}+-{get_metrics(df,'time')[1]:.4f}"
    flops = get_str_metric(get_metrics(df,'flops'))
    params = get_str_metric(get_metrics(df,'params'))
    lat = f"{get_metrics(df,'latency')[0]:.4f}+-{get_metrics(df,'latency')[1]:.4f}"
"""
    """valid = get_metrics(df, 'acc_valid')[0]
    train = get_metrics(df, 'train')[0]
    eval = get_metrics(df, 'time')[0]
    flops = get_metrics(df, 'flops')[0]
    params = get_metrics(df, 'params')[0]
    lat = get_metrics(df, 'latency')[0]"""

    valid = get_only_mean(df, 'acc_valid')
    train = get_only_mean(df, 'train')
    eval = f"{get_metrics(df, 'time')[0]:.4f}"
    flops = get_only_mean(df, 'flops')
    params = get_only_mean(df, 'params')
    lat = f"{get_metrics(df, 'latency')[0]:.4f}"

    l.append(valid)
    l.append(train)
    l.append(eval)
    l.append(flops)
    l.append(params)
    l.append(lat)
    l = [str(elem) for elem in l]
    print('&','&'.join(l))

def get_only_mean(df, col):
    if col =="acc_valid":
        return f"{df[col].mean()*100:.2f}"
    else:
        return f"{df[col].mean():.2f}"

if __name__ == '__main__':
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    df = get_df('nasbench', 'reinforce', 'cifar10', None, '10')
    best_valid = get_best(df, 'acc_valid')
    print(best_valid)
    val = get_metrics(df, 'acc_valid')
    print(val)

    """nasnet_nasbench = generate_df_metrics('nasbench', 'reinforce', 'ImageNet16-120') #ImageNet16-120
    nasnet_naswot = generate_df_metrics('nasbench', 'reinforce', 'ImageNet16-120', 'naswot')
    random_nasbench = generate_df_metrics('nasbench', 'random', 'ImageNet16-120')
    generate_acc_table([nasnet_nasbench, nasnet_naswot, random_nasbench], ["NASNET", "NASNET+NASWOT", "Random Search"])

    nasnet_nasbench = generate_df_cost('nasbench', 'reinforce', 'ImageNet16-120') #ImageNet16-120
    nasnet_naswot = generate_df_cost('nasbench', 'reinforce', 'ImageNet16-120', 'naswot')
    random_nasbench = generate_df_cost('nasbench', 'random', 'ImageNet16-120')
    generate_cost_table([nasnet_nasbench, nasnet_naswot, random_nasbench], ["NASNET","NASNET+NASWOT", "Random Search"])
"""
    """nasnet_nasbench = generate_df_metrics('nasbench', 'reinforce', 'ImageNet16-120', 'naswot')
    nasnet_nasbig = generate_df_metrics('nasbig', 'reinforce', 'ImageNet16-120', 'naswot')  # ImageNet16-120
    nasnet_nasmedium = generate_df_metrics('nasmedium', 'reinforce', 'ImageNet16-120', 'naswot')
    nasnet_naslittle = generate_df_metrics('naslittle', 'reinforce', 'ImageNet16-120', 'naswot')
    generate_acc_table([nasnet_nasbench, nasnet_nasbig, nasnet_nasmedium, nasnet_naslittle],
                       ["NAS-bench", "NAS-big", "NAS-medium", "NAS-little"])

    nasnet_nasbench = generate_df_cost('nasbench', 'reinforce', 'ImageNet16-120', 'naswot')
    nasnet_nasbig = generate_df_cost('nasbig', 'reinforce', 'ImageNet16-120', 'naswot')  # ImageNet16-120
    nasnet_nasmedium = generate_df_cost('nasmedium', 'reinforce', 'ImageNet16-120', 'naswot')
    nasnet_naslittle = generate_df_cost('naslittle', 'reinforce', 'ImageNet16-120', 'naswot')
    generate_cost_table([nasnet_nasbench, nasnet_nasbig, nasnet_nasmedium, nasnet_naslittle],
                       ["NAS-bench", "NAS-big", "NAS-medium", "NAS-little"])"""

    df = get_5_best('naslittle','random','cifar10') # ImageNet16-120
    get_all_metrics(df)
    df = get_5_best('naslittle', 'random', 'cifar100')  # ImageNet16-120
    get_all_metrics(df)
    df = get_5_best('naslittle','random','ImageNet16-120') # ImageNet16-120
    get_all_metrics(df)