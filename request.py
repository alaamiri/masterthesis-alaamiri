# Génerer un graphique reprenant l'évolution de l'accuracy des réseaux sélectionné
# Génerer une heatmap concernant la distribution des opérations séléctionnées
# Timer le temps que prend de selectionner un réseaux et en faire une moyenne sur les 5 seeds
# afficher le string de l'architecture
# sauvegarder la meilleurs architecture (fichier txt, puis simplement recup pour l'entrainer)

# Faire request nasmedium et naslittle
# Fix les seeds qui semblent ne pas bien s'appliquer
# (Box plot des réseaux sélectionnés afin de visualiser la performance de REINFORCE sur tout les espaces de recherches)
# A la toute fin, tableau récapitulatif concernant les performances des différents réseaux sur les différents espaces de recherche
from nats_bench import create
from controller import Controller
import numpy as np
import pandas as pd
import plot
import os
NATS_BENCH_TSS_PATH = "nats_bench_data/NATS-tss-v1_0-3ffb9-simple"
OUT_DIR = "./out/"
NB_NET = 500
EPOCHS = 12
api = create(NATS_BENCH_TSS_PATH, 'tss', fast_mode=True, verbose=False)

def matrix_sum(matrix_list):
    # chat-gpt
    # Make sure all matrices have the same shape
    shapes = [matrix.shape for matrix in matrix_list]
    if len(set(shapes)) != 1:
        raise ValueError("All matrices must have the same shape")

    # Sum the matrices element-wise
    result = np.zeros(shapes[0])
    for matrix in matrix_list:
        result += matrix / NB_NET

    return result


def write_model(models_path, seed, model_str):
    #chat-gpt
    path = models_path+f"/{seed}.txt"
    with open(path, 'w') as f:
        f.write(model_str)

        # Optionally, write a newline character to the file
        f.write('\n')


def read_model(model_path):
    with open(model_path, 'r') as file:
        lines = file.readlines()
        second_line = lines[1].strip()

    return second_line

def run_request(c, path, models_path,predictor,seeds):
    if not os.path.exists(path):
        os.makedirs(path)

    if not os.path.exists(models_path):
        os.makedirs(models_path)
    
    for seed in seeds:
        models, valids, iters, times = c.run(nb_iterations=NB_NET,
                                                              predictor=predictor,
                                                              seed=seed,
                                                              epochs=EPOCHS,
                                                              reset=True)
        idx_models = [api.query_index_by_arch(c.arch_to_str(model)) for model in models]
        df = pd.DataFrame({'id' : idx_models, 'acc_valid' : valids, 'time' : times})
        csv_name = f"/{seed}.csv"
        df.to_csv(models_path+csv_name, index=False)

"""
def run_request(c, path, models_path, dataset, search_space, predictor, fn, seeds):
    if not os.path.exists(path):
        os.makedirs(path)

    if not os.path.exists(models_path):
        os.makedirs(models_path)

    l_model = []
    l_valid = []
    l_dist = []
    l_time = []

    all_models = []
    all_valids = []

    for seed in seeds:
        models, valids, iters, delta_time = c.run(nb_iterations=NB_NET,
                                                              predictor=predictor,
                                                              seed=None,
                                                              epochs=EPOCHS,
                                                              reset=True)
        best_id = valids.index(max(valids))
        best_model, best_valid, best_iter = models[best_id], valids[best_id], iters[best_id]

        all_valids.append(valids)
        all_models.append([api.query_index_by_arch(c.arch_to_str(model)) for model in models])

        l_model.append(best_model)
        l_valid.append(best_valid)
        l_dist.append(c.op_dist)
        l_time.append(delta_time)
        best_str = "Best model:\n " \
                   "{:}" \
                   "\n Acc valid: {:>.3f}" \
                   "\n At iter: {:}" \
                   "\n Time: {:>.3f}".format(c.arch_to_str(best_model), best_valid, best_iter, delta_time)

        write_model(models_path, seed, best_str)
        print(best_str)

    avg_valid, std_valid, avg_time = np.average(l_valid), np.std(l_valid), np.average(l_time)
    best_valid = max(l_valid)
    best_of_best = l_model[l_valid.index(best_valid)]
    print("***************** Sum *****************")
    sum_str = "Best of best model:\n" \
              " {:}\n" \
              " acc best: {:>.3f}\n" \
              " avg: {:>.3f}\n" \
              " std: {:>.3f}\n" \
              " avg_time: {:>.3f}".format(c.arch_to_str(best_of_best),best_valid, avg_valid, std_valid, avg_time)
    print(sum_str)
    print("***************************************")
    write_model(path, 'summary', sum_str)
    sum_dist = matrix_sum(l_dist) / len(l_dist)
    print(read_model(path+"/summary.txt"))
    #plot.dist_heatmap(sum_dist,
                      #['zero', 'identity', 'conv1x1', 'conv3x3', 'avgp3x3'],
                      #path,
                      #dataset=dataset,
                      #nb_seeds=len(seeds),
                      #search_space=search_space,
                      #fn=fn)

    return all_models, all_valids
"""
# ======================================================================================================================
def run(s_space, fn, dataset, predictor, benchmark):
    if predictor is None:
        path = f"{OUT_DIR}{fn}/{s_space}/{dataset}"
    else:
        path = f"{OUT_DIR}{fn}/{s_space}/{predictor}/{dataset}"

    models_path = path+'/seeds'

    c = Controller(s_space=s_space,
                   ao_fn=fn,
                   dataset=dataset,
                   benchmark=benchmark,
                   verbose=True)

    return run_request(c, path, models_path, predictor, seeds)


def get_info(models, type, dataset):
    data = np.array(models)
    data = data.reshape(-1)

    stat = [api.get_cost_info(model, dataset)[type] for model in data]

    return stat


def nasnet_nasbench(dataset):
    ss = 'nasbench'
    fn = 'reinforce'
    run(s_space=ss, fn=fn, dataset=dataset, predictor=None, benchmark=True)


def random_nasbench(dataset):
    ss = 'nasbench'
    fn = 'random'
    run(s_space=ss, fn=fn, dataset=dataset, predictor=None, benchmark=True)


def nasnet_otherbenchs(dataset):
    fn = 'reinforce'
    run(s_space='nasbig', fn=fn, dataset=dataset, predictor=None, benchmark=True)
    run(s_space='nasmedium', fn=fn, dataset=dataset, predictor=None, benchmark=True)
    run(s_space='naslittle', fn=fn, dataset=dataset, predictor=None, benchmark=True)


def random_otherbenchs(dataset):
    fn = 'random'
    run(s_space='nasbig', fn=fn, dataset=dataset, predictor=None, benchmark=True)
    run(s_space='nasmedium', fn=fn, dataset=dataset, predictor=None, benchmark=True)
    run(s_space='naslittle', fn=fn, dataset=dataset, predictor=None, benchmark=True)


def nasnet_naswot_nasbench(dataset):
    ss = 'nasbench'
    fn = 'reinforce'
    run(s_space=ss, fn=fn, dataset=dataset, predictor='naswot', benchmark=False)


def random_naswot_nasbench(dataset):
    ss = 'nasbench'
    fn = 'random'
    run(s_space=ss, fn=fn, dataset=dataset, predictor='naswot', benchmark=False)


def nasnet_naswot_otherbenchs(dataset):
    fn = 'reinforce'
    run(s_space='nasbig', fn=fn, dataset=dataset, predictor='naswot', benchmark=False)
    run(s_space='nasmedium', fn=fn, dataset=dataset, predictor='naswot', benchmark=False)
    run(s_space='naslittle', fn=fn, dataset=dataset, predictor='naswot', benchmark=False)


def random_naswot_otherbenchs(dataset):
    fn = 'random'
    run(s_space='nasbig', fn=fn, dataset=dataset, predictor='naswot', benchmark=False)
    run(s_space='nasmedium', fn=fn, dataset=dataset, predictor='naswot', benchmark=False)
    run(s_space='naslittle', fn=fn, dataset=dataset, predictor='naswot', benchmark=False)


if __name__ == '__main__':
    # [14139, 655, 4237, 4361, 699]
    seeds = [1, 10, 100, 1000, 10000]
    dataset = 'cifar10' #ImageNet16-120

    ### NASNET --- NASBENCH
    nasnet_nasbench(dataset)
    random_nasbench(dataset)
    ### NASNET --- OTHER BENCHS (NASBIG NASMEDIUM NASLITTLE)
    #nasnet_otherbenchs(dataset)
    #random_otherbenchs(dataset)
    ### NASNET + NASWOT --- NASBENCH
    #nasnet_naswot_nasbench(dataset)
    #random_naswot_nasbench(dataset)
    ### NASNET + NASWOT --- OTHERBENCH
    #nasnet_naswot_otherbenchs(dataset)
    #random_naswot_otherbenchs(dataset)



    """#reinforce_nasbench(seeds, 'cifar10')
    reinforce_bench_c10 = run(s_space='nasbench', fn='reinforce', dataset='cifar10', predictor=None, benchmark=True)
    #plot.box_plot([reinforce_bench_c10], ['nasbench'], OUT_DIR,
                  #"Distribution of severals space searchs with reinforce", fn='reinforce')
                  
    #reinforce_nasbench_naswot(seeds, 'cifar10')
    #run(s_space='nasbench', fn='reinforce', dataset='cifar10', predictor='naswot', benchmark=False)
    #random_nasbench(seeds, 'cifar10')
    #random_bench_c10 = run(s_space='nasbench', fn='randomsearch', dataset='cifar10', predictor=None, benchmark=True)
    #random_nasbench_naswot(seeds, 'cifar10')
    #run(s_space='nasbench', fn='randomsearch', dataset='cifar10', predictor='naswot', benchmark=False)



    #reinforce_nasmedium(seeds, 'cifar10')
    reinforce_medium_c10 = run(s_space='nasmedium', fn='reinforce', dataset='cifar10', predictor=None, benchmark=True)
    #reinforce_nasmedium_naswot(seeds, 'cifar10')
    #run(s_space='nasmedium', fn='reinforce', dataset='cifar10', predictor='naswot', benchmark=False)
    #random_nasmedium(seeds, 'cifar10')
    #random_medium_c10 = run(s_space='nasmedium', fn='randomsearch', dataset='cifar10', predictor=None, benchmark=True)
    #random_nasmedium_naswot(seeds, 'cifar10')
    #run(s_space='nasmedium', fn='randomsearch', dataset='cifar10', predictor='naswot', benchmark=False)

    #reinforce_naslittle(seeds, 'cifar10')
    reinforce_little_c10 = run(s_space='naslittle', fn='reinforce', dataset='cifar10', predictor=None, benchmark=True)
    #reinforce_naslittle_naswot(seeds, 'cifar10')
    #run(s_space='naslittle', fn='reinforce', dataset='cifar10', predictor='naswot', benchmark=False)
    #random_naslittle(seeds, 'cifar10')
    #random_little_c10 = run(s_space='naslittle', fn='randomsearch', dataset='cifar10', predictor=None, benchmark=True)
    #random_naslittle_naswot(seeds, 'cifar10')
    #run(s_space='naslittle', fn='randomsearch', dataset='cifar10', predictor='naswot', benchmark=False)

    reinforce_big_c10 = run(s_space='nasbig', fn='reinforce', dataset='cifar10', predictor=None, benchmark=True)
    #random_big_c10 = run(s_space='nasbig', fn='randomsearch', dataset='cifar10', predictor=None, benchmark=True)

    #{'flops', 'params''latency', 'T-train@epoch', 'T-train@total', 'T-ori-test@epoch', 'T-ori-test@total'}
    #params = "params"
    #p = [get_info(data[0], "params", "cifar10") for data in [reinforce_bench_c10, reinforce_big_c10, reinforce_medium_c10, reinforce_little_c10]]
    #plot.bar_plot(p, ["bench", "big", "medium", "little"], OUT_DIR, params, "reinforce", "cifar10")
    plot.box_plot([reinforce_bench_c10[1], reinforce_big_c10[1], reinforce_medium_c10[1], reinforce_little_c10[1]], ['nasbench', 'nasbig', 'nasmedium', 'naslittle'], OUT_DIR,
                  "Distribution of severals space searchs with reinforce", fn='reinforce')
    plot.box_plot([random_bench_c10[1], random_big_c10[1], random_medium_c10[1], random_little_c10[1]], ['nasbench', 'nasbig', 'nasmedium', 'naslittle'], OUT_DIR,
                  "Distribution of severals space searchs with random search", fn='randomsearch')"""
