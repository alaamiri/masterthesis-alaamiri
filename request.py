# Génerer un graphique reprenant l'évolution de l'accuracy des réseaux sélectionné
# Génerer une heatmap concernant la distribution des opérations séléctionnées
# Timer le temps que prend de selectionner un réseaux et en faire une moyenne sur les 5 seeds
# afficher le string de l'architecture
# sauvegarder la meilleurs architecture (fichier txt, puis simplement recup pour l'entrainer)

# Faire request nasmedium et naslittle
# Fix les seeds qui semblent ne pas bien s'appliquer
# (Box plot des réseaux sélectionnés afin de visualiser la performance de REINFORCE sur tout les espaces de recherches)
# A la toute fin, tableau récapitulatif concernant les performances des différents réseaux sur les différents espaces de recherche

from controller import Controller
import numpy as np
import plot
import os

OUT_DIR = "./out/"
NB_NET = 200
EPOCHS = 12


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


def run_request(c, path, models_path, dataset, predictor, fn, seeds):
    if not os.path.exists(path):
        os.makedirs(path)

    if not os.path.exists(models_path):
        os.makedirs(models_path)

    l_model = []
    l_valid = []
    l_dist = []
    l_time = []
    for seed in seeds:
        best_model, best_valid, best_iter, delta_time = c.run(nb_iterations=NB_NET,
                                                              predictor=predictor,
                                                              seed=seed,
                                                              epochs=EPOCHS,
                                                              reset=True)
        l_model.append(best_model)
        l_valid.append(best_valid)
        l_dist.append(c.op_dist)
        l_time.append(delta_time)
        best_str = "Best model:\n " \
                   "{:}" \
                   "\n Acc valid: {:>.3f}" \
                   "\n At iter: {:}" \
                   "\n Time: {:>.3f}".format(c.arch_to_str(best_model), best_valid, best_iter, delta_time)

        print(best_str)
        write_model(models_path, seed, best_str)
    avg_valid, std_valid, avg_time = np.average(l_valid), np.std(l_valid), np.average(l_time)
    print("avg: {:>.3f}, std: {:>.3f}, avg time: {:>.3f}".format(avg_valid, std_valid, avg_time))
    sum_dist = matrix_sum(l_dist) / len(l_dist)
    plot.dist_heatmap(sum_dist,
                      ['zero', 'identity', 'conv1x1', 'conv3x3', 'avgp3x3'],
                      path,
                      dataset=dataset,
                      nb_seeds=len(seeds),
                      fn=fn)

# ======================================================================================================================


def reinforce_nasbench(seeds, dataset):
    fn = 'reinforce'
    path = OUT_DIR + "reinforce/nasbench/" + dataset
    models_path = path+'/seeds'

    c = Controller(s_space='nasbench',
                   rnn_fn='reinforce',
                   dataset=dataset,
                   benchmark=True,
                   verbose=True)

    run_request(c, path, models_path, dataset, None, fn, seeds)


def random_nasbench(seeds, dataset):
    fn = 'randomsearch'
    path = OUT_DIR + "randomsearch/nasbench/" + dataset
    models_path = path + '/seeds'

    c = Controller(s_space='nasbench',
                   rnn_fn='randomsearch',
                   dataset=dataset,
                   benchmark=True,
                   verbose=True)

    run_request(c, path, models_path, dataset, None, fn, seeds)


def reinforce_nasbench_naswot(seeds, dataset):
    fn = 'reinforce'
    path = OUT_DIR + "reinforce/nasbench/naswot/" + dataset
    models_path = path + '/seeds'

    c = Controller(s_space='nasbench',
                   rnn_fn='reinforce',
                   dataset=dataset,
                   benchmark=False,
                   verbose=True)

    run_request(c, path, models_path, dataset, 'naswot', fn, seeds)


def reinforce_nasmedium(seeds, dataset):
    fn = 'reinforce'
    path = OUT_DIR + "reinforce/nasmedium/" + dataset
    models_path = path + '/seeds'

    c = Controller(s_space='nasmedium',
                   rnn_fn='reinforce',
                   dataset=dataset,
                   benchmark=True,
                   verbose=True)

    run_request(c, path, models_path, dataset, None, fn, seeds)


def reinforce_nasmedium_naswot(seeds, dataset):
    fn = 'reinforce'
    path = OUT_DIR + "reinforce/nasmedium/naswot/" + dataset
    models_path = path + '/seeds'

    c = Controller(s_space='nasmedium',
                   rnn_fn='reinforce',
                   dataset=dataset,
                   benchmark=False,
                   verbose=True)

    run_request(c, path, models_path, dataset, 'naswot', fn, seeds)


def reinforce_naslittle(seeds, dataset):
    pass


def reinforce_naslittle_naswot(seeds, dataset):
    pass


if __name__ == '__main__':
    # [14139, 655, 4237, 4361, 699]
    seeds = [1, 10, 100, 1000, 10000]

    #reinforce_nasbench(seeds, 'cifar10')
    #random_nasbench(seeds, 'cifar10')
    #reinforce_nasbench_naswot(seeds, 'cifar10')
    #reinforce_nasmedium(seeds, 'cifar10')
    reinforce_nasmedium_naswot(seeds, 'cifar10')
