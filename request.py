# Génerer un graphique reprenant l'évolution de l'accuracy des réseaux sélectionné
# Génerer une heatmap concernant la distribution des opérations séléctionnées
# (Box plot des réseaux sélectionnés afin de visualiser la performance de REINFORCE sur tout les espaces de recherches)
# Timer le temps que prend de selectionner un réseaux et en faire une moyenne sur les 5 seeds
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


def reinforce_nasbench(seeds, dataset):
    path = OUT_DIR + "reinforce/nasbench/" + dataset
    if not os.path.exists(path):
        os.makedirs(path)

    c = Controller(s_space='nasbench',
                   rnn_fn='reinforce',
                   dataset=dataset,
                   benchmark=True,
                   verbose=True)
    l_model = []
    l_valid = []
    l_dist = []
    l_time = []
    for seed in seeds:
        best_model, best_valid, best_iter, delta_time = c.run(nb_iterations=NB_NET,
                                                              predictor=None,
                                                              seed=seed,
                                                              epochs=EPOCHS,
                                                              reset=True)
        l_model.append(best_model)
        l_valid.append(best_valid)
        l_dist.append(c.op_dist)
        l_time.append(delta_time)
        print("Best model: {:}"
              "\n Acc valid: {:>.3f}"
              "\n At iter: {:}"
              "\n Time: {:>.3f}".format(best_model, best_valid, best_iter, delta_time))

    avg_valid, std_valid, avg_time = np.average(l_valid), np.std(l_valid), np.average(l_time)
    print("avg: {:>.3f}, std: {:>.3f}, avg time: {:>.3f}".format(avg_valid, std_valid, avg_time))

    sum_dist = matrix_sum(l_dist) / len(l_dist)

    plot.dist_heatmap(sum_dist,
                      ['zero', 'identity', 'conv1x1', 'conv3x3', 'avgp3x3'],
                      "Operations distribution with REINFORCE in NAS-Bench for 5 seeds",
                      path)


def random_nasbench(seeds, dataset):
    c = Controller(s_space='nasbench',
                   rnn_fn='randomsearch',
                   dataset=dataset,
                   benchmark=True,
                   verbose=True)
    l_model = []
    l_valid = []
    l_dist = []
    l_time = []
    for seed in seeds:
        best_model, best_valid, best_iter, delta_time = c.run(nb_iterations=NB_NET,
                                                              predictor=None,
                                                              seed=seed,
                                                              epochs=EPOCHS,
                                                              reset=True)
        l_model.append(best_model)
        l_valid.append(best_valid)
        l_dist.append(c.op_dist)
        l_time.append(delta_time)
        print("Best model: {:}"
              "\n Acc valid: {:>.3f}"
              "\n At iter: {:}"
              "\n Time: {:>.3f}".format(best_model, best_valid, best_iter, delta_time))

    avg_valid, std_valid, avg_time = np.average(l_valid), np.std(l_valid), np.average(l_time)
    print("avg: {:>.3f}, std: {:>.3f}, avg time: {:>.3f}".format(avg_valid, std_valid, avg_time))

    sum_dist = matrix_sum(l_dist) / len(l_dist)

    plot.dist_heatmap(sum_dist,
                      ['zero', 'identity', 'conv1x1', 'conv3x3', 'avgp3x3'],
                      "Operations distribution with Random Search in NAS-Bench for 5 seeds")


def reinforce_nasbench_naswot(seeds, dataset):
    c = Controller(s_space='nasbench',
                   rnn_fn='reinforce',
                   dataset=dataset,
                   benchmark=False,
                   verbose=True)
    l_model = []
    l_valid = []
    l_dist = []
    l_time = []
    for seed in seeds:
        best_model, best_valid, best_iter, delta_time = c.run(nb_iterations=NB_NET,
                                                              predictor='naswot',
                                                              seed=seed,
                                                              epochs=EPOCHS,
                                                              reset=True)
        l_model.append(best_model)
        l_valid.append(best_valid)
        l_dist.append(c.op_dist)
        l_time.append(delta_time)
        print("Best model: {:}"
              "\n Acc valid: {:>.3f}"
              "\n At iter: {:}"
              "\n Time: {:>.3f}".format(best_model, best_valid, best_iter, delta_time))

    avg_valid, std_valid, avg_time = np.average(l_valid), np.std(l_valid), np.average(l_time)
    print("avg: {:>.3f}, std: {:>.3f}, avg time: {:>.3f}".format(avg_valid, std_valid, avg_time))

    sum_dist = matrix_sum(l_dist) / len(l_dist)

    plot.dist_heatmap(sum_dist,
                      ['zero', 'identity', 'conv1x1', 'conv3x3', 'avgp3x3'],
                      "Operations distribution with REINFORCE+NASWOT in NAS-Bench for 5 seeds")



def reinforce_nasmedium(seeds, dataset):
    pass


def reinforce_nasmedium_naswot(seeds, dataset):
    pass


def reinforce_naslittle(seeds, dataset):
    pass


def reinforce_naslittle_naswot(seeds, dataset):
    pass


if __name__ == '__main__':
    # [14139, 655, 4237, 4361, 699]
    seeds = [1, 10, 100, 1000, 10000]

    reinforce_nasbench(seeds, 'cifar10')
    #random_nasbench(seeds, 'cifar10')
    #reinforce_nasbench_naswot(seeds,'cifar10')
