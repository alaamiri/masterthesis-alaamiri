import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def accuracy_plot(accuracy_list, nb_net, nb_layers, seed=None):
    plt.plot(range(1,len(accuracy_list)+1), accuracy_list, 'c.')
    #plt.plot(range(1,len(accuracy_list)+1), accuracy_list,'b.')
    plt.hlines(np.average(accuracy_list),0,len(accuracy_list)+1, 'r')
    plt.ylabel("Accuracy")
    plt.xlabel("Iteration")
    #plt.xticks(range(1,len(accuracy_list),2))
    plt.title(f"Accuracy of models generated by the controller N={nb_net} l={nb_layers}")
    if seed is not None:
        seed = f"_[{seed}]"
    else:
        seed = ""
    plt.savefig(f"plot/acc_plot_{nb_net}_{nb_layers}{seed}.png")
    plt.show()

def loss_plot(loss_list, nb_net, nb_layers, seed=None):
    #loss_list = loss_list.detach()
    plt.plot(range(1, len(loss_list) + 1), loss_list, 'c.')
    #plt.plot(range(1, len(loss_list) + 1), loss_list, 'b.')
    plt.hlines(np.average(loss_list), 0, len(loss_list) + 1, 'r')
    plt.ylabel("Loss")
    plt.xlabel("Iteration")
    # plt.xticks(range(1,len(accuracy_list),2))
    plt.title(f"Loss of the controller N={nb_net} l={nb_layers}")
    if seed is not None:
        seed = f"_[{seed}]"
    else:
        seed = ""
    plt.savefig(f"plot/loss_plot_{nb_net}_{nb_layers}{seed}.png")
    plt.show()

def plot_several_runs(nb_run, train_l, valid_l, test_l):
    plt.plot(range(1, len(train_l) + 1), train_l, 'c.')
    plt.plot(range(1, len(valid_l) + 1), valid_l, 'g.')
    plt.plot(range(1, len(test_l) + 1), test_l, 'r.')
    plt.ylabel("%")
    plt.xlabel("Runs")
    # plt.xticks(range(1,len(accuracy_list),2))
    plt.title(f"Accuracies of the best net on {nb_run} run(s)")
    plt.legend(["Train", "Validation", "Test"])
    plt.show()

def dist_heatmap(dist_mat, op, path, fn, dataset, search_space, nb_seeds):
    sns.set()
    yticks = [f"op{i}" for i in range(len(dist_mat))]
    ax = sns.heatmap(dist_mat, xticklabels=op, yticklabels=yticks, cmap='magma')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right")
    title = f"Operations distribution on {search_space} with {fn} in {dataset} for {nb_seeds} seed(s)"
    plt.title(title)
    plt.savefig(path+"/dist_heatmap"+".png")
    plt.show()

def box_plot(data, xticks, path, title, fn):
    data = np.array(data)
    data = data.reshape(len(data), -1)
    ax = plt.boxplot(np.transpose(data))
    plt.ylabel("acc")
    plt.title(title)
    plt.xticks([i+1 for i in range(len(xticks))], xticks)
    plt.savefig(path+f"boxplot_{fn}.png")

    plt.show()

def bar_plot(data, xticks, path, title, fn):
    print(data)
    plt.bar(xticks, np.average(data))
    plt.show()