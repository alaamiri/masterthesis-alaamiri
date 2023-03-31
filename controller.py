import itertools
import torch
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from plot import *

import net
import rnn
import search_space

from nats_bench import create

seed = 1
torch.manual_seed(seed)

NATS_BENCH_TSS_PATH = "nats_bench_data/NATS-tss-v1_0-3ffb9-simple"

class Controller():
    def __init__(self, s_space, dataset='cifar10', rnn_fn='REINFORCE', benchmark=False, verbose=False):
        self.s_space = self._init_ss(s_space)
        self.benchmark = benchmark
        self.rnn_fn = rnn_fn

        self.rnn = rnn.RNN(rnn.HIDDEN_SIZE, self.s_space)
        self.loaders = self._get_dataloaders(batch_size=64)

        self.verbose = verbose
        if benchmark:
            self.api = create(NATS_BENCH_TSS_PATH,
                              'tss',
                              fast_mode=True,
                              verbose=False)

        if verbose:
            print("dataset : {:}\nbenchmark : {:}".format(dataset, benchmark))

    def generate_arch(self, nb_layer):
        if self.benchmark:
            nb_layer = 6
        arch, prob_list = self.rnn.generate_arch(nb_layer)

        return arch, prob_list

    def build_arch(self, arch):
        model = None
        if self.benchmark:
            arch = '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.format(*arch)
            model = self.api.query_index_by_arch(arch)
        else:
            model = net.Net(arch)

        return model

    def evaluate_arch(self, model, predictor, epochs):
        if self.benchmark:
            info = self.api.get_more_info(model, 'cifar10')
            r = info['train-accuracy']/100

            return r
        if predictor is not None:
            pass
        else:
            r = self.validate_arch(model, epochs)

        return r

    def validate_arch(self, model, epochs):
        for epoch in range(epochs):
            model.train(True)
            avg_loss = model.train_one_epoch(self.loaders['train'])
            model.train(False)

            r = model.test_model(self.loaders['test'])

        return r

    def update_rnn(self, r, prob_list):
        return self.rnn.reinforce(prob_list, r)

    def iterate(self, nb_layer, predictor, epochs):
        arch, prob_list = self.generate_arch(nb_layer)
        model = self.build_arch(arch)
        r = self.evaluate_arch(model, predictor, epochs)
        rnn_loss = self.update_rnn(r, prob_list)

        return arch,model,r,rnn_loss

    def run(self, nb_iterations, nb_layer, predictor=None, epochs=12):
        self.loss_list = []
        self.acc_list = []
        best_model = None
        best_r = 0
        best_iter = 0
        for i in range(nb_iterations):
            arch,model,r,rnn_loss = self.iterate(nb_layer,predictor,epochs)

            self.acc_list.append(r)
            self.loss_list.append(rnn_loss)

            if self.verbose:
                if i % 100 == 0:
                    print(f"\t[{i:>5d}/{nb_iterations:>5d}]")

            if r > best_r:
                best_r = r
                best_model = model
                best_iter = i

        if self.verbose:
            print(f"Best model : {best_model}\nAccuracy : {best_r*100}")

        accuracy_plot(self.acc_list, nb_iterations, nb_layer)
        loss_plot(self.loss_list, nb_iterations, nb_layer)

    def _init_ss(self, ss) -> dict:
        """
        Generate all the combination of hyperparameters sets

        :param ss: list
            The search space's hyperparamters
        :return: dict
            Return a dictionnary with in the form ID : layers where ID = [0,N] where N is the number of all possible
            combination
        """
        a = ss
        if isinstance(ss[0], list):
            a = list(itertools.product(*a))
        dict = {}
        for i in range(len(a)):
            dict[i] = a[i]

        return dict

    def _get_dataloaders(self, batch_size : int=64, data_type : str="MNIST") -> dict:
        """
        Generate the dataloaders used by the generated nets to train and to validate

        :param batch_size: int
            Size of the batch
        :param data_type: string
            The type of dataset
        :return: dict
            Return the loaders in a dict with keys 'train' and 'test'

        """
        #45 000
        train_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )

        #5000
        test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
        )

        indices_train = torch.arange(45000)
        indices_test = torch.arange(5000)

        train_data = data_utils.Subset(train_data, indices_train)
        test_data = data_utils.Subset(test_data, indices_test)

        loaders = {
            'train': DataLoader(train_data,
                                batch_size=batch_size,
                                shuffle=True),

            'test': DataLoader(test_data,
                               batch_size=batch_size,
                               shuffle=True)
        }

        return loaders

if __name__ == '__main__':
    nb_net = 5000
    nb_layers = 7

    c = Controller(s_space=search_space.nats_bench_tss,
                   dataset='cifar10',
                   benchmark=True,
                   verbose=True)
    c.run(nb_net,nb_layers, epochs=1)