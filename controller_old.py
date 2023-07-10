import itertools
import random
import time

import numpy as np
import torch
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn as nn

import plot
from plot import *

import rnn
from . import search_spaces
from . import ao
from predictors import naswot

from nats_bench import create


#seed = 1
#torch.manual_seed(seed)


class Controller():
    def __init__(self, s_space, dataset='cifar10', rnn_fn='reinforce', benchmark=False, verbose=False):
        if verbose:
            print("Controller"                  
                  "\n   +--- Dataset: {:}"
                  "\n   +--- Search Space: {:}"
                  "\n   +--- Benchmark mode: {:}".format(dataset, s_space,benchmark))
        self.search_space = search_spaces.ss_selector(s_space, dataset)
        self.s_tag = self.search_space.OPERATIONS
        self.nb_ops = self.search_space.NB_OPS

        self.fn = rnn_fn

        self.dataset = dataset
        self.benchmark = benchmark
        self.rnn_fn = rnn_fn

        self.rnn = rnn.RNN(rnn.HIDDEN_SIZE, self.s_tag)
        self.loaders = self._get_dataloaders(batch_size=64)

        self.verbose = verbose
        

    def generate_arch(self, optimizer):
        if optimizer=="reinforce":
            arch_l, prob_list = self.rnn.generate_arch(self.nb_ops)
        elif optimizer=="randomsearch":
            arch_l, prob_list = self.generate_random_arch(self.nb_ops)

        return arch_l, prob_list

    def generate_random_arch(self, nb_ops):
        arch_l = []
        for _ in range(nb_ops):
            i = random.randint(0,len(self.s_tag)-1)
            arch_l.append(self.s_tag[i])

        return arch_l, 0
    def build_arch(self, arch_l):
        model = None
        if self.benchmark:
            model = self.search_space.get_arch_id(arch_l)
        if not self.benchmark:
            model = self.search_space.get_model(arch_l)
            self.loss_fn = nn.CrossEntropyLoss()
            # self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-3) #optim.AdamP or SGDP or SGDW or SWATS
            self.optimizer = torch.optim.SGD(model.parameters(), lr=5e-2, momentum=0.9,
                                             weight_decay=1e-4, nesterov=True)

        return model

    def evaluate_arch(self, model, predictor, epochs):
        if predictor is not None:
            r = self.predict_acc(model, predictor)
        else:
            r = self.validate_arch(model, epochs)

        return r

    def predict_acc(self, model, predictor):
        r = None
        if predictor == 'naswot':
            p = naswot.NASWOT(self.loaders['train'], 64)
            r = p.predict(model)

        return r

    def train_one_epoch(self, model, dataloader, writer=None):
        size = len(dataloader.dataset)

        avg_loss = 0

        for batch, (X, y) in enumerate(dataloader):
            #X, y = X.to(self.device), y.to(self.device)
            # Compute prediction error
            # print(X.size())

            pred = model(X)

            loss = self.loss_fn(pred, y)

            # Backpropagation

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"\tloss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        return avg_loss

    def test_model(self, model, dataloader):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                # X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"\tTest Error: \n \t\tAccuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        return correct

    def validate_arch(self, model, epochs):
        if self.benchmark:
            r = self.search_space.get_score_from_api(model)

        else:
            for epoch in range(epochs):
                model.train(True)
                avg_loss = self.train_one_epoch(model, self.loaders['train'])
                model.train(False)

            r = self.test_model(model, self.loaders['test'])

        return r


    def update_rnn(self, r, prob_list):
        loss = self.rnn.reinforce(prob_list, r)

        return loss


    def iterate(self, optimizer, predictor, epochs):
        arch_l, prob_list = self.generate_arch(optimizer)
        model = self.build_arch(arch_l)
        r = self.evaluate_arch(model, predictor, epochs)
        if optimizer=="reinforce":
            rnn_loss = self.update_rnn(r, prob_list)
        else:
            rnn_loss = False

        return arch_l,model,r,rnn_loss

    def run(self, nb_iterations, predictor=None, seed=None, epochs=12, reset=False):
        if self.verbose:
            print("\tRunning =========================="
                  "\n\t   +--- # iterations: {:}"
                  "\n\t   +--- Predictor: {:}"
                  "\n\t   +--- seed: {:}"
                  "\n\t   +--- EPOCH: {:}"
                  "\n\t   +--- Reset hidden state: {:}".format(nb_iterations, predictor, seed, epochs, reset))

        if reset:
            self.rnn.h = self.rnn.init_hidden()

        if seed is not None:
            torch.manual_seed(seed)

        l_models = []
        l_valid = []
        l_iter = []

        self.op_dist = self._init_dist_list()
        start_time = time.time()

        for i in range(nb_iterations):
            arch,model,r,rnn_loss = self.iterate(self.fn, predictor,epochs)

            self._add_dist_layer(arch)

            l_models.append(arch)
            l_valid.append(r)
            l_iter.append(i)

            if self.verbose:
                if i % (nb_iterations//5) == 0:
                    print(f"\t[{i:>5d}/{nb_iterations:>5d}]")


        end_time = time.time()
        delta_time = end_time - start_time

        #return best_model, best_valid, best_iter, delta_time
        return  l_models, l_valid, l_iter, delta_time

    def arch_to_str(self,operations):
        return self.search_space.get_nasbench_unique(operations)


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
        train_data = datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )

        #5000
        test_data = datasets.CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
        )

        """indices_train = torch.arange(5000)
        indices_test = torch.arange(5000)

        train_data = data_utils.Subset(train_data, indices_train)
        test_data = data_utils.Subset(test_data, indices_test)"""

        loaders = {
            'train': DataLoader(train_data,
                                batch_size=batch_size,
                                shuffle=True),

            'test': DataLoader(test_data,
                               batch_size=batch_size,
                               shuffle=True)
        }

        return loaders

    def _get_dist_layers(self):
        for elem in self.op_dist:
            elem[:] = [x/sum(elem) for x in elem]

    def _show_dist(self):
        # by Chat GPT
        layers = self.s_tag
        format_string = "{:<15}" + "{:<15}" * len(layers)
        # Print the header
        print(format_string.format("", *layers))

        # Print the matrix values
        for i, row in enumerate(self.op_dist):
            print(format_string.format("Layer " + str(i), *["{:.4f}".format(val) for val in row]))

    def _add_dist_layer(self, arch):
        for i in range(len(arch)):
            curr_l = arch[i]
            self.op_dist[i][self.s_tag.index(curr_l)] +=1

    def _init_dist_list(self):
        nb_layers = self.nb_ops
        l = []
        for i in range(nb_layers):
            l.append([0]*len(self.s_tag))

        return np.array(l)


    def get_bench_best(self):
        print('There are {:} architectures on the topology search space'.format(len(self.api)))

        best_arch_index, highest_valid_accuracy = self.api.find_best(dataset='cifar10', metric_on_set='ori-tes', hp='12')
        print(best_arch_index,highest_valid_accuracy)
        #13714 84.89199999023438 cifar10-valid x-valid

if __name__ == '__main__':
    nb_net = 200
    nb_layers = 7
    nb_run = 5

    c = Controller(s_space='nas_medium',
                   dataset='cifar10',
                   benchmark=False,
                   verbose=True)
    #c.get_bench_best()

    #c.run(nb_net,nb_layers, epochs=1)

    """c.run_several(nb_run, nb_net, nb_layers)
    accuracy_plot(c.acc_list, nb_net, nb_layers)
    loss_plot(c.loss_list, nb_net, nb_layers)
    c.random_search_several(nb_run, nb_net, nb_layers)
    accuracy_plot(c.acc_list, nb_net, nb_layers)
    loss_plot(c.loss_list, nb_net, nb_layers)"""

    print(c.iterate(predictor='naswot',
                    epochs=1))


