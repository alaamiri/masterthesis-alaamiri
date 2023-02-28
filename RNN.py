import itertools

import numpy as np

from Plot import *

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.utils.data as data_utils
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import torch.optim as optim

import Net

# The RNN will output a layers depending the combination of those hyperparameter
F_HEIGHT = [1,3,5]
F_WIDTH = [1,3,5]
N_FILTERS = [24,36,48,64]
N_STRIDES = [1]

# LSTM parameters
HIDDEN_SIZE = 35
N_LAYER = 2
seed = 1
torch.manual_seed(seed)

class RNN(nn.Module):
    """
    A class representing the controller which generate the CNN depending of the search space
    """
    def __init__(self, hidden_size: int):
        super(RNN, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")

        self.type_layers = self._init_layers()

        self.input_size = len(self.type_layers)
        self.hidden_size = hidden_size
        print("# of possible layers :", self.input_size)
        self.output_size = len(self.type_layers)

        self.lstm = nn.LSTM(self.input_size, hidden_size, num_layers=N_LAYER)
        self.hidden_to_hyper = nn.Linear(hidden_size, self.output_size)

        self.x = torch.zeros(self.input_size).unsqueeze(dim=0)  # lstm need dim 3 so we dim 2 then dim 3
        self.h = self._init_hidden()

        self.optimizer = optim.Adam(self.parameters(), lr=6e-4)

    def forward(self, x, h):
        x = torch.unsqueeze(x,0)

        x_lstm, h = self.lstm(x, h)
        x = self.hidden_to_hyper(x_lstm.view(len(x_lstm),-1))

        return x,h

    def return_NNlayer(self, x: torch.Tensor, h: torch.Tensor) -> tuple:
        """
        Return a layer depending of the distribution given by the RNN's output

        :param x: tensor
            The output given by the RNN
        :param h: tensor
            The hidden state of the Rnn
        :return:
            The output and hidden state at t+1 with the selected layer and its probability
        """
        x, h = self(x, h)

        prob = F.softmax(x, dim=-1).squeeze(dim=0)
        idx = torch.distributions.Categorical(logits=prob).sample()


        return x, h, self.type_layers[int(idx)], prob[int(idx)]

    def generate_NNstring(self, nb_layer: int) -> tuple:
        """
        Generate a string coresponding to an architecture to build
        :param nb_layer: int
            Number of layers needed to generate the architecture
        :return:
            A string in a form of list designating the architecture's layers to generate, with a list of its associated
            probabilities
        """
        nn_str = []
        prob_list = []

        for _ in range(nb_layer):
            self.x, self.h, layer, prob = self.return_NNlayer(self.x, self.h)
            nn_str.append(layer)
            prob_list.append(prob) #to remove the grad_fn field

        return nn_str, torch.tensor(prob_list)

    def build_net(self, string_layer: list) -> Net.Net:
        """

        :param string_layer:
        :return:
        """
        net = Net.Net(string_layer)

        return net


    def validate_net(self, model: Net.Net) -> float:
        """
        Method which will train and validate the network

        :param model: Net.Net
            The model to train and validate
        :return:
            The accuracy of the validation
        """
        loaders = self._get_dataloaders()
        model.train_model(loaders['train'])
        accuracy = model.test_model(loaders['test'])

        return accuracy

    def reinforce(self, prob: list, reward: float) -> None:
        """
        Method implementing the REINFORCE method of

        :param prob: list
            List of probabilities of each layers of the generated net in form of tensor
        :param reward: float
            The accuracy of the net after validation
        :return: None
        """
        # Keep in memory every network its action
        # Build a tensor on the prob of those actions
        # Build the network, train it and return its accuracy of testing step
        # Compute the log of this prob and multiply it with the reward
        # do the gradient ascent

        #log_prob = np.log(prob)
        self.acc_list.append(reward)
        #self.loss = -torch.tensor(np.sum(log_prob * reward),requires_grad=True) \
        #            / len(log_prob)
        #G = torch.ones(1) * reward

        self.loss = torch.sum(-torch.log(prob) * reward).requires_grad_() / len(prob) #tester - et + log
        self.loss_list.append(self.loss.item())

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()


    def iter(self, nb_layer: int) -> tuple:
        """
        Simulate one iteration, which is a generation of a net with a given numbre of layers,
        its training and validation and the learning of the controller

        :param nb_layer: int
            Numbrer of layers of the generated nets
        :return: tuple
            Returns the models with its associated accuracy considered as the reward
        """
        nn_str, prob_list = rnn.generate_NNstring(nb_layer)
        model = rnn.build_net(nn_str)
        r = rnn.validate_net(model)
        self.reinforce(prob_list, r)

        return model, r

    def run(self, iteration: int, nb_layers: int) -> None:
        """
        Main method which will for a given number of iteration generated several net and reinforce the controller
        depending of the accuracy of the nets

        :param iteration: int
            Number of iteration i.e number of net to generate
        :param nb_layers: int
            Number of layers per net
        :return: None
        """
        print(f'Generating {iteration} CNN of {nb_layers} layers...')
        self.loss = 0
        self.loss_list = []
        self.acc_list = []
        best_model = None
        best_r = 0
        best_iter = 0
        for i in range(iteration):
            model, r = self.iter(nb_layers)
            if r > best_r:
                best_r = r
                best_model = model
                best_iter = i
            print(f"RNN loss: {self.loss:>7f}  [{i+1:>3d}/{iteration:>3d}]")
            print("=========================================================")

        print("\nEnd of iteration loss =", f"{self.loss.item():>7f}","----------")
        print("Best model at iteration",best_iter,":", best_model)
        print("With Accurary of", f"{best_r*100:>0.1f}%")

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
        train_data = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )

        #5000
        test_data = datasets.MNIST(
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


    def _init_layers(self, f_height = F_HEIGHT, f_width = F_WIDTH,
                     n_filter = N_FILTERS, n_strides = N_STRIDES) -> dict:
        """
        Generate all the combination of hyperparameters sets

        :param f_height: list
            Filter height hyperparameter list
        :param f_width: list
            Filter width hyperparameter list
        :param n_filter: list
            Number of filter hyperparameter list
        :param n_strides: list
            Number of strides hyperparameter list
        :return: dict
            Return a dictionnary with in the form ID : layers where ID = [0,N] where N is the number of all possible
            combination
        """
        a = [f_height, f_width, n_filter, n_strides]
        a = list(itertools.product(*a))
        dict = {}
        for i in range(len(a)):
            dict[i] = a[i]
        return dict

    def _init_hidden(self, r1=-0.8, r2=0.8) -> tuple:
        """
        Initialize the hidden states of the controller

        :param r1:
        :param r2:
        :return: tuple
            Hidden states of the controller
        """
        return (torch.FloatTensor(2, 1, self.hidden_size).uniform_(r1, r2),
                torch.FloatTensor(2, 1, self.hidden_size).uniform_(r1, r2))


if __name__ == '__main__':
    nb_net = 50
    nb_layers = 4
    rnn = RNN(HIDDEN_SIZE)
    rnn.run(nb_net,nb_layers)
    accuracy_plot(rnn.acc_list, nb_net, nb_layers, seed)
    loss_plot(rnn.loss_list, nb_net, nb_layers, seed)
    #print(nn_str)
