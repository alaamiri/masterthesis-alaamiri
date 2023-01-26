import itertools

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import torch.optim as optim

import Net

# The RNN will output a layers depending the combination of those hyperparameter
F_HEIGHT = [1,3,5,7]
F_WIDTH = [1,3,5,7]
N_FILTERS = [24,36,48,64]
N_STRIDES = [1]

# LSTM parameters
HIDDEN_SIZE = 35
N_LAYER = 2

#torch.manual_seed(1)

class RNN(nn.Module):
    """
    A class representing the controller which generate the CNN depending of the search space
    """
    def __init__(self, hidden_size):
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

        self.optimizer = optim.Adam(self.parameters(), lr=6e-4)

    def forward(self, x, h):
        x = torch.unsqueeze(x,0)

        x_lstm, h = self.lstm(x, h)
        x = self.hidden_to_hyper(x_lstm.view(len(x_lstm),-1))

        return x,h

    def return_NNlayer(self, x, h):
        """
        Return a layer depending of the distribution given by the RNN's output

        :param x: tensor
            The output given by the RNN
        :param h: tensor
            The hidden state of the Rnn
        :return:
            The output and hidden state at t+1 with the selected layer
        """
        x, h = self(x, h)

        idx = torch.distributions.Categorical(logits=x).sample()
        prob = F.softmax(x, dim=-1).squeeze(dim=0)


        return x, h, self.type_layers[int(idx)], prob[int(idx)]

    def generate_NNstring(self, nb_layer):
        """
        Generate a string coresponding to an architecture to build
        :param nb_layer: int
            Number of layers needed to generate the architecture
        :return:
            A string in a form of list designating the architecture to generate
        """
        nn_str = []
        prob_list = []
        # Initializing the tensor for the RNN
        x = torch.zeros(self.input_size).unsqueeze(dim=0) #lstm need dim 3 so we dim 2 then dim 3
        h = self._init_hidden()
        for _ in range(nb_layer):
            x, h, layer, prob = self.return_NNlayer(x, h)
            nn_str.append(layer)
            prob_list.append(prob)

        return nn_str, prob_list

    def build_net(self, string_layer):
        """

        :param string_layer:
        :return:
        """
        print(string_layer)
        net = Net.Net(string_layer)
        print(net)

        return net


    def validate_net(self, model):
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

    def reinforce(self, prob, reward):
        # Keep in memory every network its action
        # Build a tensor on the prob of those actions
        # Build the network, train it and return its accuracy of testing step
        # Compute the log of this prob and multiply it with the reward
        # do the gradient ascent
        log_prob = np.log(prob)
        loss = np.sum(log_prob * reward)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()






    def run(self, nb_layer=2):
        nn_str, prob_list = rnn.generate_NNstring(nb_layer)
        model = rnn.build_net(nn_str)
        r = rnn.validate_net(model)
        self.reinforce(prob_list, r)

    def _get_dataloaders(self, batch_size=64, data_type="MNIST"):
        train_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )

        test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
        )

        loaders = {
            'train': DataLoader(train_data,
                                batch_size=batch_size),

            'test': DataLoader(test_data,
                               batch_size=batch_size),
        }

        return loaders


    def _init_layers(self, f_height = F_HEIGHT, f_width = F_WIDTH,
                     n_filter = N_FILTERS, n_strides = N_STRIDES):
        a = [f_height, f_width, n_filter, n_strides]
        a = list(itertools.product(*a))
        dict = {}
        for i in range(len(a)):
            dict[i] = a[i]
        return dict

    def _init_hidden(self, r1=-0.8, r2=0.8):
        return (torch.FloatTensor(2, 1, self.hidden_size).uniform_(r1, r2),
                torch.FloatTensor(2, 1, self.hidden_size).uniform_(r1, r2))


if __name__ == '__main__':
    rnn = RNN(HIDDEN_SIZE)
    rnn.run()
    #print(nn_str)
