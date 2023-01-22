import itertools
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import torch.optim as optim

import Net

# Possible outputs of RNN
#OUTPUT_SIZE = 191
F_HEIGHT = [1,3,5,7]
F_WIDTH = [1,3,5,7]
N_FILTERS = [24,36,48,64]
N_STRIDES = [1]

# LSTM parameters
HIDDEN_SIZE = 35
N_LAYER = 2

#torch.manual_seed(1)


"""
"""
class RNN(nn.Module):
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

        x, h = self(x, h)

        idx = torch.distributions.Categorical(logits=x).sample()

        return x, h, self.type_layers[int(idx)]

    def generate_NNstring(self, nb_layer):
        nn_str = []
        x = torch.zeros(self.input_size).unsqueeze(dim=0) #lstm need dim 3 so we dim 2 then dim 3
        h = self._init_hidden()
        for _ in range(nb_layer):
            x, h, layer = self.return_NNlayer(x, h)
            nn_str.append(layer)

        return nn_str

    def build_net(self, string_layer):
        print(string_layer)
        self.net = Net.Net(string_layer)
        print(self.net)


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

    def validate_net(self):
        loaders = self._get_dataloaders()
        self.net.train_model(loaders['train'])
        self.net.test_model(loaders['test'])

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

    def test_LSTM(self):
        inputs = [torch.randn(1, 1) for _ in range(100)]
        #print(inputs)
        inputs = torch.cat(inputs).view(len(inputs), 1, -1)
        #print(inputs)
        hidden = self._init_hidden()  # clean out hidden state
        out, hidden = self.lstm(inputs, hidden)
        #print(out)
        out_space = self.hidden_to_hyper(out.view(len(inputs),-1))
        out_scores = F.softmax(out_space,dim=1)
        #print(out_scores)


if __name__ == '__main__':
    rnn = RNN(HIDDEN_SIZE)
    nn_str = rnn.generate_NNstring(4)
    rnn.build_net(nn_str)
    #rnn.train_net()
    #rnn.test_net()
    rnn.validate_net()
    #print(nn_str)
