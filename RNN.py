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
N_STRIDES = [1,2,3]

# LSTM parameters
HIDDEN_SIZE = 35
N_LAYER = 2

#torch.manual_seed(1)


"""
"""
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, train_size):
        super(RNN, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.type_layers = self._init_layers()
        print("# of possible layers :", len(self.type_layers))
        self.output_size = len(self.type_layers)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=N_LAYER)
        self.hidden_to_hyper = nn.Linear(hidden_size, self.output_size)

        self.optimizer = optim.Adam(self.parameters(), lr=6e-4)

    def forward(self, x, h):
        x = torch.cat(x).view(len(x), 1, -1)

        x_lstm, h = self.lstm(x, h)
        x = self.hidden_to_hyper(x_lstm.view(len(x_lstm),-1))

        return x,h

    def return_NNlayer(self):
        inputs = [torch.randn(1, 1) for _ in range(1)]
        x, h = self(inputs, (torch.FloatTensor(2, 1, 35).uniform_(-0.8, 0.8),
                            torch.FloatTensor(2, 1, 35).uniform_(-0.8, 0.8)))
        #print("x :", x.size())
        #print("h :", h[0].size(), h[1].size())

        idx = torch.distributions.Categorical(logits=x).sample()
        #print(idx)

        return self.type_layers[int(idx)]

    def generate_NNstring(self, nb_layer):
        nn_str = []
        for _ in range(nb_layer):
            nn_str.append(self.return_NNlayer())

        return nn_str

    def build_net(self, string_layer):
        print(string_layer)
        self.net = Net.Net(string_layer)
        print(self.net)

    def train_net(self, dataset = "MNIST"):
        training_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )

        batch_size = 64
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=1)

        size = len(train_dataloader.dataset)
        self.net.train()
        for batch, (X, y) in enumerate(train_dataloader):
            #X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = self.net(X)
            print(pred.size())
            print(y.size())
            loss = self.net.loss_fn(pred, y)

            # Backpropagation
            self.net.optimizer.zero_grad()
            loss.backward()
            self.net.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test_net(self):
        # Download test data from open datasets.
        test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
        )

        batch_size = 64
        test_dataloader = DataLoader(test_data, batch_size=batch_size)
        for X, y in test_dataloader:
            print(f"Shape of X [N, C, H, W]: {X.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break

        size = len(test_dataloader.dataset)
        num_batches = len(test_dataloader)
        self.net.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in test_dataloader:
                #X, y = X.to(device), y.to(device)
                pred = self.net(X)
                test_loss += self.net.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

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
    rnn = RNN(1, HIDDEN_SIZE, 10)
    nn_str = rnn.generate_NNstring(4)
    rnn.build_net(nn_str)
    #rnn.train_net()
    #rnn.test_net()
    rnn.validate_net()
    #print(nn_str)
