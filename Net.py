import itertools
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

class Net(nn.Module):
    def __init__(self, string_layers):
        super(Net, self).__init__()
        self.expected_input_size = (1, 1, 28, 28)
        self.string_layers = string_layers

        self.net = self._init_net()
        self.out = self._init_linear()

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)

    def __repr__(self):
        return self.net.__repr__()

    def _init_net(self):
        layers = []
        prev_channel = 1
        for layer in self.string_layers:
            layers.append(nn.Conv2d(in_channels=prev_channel,
                                    out_channels=layer[2],
                                    kernel_size=(layer[0], layer[1]), #kernel_size=(layer[0], layer[1])
                                    stride=layer[3],    #stride=layer[3]
                                    padding=0))
            prev_channel = layer[2]
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def _init_linear(self, output_features = 10):
        conv_out = self._get_output_shape(self.net, self.expected_input_size)
        # self.out = nn.Linear(prev_channel * 12 * 12, 10)
        return nn.Linear(conv_out, out_features=output_features)

    """
    code from https://discuss.pytorch.org/t/linear-layer-input-neurons-number-calculation-after-conv2d/28659/3
    """
    def _get_output_shape(self,model, image_dim):
        return np.prod(model(torch.rand(*(image_dim))).data.shape)

    def forward(self, x):
        y = self.net(x)
        #print(y.size())
        y = y.view(y.size(0),-1)
        #print(y.size())
        out_y = self.out(y)

        return out_y

    def train_model(self, dataloader):
        size = len(dataloader.dataset)
        self.net.train()
        for batch, (X, y) in enumerate(dataloader):
            # X, y = X.to(self.device), y.to(self.device)
            # Compute prediction error
            #print(X.size())
            pred = self(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test_model(self, dataloader):
        for X, y in dataloader:
            print(f"Shape of X [N, C, H, W]: {X.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break

        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                # X, y = X.to(device), y.to(device)
                pred = self(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        print("For model: \n", self)

        return correct

