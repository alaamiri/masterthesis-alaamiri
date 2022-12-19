import itertools
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import torch.optim as optim

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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, train_size):
        super(RNN, self).__init__()
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

    def return_NNstring(self):
        inputs = [torch.randn(1, 1) for _ in range(1)]
        x, h = self(inputs, (torch.FloatTensor(2, 1, 35).uniform_(-0.8, 0.8),
                            torch.FloatTensor(2, 1, 35).uniform_(-0.8, 0.8)))
        print("x :", x.size())
        print("h :", h[0].size(), h[1].size())

        idx = torch.distributions.Categorical(logits=x).sample()
        print(idx)

        return self.type_layers[int(idx)]

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
    print(rnn.return_NNstring())

