import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

class RNN(nn.Module):
    """
    A class representing the controller which generate the CNN depending of the search space
    """
    def __init__(self, input_size: int, hidden_size: int, n_layers: int):
        super(RNN, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = input_size

        self.lstm = nn.LSTM(self.input_size, hidden_size, num_layers= n_layers)
        self.hidden_to_hyper = nn.Linear(hidden_size, self.output_size)

        self.optimizer = optim.Adam(self.lstm.parameters(), lr=1e-2)
        #self.optimizer = optim.Adam(self.parameters(), lr=6e-4)

        
    def forward(self, x, h):
        x = torch.unsqueeze(x,0)

        x_lstm, h = self.lstm(x, h)
        x = self.hidden_to_hyper(x_lstm.view(len(x_lstm),-1))

        return x,h
    

    def init_hidden(self, r1=-0.8, r2=0.8) -> tuple:
        """
        Initialize the hidden states of the controller

        :param r1:
        :param r2:
        :return: tuple
            Hidden states of the controller
        """
        return (torch.FloatTensor(2, 1, self.hidden_size).uniform_(r1, r2),
                torch.FloatTensor(2, 1, self.hidden_size).uniform_(r1, r2))
