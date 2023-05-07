import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from search_spaces import search_space

# LSTM parameters
HIDDEN_SIZE = 35
N_LAYER = 2

EPOCHS = 5

# 1097.6585461702298
# Accuracy: 78.0%, Avg loss: 0.649093
"""
[[82768608. 57117771. 56746112. ... 57496766. 56501693. 56924064.]
 [57117771. 82768608. 56708583. ... 56745599. 56432870. 56861305.]
 [56746112. 56708583. 82768608. ... 56614470. 56420361. 56639482.]
 ...
 [57496766. 56745599. 56614470. ... 82768608. 56553243. 57011024.]
 [56501693. 56432870. 56420361. ... 56553243. 82768608. 57076075.]
 [56924064. 56861305. 56639482. ... 57011024. 57076075. 82768608.]]
"""

class RNN(nn.Module):
    """
    A class representing the controller which generate the CNN depending of the search space
    """
    def __init__(self, hidden_size: int, s_space : list):
        super(RNN, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")

        self.s_space = s_space

        self.input_size = len(self.s_space)
        self.hidden_size = hidden_size
        self.output_size = len(self.s_space)

        self.lstm = nn.LSTM(self.input_size, hidden_size, num_layers=N_LAYER)
        self.hidden_to_hyper = nn.Linear(hidden_size, self.output_size)

        self.x = torch.zeros(self.input_size).unsqueeze(dim=0)  # lstm need dim 3 so we dim 2 then dim 3
        self.h = self.init_hidden()

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        #self.optimizer = optim.Adam(self.parameters(), lr=6e-4)

        self.prev_ema = -1
        self.alpha = 0.85

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


        return x, h, self.s_space[int(idx)], prob[int(idx)]

    def generate_arch(self, nb_layer: int=4) -> tuple:
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
            prob_list.append(prob)

        return nn_str, torch.tensor(prob_list)

    def ema(self, r, prev_ema):
        return self.alpha * r + (1-self.alpha) * prev_ema

    def reinforce(self, prob: list, reward: float) -> None:
        """
        Method implementing the REINFORCE method of

        :param prob: list
            List of probabilities of each layers of the generated net in form of tensor
        :param reward: float
            The accuracy of the net after validation
        :return: None
        """

        #log_prob = np.log(prob)
        #self.acc_list.append(reward)
        #self.loss = -torch.tensor(np.sum(log_prob * reward),requires_grad=True) \
        #            / len(log_prob)
        #G = torch.ones(1) * reward

        if self.prev_ema == -1:
            self.prev_ema = reward

        #print("prev_ema ", self.prev_ema)
        self.curr_ema = self.ema(reward, self.prev_ema)
        #print("curr_ema ", self.ema(reward, self.prev_ema))
        val = torch.sum(torch.log(prob) * (reward-self.curr_ema)).requires_grad_()
        self.loss = -val
        #print("loss ", self.loss)

        self.prev_ema = self.curr_ema
        #input()

        return self.loss.item()


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


if __name__ == '__main__':
    nb_net = 1000

    nb_layers = 7
    rnn = RNN(HIDDEN_SIZE, s_space=search_space.nats_bench_tss, benchmark=True)

