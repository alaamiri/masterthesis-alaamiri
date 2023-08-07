import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from ao import AbsAO
from .rnn import RNN

HIDDEN_SIZE = 35
N_LAYER = 2

class NASNET(AbsAO):
    def __init__(self, ss) -> None:
        super(NASNET, self).__init__(ss)

        self.rnn = RNN(len(self.search_space), HIDDEN_SIZE, N_LAYER)
        
        self.input_size = len(ss)
        self.rnn_x = torch.zeros(self.input_size).unsqueeze(dim=0)  # lstm need dim 3 so we dim 2 then dim 3
        self.rnn_h = self.rnn.init_hidden()
        
        self.prev_ema = -1
        self.alpha = 0.85
    
    
    def return_op(self, x, h):
        """
        Return a layer depending of the distribution given by the RNN's output

        :param x: tensor
            The output given by the RNN
        :param h: tensor
            The hidden state of the Rnn
        :return:
            The output and hidden state at t+1 with the selected layer and its probability
        """
        x, h = self.rnn(x, h)

        prob = F.softmax(x, dim=-1).squeeze(dim=0)
        idx = torch.distributions.Categorical(logits=prob).sample()
        
        return x, h, self.search_space[int(idx)], prob[int(idx)]
    
    def reset_param(self):
        self.rnn_h = self.rnn.init_hidden()
    
    def generate_arch(self, n_ops, reset_param = True):
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

        if reset_param:
            self.rnn_x = torch.zeros(self.input_size).unsqueeze(dim=0)  # lstm need dim 3 so we dim 2 then dim 3
            self.rnn_h = self.rnn.init_hidden()
        
        for _ in range(n_ops):
            self.rnn_x, self.rnn_h, layer, prob = self.return_op(self.rnn_x, self.rnn_h)
            nn_str.append(layer)
            prob_list.append(prob)

        return nn_str, torch.tensor(prob_list)
    
    
    def ema(self, r, prev_ema):
        return self.alpha * r + (1-self.alpha) * prev_ema


    def update(self, prob: list, reward: float) -> None :
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
        #val = torch.sum(torch.log(prob) * reward).requires_grad_() / len(prob)
        self.rnn.loss = -val
        #print("loss ", self.loss)

        self.prev_ema = self.curr_ema

        self.rnn.optimizer.zero_grad()
        self.rnn.loss.backward()

        self.rnn.optimizer.step()
        #input()

        return self.rnn.loss.item()
