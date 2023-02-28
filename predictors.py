import torch
import torch.nn as nn

"""
By ChatGPT
"""
class PeepholePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PeepholePredictor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

        # Initialize the peephole connections
        self.W_ci = nn.Parameter(torch.Tensor(hidden_size))
        self.W_cf = nn.Parameter(torch.Tensor(hidden_size))
        self.W_co = nn.Parameter(torch.Tensor(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.W_ci)
        nn.init.normal_(self.W_cf)
        nn.init.normal_(self.W_co)

    def forward(self, input, hidden=None):
        # If the hidden state is not provided, initialize it to zeros
        if hidden is None:
            hidden = (torch.zeros(1, self.hidden_size),
                      torch.zeros(1, self.hidden_size))

        # Compute the LSTM cell outputs and hidden states
        hx, cx = self.lstm_cell(input, hidden)

        # Compute the peephole connections
        i_gate = torch.sigmoid(hx + self.W_ci * cx)
        f_gate = torch.sigmoid(hx + self.W_cf * cx)
        o_gate = torch.sigmoid(hx + self.W_co * cx)

        # Compute the output
        output = self.linear(hx)

        # Return the output, hidden state, and peephole connections
        return output, (hx, cx, i_gate, f_gate, o_gate)
