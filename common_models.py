import torch

from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SelfAttention(nn.Module):
    """Self Attention Layer"""
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        query = self.query(x).view(batch_size, seq_len, self.hidden_dim)
        key = self.key(x).view(batch_size, seq_len, self.hidden_dim)
        value = self.value(x).view(batch_size, seq_len, self.hidden_dim)

        attention_weights = torch.bmm(query, key.transpose(1, 2))
        attention_weights = torch.softmax(attention_weights, dim=2)

        attended_values = torch.bmm(attention_weights, value)

        return attended_values


class GRUWithLinear(torch.nn.Module):
    """Implements a GRU with Linear Post-Processing."""
    def __init__(self, indim, hiddim, outdim, dropout=False, dropoutp=0.1, flatten=False, has_padding=False, output_each_layer=False, batch_first=False):
        """Initialize GRUWithLinear Module.
        Args:
            indim (int): Input Dimension
            hiddim (int): Hidden Dimension
            outdim (int): Output Dimension
            dropout (bool, optional): Whether to apply dropout or not. Defaults to False.
            dropoutp (float, optional): Dropout probability. Defaults to 0.1.
            flatten (bool, optional): Whether to flatten output before returning. Defaults to False.
            has_padding (bool, optional): Whether input has padding. Defaults to False.
            output_each_layer (bool, optional): Whether to return the output of every intermediate layer. Defaults to False.
            batch_first (bool, optional): Whether to apply batching before GRU. Defaults to False.
        """
        super(GRUWithLinear, self).__init__()
        self.gru = nn.GRU(indim, hiddim, batch_first=batch_first)
        self.linear = nn.Linear(hiddim, outdim)
        self.dropout = dropout
        self.dropout_layer = torch.nn.Dropout(dropoutp)
        self.flatten = flatten
        self.has_padding = has_padding
        self.output_each_layer = output_each_layer
        self.lklu = nn.LeakyReLU(0.2)
        self.attention = SelfAttention(hiddim)

    def forward(self, x):
        """Apply GRUWithLinear to Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        if self.has_padding:
            x = pack_padded_sequence(
                x[0], x[1], batch_first=True, enforce_sorted=False)
            hidden = self.gru(x)[1][-1]
        else:
            hidden = self.gru(x)[0]
        if self.dropout:
            hidden = self.dropout_layer(hidden)
        out = self.linear(hidden)
        if self.flatten:
            out = torch.flatten(out, 1)
        if self.output_each_layer:
            return [0, torch.flatten(x, 1), torch.flatten(hidden, 1), self.lklu(out)]
        return out


class MLP(torch.nn.Module):
    """Two layered perceptron."""

    def __init__(self, indim, hiddim, outdim, dropout=False, dropoutp=0.1, output_each_layer=False):
        """Initialize two-layered perceptron.

        Args:
            indim (int): Input dimension
            hiddim (int): Hidden layer dimension
            outdim (int): Output layer dimension
            dropout (bool, optional): Whether to apply dropout or not. Defaults to False.
            dropoutp (float, optional): Dropout probability. Defaults to 0.1.
            output_each_layer (bool, optional): Whether to return outputs of each layer as a list. Defaults to False.
        """
        super(MLP, self).__init__()
        self.fc = nn.Linear(indim, hiddim)
        self.fc2 = nn.Linear(hiddim, outdim)
        self.dropout_layer = torch.nn.Dropout(dropoutp)
        self.dropout = dropout
        self.output_each_layer = output_each_layer
        self.lklu = nn.LeakyReLU(0.2)

    def forward(self, x):
        """Apply MLP to Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        output = F.relu(self.fc(x))
        if self.dropout:
            output = self.dropout_layer(output)
        output2 = self.fc2(output)
        if self.dropout:
            output2 = self.dropout_layer(output)
        if self.output_each_layer:
            return [0, x, output, self.lklu(output2)]
        return output2