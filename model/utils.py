import math
import torch
from torch import nn, Tensor


class MLP(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_hid: int,
            d_out: int = None,
            num_layers: int = 3,
            dropout: float = 0.5
    ):
        super().__init__()
        if d_out is None:
            d_out = d_hid

        layers = []
        layers.append(nn.Linear(d_model, d_hid))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(d_hid, d_hid))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(d_hid, d_out))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.5,
        max_len: int = 10_000,
        batch_first: bool = False
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        position = torch.arange(max_len).unsqueeze(1)   # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        if self.batch_first:
            seq_len = x.size(1)
            x = x + self.pe[:seq_len].transpose(0, 1)
        else:
            seq_len = x.size(0)
            x = x + self.pe[:seq_len]

        return self.dropout(x)
