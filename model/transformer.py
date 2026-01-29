import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .utils import MLP, PositionalEncoding


class HIMTransfomerNet(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hid: int = 128,
        n_heads: int = 2,
        n_layers: int = 2,
        dropout: float = 0.5,
        batch_first: bool = True,
        mlp_kwargs: dict = {},
        pe_kwargs: dict = {}
    ):
        super().__init__()

        # Set default MLP and Positional Encoding parameters
        mlp_kwargs.setdefault('d_model', d_model)
        mlp_kwargs.setdefault('d_hid', d_hid)
        mlp_kwargs.setdefault('dropout', dropout)

        pe_kwargs.setdefault('d_model', d_hid)
        pe_kwargs.setdefault('dropout', dropout)
        pe_kwargs.setdefault('batch_first', batch_first)

        # Initialize modules
        self.mlp_encoder = MLP(**mlp_kwargs)
        self.pos_encoder = PositionalEncoding(**pe_kwargs)

        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=d_hid,
                nhead=n_heads,
                dim_feedforward=d_hid * 2,
                dropout=dropout,
                batch_first=batch_first
            ),
            num_layers=n_layers
        )

    def forward(self, src: Tensor) -> Tensor:
        src = self.mlp_encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output
