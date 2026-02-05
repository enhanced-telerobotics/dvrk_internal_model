import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .utils import MLP, PositionalEncoding


class HIMTransfomerNet(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_out: int,
        d_hid: int = 128,
        n_heads: int = 2,
        n_layers: int = 2,
        dropout: float = 0.5,
        batch_first: bool = True,
        mlp_kwargs: dict = {},
        pe_kwargs: dict = {}
    ):
        super().__init__()
        self.batch_first = batch_first

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
                dim_feedforward=d_hid * 4,
                dropout=dropout,
                batch_first=batch_first
            ),
            num_layers=n_layers
        )

        self.head = nn.Linear(d_hid, d_out)

    def forward(self,
                src: Tensor,
                batch_mask: Tensor = None) -> Tensor:
        src = self.mlp_encoder(src)
        src = self.pos_encoder(src)

        if self.batch_first:
            T = src.size(1)
        else:
            T = src.size(0)

        causal_mask = torch.triu(
            torch.ones((T, T), device=src.device), diagonal=1).bool()
        
        output = self.transformer_encoder(
            src,
            mask=causal_mask,
            src_key_padding_mask=batch_mask)
        
        output = self.head(output)
        return output
