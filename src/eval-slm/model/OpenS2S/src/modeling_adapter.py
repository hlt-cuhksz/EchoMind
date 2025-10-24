import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

import logging

from .utils import length_to_attention_mask

logger = logging.getLogger(__name__)

class Conv1dSubsampler(nn.Module):
    """Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)
    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = (3, 3),
    ):
        super(Conv1dSubsampler, self).__init__()
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=2,
                padding=k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(self.n_layers):
            out = ((out.float() - 1) / 2 + 1).floor().long()
        return out

    def forward(self, src_tokens, src_lengths):
        bsz, in_seq_len, _ = src_tokens.size()  # B x T x (C x D)
        x = src_tokens.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        _, _, out_seq_len = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)
        return x, self.get_out_seq_lens_tensor(src_lengths)


class Subsampler(nn.Module):
    def __init__(
        self,
        in_dim: int,
        mid_dim: int,
        out_dim: int,
        conv_kernel_sizes: str="5,5,5",
    ):
        super(Subsampler, self).__init__()
        self.subsampler = Conv1dSubsampler(
            in_dim,
            2 * in_dim,
            out_dim,
            [int(k) for k in conv_kernel_sizes.split(",")],
        )

        self.fc1 = nn.Linear(out_dim, mid_dim, bias=False)
        self.fc2 = nn.Linear(mid_dim, out_dim, bias=False)
        self.activation = nn.GELU()
        self.ln = torch.nn.LayerNorm(out_dim, 1e-5, True)

    def forward(self, x, attention_mask):
        x, lengths = self.subsampler(x, attention_mask.sum(dim=-1)) # B x T x H -> T x B x C
        attention_mask = length_to_attention_mask(lengths)
        x = x.transpose(0,1) # T x B x C -> B x T x C

        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x) + residual

        x = self.ln(x)

        return x, attention_mask


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_dim: int,
        mid_dim: int,
        out_dim: int,
    ):
        super(Bottleneck, self).__init__()

        self.fc = nn.Linear(in_dim, out_dim, bias=False)

        self.fc1 = nn.Linear(out_dim, mid_dim, bias=False)
        self.fc2 = nn.Linear(mid_dim, out_dim, bias=False)
        self.activation = nn.GELU()
        self.ln = torch.nn.LayerNorm(out_dim, 1e-5, True)

    def forward(self, x, attention_mask=None):
        x = self.fc(x)
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x) + residual

        x = self.ln(x)

        return x, attention_mask