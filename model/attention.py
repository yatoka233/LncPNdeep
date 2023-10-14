import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class CausalSelfAttention(nn.Module):


    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        if config.Mask:
            self.register_buffer("mask", torch.tril(torch.ones(config.max_size, config.max_size))
                                        .view(1, 1, config.max_size, config.max_size))
        self.n_head = config.n_head
        self.Mask = config.Mask

    def forward(self, x, pad_mask):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if self.Mask:
            att = att.masked_fill(self.mask[:,:,:T,:T] == 0, -1e10) # todo: just use float('-inf') instead?
        pad_mask = torch.unsqueeze(torch.unsqueeze(pad_mask, 1),3)  ## (B, T) -> (B, 1, T, 1)
        pad_mask = pad_mask.expand(B, self.n_head, T, T)  ## (B, 1, T, 1) -> (B, nh, T, T)
        pad_mask_T = pad_mask.transpose(-2, -1)
        pad_mask = pad_mask * pad_mask_T
        att = att.masked_fill(pad_mask[:,:,:T,:T] == 0, -1e10) # todo: just use float('-inf') instead?
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

