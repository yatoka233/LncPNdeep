
"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import sigmoid_focal_loss

import model.attention as attention


class BERTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    num_class = 2

    def __init__(self, vocab_size, max_size, **kwargs):
        self.vocab_size = vocab_size
        self.max_size = max_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class BERToutput:
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class BERT1Config(BERTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        self.attn = attention.CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, input):
        x = input[0]
        pad_mask = input[1]
        x = x + self.attn(self.ln1(x), pad_mask)
        x = x + self.mlp(self.ln2(x))
        return x, pad_mask

class BERT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()
        self.pretrain = config.pretrain
        self.num_class = config.num_class
        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_size, config.n_embd)) # position embedding parameter
                ## try sin cos position embedding ?
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, self.num_class)

        self.max_size = config.max_size
        self.apply(self._init_weights)

        print("number of parameters: {}".format(sum(p.numel() for p in self.parameters())))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_max_size(self):
        return self.max_size

    def forward(self, input_ids, labels, attention_mask):
        x, label, pad_mask = input_ids, labels, attention_mask
        b, t = x.size()
        assert t <= self.max_size, "Cannot forward, model block size is exhausted."

        # forward the BERT model
        token_embeddings = self.tok_emb(x) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        x, _ = self.blocks((x, pad_mask))
        last_h = x  # [b, t, h]
        x = self.ln_f(x)
        logits = self.head(x)   # [b, t, class]
        if not self.pretrain:
            logits = logits[:, 0, :]
            loss = None
            ## cross entropy
            # print("logits: ", logits)
            # print("label: : ", label)
            loss = F.binary_cross_entropy_with_logits(logits.view(b, self.num_class), label)
            ## focal loss
            # loss = sigmoid_focal_loss(logits.view(b, self.num_class).gather(1, label.view(-1,1)).view(b), label.float())
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), label.view(-1), ignore_index=0)

        return BERToutput(logits=logits, loss=loss, last_h=last_h)

