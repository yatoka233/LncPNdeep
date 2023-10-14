import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import LongformerModel, LongformerConfig
from transformers import BigBirdModel, BigBirdConfig
from utils import sigmoid_focal_loss

class MYConfig:
    """ base GPT config, params common to all GPT versions """

    def __init__(self, n_embd, num_class, **kwargs):
        self.n_embd = n_embd
        self.num_class = num_class

        for k,v in kwargs.items():
            setattr(self, k, v)

class Modeloutput:
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Mymodel(nn.Module):
    def __init__(self, config):
        ## config: model, num_class, n_embd
        super().__init__()
        if config.model == "Bigbird":
            self.model = BigBirdModel(config.config)
        if config.model == "Longformer":
            self.model = LongformerModel(config.config)
        self.pretrain = config.pretrain ## control trainning mode: train or pretrain
        self.num_class = config.num_class

        self.dropout = nn.Dropout(0.1)
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.dense = nn.Linear(config.n_embd, config.n_embd)
        self.head = nn.Linear(config.n_embd, self.num_class)

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

    def forward(self, input_ids, labels, attention_mask=None):
        b, t = input_ids.size()
        input = dict(input_ids=input_ids, attention_mask=attention_mask)
        outputs = self.model(**input, return_dict=True)
        # pooler_logits = outputs.pooler_output
        last_h = outputs.last_hidden_state  # [b, t, h]
        pooler = outputs.pooler_output  # [b, h]

        # logits = self.dropout(last_h)
        # logits = self.dense(logits)
        # logits = F.relu(logits)
        # logits = self.dropout(logits)
        # logits = self.head(logits)
        if not self.pretrain:
            # logits = logits[:, 0, :]
            logits = self.head(pooler)
            loss = None
            ## cross entropy
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            ## focal loss
            # loss = sigmoid_focal_loss(logits.view(b, self.num_class).gather(1, label.view(-1,1)).view(b), label.float())
        else:
            logits = self.head(last_h)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=0)

        output = Modeloutput(logits=logits, loss=loss, last_h=last_h)
        return output
