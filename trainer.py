"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.

We suggest not changing anything in this file.
"""

import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, valid_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self, epoch):
        if self.config.ckpt_path is not None:
            ckpt_model = self.model.module if hasattr(self.model, "module") else self.model
            logger.info("saving %s", self.config.ckpt_path)
            torch.save(ckpt_model.state_dict(), self.config.ckpt_path + f".epoch{epoch + 1}.params")

    def train(self):
        model, config = self.model, self.config

        # create the optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
        params_nodecay = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
        optim_groups = [
            {"params": params_decay, "weight_decay": config.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]
        optimizer = optim.AdamW(optim_groups, lr=config.learning_rate, betas=config.betas)
        tb_writer = SummaryWriter()

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.valid_dataset
            loader = DataLoader(data, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True, collate_fn=data.collate_func)

            losses = []
            predict = None if is_train else []
            acc = 0
            train_num = 0
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, batch in pbar:
                # place data on the correct
                x = batch['input_ids'].to(self.device)
                batch_size = x.shape[0]
                label = batch['labels'].to(self.device)
                mask = batch['attention_mask'].to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    outputs = model(input_ids=x, labels=label, attention_mask=mask)
                    logits = outputs.logits
                    loss = outputs.loss
                    # loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())
                    if not self.config.pretrain:
                        predict_y = torch.max(logits, dim=1)[1]
                        # if is_train:
                            # print(logits)

                        acc += torch.eq(predict_y, label).sum().item()
                        train_num += batch_size
                        acc_cum = acc / train_num
                    else:
                        logits = logits.view(-1, logits.size(-1))
                        predict_y = torch.max(logits, dim=1)[1]
                        label = label.view(-1)
                        index = torch.nonzero(label)
                        acc += torch.eq(predict_y[index], label[index]).sum().item()
                        train_num += index.size(0)
                        acc_cum = acc / train_num


                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (x > 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    if is_train:
                        pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.4f} "
                                             f"train cumulate acc {acc_cum:.4f}. lr {lr:.4e}")
                    if not is_train:
                        pbar.set_description(f"epoch {epoch + 1} iter {it}: valid loss {loss.item():.4f} "
                                             f"valid cumulate acc {acc_cum:.4f}. lr {lr:.4e}")

            ## after one epoch
            if is_train:
                tags = ["loss", "accuracy", "learning_rate"]
                tb_writer.add_scalar(tags[0], np.mean(losses), epoch+1)
                tb_writer.add_scalar(tags[1], acc_cum, epoch+1)
                tb_writer.add_scalar(tags[2], lr, epoch+1)


            if not is_train:
                logger.info("valid loss: %f", np.mean(losses))
                tags = ["valid_loss", "valid_accuracy"]
                tb_writer.add_scalar(tags[0], np.mean(losses), epoch + 1)
                tb_writer.add_scalar(tags[1], acc_cum, epoch + 1)
                print(f"epoch {epoch + 1} : valid loss {np.mean(losses):.5f} "f"valid cumulate acc {acc_cum:.5f}.")



        self.tokens = 0 # counter used for learning rate decay
        for epoch in range(config.max_epochs):

            run_epoch('train')
            if self.valid_dataset is not None:
                run_epoch('test')

            if (epoch+1)%1 == 0:
                self.save_checkpoint(epoch)
