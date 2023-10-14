import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
import random
import argparse
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data.dataloader import DataLoader

from transformers import LongformerModel, LongformerConfig
from transformers import BigBirdModel, BigBirdConfig
from LncRNA.model.mybigbird import MYConfig, Mymodel

import dataset



if __name__ == '__main__':

    argp = argparse.ArgumentParser()
    ## model argument
    argp.add_argument('--model',
        help="Choose one of the models", default="Bigbert", choices=["Bigbert", "Longformer"])
    argp.add_argument('--writing_params_path',
        help="Path to save the model after pretraining/finetuning", default=None)
    ## data argument
    argp.add_argument('--Coding_RNA_path', default="data/train_coding.csv")
    argp.add_argument('--Lnc_RNA_path', default="data/train_lnc.csv")
    argp.add_argument('--batch_size', default=4)

    argp.add_argument('--outputs_path', default=None)
    args = argp.parse_args()

    # Save the device
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    ## Load dataset
    Coding_RNA_dataset = dataset.RNADataset(open(args.Coding_RNA_path, encoding='UTF-8').read(), lnc=False)
    Lnc_RNA_dataset = dataset.RNADataset(open(args.Lnc_RNA_path, encoding='UTF-8').read(), lnc=True)
    RNA_dataset = dataset.CombineDataset(Coding_RNA_dataset, Lnc_RNA_dataset)

    dataloader = DataLoader(RNA_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True,
                        collate_fn=RNA_dataset.collate_func)
    # Load model
    if args.model == "Longformer":
        config = LongformerConfig(num_hidden_layers=2, num_attention_heads=4, hidden_size=128, max_position_embeddings=4096)
        config.attention_mode = 'tvm'   # choose the attention mode 'n2', 'tvm' or 'sliding_chunks'
        Lformer = LongformerModel(config)
        mconf = MYConfig(model=Lformer, n_embd=config.hidden_size, num_class=2, pretrain=False)
        model = Mymodel(mconf)

    elif args.model == "Bigbert":
        # model = BigBirdModel.from_pretrained("google/bigbird-roberta-large",num_labels=2, block_size=16, num_random_blocks=2)
        config = BigBirdConfig(num_hidden_layers=2, num_attention_heads=4, hidden_size=128, max_position_embeddings=4096)
        Bigbert = BigBirdModel(config)
        mconf = MYConfig(model=Bigbert, n_embd=config.hidden_size, num_class=2, pretrain=False)
        model = Mymodel(mconf)


    num_epochs = 3
    num_training_steps = num_epochs * len(dataloader)
    optimizer = AdamW(model.parameters(), lr=6e-5, weight_decay=0.1)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=600, num_training_steps=num_training_steps)

    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        acc = 0
        train_num = 0
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        # training
        model.train()
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for batch_i, batch in pbar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            input = dict(input_ids=input_ids, attention_mask=attention_mask) # labels=labels
            model.to(device)

            outputs = model(labels, **input)
            logits = outputs[0]
            loss = outputs[1]
            loss = loss.mean()
            predict_y = torch.max(logits, dim=1)[1]
            acc += torch.eq(predict_y, labels).sum().item()
            train_num += input_ids.shape[0]
            acc_cum = acc / train_num

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            pbar.set_description(f"epoch {epoch + 1} iter {batch_i}: train loss {loss.item():.5f} train cumulate acc {acc_cum:.5f}.")

        # # validation
        # model.eval()
        # loss = 0
        # for batch_i, batch in enumerate(eval_dataloader):
        #     with torch.no_grad():
        #         output = model(**batch)
        #     loss += output.loss
        #
        # avg_val_loss = loss / len(eval_dataloader)
        # print(f"Validation loss: {avg_val_loss}")
        # if avg_val_loss < best_val_loss:
        #     print("Saving checkpoint!")
        #     best_val_loss = avg_val_loss
            # torch.save({
            #    'epoch': epoch,
            #    'model_state_dict': model.state_dict(),
            #    'optimizer_state_dict': optimizer.state_dict(),
            #    'val_loss': best_val_loss,
            #    },
            #    f"checkpoints/epoch_{epoch}.pt"
            # )