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

import pre_dataset
import model.bert
import trainer


if __name__ == '__main__':

    argp = argparse.ArgumentParser()
    ## model argument
    argp.add_argument('--model',
        help="Choose one of the models", default="GPT")
    argp.add_argument('--reading_params_path',
        help="If specified, path of the model to load before finetuning/evaluation",
        default=None)
    argp.add_argument('--writing_params_path',
        help="Path to save the model after pretraining/finetuning", default="weights/vanilla.train.params")
    ## data argument
    argp.add_argument('--Train_Coding_path', default="data/train_coding.csv")
    argp.add_argument('--Train_Lnc_path', default="data/train_lnc.csv")
    argp.add_argument('--Valid_Coding_path', default="data/valid_coding.csv")
    argp.add_argument('--Valid_Lnc_path', default="data/valid_lnc.csv")

    argp.add_argument('--outputs_path', default=None)
    args = argp.parse_args()

    # Save the device
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    ## Load dataset
    Train_Coding_dataset = pre_dataset.Pre_RNADataset(open(args.Train_Coding_path, encoding='UTF-8').read(), lnc=False)
    Train_Lnc_dataset = pre_dataset.Pre_RNADataset(open(args.Train_Lnc_path, encoding='UTF-8').read(), lnc=True)
    Train_dataset = pre_dataset.Pre_CombineDataset(Train_Coding_dataset, Train_Lnc_dataset)

    Valid_Coding_dataset = pre_dataset.Pre_RNADataset(open(args.Valid_Coding_path, encoding='UTF-8').read(), lnc=False)
    Valid_Lnc_dataset = pre_dataset.Pre_RNADataset(open(args.Valid_Lnc_path, encoding='UTF-8').read(), lnc=True)
    Valid_dataset = pre_dataset.Pre_CombineDataset(Valid_Coding_dataset, Valid_Lnc_dataset)

    # Load model
    if args.model == "GPT":
        mconf = model.gpt.BERTConfig(Train_dataset.vocab_size, max_size=2100, num_class=67,
                                     n_layer=2, n_head=4, n_embd=128, pretrain=True, Mask=True) # 4 8 256

        model = model.gpt.BERT(mconf)

    ## Training
    if args.reading_params_path is None:
        train_config = trainer.TrainerConfig(max_epochs=10, batch_size=1, learning_rate=6e-4, lr_decay=True,
                                             warmup_tokens=int(0.1 * Train_dataset.total_chars),
                                             final_tokens=int(0.9 * Train_dataset.total_chars),
                                             ckpt_path=args.writing_params_path, num_workers=0, pretrain=True) ## 75
    else:
        train_config = trainer.TrainerConfig(max_epochs=10, batch_size=1, learning_rate=6e-4, lr_decay=True,
                                             warmup_tokens=int(0.1 * Train_dataset.total_chars),
                                             final_tokens=int(0.9 * Train_dataset.total_chars),
                                             ckpt_path=args.writing_params_path, num_workers=0, pretrain=True)
        model.load_state_dict(torch.load(args.reading_params_path))

    Mytrainer = trainer.Trainer(model, Train_dataset, Valid_dataset, train_config)
    Mytrainer.train()