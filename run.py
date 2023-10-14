import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
import random
import argparse

from transformers import LongformerModel, LongformerConfig, LongformerForSequenceClassification
from transformers import BigBirdModel, BigBirdConfig, BigBirdForSequenceClassification
from transformers import BertModel, BertConfig, BertForSequenceClassification

import dataset
from model.bert import BERTConfig, BERT
from model.mybigbird import MYConfig, Mymodel
import trainer

random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)


if __name__ == '__main__':

    argp = argparse.ArgumentParser()
    ## model argument
    argp.add_argument('--model',
        help="Choose one of the models", default="Bigbird", choices=["BERT", "Longformer", "Bigbird"])
    argp.add_argument('--reading_params_path',
        help="If specified, path of the model to load before finetuning/evaluation",
        default="/root/autodl-tmp/weights/pretrain/save.bigbird.pretrain.epoch20.params")
    argp.add_argument('--writing_params_path',
        help="Path to save the model after pretraining/finetuning", 
        default="/root/autodl-tmp/weights/train/bigbird.train")
    ## data argument
    argp.add_argument('--Train_Coding_path', default="data/coding_train_seq.txt")
    argp.add_argument('--Train_Lnc_path', default="data/lnc_train_seq.txt")
    argp.add_argument('--Valid_Coding_path', default="data/coding_valid_seq.txt")
    argp.add_argument('--Valid_Lnc_path', default="data/lnc_valid_seq.txt")

    argp.add_argument('--outputs_path', default=None)
    args = argp.parse_args()

    # Save the device
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    ## Load dataset
    Train_Coding_dataset = dataset.RNADataset(open(args.Train_Coding_path, encoding='UTF-8').readlines()[:-1], lnc=False)
    Train_Lnc_dataset = dataset.RNADataset(open(args.Train_Lnc_path, encoding='UTF-8').readlines()[:-1], lnc=True)
    Train_dataset = dataset.CombineDataset(Train_Coding_dataset, Train_Lnc_dataset)

    Valid_Coding_dataset = dataset.RNADataset(open(args.Valid_Coding_path, encoding='UTF-8').readlines()[:-1], lnc=False)
    Valid_Lnc_dataset = dataset.RNADataset(open(args.Valid_Lnc_path, encoding='UTF-8').readlines()[:-1], lnc=True)
    Valid_dataset = dataset.CombineDataset(Valid_Coding_dataset, Valid_Lnc_dataset)

    # Load model
    if args.model == "BERT":
        # mconf = BERTConfig(Train_dataset.vocab_size, max_size=2100,
        #                              n_layer=4, n_head=4, n_embd=256, pretrain=False, Mask=False) # 4 8 256

        # model = BERT(mconf)
        config = BertConfig(num_hidden_layers=2, num_attention_heads=4, hidden_size=128, max_position_embeddings=4096,
                               vocab_size=8,
                               eos_token_id=None,
                               sep_token_id=None)
        model = BertForSequenceClassification(config)

    elif args.model == "Longformer":
        config = LongformerConfig(num_hidden_layers=2, num_attention_heads=4, hidden_size=128, max_position_embeddings=4096,
                                  vocab_size=8,
                                  eos_token_id=None,
                                  sep_token_id=None)
        config.attention_mode = 'tvm'  # choose the attention mode 'n2', 'tvm' or 'sliding_chunks'
        model = LongformerForSequenceClassification(config)
        # mconf = MYConfig(model="Longformer", config=config, n_embd=config.hidden_size, num_class=2, pretrain=False)
        # model = Mymodel(mconf)

    elif args.model == "Bigbird":
        # model = BigBirdModel.from_pretrained("google/bigbird-roberta-large",num_labels=2, block_size=16, num_random_blocks=2)
        config = BigBirdConfig(num_hidden_layers=4, num_attention_heads=8, hidden_size=256, max_position_embeddings=4096,
                               vocab_size=76,
                               eos_token_id=None,
                               sep_token_id=None,
                               num_labels=2)
        model = BigBirdForSequenceClassification(config)
        # model = BigBirdForSequenceClassification.from_pretrained("l-yohai/bigbird-roberta-base-mnli", num_labels=2,ignore_mismatched_sizes=True)
        # mconf = MYConfig(model="Bigbert", config=config, n_embd=config.hidden_size, num_class=2, pretrain=False)
        # model = Mymodel(mconf)

    ## Training
    if args.reading_params_path is None:
        train_config = trainer.TrainerConfig(max_epochs=20, batch_size=16, learning_rate=6e-4, lr_decay=True,
                                             warmup_tokens=int(0.1 * Train_dataset.total_chars),
                                             final_tokens=int(0.9 * Train_dataset.total_chars),
                                             ckpt_path=args.writing_params_path, num_workers=4, pretrain=False) ## 75
    else:
        train_config = trainer.TrainerConfig(max_epochs=5, batch_size=16, learning_rate=6e-5, lr_decay=True,
                                             warmup_tokens=int(0.1 * Train_dataset.total_chars),
                                             final_tokens=int(0.9 * Train_dataset.total_chars),
                                             ckpt_path=args.writing_params_path, num_workers=4, pretrain=False)
        model.load_state_dict(torch.load(args.reading_params_path), strict=False)
        print("Load Complete!")

    Mytrainer = trainer.Trainer(model, Train_dataset, Valid_dataset, train_config)
    Mytrainer.train()


    # python run.py --model="BERT" --writing_params_path=weights/bert.train
    # python run.py --model="Longformer" --writing_params_path=weights/longformer.train