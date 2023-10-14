import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
import random
import argparse

from transformers import LongformerModel, LongformerConfig
from transformers import BigBirdModel, BigBirdConfig, BigBirdForMaskedLM
import pre_dataset
from model.bert import BERTConfig, BERT
from model.mybigbird import MYConfig, Mymodel
import trainer

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


if __name__ == '__main__':

    argp = argparse.ArgumentParser()
    ## model argument
    argp.add_argument('--model',
        help="Choose one of the models", default="BERT", choices=["BERT", "Longformer", "Bigbird"])
    argp.add_argument('--reading_params_path',
        help="If specified, path of the model to load before finetuning/evaluation",
        default=None)
    argp.add_argument('--writing_params_path',
        help="Path to save the model after pretraining/finetuning", default="weights/pretrain/bert.pretrain")
    ## data argument
    argp.add_argument('--Pretrain_train_path', default="data/kmer1/pre_train_kmer1.txt")
    argp.add_argument('--Pretrain_valid_path', default="data/kmer1/pre_valid_kmer1.txt")

    argp.add_argument('--outputs_path', default=None)
    args = argp.parse_args()

    # Save the device
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    ## Load dataset
    Train_dataset = pre_dataset.Pre_RNADataset(open(args.Pretrain_train_path, encoding='UTF-8').readlines()[:-1])  # -1
    Valid_dataset = pre_dataset.Pre_RNADataset(open(args.Pretrain_valid_path, encoding='UTF-8').readlines()[:-1])  # -1

    # Load model
    if args.model == "BERT":
        mconf = BERTConfig(Train_dataset.vocab_size, max_size=1000, num_class=8,
                                     n_layer=4, n_head=8, n_embd=256, pretrain=True, Mask=False) # 4 8 256

        model = BERT(mconf)

    elif args.model == "Longformer":
        config = LongformerConfig(num_hidden_layers=12, num_attention_heads=12, hidden_size=768, max_position_embeddings=4096,
                                  vocab_size=8,
                                  eos_token_id=None,
                                  sep_token_id=None)
        config.attention_mode = 'tvm'  # choose the attention mode 'n2', 'tvm' or 'sliding_chunks'
        mconf = MYConfig(model="Longformer", config=config, n_embd=config.hidden_size, num_class=8, pretrain=True)
        model = Mymodel(mconf)

    elif args.model == "Bigbird":
        # 2 4 128 模型未拟合
        # 4 8 256
        # 8 8 512
        # 12 12 768
        # model = BigBirdModel.from_pretrained("google/bigbird-roberta-large",num_labels=2, block_size=16, num_random_blocks=2)
        config = BigBirdConfig(num_hidden_layers=12, num_attention_heads=12, hidden_size=768, max_position_embeddings=4096,
                               vocab_size=8,
                               eos_token_id=None,
                               sep_token_id=None)
        mconf = MYConfig(model="Bigbird", config=config, n_embd=config.hidden_size, num_class=8, pretrain=True)
        model = Mymodel(mconf)

    ## print(model)
    ## Training
    if args.reading_params_path is None:
        train_config = trainer.TrainerConfig(max_epochs=20, batch_size=8, learning_rate=6e-5, lr_decay=True,
                                             warmup_tokens=int(0.1 * Train_dataset.total_chars),
                                             final_tokens=int(0.9 * Train_dataset.total_chars),
                                             ckpt_path=args.writing_params_path, num_workers=4, pretrain=True) ## 75
    else:
        train_config = trainer.TrainerConfig(max_epochs=10, batch_size=1, learning_rate=6e-5, lr_decay=True,
                                             warmup_tokens=int(0.1 * Train_dataset.total_chars),
                                             final_tokens=int(0.9 * Train_dataset.total_chars),
                                             ckpt_path=args.writing_params_path, num_workers=4, pretrain=True)
        model.load_state_dict(torch.load(args.reading_params_path))

    Mytrainer = trainer.Trainer(model, Train_dataset, Valid_dataset, train_config)
    Mytrainer.train()