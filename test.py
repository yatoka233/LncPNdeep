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
from utils import sequence_split
import trainer

random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)

class Tester:

    def __init__(self, model, Test_Coding_path, Test_Lnc_path, itos, stoi):
        self.model = model
        self.coding_dataset = open(Test_Coding_path, encoding='UTF-8').read().split('\n')[1:-1]
        self.lnc_dataset = open(Test_Lnc_path, encoding='UTF-8').read().split('\n')[1:-1]
        self.itos = itos
        self.stoi = stoi

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def test(self):
        model = self.model
        model.train(False)
        acc = 0
        test_num = 0


        pbar0 = tqdm(enumerate(self.coding_dataset), total=len(self.coding_dataset))
        for it, doc in pbar0:
            seq = doc.split(',')[-1]  ## for csv dataset
            seq = seq.split(' ')[:-1]  ## exclude the last one
            s_data = sequence_split(seq)    ## sample for a sequence
            prediction = []     # store the predicted probability after softmax
            for s in s_data:
                x = torch.tensor([self.stoi[c] for c in s], dtype=torch.long).view(1, len(s)).to(self.device)
                y = torch.tensor([0], dtype=torch.long).to(self.device)
                mask = torch.tensor([1.]*len(s), dtype=torch.long).view(1, len(s)).to(self.device)

                outputs = model(input_ids=x, labels=y, attention_mask=mask)
                logits = outputs.logits
                loss = outputs.loss
                logits = F.softmax(logits, dim=-1)
                predict_y = logits[:,-1].detach().squeeze().cpu().numpy()   # probability to be 1
                prediction.append(predict_y)

            predict_p = np.mean(np.array(prediction))
            predict_label = 1 if predict_p > 0.5 else 0
            acc += predict_label == 0
            test_num += 1
            acc_cum = acc / test_num
            pbar0.set_description(f"iter {it}: predict_p {predict_p:.4f} test cumulate acc {acc_cum:.4f}.")

        pbar1 = tqdm(enumerate(self.lnc_dataset), total=len(self.lnc_dataset))
        for it, doc in pbar1:
            seq = doc.split(',')[-1]  ## for csv dataset
            seq = seq.split(' ')[:-1]  ## exclude the last one
            s_data = sequence_split(seq)
            prediction = []
            for s in s_data:
                x = torch.tensor([self.stoi[c] for c in s], dtype=torch.long).view(1, len(s)).to(self.device)
                y = torch.tensor([1], dtype=torch.long).to(self.device)
                mask = torch.tensor([1.] * len(s), dtype=torch.long).view(1, len(s)).to(self.device)

                outputs = model(input_ids=x, labels=y, attention_mask=mask)
                logits = outputs.logits
                loss = outputs.loss
                logits = F.softmax(logits, dim=-1)
                predict_y = logits[:,-1].detach().squeeze().cpu().numpy()   # probability to be 1
                prediction.append(predict_y)

            predict_p = np.mean(np.array(prediction))
            predict_label = 1 if predict_p > 0.5 else 0
            acc += predict_label == 1
            test_num += 1
            acc_cum = acc / test_num
            pbar1.set_description(f"iter {it}: predict_p {predict_p:.4f} test cumulate acc {acc_cum:.4f}.")



if __name__ == '__main__':

    argp = argparse.ArgumentParser()
    ## model argument
    argp.add_argument('--model',
        help="Choose one of the models", default="Longformer", choices=["BERT", "Longformer", "Bigbird"])
    argp.add_argument('--reading_params_path',
        help="If specified, path of the model to load before finetuning/evaluation",
        default="/root/autodl-tmp/weights/train/longformer.train.epoch20.params")
    ## data argument
    argp.add_argument('--Train_Coding_path', default="data/coding_train_seq.txt")
    argp.add_argument('--Train_Lnc_path', default="data/lnc_train_seq.txt")
    argp.add_argument('--Test_Coding_path', default="data/test_coding.csv")
    argp.add_argument('--Test_Lnc_path', default="data/test_lnc.csv")

    argp.add_argument('--outputs_path', default=None)
    args = argp.parse_args()

    # Save the device
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    # Train data
    Train_Coding_dataset = dataset.RNADataset(open(args.Train_Coding_path, encoding='UTF-8').readlines(), lnc=False)
    Train_Lnc_dataset = dataset.RNADataset(open(args.Train_Lnc_path, encoding='UTF-8').readlines(), lnc=True)
    Train_dataset = dataset.CombineDataset(Train_Coding_dataset, Train_Lnc_dataset)
    stoi = Train_dataset.stoi
    itos = Train_dataset.itos

    # Load model
    if args.model == "BERT":
        # mconf = BERTConfig(vocab_size=76, max_size=2100,
        #                    n_layer=2, n_head=4, n_embd=128, pretrain=False, Mask=False)  # 4 8 256
        # 
        # model = BERT(mconf)
        config = BertConfig(num_hidden_layers=2, num_attention_heads=4, hidden_size=128, max_position_embeddings=4096,
                               vocab_size=76,
                               eos_token_id=None,
                               sep_token_id=None)
        model = BertForSequenceClassification(config)

    elif args.model == "Longformer":
        config = LongformerConfig(num_hidden_layers=2, num_attention_heads=4, hidden_size=128, max_position_embeddings=4096,
                                  vocab_size=76,
                                  eos_token_id=None,
                                  sep_token_id=None)
        config.attention_mode = 'tvm'  # choose the attention mode 'n2', 'tvm' or 'sliding_chunks'
        model = LongformerForSequenceClassification(config)
        # mconf = MYConfig(model=Lformer, n_embd=config.hidden_size, num_class=2, pretrain=False)
        # model = Mymodel(mconf)

    elif args.model == "Bigbird":
        # model = BigBirdModel.from_pretrained("google/bigbird-roberta-large",num_labels=2, block_size=16, num_random_blocks=2)
        config = BigBirdConfig(num_hidden_layers=2, num_attention_heads=4, hidden_size=128, max_position_embeddings=4096,
                               vocab_size=76,
                               eos_token_id=None,
                               sep_token_id=None)
        model = BigBirdForSequenceClassification(config)
        # mconf = MYConfig(model=Bigbert, n_embd=config.hidden_size, num_class=2, pretrain=False)
        # model = Mymodel(mconf)

    model.load_state_dict(torch.load(args.reading_params_path))

    Mytester = Tester(model, args.Test_Coding_path, args.Test_Lnc_path, itos, stoi)
    Mytester.test()

    # python test.py --model=BERT --reading_params_path=weights/bert.train.epoch9.params
    # python test.py --model=Bigbird --reading_params_path=weights/bigbert.train.epoch14.params
    # python test.py --model=Longformer --reading_params_path=/root/autodl-tmp/weights/train/longformer.train.epoch20.params
