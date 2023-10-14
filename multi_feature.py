import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
import random
import argparse
import os

from transformers import LongformerModel, LongformerConfig
from transformers import BigBirdModel, BigBirdConfig, BigBirdForMaskedLM
import pre_dataset
from model.bert import BERTConfig, BERT
from model.mybigbird import MYConfig, Mymodel
from utils import sequence_split
import trainer

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)



class Feature_Generator:

    def __init__(self, model, Test_Coding_path, Test_Lnc_path, itos, stoi, Coding_w_path, Lnc_w_path):
        self.model = model
        self.coding_dataset = open(Test_Coding_path, encoding='UTF-8').readlines()
        self.lnc_dataset = open(Test_Lnc_path, encoding='UTF-8').readlines()
        self.write_path0 = Coding_w_path    ## write test coding features
        self.write_path1 = Lnc_w_path   ## write test lnc features
        self.itos = itos
        self.stoi = stoi

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)


    def feature(self):
        if not os.path.exists(os.path.dirname(self.write_path0 + ".txt")):
            os.makedirs(os.path.dirname(self.write_path0 + ".txt"))
        if not os.path.exists(os.path.dirname(self.write_path1 + ".txt")):
            os.makedirs(os.path.dirname(self.write_path1 + ".txt"))
        
        with open(self.write_path0 + ".txt", 'a+', encoding='utf-8') as test:
            test.truncate(0)
        with open(self.write_path1 + ".txt", 'a+', encoding='utf-8') as test:
            test.truncate(0)

        model = self.model
        model.train(False)
        acc = 0
        test_num = 0

        pbar0 = tqdm(enumerate(self.coding_dataset), total=len(self.coding_dataset))
        for it, doc in pbar0:
            seq = doc.split(',')[:-1]  ## for txt dataset
            seq[0] = seq[0].strip("\"")
            seq[-1] = seq[-1].strip("\"")
            s_data = sequence_split(seq, repeat=10)    ## sample for a sequence
            prediction = []     # store the predicted probability after softmax
            features = []       # store the feature for each split seq
            for s in s_data:
                s.insert(0, u"[CLS]")
                x = torch.tensor([self.stoi[c] for c in s], dtype=torch.long).view(1, len(s)).to(self.device)
                y = torch.tensor([0]*len(s), dtype=torch.long).view(1, len(s)).to(self.device)
                mask = torch.tensor([1.]*len(s), dtype=torch.long).view(1, len(s)).to(self.device)

                outputs = model(input_ids=x, labels=y, attention_mask=mask)
                logits = outputs.logits
                loss = outputs.loss
                last_h = outputs.last_h[:,0,:]  ## [1, t, h]
                logits = F.softmax(logits, dim=-1)
                predict_y = logits[:,-1].detach().squeeze().cpu().numpy()   # probability to be 1
                prediction.append(predict_y)

                features.append(last_h.detach().squeeze().cpu().numpy()) ## [1, h]


            ## write feature
            features = np.mean(np.vstack(features), axis=0)  ## [1, h] -> [s, h] -> [h]
            # print(features.shape)
            feature = ' '.join([str(a) for a in features.tolist()])
            with open(self.write_path0 + ".txt", 'a+', encoding='utf-8') as write:
                    write.write(feature+'\n')

            predict_p = np.mean(np.array(prediction))
            predict_label = 1 if predict_p > 0.5 else 0
            acc += predict_label == 0
            test_num += 1
            acc_cum = acc / test_num
            pbar0.set_description(f"iter {it}: predict_p {predict_p:.4f} test cumulate acc {acc_cum:.4f}.")


        pbar1 = tqdm(enumerate(self.lnc_dataset), total=len(self.lnc_dataset))
        for it, doc in pbar1:
            seq = doc.split(',')[:-1]  ## for txt dataset
            seq[0] = seq[0].strip("\"")
            seq[-1] = seq[-1].strip("\"")
            s_data = sequence_split(seq)    ## sample for a sequence
            prediction = []     # store the predicted probability after softmax
            features = []
            for s in s_data:
                s.insert(0, u"[CLS]")
                x = torch.tensor([self.stoi[c] for c in s], dtype=torch.long).view(1, len(s)).to(self.device)
                y = torch.tensor([0]*len(s), dtype=torch.long).view(1, len(s)).to(self.device)
                mask = torch.tensor([1.]*len(s), dtype=torch.long).view(1, len(s)).to(self.device)

                outputs = model(input_ids=x, labels=y, attention_mask=mask)
                logits = outputs.logits
                loss = outputs.loss
                last_h = outputs.last_h[:,0,:]  ## [1, t, h]
                logits = F.softmax(logits, dim=-1)
                predict_y = logits[:,-1].detach().squeeze().cpu().numpy()   # probability to be 1
                prediction.append(predict_y)

                features.append(last_h.detach().squeeze().cpu().numpy()) ## [1, h]


            ## write feature
            features = np.mean(np.vstack(features), axis=0)  ## [1, h] [h]
            feature = ' '.join([str(a) for a in features.tolist()])
            with open(self.write_path1 + ".txt", 'a+', encoding='utf-8') as write:
                    write.write(feature+'\n')

            predict_p = np.mean(np.array(prediction))
            predict_label = 1 if predict_p > 0.5 else 0
            acc += predict_label == 0
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
        default="weights/pretrain/save.Longformer.pretrain.epoch20.params")
    ## data argument
    argp.add_argument('--Pretrain_path', default="data/kmer1/pre_valid_kmer1.txt")
    
    argp.add_argument('--multi_type', default=None)
    argp.add_argument('--name', default=None)
    
    argp.add_argument('--Test_Coding_path', default=None)
    argp.add_argument('--Test_Lnc_path', default=None)

    argp.add_argument('--outputs_path0', default=None)
    argp.add_argument('--outputs_path1', default=None)
    args = argp.parse_args()

    multi_type = args.multi_type
    name = args.name
    args.Test_Coding_path = "data/codingRNA/" + multi_type + "/" + name + "_codingRNA.txt.txt"
    args.Test_Lnc_path = "data/lncRNA/" + multi_type + "/" + name + "_lncRNA.txt.txt"
    args.outputs_path0 = "data/output/multi_RNA/Longformer256/" + multi_type + "/" + name + "_codingRNA_feature"
    args.outputs_path1 = "data/output/multi_RNA/Longformer256/" + multi_type + "/" + name + "_lncRNA_feature"


    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    # from pretrain data get stoi and itos
    Valid_dataset = pre_dataset.Pre_RNADataset(open(args.Pretrain_path, encoding='UTF-8').readlines()[:-1])
    stoi = Valid_dataset.stoi
    itos = Valid_dataset.itos

    # Load model
    if args.model == "BERT":
        mconf = BERTConfig(Valid_dataset.vocab_size, max_size=1000, num_class=8,
                                     n_layer=2, n_head=4, n_embd=128, pretrain=True, Mask=False) # 4 8 256

        model = BERT(mconf)

    elif args.model == "Longformer":
        config = LongformerConfig(num_hidden_layers=4, num_attention_heads=8, hidden_size=256, max_position_embeddings=4096,
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

    ## Training
    model.load_state_dict(torch.load(args.reading_params_path))

    Feature_G = Feature_Generator(model, args.Test_Coding_path, args.Test_Lnc_path, itos, stoi, args.outputs_path0, args.outputs_path1)
    Feature_G.feature()









