from utils import sequence_split
import os
import argparse
from tqdm import tqdm

def split_csv(coding_path, lnc_path, train_name):
    data1 = open(coding_path, encoding='UTF-8').read().split('\n')[1:-1]
    data2 = open(lnc_path, encoding='UTF-8').read().split('\n')[1:-1]

    with open("data/kmer1/coding_"+train_name+".txt", 'a+', encoding='utf-8') as test:
        test.truncate(0)
    with open("data/kmer1/lnc_"+train_name+".txt", 'a+', encoding='utf-8') as test:
        test.truncate(0)

    pbar = tqdm(enumerate(data1), total=len(data1))
    for i,seq in pbar:
        ## for kmer=3
        # seq = seq.split(',')[-1]  ## for csv dataset
        # seq = seq.split(' ')[:-1]  ## exclude the last one

        ## for kmer=1
        seq = seq.split(',')[2:]  ## for csv dataset
        seq[0] = seq[0].strip("\"")
        seq[-1] = seq[-1].strip("\"")
        if len(seq) > 200:
            s_data = sequence_split(seq, repeat=1)
            for s in s_data:
                s = ' '.join(s)
                with open("data/kmer1/coding_" + train_name + ".txt", 'a+', encoding='utf-8') as write:
                    write.write(s+'\n')

    pbar = tqdm(enumerate(data2), total=len(data2))
    for i, seq in pbar:
        ## for kmer=3
        # seq = seq.split(',')[-1]  ## for csv dataset
        # seq = seq.split(' ')[:-1]  ## exclude the last one

        ## for kmer=1
        seq = seq.split(',')[2:]  ## for csv dataset
        seq[0] = seq[0].strip("\"")
        seq[-1] = seq[-1].strip("\"")
        if len(seq) > 200:
            s_data = sequence_split(seq)
            for s in s_data:
                s = ' '.join(s)
                with open("data/kmer1/lnc_" + train_name + ".txt", 'a+', encoding='utf-8') as write:
                    write.write(s + '\n')


def split_csv_valid(coding_path, lnc_path, valid_name):
    data1 = open(coding_path, encoding='UTF-8').read().split('\n')[1:-1]
    data2 = open(lnc_path, encoding='UTF-8').read().split('\n')[1:-1]

    with open("data/kmer1/coding_"+valid_name+".txt", 'a+', encoding='utf-8') as test:
        test.truncate(0)
    with open("data/kmer1/lnc_"+valid_name+".txt", 'a+', encoding='utf-8') as test:
        test.truncate(0)

    pbar = tqdm(enumerate(data1), total=len(data1))
    for i,seq in pbar:
        ## for kmer=3
        # seq = seq.split(',')[-1]  ## for csv dataset
        # seq = seq.split(' ')[:-1]  ## exclude the last one

        ## for kmer=1
        seq = seq.split(',')[2:]  ## for csv dataset
        seq[0] = seq[0].strip("\"")
        seq[-1] = seq[-1].strip("\"")
        if len(seq) > 200:
            s_data = sequence_split(seq, repeat=1)
            for s in s_data:
                s = ' '.join(s)
                with open("data/kmer1/coding_" + valid_name + ".txt", 'a+', encoding='utf-8') as write:
                    write.write(s+'\n')

    pbar = tqdm(enumerate(data2), total=len(data2))
    for i, seq in pbar:
        ## for kmer=3
        # seq = seq.split(',')[-1]  ## for csv dataset
        # seq = seq.split(' ')[:-1]  ## exclude the last one

        ## for kmer=1
        seq = seq.split(',')[2:]  ## for csv dataset
        seq[0] = seq[0].strip("\"")
        seq[-1] = seq[-1].strip("\"")
        if len(seq) > 200:
            s_data = sequence_split(seq)
            for s in s_data:
                s = ' '.join(s)
                with open("data/kmer1/lnc_" + valid_name + ".txt", 'a+', encoding='utf-8') as write:
                    write.write(s + '\n')



def split_csv_pretrain(pretrain_coding_path, pretrain_lnc_path, valid_coding_path, valid_lnc_path, pretrain_name, valid_name):
    with open("data/kmer1/pre_train_"+pretrain_name+".txt", 'a+', encoding='utf-8') as test:
        test.truncate(0)
    with open("data/kmer1/pre_valid_"+valid_name+".txt", 'a+', encoding='utf-8') as test:
        test.truncate(0)

    data1 = open(pretrain_coding_path, encoding='UTF-8').read().split('\n')[1:-1]
    data2 = open(pretrain_lnc_path, encoding='UTF-8').read().split('\n')[1:-1]
    ## pretrain train
    pbar = tqdm(enumerate(data1), total=len(data1))
    for i,seq in pbar:
        seq = seq.split(',')[2:]  ## for csv dataset
        seq[0] = seq[0].strip("\"")
        seq[-1] = seq[-1].strip("\"")
        if len(seq) > 200:
            s_data = sequence_split(seq)
            for s in s_data:
                s = ' '.join(s)
                with open("data/kmer1/pre_train_" + pretrain_name + ".txt", 'a+', encoding='utf-8') as write:
                    write.write(s+'\n')

    pbar = tqdm(enumerate(data2), total=len(data2))
    for i, seq in pbar:
        seq = seq.split(',')[2:]  ## for csv dataset
        seq[0] = seq[0].strip("\"")
        seq[-1] = seq[-1].strip("\"")
        if len(seq) > 200:
            s_data = sequence_split(seq)
            for s in s_data:
                s = ' '.join(s)
                with open("data/kmer1/pre_train_" + pretrain_name + ".txt", 'a+', encoding='utf-8') as write:
                    write.write(s + '\n')

    data1 = open(valid_coding_path, encoding='UTF-8').read().split('\n')[1:-1]
    data2 = open(valid_lnc_path, encoding='UTF-8').read().split('\n')[1:-1]
    ## pretrain valid
    pbar = tqdm(enumerate(data1), total=len(data1))
    for i, seq in pbar:
        seq = seq.split(',')[2:]  ## for csv dataset
        seq[0] = seq[0].strip("\"")
        seq[-1] = seq[-1].strip("\"")
        if len(seq) > 200:
            s_data = sequence_split(seq)
            for s in s_data:
                s = ' '.join(s)
                with open("data/kmer1/pre_valid_" + valid_name + ".txt", 'a+', encoding='utf-8') as write:
                    write.write(s + '\n')

    pbar = tqdm(enumerate(data2), total=len(data2))
    for i, seq in pbar:
        seq = seq.split(',')[2:]  ## for csv dataset
        seq[0] = seq[0].strip("\"")
        seq[-1] = seq[-1].strip("\"")
        if len(seq) > 200:
            s_data = sequence_split(seq)
            for s in s_data:
                s = ' '.join(s)
                with open("data/kmer1/pre_valid_" + valid_name + ".txt", 'a+', encoding='utf-8') as write:
                    write.write(s + '\n')


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('--Coding_train_path', default="data/kmer1/train_coding_kmer=1.csv")
    argp.add_argument('--Lnc_train_path', default="data/kmer1/train_lnc_kmer=1.csv")
    argp.add_argument('--Coding_valid_path', default="data/kmer1/valid_coding_kmer=1.csv")
    argp.add_argument('--Lnc_valid_path', default="data/kmer1/valid_lnc_kmer=1.csv")

    argp.add_argument('--Pre_train_coding', default="data/kmer1/train_coding_kmer=1.csv")
    argp.add_argument('--Pre_train_lnc', default="data/kmer1/train_lnc_kmer=1.csv")
    argp.add_argument('--Pre_valid_coding', default="data/kmer1/valid_coding_kmer=1.csv")
    argp.add_argument('--Pre_valid_lnc', default="data/kmer1/valid_lnc_kmer=1.csv")
    args = argp.parse_args()


    split_csv(args.Coding_train_path, args.Lnc_train_path, train_name='train_seq_kmer1')
    split_csv_valid(args.Coding_valid_path, args.Lnc_valid_path, valid_name='valid_seq_kmer1')
    # split_csv_pretrain(args.Pre_train_coding, args.Pre_train_lnc, args.Pre_valid_coding, args.Pre_valid_lnc, pretrain_name='kmer1', valid_name="kmer1")