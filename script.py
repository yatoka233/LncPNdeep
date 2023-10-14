import torch

if __name__ == '__main__':
     x = open("data/kmer1/lnc_valid_seq_kmer1.txt", encoding='UTF-8').readlines()[:5000]
     for i,seq in enumerate(x):
          print(seq)
          break