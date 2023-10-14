import random
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import argparse

"""
Load data from one txt file.
Could be codingRNA sequences or LncRNA sequences.
Create special tokens and dictionaries, count numbers.
"""


class RNADataset(Dataset):
    def __init__(self, data, lnc=False):
        self.MASK_CHAR = u"\u2047"  # the doublequestionmark ⁇ character, for mask
        self.PAD_CHAR = u"\u25A1"  # the empty square character □, for pad
        self.CLS_CHAR = u"[CLS]"
        self.LNC = lnc
        self.data = data

        print("Loading data...")
        chars = []
        total_chars = 0
        max_size = 0
        min_size = 2000
        for seq in data:
            # seq = seq.split(',')[-1]  ## for csv dataset
            seq = seq.split(' ')[:-1]  ## delete \n
            total_chars = total_chars + len(seq)
            chars.extend(seq)
            chars = list(sorted(list(set(chars))))
            max_size = len(seq) if len(seq) > max_size else max_size
        print("Complete!")
        self.total_chars = total_chars
        self.max_size = max_size + 1  ## [CLS]
        # print("max_size", self.max_size)
        # print("min_size", min_size)

        chars = list(sorted(list(set(chars))))
        assert self.MASK_CHAR not in chars
        assert self.CLS_CHAR not in chars
        assert self.PAD_CHAR not in chars

        self.chars_unique = chars.copy()

        data_size, vocab_size = len(self.data), len(self.chars_unique)
        print('data has %d sequences, %d unique.' % (data_size, vocab_size))
        self.vocab_size = vocab_size

    def __len__(self):
        # returns the length of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]  # get a line
        # label is 1 for lncRNA
        if self.LNC:
            y = torch.tensor([1], dtype=torch.long)
        else:
            y = torch.tensor([0], dtype=torch.long)
        return x, y

    def collate_func(self, batch_dic):
        batch_len = len(batch_dic)
        max_seq_length = max([dic[2] for dic in batch_dic])  # longest length in a batch
        mask_batch = torch.zeros((batch_len, max_seq_length))  # mask
        x_batch = []
        label_batch = []
        for i in range(len(batch_dic)):
            dic = batch_dic[i]
            x_batch.append(dic[0])
            label_batch.append(dic[1])
            mask_batch[i, :dic[2]] = 1  # mask
        res = {}
        res['x'] = pad_sequence(x_batch, batch_first=True, padding_value=self.stoi[self.PAD_CHAR])
        res['label'] = torch.tensor(label_batch)
        res['mask'] = mask_batch
        return res


"""
Combine 2 RNAdatasets
Merge the dictionarys
"""


class CombineDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.MASK_CHAR = u"\u2047"  # the doublequestionmark ⁇ character, for mask
        self.PAD_CHAR = u"\u25A1"  # the empty square character □, for pad
        self.CLS_CHAR = u"[CLS]"

        self.chars_unique = list(sorted(list(set(dataset1.chars_unique + dataset2.chars_unique))))
        assert self.MASK_CHAR not in self.chars_unique
        assert self.CLS_CHAR not in self.chars_unique
        assert self.PAD_CHAR not in self.chars_unique
        self.chars_unique.insert(0, self.MASK_CHAR)
        self.chars_unique.insert(0, self.CLS_CHAR)
        self.chars_unique.insert(0, self.PAD_CHAR)

        self.stoi = {ch: i for i, ch in enumerate(self.chars_unique)}
        self.itos = {i: ch for i, ch in enumerate(self.chars_unique)}

        data_size, vocab_size = len(self.dataset1) + len(self.dataset2), len(self.stoi)
        print('total data has %d sequences, %d unique.' % (data_size, vocab_size))
        self.data_size = data_size
        self.vocab_size = vocab_size
        self.total_chars = dataset1.total_chars + dataset2.total_chars
        self.max_size = max(dataset1.max_size, dataset2.max_size)

    def __len__(self):
        # returns the length of the dataset
        return self.data_size

    def __getitem__(self, idx):
        if idx < len(self.dataset1):
            seq = self.dataset1[idx][0].split(' ')[:-1]  ## delete \n
            # print(seq)
            x = [self.CLS_CHAR] + seq + [self.MASK_CHAR]
            x = torch.tensor([self.stoi[c] for c in x], dtype=torch.long)
            y = self.dataset1[idx][1]
            return x, y, len(x)
        else:
            seq = self.dataset2[idx - len(self.dataset1)][0].split(' ')[:-1]
            # print(seq)
            x = [self.CLS_CHAR] + seq + [self.MASK_CHAR]
            x = torch.tensor([self.stoi[c] for c in x], dtype=torch.long)
            y = self.dataset2[idx - len(self.dataset1)][1]
            return x, y, len(x)

    def collate_func(self, batch_dic):
        batch_len = len(batch_dic)
        max_seq_length = max([dic[2] for dic in batch_dic])  # longest length in a batch
        mask_batch = torch.zeros((batch_len, max_seq_length))  # mask
        x_batch = []
        label_batch = []
        for i in range(len(batch_dic)):
            dic = batch_dic[i]
            x_batch.append(dic[0])
            label_batch.append(dic[1])
            mask_batch[i, :dic[2]] = 1  # mask
        res = {}
        res['input_ids'] = pad_sequence(x_batch, batch_first=True, padding_value=self.stoi[self.PAD_CHAR])
        res['labels'] = torch.tensor(label_batch)
        res['attention_mask'] = mask_batch
        return res


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('--Coding_RNA_path', default="data/coding_valid_seq.txt")
    argp.add_argument('--Lnc_RNA_path', default="data/lnc_train_seq.txt")
    args = argp.parse_args()

    Coding_RNA_dataset = RNADataset(open(args.Coding_RNA_path, encoding='UTF-8').readlines()[:-1], lnc=False)
    Lnc_RNA_dataset = RNADataset(open(args.Lnc_RNA_path, encoding='UTF-8').readlines()[:-1], lnc=True)
    RNA_dataset = CombineDataset(Coding_RNA_dataset, Lnc_RNA_dataset)
    # print(Lnc_RNA_dataset.stoi)
    print(RNA_dataset.stoi)

    ## test dataset
    for _, example in zip(range(1), RNA_dataset):
        x, y, l = example
        print('x:', ' '.join([RNA_dataset.itos[int(c)] for c in x]))
        print('y:', y)
        print('length:', l)

    ## test dataloader
    loader = DataLoader(RNA_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=RNA_dataset.collate_func)
    batch = next(iter(loader))
    print(batch)
