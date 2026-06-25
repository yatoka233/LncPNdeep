import random
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import argparse

class Pre_RNADataset(Dataset):
    def __init__(self, data, lnc=False):
        self.MASK_CHAR = u"\u2047"  # the doublequestionmark ⁇ character, for mask
        self.PAD_CHAR = u"\u25A1"  # the empty square character □, for pad
        self.CLS_CHAR = u"[CLS]"
        self.LNC = lnc  # True of False
        data = data.split('\n')   # split data into documents. Each documents should have many sequences which need to be generated in below
        self.data = []

        print("Loading data...")
        chars = []
        total_chars = 0
        max_size = 0
        for seq in data:
            seq = seq.split(',')[-1]   ## for csv dataset
            if len(seq) > 200 and len(seq) < 2001:
                seq = seq.split(' ')[:-1]  ## exclude the last one
                chars.extend(seq)
                self.data.append(seq)
                chars = list(sorted(list(set(chars))))
                total_chars = total_chars + len(seq)
                max_size = len(seq) if len(seq) > max_size else max_size
        print("Complete!")
        self.total_chars = total_chars
        self.max_size = max_size + 3  ## [⁇] [⁇] [⁇]
        # print("max_size", self.max_size)

        chars = list(sorted(list(set(chars))))
        assert self.MASK_CHAR not in chars
        assert self.CLS_CHAR not in chars
        assert self.PAD_CHAR not in chars

        self.chars_unique = chars.copy()

        data_size, vocab_size = len(self.data), len(chars)
        print('data has %d sequences, %d unique.' % (data_size, vocab_size))
        self.vocab_size = vocab_size

    def __len__(self):
        # returns the length of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]  # get a line


class Pre_CombineDataset(Dataset):
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
            seq = self.dataset1[idx]
        else:
            seq = self.dataset2[idx - len(self.dataset1)]
        t = random.randint(1, len(seq)-1)
        mask = seq[t]
        seq[t] = self.MASK_CHAR
        seq = seq + [self.MASK_CHAR] + [mask] + [self.MASK_CHAR]
        y = seq[1:]
        x = seq[:-1]
        x = torch.tensor([self.stoi[c] for c in x], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in y], dtype=torch.long)  ## let y from 0 - 63
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
        res['labels'] = pad_sequence(label_batch, batch_first=True, padding_value=self.stoi[self.PAD_CHAR])
        res['attention_mask'] = mask_batch
        return res

if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('--Coding_RNA_path', default="data/human_coding_Kmer_divide.txt")
    argp.add_argument('--Lnc_RNA_path', default="data/human_lncRNA_Kmer_divide.txt")
    args = argp.parse_args()

    Coding_RNA_dataset = Pre_RNADataset(open(args.Coding_RNA_path, encoding='UTF-8').read(), lnc=False)
    Lnc_RNA_dataset = Pre_RNADataset(open(args.Lnc_RNA_path, encoding='UTF-8').read(), lnc=True)
    RNA_dataset = Pre_CombineDataset(Coding_RNA_dataset, Lnc_RNA_dataset)

    ## test dataset
    for _, example in zip(range(1), RNA_dataset):
        x, y, l = example
        print('x:', ' '.join([RNA_dataset.itos[int(c)] for c in x]))
        print('y:', y)
        print('length:', l)

    ## test dataloader
    loader = DataLoader(RNA_dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=RNA_dataset.collate_func)
    batch = next(iter(loader))
    print(batch)