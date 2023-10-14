import random
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import argparse
from utils import sequence_split
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

class Pre_RNADataset(Dataset):
    def __init__(self, data):
        self.MASK_CHAR = u"\u2047"  # the doublequestionmark ⁇ character, for mask
        self.PAD_CHAR = u"\u25A1"  # the empty square character □, for pad
        self.CLS_CHAR = u"[CLS]"

        self.data = data

        print("Loading data...")
        chars = []
        total_chars = 0
        max_size = 0
        for seq in data:
            seq = seq.split(' ')[:-1]  ## delete \n
            total_chars = total_chars + len(seq)
            chars.extend(seq)
            chars = list(sorted(list(set(chars))))
            max_size = len(seq) if len(seq) > max_size else max_size
            # self.data.append(seq)
        print("Complete!")
        self.total_chars = total_chars
        self.max_size = max_size + 1 # [CLS]
        # print("max_size", self.max_size)

        chars = list(sorted(list(set(chars))))
        assert self.MASK_CHAR not in chars
        assert self.CLS_CHAR not in chars
        assert self.PAD_CHAR not in chars
        if "N" not in chars:
            chars.append("N")
            chars = list(sorted(list(set(chars))))
        self.chars_unique = chars.copy()
        self.chars_choice = self.chars_unique.copy()  # for random word exchange, without special tokens

        self.chars_unique.insert(0, self.MASK_CHAR)
        self.chars_unique.insert(0, self.CLS_CHAR)
        self.chars_unique.insert(0, self.PAD_CHAR)

        self.stoi = {ch: i for i, ch in enumerate(self.chars_unique)}
        self.itos = {i: ch for i, ch in enumerate(self.chars_unique)}

        data_size, vocab_size = len(self.data), len(self.chars_unique)
        print('data has %d sequences, %d unique.' % (data_size, vocab_size))
        self.vocab_size = vocab_size

    def __len__(self):
        # returns the length of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]  # get a line
        seq = seq.split(' ')[:-1]  ## delete \n
        x, y = self.random_word(seq)
        x = torch.tensor([self.stoi[c] for c in x], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in y], dtype=torch.long)
        return x, y, len(x)

    def random_word(self, sentence):
        output_label = [self.PAD_CHAR]
        sentence.insert(0, self.CLS_CHAR)
        for i,s in enumerate(sentence):
            if i==0:
                continue
            prob = random.random()
            if prob < 0.15:
                output_label.append(s)
                prob /= 0.15
                # 80% randomly change token to mask token
                if prob < 0.8:
                    sentence[i] = self.MASK_CHAR

                # 10% randomly change token to random token
                elif prob < 0.9:
                    sentence[i] = random.choice(self.chars_choice)

                # 10% randomly change token to current token

            else:
                output_label.append(self.PAD_CHAR)

        return sentence, output_label

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
    argp.add_argument('--Pretrain_RNA_path', default="data/kmer1/pre_train_kmer1.txt")
    args = argp.parse_args()

    RNA_dataset = Pre_RNADataset(open(args.Pretrain_RNA_path, encoding='UTF-8').readlines()[:-1])
    print(RNA_dataset.chars_unique)

    ## test dataset
    for _, example in zip(range(1), RNA_dataset):
        x, y, l = example
        print('x:', ' '.join([RNA_dataset.itos[int(c)] for c in x]))
        print('y:', y)
        print('length:', l)

    ## test dataloader
    loader = DataLoader(RNA_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=RNA_dataset.collate_func)
    batch = next(iter(loader))
    print(batch)