import torch
from collections import defaultdict
from torch import nn, cuda, optim
from torch.nn import functional
import random
import time
import math
import numpy as np

import torch.nn.functional as F

# much of the beginning is the same as the text retrieval
# format of files: each line is "word1 word2 ..." aligned line-by-line
DATA_PATH = "../.."
train_src_file = f"{DATA_PATH}/data/parallel/train.ja"
train_trg_file = f"{DATA_PATH}/data/parallel/train.en"
dev_src_file = f"{DATA_PATH}/data/parallel/dev.ja"
dev_trg_file = f"{DATA_PATH}/data/parallel/dev.en"
test_src_file = f"{DATA_PATH}/data/parallel/test.ja"
test_trg_file = f"{DATA_PATH}/data/parallel/test.en"

w2i_src = defaultdict(lambda: len(w2i_src))
w2i_trg = defaultdict(lambda: len(w2i_trg))


def read(fname_src, fname_trg):
    """
    Read parallel files where each line lines up
    """
    with open(fname_src, "r") as f_src, open(fname_trg, "r") as f_trg:
        for line_src, line_trg in zip(f_src, f_trg):
            # need to append EOS tags to at least the target sentence
            sent_src = [
                w2i_src[x]
                for x in line_src.strip().split() + ['</s>']
            ]
            sent_trg = [
                w2i_trg[x]
                for x in ['<s>'] + line_trg.strip().split() + ['</s>']
            ]
            yield (sent_src, sent_trg)


# for sent_src, sent_trg in read(train_src_file, train_trg_file):
#     print(sent_src, sent_trg)
#     break

# Read the data
train = list(read(train_src_file, train_trg_file))
unk_src = w2i_src["<unk>"]
pad_src = w2i_src["<pad>"]
eos_src = w2i_src['</s>']
w2i_src = defaultdict(lambda: unk_src, w2i_src)
unk_trg = w2i_trg["<unk>"]
eos_trg = w2i_trg['</s>']
sos_trg = w2i_trg['<s>']
w2i_trg = defaultdict(lambda: unk_trg, w2i_trg)
i2w_trg = {v: k for k, v in w2i_trg.items()}

nwords_src = len(w2i_src)
nwords_trg = len(w2i_trg)
dev = list(read(dev_src_file, dev_trg_file))
test = list(read(test_src_file, test_trg_file))


class CopyNetEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dropout=0.2):
        super(CopyNetEncoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, enc_hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        # 双向
        self.fc = nn.Linear(enc_hidden_size * 2, dec_hidden_size)

    def forward(self, x, lengths):
        sorted_len, sorted_idx = lengths.sort(0, descending=True)
        x_sorted = x[sorted_idx]
        embedded = self.dropout(self.embed(x_sorted))

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_len,
                                                            batch_first=True)

        packed_out, hid = self.rnn(packed_embedded)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        _, original_idx = sorted_idx.sort(0, descending=False)
        out = out[original_idx.long()].contiguous()
        hid = hid[:, original_idx.long()].contiguous()
        # hid (layers=1*directions) x batch x dim.
        # We need to convert it to layers x batch x (directions*dim).
        hid = torch.cat((hid[-2], hid[-1]), dim=1)
        # add layer=1 dim
        hid = torch.tanh(self.fc(hid)).unsqueeze(0)
        # out: (batch_size, seq_len, num_directions*enc_hidden_size)
        # hid: (1, batch_size, dec_hidden_size)
        return out, hid


class CopyNetDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dropout=0.2):
        super(CopyNetDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        # self.attention = Attention(enc_hidden_size, dec_hidden_size)
        self.rnn = nn.GRU(2 * enc_hidden_size + embed_size, dec_hidden_size, batch_first=True)
        self.out = nn.Linear(dec_hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ctx, ctx_lengths, y, y_lengths, hid):
        sorted_len, sorted_idx = y_lengths.sort(0, descending=True)
        y_sorted = y[sorted_idx.long()]
        hid = hid[:, sorted_idx.long()]

        y_sorted = self.dropout(self.embed(y_sorted))  # batch_size, output_length, embed_size

        emb = self.embed(y_sorted)
        # emb: [batch, seq_len, embed_size]
        for emb_t in emb.split(1):
        # 单个 embedding 输入 y(t_1)

        pass
