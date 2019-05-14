# 针对 enc_dec-torch 的单句子训练的版本，本版本提供mini-batch
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


def get_minibatches(n, minibatch_size, shuffle=True):
    idx_list = np.arange(0, n, minibatch_size)  # [0, 1, ..., n-1]
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    for idx in idx_list:
        minibatches.append(np.arange(idx, min(idx + minibatch_size, n)))
    return minibatches


def prepare_data(seqs):
    lengths = [len(seq) for seq in seqs]
    n_samples = len(seqs)
    max_len = np.max(lengths)

    x = np.zeros((n_samples, max_len)).astype('int32')
    x_lengths = np.array(lengths).astype("int32")
    for idx, seq in enumerate(seqs):
        x[idx, :lengths[idx]] = seq
    return x, x_lengths  # x_mask


def gen_examples(data, batch_size):
    minibatches = get_minibatches(len(data), batch_size)
    all_ex = []
    for minibatch in minibatches:
        mb_src_sentences = [data[t][0] for t in minibatch]
        mb_trg_sentences = [data[t][1] for t in minibatch]
        mb_x, mb_x_len = prepare_data(mb_src_sentences)
        mb_y, mb_y_len = prepare_data(mb_trg_sentences)
        all_ex.append((mb_x, mb_x_len, mb_y, mb_y_len))
    return all_ex


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Model parameters
EMBED_SIZE = 64
HIDDEN_SIZE = 128
BATCH_SIZE = 16


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PlainEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, dropout=0.2):
        super(PlainEncoder, self).__init__()
        self.hidden_size = hidden_size
        # embed_size =
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # uniform initialization
        torch.nn.init.uniform_(self.embedding.weight, -0.25, 0.25)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, lengths: torch.Tensor):
        """
        :param inputs: (batch_size, seq_len)
        :param lengths: batch_size
        :return:
        """
        # lengths (batch_size)
        sorted_len, sorted_idx = lengths.sort(0, descending=True)
        inputs_sorted = inputs[sorted_idx.long()]
        embedded = self.dropout(self.embedding(inputs_sorted))  # batch_size, seq_len, embed_size
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_len, batch_first=True)
        # packed_out(batch, seq_len, num_directions * hidden_size)
        # hid(num_layers * num_directions, batch, hidden_size)
        packed_out, hid = self.gru(packed_embedded)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        _, original_idx = sorted_idx.sort(0, descending=False)
        out = out[original_idx]
        hid = hid[:, original_idx]
        # The encoder hidden is  (layers*directions) x batch x dim.
        # We need to convert it to layers x batch x (directions*dim).
        # 此处 directions = 1, layers = 1
        return out, hid


class PlainDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, dropout=0.2):
        super(PlainDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # uniform initialization
        torch.nn.init.uniform_(self.embedding.weight, -0.25, 0.25)

        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        # self.softmax = nn.LogSoftmax(dim=-1)  # batch_size, seq_len, vocab_size

    def forward(self, inputs, lengths, hidden):
        """
        :param inputs: (batch_size, seq_len)
        :param lengths: (batch_size)
        :param hidden: encoder的hidden_state
        :return:
        """
        # lengths (batch_size)
        sorted_len, sorted_idx = lengths.sort(0, descending=True)
        inputs_sorted = inputs[sorted_idx.long()]
        embedded = self.dropout(self.embedding(inputs_sorted))  # batch_size, seq_len, embed_size
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_len, batch_first=True)
        packed_out, hid = self.gru(packed_embedded, hidden)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        _, original_idx = sorted_idx.sort(0, descending=False)
        out = out[original_idx]
        hid = hid[:, original_idx]
        # batch_size, seq_len, vocab_size
        output = F.log_softmax(self.out(out), -1)

        return output, hid


class PlainSeq2Seq(nn.Module):
    def __init__(self, encoder: PlainEncoder, decoder: PlainDecoder):
        super(PlainSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, x_lengths, y, y_lengths):
        """

        :param x: batch_size, seq_len
        :param x_lengths:
        :param y:
        :param y_lengths:
        :return:
        """
        encoder_out, hid = self.encoder(x, x_lengths)
        output, hid = self.decoder(inputs=y,
                                   lengths=y_lengths,
                                   hidden=hid)
        return output

    def translate(self, x, x_lengths, y, max_length=10):
        encoder_out, hid = self.encoder(x, x_lengths)
        preds = []
        batch_size = x.shape[0]
        for i in range(max_length):
            output, hid = self.decoder(y=y,
                                       y_lengths=torch.ones(batch_size),
                                       hid=hid)
            # output (batch_size, 1, vocab_size)
            # max return (val, index)
            y = output.max(2)[1].view(batch_size, 1)
            preds.append(y)
        # preds (batch_size, seq_len)
        return torch.cat(preds, 1)


# masked cross entropy loss
class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, logits, target, mask):
        # input: (batch_size , seq_len, vocab_size
        logits = logits.view(-1, logits.size(2))  # batch_size*seq_len, vocab_size
        # target: batch_size , seq_len, 1
        # mask: batch_size , seq_len, 1
        target = target.view(-1, 1)
        mask = mask.view(-1, 1)
        output = -logits.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


# 定义模型
encoder = PlainEncoder(vocab_size=nwords_src, embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE)
decoder = PlainDecoder(vocab_size=nwords_trg, embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE)
model = PlainSeq2Seq(encoder, decoder)
criterion = LanguageModelCriterion()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

dtype = torch.LongTensor
use_cuda = torch.cuda.is_available()

if use_cuda:
    dtype = torch.cuda.LongTensor
    model.to(device)


def train_method(model, data, num_epochs=20):
    data = gen_examples(data, BATCH_SIZE)
    for epoch in range(num_epochs):
        start = time.time()
        model.train()
        total_num_words = total_loss = 0.
        for it, (mb_x, mb_x_len, mb_y, mb_y_len) in enumerate(data):
            mb_x = torch.Tensor(mb_x).to(device).long()

            mb_x_len = torch.Tensor(mb_x_len).to(device).long()
            mb_input = torch.Tensor(mb_y[:, :-1]).to(device).long()
            mb_output = torch.Tensor(mb_y[:, 1:]).to(device).long()
            mb_y_len = torch.Tensor(mb_y_len - 1).to(device).long()
            mb_y_len[mb_y_len <= 0] = 1

            mb_pred = model(mb_x, mb_x_len, mb_input, mb_y_len)

            mb_out_mask = torch.arange(mb_y_len.max().item(), device=device)[None, :] < mb_y_len[:, None]
            mb_out_mask = mb_out_mask.float()

            loss = criterion(mb_pred, mb_output, mb_out_mask)

            num_words = torch.sum(mb_y_len).item()
            total_loss += loss.item() * num_words
            total_num_words += num_words

            # 更新模型
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()

            if (it + 1) % 1000 == 0:
                print("Epoch", epoch, "iteration", it + 1, "loss", loss.item())

        print("Epoch %r: train loss/word=%.4f,  ppl=%.4f, time=%.2fs" % (
            epoch, total_loss / total_num_words, math.exp(total_loss / total_num_words), time.time() - start))
        # print("Epoch", epoch, "Training loss", total_loss / total_num_words)
        if epoch % 5 == 0:
            eval_method(model, dev)


def eval_method(model, data):
    data = gen_examples(data, BATCH_SIZE)
    model.eval()
    start = time.time()
    total_num_words = total_loss = 0.
    for it, (mb_x, mb_x_len, mb_y, mb_y_len) in enumerate(data):
        mb_x = torch.Tensor(mb_x).to(device).long()

        mb_x_len = torch.Tensor(mb_x_len).to(device).long()
        mb_input = torch.Tensor(mb_y[:, :-1]).to(device).long()
        mb_output = torch.Tensor(mb_y[:, 1:]).to(device).long()
        mb_y_len = torch.Tensor(mb_y_len - 1).to(device).long()
        mb_y_len[mb_y_len <= 0] = 1

        mb_pred = model(mb_x, mb_x_len, mb_input, mb_y_len)

        mb_out_mask = torch.arange(mb_y_len.max().item(), device=device)[None, :] < mb_y_len[:, None]
        mb_out_mask = mb_out_mask.float()

        loss = criterion(mb_pred, mb_output, mb_out_mask)

        num_words = torch.sum(mb_y_len).item()
        total_loss += loss.item() * num_words
        total_num_words += num_words
        print("dev loss/word=%.4f,  ppl=%.4f, time=%.2fs" % (
            total_loss / total_num_words, math.exp(total_loss / total_num_words), time.time() - start))
        # print("Epoch", epoch, "Training loss", total_loss / total_num_words)


train_method(model, train)
