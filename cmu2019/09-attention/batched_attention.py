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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dropout=0.2):
        super(Encoder, self).__init__()
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
        # hid (layers*directions) x batch x dim.
        # We need to convert it to layers x batch x (directions*dim).
        hid = torch.cat((hid[-2], hid[-1]), dim=1)
        # add layer=1 dim
        hid = torch.tanh(self.fc(hid)).unsqueeze(0)
        return out, hid


class Attention(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super(Attention, self).__init__()

        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

        self.linear_in = nn.Linear(enc_hidden_size * 2, dec_hidden_size, bias=False)
        self.linear_out = nn.Linear(enc_hidden_size * 2 + dec_hidden_size, dec_hidden_size)

    def forward(self, output, context, mask):
        # output: batch_size, output_len, dec_hidden_size
        # context: batch_size, context_len, 2*enc_hidden_size

        batch_size = output.size(0)
        output_len = output.size(1)
        input_len = context.size(1)

        context_in = self.linear_in(context.view(batch_size * input_len, -1)).view(
            batch_size, input_len, -1)  # batch_size, context_len, dec_hidden_size

        # context_in.transpose(1,2): batch_size, dec_hidden_size, context_len
        # output: batch_size, output_len, dec_hidden_size
        attn = torch.bmm(output, context_in.transpose(1, 2))
        # batch_size, output_len, context_len

        attn.data.masked_fill(mask, -1e6)

        attn = F.softmax(attn, dim=2)
        # attn: batch_size, output_len, context_len
        # context: batch_size, context_len, 2*enc_hidden_size
        context = torch.bmm(attn, context)
        # batch_size, output_len, 2*enc_hidden_size

        output = torch.cat((context, output), dim=2)  # batch_size, output_len, enc_hidden_size*2 + dec_hidden_size

        output = output.view(batch_size * output_len, -1)
        output = torch.tanh(self.linear_out(output))
        output = output.view(batch_size, output_len, -1)
        return output, attn


# decoder会根据已经翻译的句子内容，和context vectors，来决定下一个输出的单词
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(enc_hidden_size, dec_hidden_size)
        self.rnn = nn.GRU(embed_size, dec_hidden_size, batch_first=True)
        self.out = nn.Linear(dec_hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def create_mask(x_len, y_len):
        # a mask of shape x_len * y_len
        # device = x_len.device
        max_x_len = x_len.max()
        max_y_len = y_len.max()
        x_mask = torch.arange(max_x_len, device=x_len.device)[None, :] < x_len[:, None]
        y_mask = torch.arange(max_y_len, device=x_len.device)[None, :] < y_len[:, None]
        mask = (1 - x_mask[:, :, None] * y_mask[:, None, :]).byte()
        return mask

    def forward(self, ctx, ctx_lengths, y, y_lengths, hid):
        sorted_len, sorted_idx = y_lengths.sort(0, descending=True)
        y_sorted = y[sorted_idx.long()]
        hid = hid[:, sorted_idx.long()]

        y_sorted = self.dropout(self.embed(y_sorted))  # batch_size, output_length, embed_size

        packed_seq = nn.utils.rnn.pack_padded_sequence(y_sorted, sorted_len.long(), batch_first=True)
        out, hid = self.rnn(packed_seq, hid)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        _, original_idx = sorted_idx.sort(0, descending=False)
        output_seq = unpacked[original_idx.long()].contiguous()
        hid = hid[:, original_idx.long()].contiguous()

        mask = self.create_mask(y_lengths, ctx_lengths)

        output, attn = self.attention(output_seq, ctx, mask)
        output = F.log_softmax(self.out(output), -1)

        return output, hid, attn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, x_lengths, y, y_lengths):
        encoder_out, hid = self.encoder(x, x_lengths)
        output, hid, attn = self.decoder(ctx=encoder_out,
                                         ctx_lengths=x_lengths,
                                         y=y,
                                         y_lengths=y_lengths,
                                         hid=hid)
        return output, attn

    def translate(self, x, x_lengths, y, max_length=100):
        encoder_out, hid = self.encoder(x, x_lengths)
        preds = []
        batch_size = x.shape[0]
        attns = []
        for i in range(max_length):
            output, hid, attn = self.decoder(ctx=encoder_out,
                                             ctx_lengths=x_lengths,
                                             y=y,
                                             y_lengths=torch.ones(batch_size).long().to(y.device),
                                             hid=hid)
            y = output.max(2)[1].view(batch_size, 1)
            preds.append(y)
            attns.append(attn)
        return torch.cat(preds, 1), torch.cat(attns, 1)


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


dropout = 0.2
embed_size = hidden_size = 100
encoder = Encoder(vocab_size=nwords_src,
                  embed_size=embed_size,
                  enc_hidden_size=hidden_size,
                  dec_hidden_size=hidden_size,
                  dropout=dropout)
decoder = Decoder(vocab_size=nwords_trg,
                  embed_size=embed_size,
                  enc_hidden_size=hidden_size,
                  dec_hidden_size=hidden_size,
                  dropout=dropout)
model = Seq2Seq(encoder, decoder)
model = model.to(device)
loss_fn = LanguageModelCriterion().to(device)
optimizer = torch.optim.Adam(model.parameters())

BATCH_SIZE = 16


def train_method(model, data, num_epochs=20):
    data = gen_examples(data, BATCH_SIZE)
    for epoch in range(num_epochs):
        start = time.time()
        model.train()
        total_num_words = total_loss = 0.
        for it, (mb_x, mb_x_len, mb_y, mb_y_len) in enumerate(data):
            mb_x = torch.Tensor(mb_x).long().to(device)

            mb_x_len = torch.Tensor(mb_x_len).long().to(device)
            mb_input = torch.Tensor(mb_y[:, :-1]).long().to(device)
            mb_output = torch.Tensor(mb_y[:, 1:]).long().to(device)
            mb_y_len = torch.Tensor(mb_y_len - 1).long().to(device)
            mb_y_len[mb_y_len <= 0] = 1

            mb_pred, attn = model(mb_x, mb_x_len, mb_input, mb_y_len)

            mb_out_mask = torch.arange(mb_y_len.max().item(), device=device)[None, :] < mb_y_len[:, None]
            mb_out_mask = mb_out_mask.float().to(device)

            loss = loss_fn(mb_pred, mb_output, mb_out_mask)

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

        mb_pred, attn = model(mb_x, mb_x_len, mb_input, mb_y_len)

        mb_out_mask = torch.arange(mb_y_len.max().item(), device=device)[None, :] < mb_y_len[:, None]
        mb_out_mask = mb_out_mask.float()

        loss = loss_fn(mb_pred, mb_output, mb_out_mask)

        num_words = torch.sum(mb_y_len).item()
        total_loss += loss.item() * num_words
        total_num_words += num_words
    print("dev loss/word=%.4f,  ppl=%.4f, time=%.2fs" % (
        total_loss / total_num_words, math.exp(total_loss / total_num_words), time.time() - start))
    # print("Epoch", epoch, "Training loss", total_loss / total_num_words)


train_method(model, train)
