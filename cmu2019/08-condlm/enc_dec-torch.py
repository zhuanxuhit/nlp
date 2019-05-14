# iter 0: train loss/word=67.3370,  ppl=175417269135850746718049009664.0000, time=52.02s
# iter 0: dev loss/word=76.0325, ppl=1048341723867340797217467051016192.0000 (word/sec=6403.27)
import torch
from collections import defaultdict
from torch import nn, cuda, optim
from torch.nn import functional
import random
import time
import math


# import torch.nn.functional as F

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

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters
EMBED_SIZE = 64
HIDDEN_SIZE = 128
BATCH_SIZE = 16


class PlainEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(PlainEncoder, self).__init__()
        self.hidden_size = hidden_size
        # embed_size =
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # uniform initialization
        torch.nn.init.uniform_(self.embedding.weight, -0.25, 0.25)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, inputs: torch.Tensor):
        # x has size(seq_len)
        inputs = inputs.view((1, -1))  # batch_size, seq_len
        embedded = self.embedding(inputs)  # batch_size, seq_len, embed_size
        # output(batch, seq_len, num_directions * hidden_size)
        # hidden(num_layers * num_directions, batch, hidden_size)
        output, hidden = self.gru(embedded)
        return output, hidden


class PlainDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(PlainDecoder, self).__init__()
        # self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # uniform initialization
        torch.nn.init.uniform_(self.embedding.weight, -0.25, 0.25)

        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=2)  # batch_size, seq_len, vocab_size

    def forward(self, inputs, hidden):
        """
        :param inputs:
        :param hidden: encoder的hidden_state
        :return:
        """
        # x has size(seq_len)
        inputs = inputs.view((1, -1))  # batch_size, seq_len
        embedded = self.embedding(inputs)  # batch_size, seq_len, embed_size
        output, hidden = self.gru(embedded, hidden)
        output = self.softmax(self.out(output))
        # batch_size, seq_len, vocab_size
        return output, hidden


MAX_LEN = 100


class PlainSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(PlainSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        _, hid = self.encoder(src)
        # output_has size (batch_size=1, seq_len, vocab_size)
        output, _ = self.decoder(trg, hid)
        output = output.squeeze(0)  # 去除batch_size
        return output

    def translate(self, src):
        # encoder_out(batch, seq_len, num_directions * hidden_size)
        # encoder_hidden(batch, num_layers * num_directions, hidden_size)
        encoder_out, encoder_hidden = self.encoder(src)
        decoded_words = []
        decoder_input = torch.Tensor([sos_trg]).type(dtype)
        decoder_hidden = encoder_hidden

        for i in range(MAX_LEN):
            # decoder_output(batch_size, seq_len, vocab_size)
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_input.data.topk(1)
            # topi (batch_size=1, 1, 1)
            if topi.view(-1).item() == eos_trg:
                decoded_words.append("</s>")
                break
            else:
                decoded_words.append(i2w_trg[topi.view(-1).item()])
        return decoded_words


# masked cross entropy loss
class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, logits, target):
        # output: seq_len * vocab_size
        # target: seq_len, 1
        target = target.view(-1, 1)
        logits = -logits.gather(1, target)
        loss = torch.sum(logits.view(-1))
        return loss


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
    model.cuda()


def calc_sent_loss(sent):
    src = sent[0]
    trg = sent[1]

    x = torch.Tensor(src).type(dtype)
    y = torch.Tensor(trg[:-1]).type(dtype)
    target = torch.Tensor(trg[1:]).type(dtype)
    logits = model(x, y)
    # logits (seq_len, vocab_size)
    # target (seq_len)
    loss = criterion(logits, target)
    return loss

last_dev = 1e20
best_dev = 1e20

for ITER in range(5):
    print("started iter %r" % ITER)
    # Perform training
    random.shuffle(train)
    # set the model to training mod
    model.train()
    train_words, train_loss = 0, 0.0
    start = time.time()
    for sent_id, sent in enumerate(train):
        my_loss = calc_sent_loss(sent)
        train_loss += my_loss.item()
        train_words += len(sent[1])  # target_loss
        # Taking the step after calculating loss for all the words in the sentence
        optimizer.zero_grad()
        my_loss.backward()
        optimizer.step()
        if (sent_id + 1) % 5000 == 0:
            print("--finished %r sentences" % (sent_id + 1))
    print("iter %r: train loss/word=%.4f,  ppl=%.4f, time=%.2fs" % (
        ITER, train_loss / train_words, math.exp(train_loss / train_words), time.time() - start))

    # Evaluate on dev set
    # set the model to evaluation mode
    model.eval()
    dev_words, dev_loss = 0, 0.0
    start = time.time()
    for sent_id, sent in enumerate(dev):
        my_loss = calc_sent_loss(sent)
        dev_loss += my_loss.item()
        dev_words += len(sent[1])

    # Keep track of the development accuracy and reduce the learning rate if it got worse
    # if last_dev < dev_loss:
    #     optimizer.learning_rate /= 2
    # last_dev = dev_loss

    # Keep track of the best development accuracy, and save the model only if it's the best one
    # if best_dev > dev_loss:
    #     torch.save(model, "model.pt")
    #     best_dev = dev_loss

    # Save the model
    print("iter %r: dev loss/word=%.4f, ppl=%.4f (word/sec=%.2f)" % (
        ITER, dev_loss / dev_words, math.exp(dev_loss / dev_words), dev_words / (time.time() - start)))

    # Generate a few sentences
    for i in range(5):
        x = torch.Tensor(dev[i][0]).type(dtype)
        sent = model.translate(x)
        print(sent)
