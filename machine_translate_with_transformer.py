from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from timeit import default_timer as timer
import torch
import math
from torch import Tensor
import numpy as np

# torch.set_printoptions(threshold=np.inf)
torch.manual_seed(0)

EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_BLOCK_LAYERS = 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 18
learning_rate = 0.0001


# DEVICE = "cpu"


class PositionalEncoder(torch.nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super().__init__()
        self.d_model = d_model  # 每个词向量的维数
        # 根据 pos 和 i 创建一个常量 PE 矩阵
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 使得单词嵌入表示相对大一些
        x = x * math.sqrt(self.d_model)
        # 增加位置常量到单词嵌入表示中
        seq_len = x.size(1)  # 第0维度为batch_size, 第1维度为序列长度，第2维度为词向量的维数
        # self.pe[:, :seq_len]等价于self.pe[:, :seq_len，:],  pytorch应该重载了[]，[,,]相当于[][][]
        x = x + torch.tensor(self.pe[:, :seq_len], requires_grad=False)  # 这一步应该会对PE广播
        return x


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads  # 总维数应该要能整除注意力头数， d_k为每个注意力头的维数。
        self.h = heads
        self.q_linear = torch.nn.Linear(d_model, d_model).to(DEVICE)  # 可以理解为词向量内部的多个特征之间的组合
        self.k_linear = torch.nn.Linear(d_model, d_model).to(DEVICE)  # 可以理解为词向量内部的多个特征之间的组合
        self.v_linear = torch.nn.Linear(d_model, d_model).to(DEVICE)  # 可以理解为词向量内部的多个特征之间的组合
        self.dropout = torch.nn.Dropout(dropout)
        self.out = torch.nn.Linear(d_model, d_model).to(DEVICE)

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        # src_mask.size = (batch_size, num_head, src_seq_L, src_seq_L)
        # tgt_mask.size = (batch_size, num_head, tgt_seq_L, tgt_seq_L)
        # mem_mask.size = (batch_size, num_head, tgt_seq_L, src_seq_L)
        # 原始掩码 = (batch_size, seq_L) ---> (batch_size, num_head, seq_L, seq_L)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        # print("score.size=", scores.size())
        # 掩盖掉那些为了填补长度增加的单元,使其通过 softmax 计算后为 0
        # if mask is not None:
        #     mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 1, float("-inf"))  # 如果为1,就填充成一个趋近于负无穷。这样softmax之后为0
        # 把最后一维归一化。 相当于按行归一化。q,k,v的size都是(bs, seq_L, d_k)
        # scores 的维度是(batch_size, seq_L, seq_L)
        scores = torch.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)
        output = torch.matmul(scores, v)  # 最低的两维做矩阵乘法。
        return output

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)  # batch_size
        # 进行线性操作划分为成 h 个头， -1那一维就是序列长度seq_L, -1表示自动计算出来。h是head_num，d_k是head_dim
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # 矩阵转置（bs, seq_L, head_num, head_dim）-> (bs, head_num, seq_L, head_dim), 为了用最后两维后面做矩阵乘法。
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # print("q.size=", q.size())
        # print("k.size=", k.size())
        # print("v.size=", v.size())
        # 计算 attention
        att = self.attention(q, k, v, self.d_k, mask, self.dropout)
        # print("att.size=", att.size())
        # 连接多个头并输入到最后的线性层
        # 注意力层输出是计算的到的新tensor, 是连续的。 转置之后又不连续了，所以需要contiguous一下。
        concat = att.transpose(1, 2).contiguous().view(bs, -1, self.d_model)  # -1 那一维对应seq_L
        output = self.out(concat)
        return output


class FeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        # d_ff 默认设置为 2048
        self.linear_1 = torch.nn.Linear(d_model, d_ff).to(DEVICE)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear_2 = torch.nn.Linear(d_ff, d_model).to(DEVICE)

    def forward(self, x):
        x = self.dropout(torch.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class NormLayer(torch.nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model
        # 层归一化包含两个可以学习的参数
        self.alpha = torch.nn.Parameter(torch.ones(self.size)).to(DEVICE)
        self.bias = torch.nn.Parameter(torch.zeros(self.size)).to(DEVICE)
        self.eps = eps

    def forward(self, x):
        # 这里的减法和除法是逐元素操作。# -1表示沿最后一个维度
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = NormLayer(d_model)
        self.norm_2 = NormLayer(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = torch.nn.Dropout(dropout)
        self.dropout_2 = torch.nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class Encoder(torch.nn.Module):
    # N为EncoderLayer的数量
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)  # max_seq_len=默认
        self.layers = [EncoderLayer(d_model, heads, dropout) for i in range(N)]
        self.norm = NormLayer(d_model)

    def forward(self, src, src_mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, src_mask)
        return self.norm(x)


class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = NormLayer(d_model)
        self.norm_2 = NormLayer(d_model)
        self.norm_3 = NormLayer(d_model)
        self.dropout_1 = torch.nn.Dropout(dropout)
        self.dropout_2 = torch.nn.Dropout(dropout)
        self.dropout_3 = torch.nn.Dropout(dropout)
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    # src_mask是方的，trg_mask是下(上)三角的
    def forward(self, x, mem, mem_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, mem, mem, mem_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


class Decoder(torch.nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)  # max_seq_len is default
        self.layers = [DecoderLayer(d_model, heads, dropout) for i in range(N)]
        self.norm = NormLayer(d_model)

    def forward(self, trg, mem, mem_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mem, mem_mask, trg_mask)
        return self.norm(x)


class Transformer(torch.nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, N, heads, dropout=0):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab_size, d_model, N, heads, dropout)
        self.out = torch.nn.Linear(d_model, trg_vocab_size)

    def forward(self, src, trg, src_mask, mem_mask, trg_mask):
        mem = self.encoder(src, src_mask)
        d_output = self.decoder(trg, mem, mem_mask, trg_mask)
        output = self.out(d_output)
        return output

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.encoder(src, src_mask)

    def decode(self, tgt: Tensor, mem: Tensor, mem_mask, tgt_mask: Tensor):
        return self.decoder(tgt, mem, mem_mask, tgt_mask)


# We need to modify the URLs for the dataset since the links to the original dataset are broken
# Refer to https://github.com/pytorch/text/issues/1756#issuecomment-1163664163 for more info
multi30k.URL[
    "train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL[
    "valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

# Place-holders
token_transform = {}
vocab_transform = {}

token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')


# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])


# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # Training data Iterator
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    # Create torchtext's Vocab object
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

# Set ``UNK_IDX`` as the default index. This index is returned when the token is not found.
# If not set, it throws ``RuntimeError`` when the queried token is not found in the Vocabulary.
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[ln].set_default_index(UNK_IDX)

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)


# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))


# ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln],  # Tokenization
                                               vocab_transform[ln],  # Numericalization
                                               tensor_transform)  # Add BOS/EOS and create tensor


# function to collate data samples into batch tensors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    src_batch = src_batch.transpose(0, 1)
    tgt_batch = tgt_batch.transpose(0, 1)
    return src_batch, tgt_batch


def generate_square_subsequent_mask(sz):
    return (torch.ones((sz, sz)) - torch.Tensor(torch.tril(torch.ones((sz, sz))))).to(DEVICE)


def create_mask(src, tgt, num_head):
    batch_size = src.shape[0]
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE)

    src_padding_mask = (src == PAD_IDX)
    tgt_padding_mask = (tgt == PAD_IDX)

    mem_mask = torch.Tensor(src_padding_mask).contiguous().view(batch_size, 1, src_seq_len).expand(-1,
                                                                                                   tgt_seq_len,
                                                                                                   -1)

    src_mask = src_mask.contiguous().view(1, src_seq_len, src_seq_len).expand(batch_size, -1, -1)
    src_padding_mask = torch.Tensor(src_padding_mask).contiguous().view(batch_size, 1, src_seq_len).expand(-1,
                                                                                                           src_seq_len,
                                                                                                           -1)
    tgt_mask = tgt_mask.contiguous().view(1, tgt_seq_len, tgt_seq_len).expand(batch_size, -1, -1)
    tgt_padding_mask = torch.Tensor(tgt_padding_mask).contiguous().view(batch_size, 1, tgt_seq_len).expand(-1,
                                                                                                           tgt_seq_len,
                                                                                                           -1)
    final_src_mask = src_mask.type(torch.bool) + src_padding_mask.type(torch.bool)
    final_tgt_mask = tgt_mask.type(torch.bool) + tgt_padding_mask.type(torch.bool)
    final_mem_mask = mem_mask.type(torch.bool)

    final_src_mask = final_src_mask.view(batch_size, 1, src_seq_len, src_seq_len).expand(-1, num_head, -1, -1).to(
        DEVICE)
    final_tgt_mask = final_tgt_mask.view(batch_size, 1, tgt_seq_len, tgt_seq_len).expand(-1, num_head, -1, -1).to(
        DEVICE)
    final_mem_mask = final_mem_mask.view(batch_size, 1, tgt_seq_len, src_seq_len).expand(-1, num_head, -1, -1).to(
        DEVICE)

    # print("final_src_mask=", final_src_mask)
    # print("final_tgt_mask=", final_tgt_mask)
    # print("final_tgt_mask=", final_mem_mask)
    # print("final_src_mask.size=", final_src_mask.size())
    # print("final_tgt_mask.size=", final_tgt_mask.size())
    # print("final_mem_mask.size=", final_mem_mask.size())
    # print(final_mem_mask)

    return final_src_mask, final_mem_mask, final_tgt_mask


def train_epoch(model, optimizer):
    model: Transformer
    model.train()
    losses = 0
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        tgt_input = tgt[:, :-1]
        src_mask, mem_mask, tgt_mask = create_mask(src, tgt_input, NHEAD)
        logits = model(src, tgt_input, src_mask, mem_mask, tgt_mask)
        # print("logits.size = ", logits.size())
        # print(logits)
        optimizer.zero_grad()
        tgt_out = tgt[:, 1:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_dataloader))


def evaluate(model):
    model.eval()
    losses = 0
    val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        tgt_input = tgt[:, :-1]
        src_mask, mem_mask, tgt_mask = create_mask(src, tgt_input, NHEAD)
        logits = model(src, tgt_input, src_mask, mem_mask, tgt_mask)
        tgt_out = tgt[:, 1:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
    return losses / len(list(val_dataloader))


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask, _, _ = create_mask(src, src, NHEAD)
    src_mask = src_mask.to(DEVICE)
    tgt = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    # print(tgt)
    mem = model.encode(src, src_mask)

    for i in range(max_len - 1):
        mem = mem.to(DEVICE)
        _, mem_mask, tgt_mask = create_mask(src, tgt, NHEAD)
        out = model.decode(tgt=tgt, mem=mem, mem_mask=mem_mask, tgt_mask=tgt_mask)
        # print("out.size()=", out.size())
        # out.size=(batch_size, seq_L, embedding_size) = (1, seq_L, embedding_size)

        # print("out[:, -1].size()=", out[:, -1].size())
        prob = model.out(out[:, -1])  # 查看最后一个词的下一个词。

        _, next_word = torch.max(prob, dim=1)
        # print("next_word.size()=", next_word.size())
        next_word = next_word.item()
        tgt = torch.cat([tgt, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        if next_word == EOS_IDX:
            break
    return tgt


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(1, -1)
    num_tokens = src.shape[1]
    tgt_tokens = greedy_decode(model, src, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>",
                                                                                                         "").replace(
        "<eos>", "")


START_EPOCH = 1
model = Transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, EMB_SIZE, NUM_BLOCK_LAYERS, NHEAD, dropout=0.2)

# 初始化参数
for p in model.parameters():
    if p.dim() > 1:
        torch.nn.init.xavier_uniform_(p)

model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)

for epoch in range(START_EPOCH, NUM_EPOCHS + 1):
    start_time = timer()
    train_loss = train_epoch(model, optimizer)
    end_time = timer()
    val_loss = evaluate(model)
    print((
        f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    print(translate(model, "Eine Gruppe von Menschen steht vor einem Iglu ."))
    print(translate(model, "Mir gefällt die Art, wie Du zu mir kommst, auch wenn Du nur vorbeikommst."))
    print(translate(model, "Wenn du mich liebst, dann werde ich dich lieben"))
:w

