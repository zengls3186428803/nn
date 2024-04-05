import math

import torch
from torch import Tensor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
