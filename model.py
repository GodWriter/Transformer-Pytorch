import math
import numpy as np

import torch
import torch.nn as nn

from dataloader import *


D_MODEL = 512  # 词向量维度
D_FF = 2048  # 全连接层维度
D_K = D_V = 64  # Q，K矩阵维度
N_LAYERS = 6  # 编码层和解码层层数
N_HEADS = 8  # 多头注意力个数


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        位置编码函数
        :param d_model: 词嵌入的维度
        :param max_len: 句子的最长长度
        """

        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # pe表示了一个句子中所有词向量的集合
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0)/d_model))  # 负号是为了后面计算的时候直接乘；log(x) 是以自然底数 e 来计算

        pe[:, 0::2] = torch.sin(position * div_term)  # div_term.shape = 1/2 max_len
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # 在内存中定一个常量，同时，模型保存和加载的时候可以写入和读出

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]  # x->[seq_len, batch_size, d_model]
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    """
    由于各句子长度不一样，故需要通过PAD将所有句子填充到指定长度；
    故用于填充的PAD在句子中无任何含义，无需注意力关注；
    注意力掩码函数，可用于屏蔽单词位置为PAD的位置，将注意力放在其他单词上。
    :param seq_q: [batch_size, seq_len]
    :param seq_k:  [batch_size, seq_len]
    """

    batch_size, len_q = seq_q.size()
    _, len_k = seq_k.size()

    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], 0代表PAD，eq(0)返回和seq_k同等维度的矩阵
                                                   # 若是seq_k某个位置上的元素为0，那么该位置为True，否则为False
                                                   # [1, 2, 3, 0] -> [F, F, F, T]

    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


def get_attn_subsequence_mask(seq):
    """
    用于Decoder，屏蔽未来时刻单词的信息
    :param seq: [batch_size, tgt_len]
    """

    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]

    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 上三角矩阵，np.triu制造上三角矩阵，k为正表示上移对角线，为负表示下移对角线
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()

    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        """
        :param Q:  [batch_size, n_heads, len_q, d_k]
        :param K:  [batch_size, n_heads, len_k, d_k]
        :param V:  [batch_size, n_heads, len_v(=len_k), d_v]
        :param attn_mask:  [batch_size, n_heads, seq_len, seq_len]
        """

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(D_K)  # [batch_size, n_heads, len_q, len_k]
        scores.masked_fill(attn_mask, -1e9)  # 给mask位置为True的地方，赋值-1e9

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]

        return context, attn


class MultiHeadAttention(nn.Module):
    """
    Encoder Layer 调用一次，传入 input_Q, input_K, input_V
    Decoder Layer 第一次调用，传入 dec_inputs, dec_inputs, dec_inputs
    Decoder Layer 第二次调用，传入 dec_outputs, enc_outputs, enc_outputs
    """

    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(D_MODEL, D_K * N_HEADS, bias=False)
        self.W_K = nn.Linear(D_MODEL, D_K * N_HEADS, bias=False)
        self.W_V = nn.Linear(D_MODEL, D_V * N_HEADS, bias=False)
        self.fc = nn.Linear(N_HEADS * D_V, D_MODEL, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        """
        :param input_Q: [batch_size, len_q, d_model]
        :param input_K: [batch_size, len_k, d_model]
        :param input_V: [batch_size, len_v(=len_k), d_model]
        :param attn_mask:  [batch_size, seq_len, seq_len]
        """

        residual, batch_size = input_Q, input_Q.size(0)

        Q = self.W_Q(input_Q).view(batch_size, -1, N_HEADS, D_K).transpose(1, 2)  # [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, N_HEADS, D_K).transpose(1, 2)  # [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, N_HEADS, D_V).transpose(1, 2)  # [batch_size, n_heads, len_v(=len_k), d_k]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, N_HEADS, 1, 1)  # [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, N_HEADS * D_V)  # [batch_size, len_q, n_heads * d_v]
                                                                                  # 多头注意力机制得到的特征直接拼接在一起

        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(D_MODEL).cuda()(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    """
    做两次线性变换，残差连接后，再加一个Layer Norm
    """

    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(D_MODEL, D_FF, bias=False),
                                nn.ReLU(),
                                nn.Linear(D_FF, D_MODEL, bias=False))

    def forward(self, inputs):
        """
        :param inputs: [batch_size, seq_len, d_model]
        """

        residual = inputs
        output = self.fc(inputs)

        return nn.LayerNorm(D_MODEL).cuda()(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()

        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        :param enc_inputs: [batch_size, src_len, d_model]
        :param enc_self_attn_mask:  [batch_size, src_len, src_len]
        """

        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)  # [batch_size, src_len, d_model]

        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size):
        super(Encoder, self).__init__()

        self.src_emb = nn.Embedding(src_vocab_size, D_MODEL)
        self.pos_emb = PositionalEncoding(D_MODEL)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(N_LAYERS)])

    def forward(self, enc_inputs):
        """
        :param enc_inputs: [batch_size, src_len]
        """

        enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)  # [batch_size, src_len, src_len]

        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        return enc_outputs, enc_self_attns


class DecoderLayer(nn.Module):
    """
    DecoderLayer中将两次调用MultiHeadAttention：
        1）计算Decoder Input的self-attention，得到dec_outputs
        2）将dec_outputs作为生成Q的元素，将enc_outputs作为生成K和V的元素，再调用一次MultiHeadAttention，得到Encoder和Decoder Layer之间的context vector
    最后将dec_outputs做一次维度变换
    """

    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        :param dec_inputs: [batch_size, tgt_len, d_model]
        :param enc_outputs: [batch_size, src_len, d_model]
        :param dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        :param dec_env_attn_mask:  [batch_size, tgt_len, src_len]
        """

        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)

        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)  # [batch_size, tgt_len, d_model]

        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size):
        super(Decoder, self).__init__()

        self.tgt_emb = nn.Embedding(tgt_vocab_size, D_MODEL)
        self.pos_emb = PositionalEncoding(D_MODEL)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(N_LAYERS)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        """
        :param dec_inputs: [batch_size, tgt_len]
        :param enc_inputs: [batch_size, src_len]
        :param enc_outputs: [batch_size, src_len, d_model]
        """

        # 1. 分别得到词向量和位置向量并相加
        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).cuda()  # [batch_size, tgt_len, d_model]

        # 2. 得到屏蔽PAD的掩码
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).cuda()  # [batch_size, tgt_len, tgt_len]

        # 3. 未来时刻信息的掩码，相加后大于0的位置，要么为PAD无需关注的位置，要么为未来时刻信息的位置。若该位置大于0，那么值为1，否则为0
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).cuda()  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask.type(torch.uint8) + dec_self_attn_subsequence_mask.type(torch.uint8)), 0).cuda()  # [batch_size, tgt_len, tgt_len]

        # 4. encoder输入和Decoder输入之间的掩码操作，PAD为0，则掩盖
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)  # [batch_size, tgt_len, src_len]

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model]
            # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
            # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)

            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)

        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size):
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_vocab_size).cuda()
        self.decoder = Decoder(tgt_vocab_size).cuda()
        self.projection = nn.Linear(D_MODEL, tgt_vocab_size, bias=False).cuda()

    def forward(self, enc_inputs, dec_inputs):
        """
        :param enc_inputs: [batch_size, src_len]
        :param dec_inputs: [batch_size, tgt_len]
        """

        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)

        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns


if __name__ == '__main__':
    inputs, src_vocab_size, tgt_vocab_size, idx2word = create_data()
    enc_inputs, dec_inputs, dec_outputs = make_data(*inputs)

    data_loader = Data.DataLoader(dataset=MyDataSet(enc_inputs, dec_inputs, dec_outputs),
                                  batch_size=2,
                                  shuffle=True)

    transformer = Transformer(src_vocab_size, tgt_vocab_size)

    for i, data in enumerate(data_loader):
        outputs = transformer(data[0].cuda(), data[1].cuda())
        print("outputs: ", outputs)
