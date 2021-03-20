import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        :param d_model: 词嵌入的维度
        :param max_len: 句子的最长长度
        """

        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # pe表示了一个句子中所有词向量的集合
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0)/d_model))  # 负号是为了后面计算的时候直接乘；log(x) 是以自然底数 e 来计算

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # 在内存中定一个常量，同时，模型保存和加载的时候可以写入和读出

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]  # x->[seq_len, batch_size, d_model]
        return self.dropout(x)