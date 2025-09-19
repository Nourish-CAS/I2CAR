import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        #
        # pe = torch.zeros(max_len, d_model)
        # position = torch.arange(0, max_len).unsqueeze(1).float()
        # div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        #
        # # 对奇偶位置分别进行 sin 和 cos 操作
        # pe[:, 0::2] = torch.sin(position * div_term)
        # if d_model % 2 == 0:
        #     pe[:, 1::2] = torch.cos(position * div_term)
        # else:
        #     pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])  # 方法 1：直接调整索引
        #
        # self.register_buffer('pe', pe.unsqueeze(0))


    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        # self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
        #                            kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='zeros', bias=False)
        # self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
        #                            kernel_size=3, padding=1, padding_mode='zeros', bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):

        # print("Input x before embedding:", x)
        # print("Input x before embedding - min:", x.min(), "max:", x.max())
        # assert not torch.isnan(x).any(), "Input data contains NaN values"

        # x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        x = self.tokenConv(x.permute(0, 2, 1))
        # print("Input x after conv - min:", x.min(), "max:", x.max())
        # print("Does x contain NaN after conv?", torch.isnan(x).any())

        x = x.transpose(1, 2)
        # with torch.no_grad():
        #     print("Conv1d weights - min:", self.tokenConv.weight.min(), "max:", self.tokenConv.weight.max())
        #     print("Conv1d weights contain NaN?", torch.isnan(self.tokenConv.weight).any())

        # print("Input x After embedding:", x)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.05):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x = self.value_embedding(x) + self.position_embedding(x)
        value_emb = self.value_embedding(x)
        pos_emb = self.position_embedding(x)
        # print("Value Embedding:", value_emb)
        # print("Position Embedding:", pos_emb)
        x = value_emb + pos_emb
        return self.dropout(x)
    

    

