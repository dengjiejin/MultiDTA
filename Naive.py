import torch
import torch.nn as nn
import math

class NaiveTransformerLayer(torch.nn.Module):
    def __init__(self, dim=128):
        super(NaiveTransformerLayer, self).__init__()
        self.dim = dim
        self.Wq = nn.Linear(self.dim, self.dim, bias=False)
        self.Wk = nn.Linear(self.dim, self.dim, bias=False)
        self.Wv = nn.Linear(self.dim, self.dim, bias=False)
        self.lm = nn.LayerNorm(self.dim)
        self.ffn1 = nn.Linear(self.dim, self.dim*4)
        self.fnn2 = nn.Linear(self.dim*4, self.dim)
        self.act = nn.GELU()
        self.lm_ffn = nn.LayerNorm(self.dim)

    def SelfAttention(self, x):
        """
        :param x: nxd
        :return: output nxd
        """

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        attention_score = torch.mm(Q, K.transpose(0,1))/math.sqrt(self.dim)
        attention_score = nn.Softmax(dim=1)(attention_score)
        O = torch.mm(attention_score, V)
        O = self.lm(x + O)
        return O

    def FNN(self, x):
        hidden = self.act(self.ffn1(x))
        output = self.fnn2(hidden)
        output = self.lm_ffn(x + output)
        return output

    def forward(self, x):
        """
        :param x:nxd
        :return: nxd
        """
        x = self.SelfAttention(x)
        x = self.FNN(x)
        return x
