import torch
import math
import torch.nn as nn


class NaiveTransformerLayer(nn.Module):
    def __init__(self):
        super(NaiveTransformerLayer, self).__init__()
        self.dim = 512
        self.att_drop_rate = 0.1
        self.state_drop_rate = 0.5
        self.Wq = nn.Linear(self.dim, self.dim, bias=False)
        self.Wk = nn.Linear(self.dim, self.dim, bias=False)
        self.Wv = nn.Linear(self.dim, self.dim, bias=False)
        self.lm = nn.LayerNorm(self.dim)
        self.ffn1 = nn.Linear(self.dim, self.dim*4)
        self.ffn2 = nn.Linear(self.dim*4, self.dim)
        self.act = nn.GELU()
        self.lm_ffn = nn.LayerNorm(self.dim)
        self.att_drop = nn.Dropout(self.att_drop_rate)
        self.state_drop = nn.Dropout(self.state_drop_rate)

    def SelfAttention(self, x):
        """
        :param x: b*n*d
        :return: b*n*d
        """
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        attention_score = torch.bmm(Q, K.transpose(1, 2))/math.sqrt(self.dim)
        attention_score = nn.Softmax(dim=2)(attention_score)
        attention_score = self.att_drop(attention_score)
        output = torch.bmm(attention_score, V)
        output = self.state_drop(output)
        output = self.lm(x + output)
        return output

    def FFN(self, x):
        hidden = self.act(self.ffn1(x))
        output = self.ffn2(hidden)
        output = self.state_drop(output)
        output = self.lm_ffn(x+output)
        return output

    def forward(self, x):
        """
        :param x: b*n*d
        :return: b*n*d
        """
        x = self.SelfAttention(x)
        x = self.FNN(x)
        return x


class MultiTransformerLayer(nn.Module):
    def __init__(self):
        super(MultiTransformerLayer, self).__init__()
        self.dim = 512
        self.att_drop_rate = 0.1
        self.state_drop_rate = 0.5
        self.num_heads = 12
        self.size_per_head = self.dim // self.num_head   #64
        self.Wq = nn.Linear(self.dim, self.num_heads * self.size_per_head, bias=False)
        self.Wk = nn.Linear(self.dim, self.num_heads * self.size_per_head, bias=False)
        self.Wv = nn.Linear(self.dim, self.num_heads * self.size_per_head, bias=False)
        self.W = nn.Linear(self.num_heads * self.size_per_head, self.dim)
        self.lm = nn.LayerNorm(self.dim)
        self.ffn1 = nn.Linear(self.dim, self.dim*4)
        self.ffn2 = nn.Linear(self.dim*4, self.dim)
        self.act = nn.GELU()
        self.lm_ffn = nn.LayerNorm(self.dim)
        self.att_drop = nn.Dropout(self.att_drop_rate)
        self.state_drop = nn.Dropout(self.state_drop_rate)

    def calc_mask_score(self, attention_mask):
        """
        :param attention_mask: b x n
        :return: b x h x n x n
        """
        mask_score = torch.zeros(attention_mask.size(0), self.num_heads, attention_mask.size(1), attention_mask.size(1))
        mask_score = mask_score + attention_mask[:, None, None, :]
        mask_score = (1.0 - mask_score) * -10000.
        return mask_score

    def SelfAttention(self, x, attention_mask):
        """
        :param x: b*n*d
        Q,K,V  b x n x (h x s)-->b x n x h x s--> b x h x n x s
        attention_mask # b x n
            1 normal token
            0 masked token
        :return: b*n*d
        """

        new_size = x.size()[:-1] + (self.num_heads, self.size_per_head)  # b, n , h, s
        Q = self.Wq(x).view(*new_size).permute(0, 2, 1, 3)
        K = self.Wk(x).view(*new_size).permute(0, 2, 1, 3)
        V = self.Wv(x).view(*new_size).permute(0, 2, 1, 3)
        attention_score = torch.matmul(Q, K.transpose(2, 3))/math.sqrt(self.dim)
        # attention mask here
        attention_score = attention_score + self.calc_mask_score(attention_score)
        attention_score = nn.Softmax(dim=3)(attention_score)
        attention_score = self.att_drop(attention_score)
        output = torch.mm(attention_score, V)
        output = self.W(output.permute(0, 2, 1, 3))  # b x n x d
        output = self.state_drop(output)
        output = self.lm(x + output)
        return output

    def FFN(self, x):
        hidden = self.act(self.ffn1(x))
        output = self.ffn2(hidden)
        output = self.state_drop(output)
        output = self.lm_ffn(x+output)
        return output

    def forward(self, x):
        """
        :param x: b*n*d
        :return: b*n*d
        """
        x = self.SelfAttention(x)
        x = self.FNN(x)
        return x
